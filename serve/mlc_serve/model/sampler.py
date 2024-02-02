import torch
import numpy as np
import structlog
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple
import tvm
from ..engine import (
    SamplingParams,
    SamplingType,
    SAMPLING_EPS,
    LOGPROB_TOP_K_MAX,
    RawLogprobsInfo,
    RawLogprobsInfos,
)

LOG = structlog.stdlib.get_logger(__name__)


def _apply_top_p_top_k(logits, top_ps, top_ks):
    p = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort, dim=-1, index=torch.argsort(logits_idx, dim=-1))
    return logits


# TODO(@sunggg): Test its performance and drop if unnecesary
# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
):
    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        # This allows us to do sampling with replacement by creating
        # num_samples copies of each row in the tensor, and then
        # batch sampling the resulting tensor.
        probs = (
            probs[:, None, :]
            .expand(probs.shape[0], num_samples, probs.shape[1])
            .contiguous()
            .view(-1, probs.shape[1])
        )
    q = torch.empty_like(probs).exponential_(1)
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


def get_logprob_infos(
    i: int,
    logprob_infos: Optional[RawLogprobsInfos],
) -> Optional[RawLogprobsInfos]:
    if logprob_infos is None or logprob_infos[i] is None:
        return None
    return [logprob_infos[i]]


def get_raw_logprob_info(
    logits,
    token_id,
    top_logprobs_num,
) -> RawLogprobsInfo:
    logprobs = torch.log_softmax(logits, dim=-1)
    res_logprob = logprobs[token_id]

    if top_logprobs_num == 0:
        top_logprobs = None
        top_tokens = None
    else:
        assert top_logprobs_num <= LOGPROB_TOP_K_MAX, "Invalid input top_logprobs"
        top_logprobs, top_tokens = torch.topk(
            logprobs, k=top_logprobs_num, dim=-1, largest=True, sorted=True
        )
        top_tokens = top_tokens.cpu().numpy()
        top_logprobs = top_logprobs.cpu().numpy()

    # Set to raw logprob info
    return RawLogprobsInfo(
        current_token_id=token_id,
        current_logprob=res_logprob,
        top_token_ids=top_tokens,
        top_logprobs=top_logprobs,
    )


def get_logprob_indices(
    sampling_params: List[SamplingParams],
    num_seq: int,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    lgp_inds_greedy: List[Tuple[int, int, int]] = []
    lgp_inds_random: List[Tuple[int, int, int]] = []

    g_ind = 0
    r_ind = 0
    for i in range(num_seq):
        sampling_param = sampling_params[i]
        if sampling_param.sampling_type == SamplingType.RANDOM:
            if sampling_param.logprobs:
                lgp_inds_random.append((i, r_ind, sampling_param.top_logprobs))
            r_ind = r_ind + 1
        else:
            if sampling_param.logprobs:
                lgp_inds_greedy.append((i, g_ind, sampling_param.top_logprobs))
            g_ind = g_ind + 1

    return lgp_inds_greedy, lgp_inds_random


def get_raw_logprob_infos(
    logprob_infos: RawLogprobsInfos,
    indices: List[Tuple[int, int, int]],
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> RawLogprobsInfos:
    for i, ind, top_logprobs in indices:
        logprob_infos[i] = get_raw_logprob_info(
            logits[ind],
            token_ids[ind],
            top_logprobs,
        )

    return logprob_infos


def check_logprob_infos(
    logprob_infos: RawLogprobsInfos,
) -> Optional[RawLogprobsInfos]:
    check = False
    for info in logprob_infos:
        if info is not None:
            check = True
            break
    if check:
        return logprob_infos
    return None


# TODO: Add logprob
@dataclass
class SamplingTensors:
    mask_random: torch.Tensor
    mask_greedy: torch.Tensor
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    logit_bias_indices: torch.Tensor
    logit_bias_values: torch.Tensor
    past_output_tokens: torch.Tensor

    @classmethod
    def from_lists(
        cls,
        dtype,
        dev,
        list_mask_random: List[bool],
        list_temperatures: List[float],
        list_top_ps: List[float],
        list_top_ks: List[int],
        list_frequency_penalties: List[float],
        list_presence_penalties: List[float],
        list_repetition_penalties: List[float],
        list_logit_bias_indices: List["long"],
        list_logit_bias_values: List[float],
        list_past_output_tokens: List["long"],
    ):
        # NOTE: Keep `mask_random_t` and `mask_greedy_t` tensors in CPU.
        #       Moving them to gpu showed a small performance regression.
        mask_random = torch.tensor(
            list_mask_random,
            dtype=torch.bool,
            device="cpu",
        )
        mask_greedy = torch.logical_not(
            mask_random,
        )
        temp = torch.tensor(
            list_temperatures,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        top_ps = torch.tensor(
            list_top_ps,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        top_ks = torch.tensor(
            list_top_ks,
            dtype=torch.int,
            device="cpu",
            pin_memory=True,
        )
        frequency_penalties = torch.tensor(
            list_frequency_penalties,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        presence_penalties = torch.tensor(
            list_presence_penalties,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        repetition_penalties = torch.tensor(
            list_repetition_penalties,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        logit_bias_indices = torch.tensor(
            list_logit_bias_indices,
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )
        logit_bias_values = torch.tensor(
            list_logit_bias_values,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        past_output_tokens = torch.tensor(
            list_past_output_tokens,
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )
        return cls(
            mask_random,
            mask_greedy,
            temp.to(device=dev, non_blocking=True),
            top_ps.to(device=dev, non_blocking=True),
            top_ks.to(device=dev, non_blocking=True),
            frequency_penalties.to(device=dev, non_blocking=True),
            presence_penalties.to(device=dev, non_blocking=True),
            repetition_penalties.to(device=dev, non_blocking=True),
            logit_bias_indices.to(device=dev, non_blocking=True),
            logit_bias_values.to(device=dev, non_blocking=True),
            past_output_tokens.to(device=dev, non_blocking=True),
        )


@dataclass
class SamplingMetadata:
    has_random: bool
    has_greedy: bool
    apply_top_p_top_k: bool
    apply_penalty: bool
    apply_bias: bool
    sampling_tensors: SamplingTensors

    @classmethod
    def from_sampling_params(
        cls,
        sampling_params: SamplingParams,
        dtype: torch.dtype,
        dev: str,
        vocab_size: int,
    ):
        list_mask_random = []
        list_temperatures = []
        list_top_ps = []
        list_top_ks = []
        do_top_p = False
        do_top_k = False
        has_random = False
        has_greedy = False
        apply_penalty = False
        apply_bias = False
        list_frequency_penalties = []
        list_presence_penalties = []
        list_repetition_penalties = []
        list_logit_bias_indices = []
        list_logit_bias_values = []
        list_past_output_tokens = []
        for param in sampling_params:
            # Prepare temperature
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            list_temperatures.append(
                param.temperature if param.temperature >= SAMPLING_EPS else 1.0
            )

            if param.sampling_type == SamplingType.RANDOM:
                list_mask_random.append(True)
                has_random |= True
                list_top_ps.append(param.top_p)
                list_top_ks.append(param.top_k if param.top_k != -1 else vocab_size)
                do_top_p |= list_top_ps[-1] < 1.0 - SAMPLING_EPS
                do_top_k |= list_top_ks[-1] != vocab_size
            else:
                list_mask_random.append(False)
                has_greedy |= True

            list_past_output_tokens.append(param.output_tokens)

            apply_penalty |= (
                abs(param.presence_penalty) >= SAMPLING_EPS
                or abs(param.frequency_penalty >= SAMPLING_EPS)
                or abs(param.repetition_penalty - 1.0) >= SAMPLING_EPS
            )
            list_frequency_penalties.append(param.frequency_penalty)
            list_presence_penalties.append(param.presence_penalty)
            list_repetition_penalties.append(param.repetition_penalty)

            if param.logit_bias_index:
                assert param.logit_bias_value
                apply_bias |= True
                list_logit_bias_indices.append(param.logit_bias_index)
                list_logit_bias_values.append(param.logit_bias_value)
            else:
                list_logit_bias_indices.append([])
                list_logit_bias_values.append([])

        apply_top_p_top_k = do_top_p | do_top_k
        sampling_tensors = SamplingTensors.from_lists(
            dtype,
            dev,
            list_mask_random,
            list_temperatures,
            list_top_ps,
            list_top_ks,
            list_frequency_penalties,
            list_presence_penalties,
            list_repetition_penalties,
            list_logit_bias_indices,
            list_logit_bias_values,
            list_past_output_tokens,
        )
        return cls(
            has_random,
            has_greedy,
            apply_top_p_top_k,
            apply_penalty,
            apply_bias,
            sampling_tensors,
        )


def adjust_logits(logits, sampling_metadata, batch_size, vocab_size):
    (
        apply_top_p_top_k,
        apply_penalty,
        apply_bias,
        sampling_tensors,
    ) = (
        sampling_metadata.apply_top_p_top_k,
        sampling_metadata.apply_penalty,
        sampling_metadata.apply_bias,
        sampling_metadata.sampling_tensors,
    )
    (
        temp_t,
        top_ps_t,
        top_ks_t,
        frequency_penalties_t,
        repetition_penalties_t,
        presence_penalties_t,
        past_output_tokens_t,
        logit_bias_indices_t,
        logit_bias_values_t,
    ) = (
        sampling_tensors.temperatures,
        sampling_tensors.top_ps,
        sampling_tensors.top_ks,
        sampling_tensors.frequency_penalties,
        sampling_tensors.repetition_penalties,
        sampling_tensors.presence_penalties,
        sampling_tensors.past_output_tokens,
        sampling_tensors.logit_bias_indices,
        sampling_tensors.logit_bias_values,
    )

    # TODO(vvchernov): need to strictly define order of using penalties and logit bias or
    # prohibit simultaneous using of them. At the latter case it can be LogitProcessor
    if apply_penalty:
        repetition_penalties_t = repetition_penalties_t[:, None].repeat(1, vocab_size)
        logits = torch.where(
            logits > 0, logits / repetition_penalties_t, logits * repetition_penalties_t
        )
        bin_counts = torch.zeros(
            (batch_size, vocab_size + 1), dtype=torch.long, device=logits.device
        )
        bin_counts.scatter_add_(
            1, past_output_tokens_t, torch.ones_like(past_output_tokens_t)
        )
        bin_counts = bin_counts[:, :vocab_size]
        mask = bin_counts > 0
        logits -= frequency_penalties_t.unsqueeze_(dim=1) * bin_counts
        logits -= presence_penalties_t.unsqueeze_(dim=1) * mask

    # Adjust temperature
    logits.div_(temp_t.unsqueeze(dim=1))
    if apply_top_p_top_k:
        logits = _apply_top_p_top_k(logits, top_ps_t, top_ks_t)

    if apply_bias:
        logits.scatter_add_(
            1, logit_bias_indices_t, torch.ones_like(logit_bias_values_t)
        )
    return logits


def sample(
    sequence_ids,
    logits: Union[tvm.nd.NDArray, torch.Tensor],
    sampling_metadata,
    vocab_size,
    check_safety=False,
) -> Optional[np.ndarray]:
    def _is_safe_to_sample(prob_like):
        return (
            torch.sum(torch.isnan(prob_like) | torch.isinf(prob_like) | (prob_like < 0))
            == 0
        )

    # Convert to torch tensors if logits are in tvm ndarray
    if isinstance(logits, tvm.nd.NDArray):
        logits = torch.from_dlpack(logits)

    batch_size = len(sequence_ids)
    logits = adjust_logits(logits, sampling_metadata, batch_size, vocab_size)

    res_greedy, res_random = None, None
    sampling_tensors = sampling_metadata.sampling_tensors
    mask_greedy_t, mask_random_t = (
        sampling_tensors.mask_greedy,
        sampling_tensors.mask_random,
    )
    if sampling_metadata.has_greedy:
        logits_greedy = logits[mask_greedy_t]
        res_greedy = torch.argmax(logits_greedy, -1)

    if sampling_metadata.has_random:
        logits_random = logits[mask_random_t]
        probs = torch.softmax(logits_random, dim=-1)
        if check_safety and not _is_safe_to_sample(probs):
            return None

        res_random = _multinomial(probs, 1)[:, 0]

    # Prepare output
    # Send results to CPU and convert them into numpy
    res_greedy = res_greedy.cpu().numpy() if res_greedy is not None else np.array([])
    res_random = res_random.cpu().numpy() if res_random is not None else np.array([])

    sequence_ids = np.array(sequence_ids)
    sequence_ids_greedy = sequence_ids[mask_greedy_t.cpu().numpy()].tolist()
    sequence_ids_random = sequence_ids[mask_random_t.cpu().numpy()].tolist()
    new_sequence_ids = [
        *sequence_ids_greedy,
        *sequence_ids_random,
    ]
    next_tokens = [*res_greedy, *res_random]
    return zip(new_sequence_ids, next_tokens)

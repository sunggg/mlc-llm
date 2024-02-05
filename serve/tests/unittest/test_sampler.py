import torch
import pytest
from mlc_serve.model.sampler import SamplingMetadata, adjust_logits
from mlc_serve.engine import SamplingParams, SAMPLING_EPS

dtype = torch.float32
dev = "cuda"
vocab_size = 32000


def get_sampling_metadata(sampling_params):
    _copy_stream: torch.cuda.Stream = torch.cuda.Stream()
    with torch.cuda.stream(_copy_stream):
        sampling_metadata = SamplingMetadata.from_sampling_params(
            sampling_params,
            list_past_output_tokens=[[]],  # pass empty list
            dtype=dtype,
            dev=dev,
            vocab_size=vocab_size,
        )
    torch.cuda.current_stream().wait_stream(_copy_stream)
    return sampling_metadata


def _test_temperature(temp=0, batch_size=1):
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    sampling_param = SamplingParams(
        temperature=temp,
    )

    sampling_metadata = get_sampling_metadata([sampling_param])

    expected = logits / temp if abs(temp) > SAMPLING_EPS else logits
    new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
    assert torch.allclose(expected, new_logits)


def _test_logit_bias_checker():
    # logit bias must be [-100, 100]
    with pytest.raises(ValueError):
        logit_bias = {1: 2, 3: 105, 2: 2}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_metadata([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {1: 99, 3: -101, 2: 2}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_metadata([sampling_param])

    logit_bias = {1: 100, 3: -100, 2: 2}
    sampling_param = SamplingParams(logit_bias=logit_bias)
    get_sampling_metadata([sampling_param])

    # TODO(@team): it seems like the valid range is [1,vocab_size]. Double check.
    logit_bias = {1: 10, 3: -10, vocab_size: 2}
    sampling_param = SamplingParams(logit_bias=logit_bias)
    get_sampling_metadata([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {0: 10, 3: -10}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_metadata([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {1: 10, 3: -10, vocab_size + 100: 2}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_metadata([sampling_param])

    with pytest.raises(ValueError):
        logit_bias = {1: 10, -1: -10}
        sampling_param = SamplingParams(logit_bias=logit_bias)
        get_sampling_metadata([sampling_param])


def _test_logit_bias():
    # test single batch
    batch_size = 1
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    logit_bias = {1: -1, 3: 1, 2: 2}
    sampling_param = SamplingParams(logit_bias=logit_bias)
    sampling_metadata = get_sampling_metadata([sampling_param])

    expected = torch.clone(logits)
    for idx, val in logit_bias.items():
        expected[0][idx - 1] += val
    new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
    assert torch.allclose(expected, new_logits)

    batch_size = 3
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    list_logit_bias = [{1: -1, 3: 1, 2: 2}, {4: 2, 5: 1}, {1: -10}]
    sampling_params = [
        SamplingParams(logit_bias=logit_bias) for logit_bias in list_logit_bias
    ]
    sampling_metadata = get_sampling_metadata(sampling_params)

    expected = torch.clone(logits)
    for batch_size in range(batch_size):
        logit_bias = list_logit_bias[batch_size]
        for idx, val in logit_bias.items():
            expected[batch_size][idx - 1] += val
    new_logits = adjust_logits(logits, sampling_metadata, vocab_size)
    assert torch.allclose(expected, new_logits)


def _test_repetition_penalty():
    pass


def _test_presence_frequency_penalty():
    batch_size = 1
    shape = (batch_size, vocab_size)
    logits = torch.rand(shape, dtype=dtype, device=dev)
    past_output_tokens = [[]]
    sampling_param = SamplingParams(logit_bias=logit_bias)


def _test_top_p():
    pass


def _test_top_k():
    pass


def _test_multinomial():
    pass


if __name__ == "__main__":
    _test_temperature()
    _test_logit_bias_checker()
    _test_logit_bias()

"""Benchmark offline user metric."""
import argparse
import time, numpy as np
from mlc_serve.engine import (
    DebugOptions,
    Request,
    SamplingParams,
    StoppingCriteria,
)
from mlc_serve.utils import (
    get_default_mlc_serve_argparser,
    postproc_mlc_serve_args,
    create_engine_and_tokenizer_module,
)

# /opt/bin/cuda-reserve.py --num-gpu 2 nsys profile --stats true -o mlc-serve-mixtral-sampling-before -w true -t cuda,nvtx,osrt,cudnn,cublas python3 serve/benchmarks/benchmark_throughput.py --local-id Mixtral-8x7B-Instruct-v0.1-q0f16-presharded-2gpu/  --max-num-sequences 1 --max-input-len 32000 --num-prompts 1 --dataset /opt/models/dataset/ShareGPT_V3_unfiltered_cleaned_split.json


def main(args: argparse.Namespace):
    print(args)

    engine = create_engine_and_tokenizer_module(args)[0]
    engine.add(
        [
            Request(
                request_id="0",
                messages=None,
                sampling_params=SamplingParams(temperature=0.0),
                stopping_criteria=StoppingCriteria(
                    max_tokens=args.num_output_tokens, stop_sequences=None
                ),
                debug_options=DebugOptions(
                    ignore_eos=True, prompt_token_ids=[3] * args.num_input_tokens
                ),
                num_sequences=args.num_sequences_to_sample,
            )
        ]
    )

    latencies = []
    while engine.has_pending_requests():
        t0 = time.perf_counter()
        engine.step()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    if args.use_staging_engine:
        engine.stop()

    ttft = latencies[0]  # time to first token
    itl = np.mean(latencies[1:])  # inter-token latency for subsequent tokens
    e2e = np.sum(latencies)

    print(
        f"User side metrics\n"
        f"* number of input tokens: {args.num_input_tokens}, number of output tokens: {args.num_output_tokens}\n"
        f"* Time To First Token (TTFT): {ttft*1000:.3f} ms\n"
        f"* Inter-Subsequent-Token-Latency (ISTL): {itl*1000:.3f} ms ({1/itl:.3f} tok/s)\n"
        f"* End-to-end latency: {e2e:.3f} s\n"
    )


if __name__ == "__main__":
    parser = get_default_mlc_serve_argparser(description="Benchmark the throughput.")
    parser.add_argument("--num-input-tokens", type=int, default=128)
    parser.add_argument("--num-output-tokens", type=int, default=128)
    args = parser.parse_args()
    args = postproc_mlc_serve_args(args)

    main(args)

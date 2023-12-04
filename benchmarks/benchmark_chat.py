"""
This file attempts to be a realistic benchmark of a typical chat workload for an
LLM-serving system. It takes the ShareGPT dataset and has a back-and forth conversation
with the model-serving server. It measures both latency and throughput.
"""
import argparse
import asyncio
from itertools import zip_longest
import json
import time
import random
import aiohttp
import numpy as np

from typing import AsyncGenerator, Tuple
from pathlib import Path

from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

REQUEST_LATENCY: list[Tuple[int, int, float]] = []
TOTAL_REQUESTS = 0

# <https://docs.python.org/3/library/itertools.html#itertools-recipes>
def grouper(iterable, n, *, incomplete='ignore', fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    args = [iter(iterable)] * n
    if incomplete == 'fill':
        return zip_longest(*args, fillvalue=fillvalue)
    if incomplete == 'strict':
        return zip(*args, strict=True)
    if incomplete == 'ignore':
        return zip(*args)
    else:
        raise ValueError('Expected fill, strict, or ignore')

def sample_chats(
    dataset_path: Path,
    num_chats: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[list[Tuple[str, int, int]]]:
    with dataset_path.open() as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # Simplify the format of the dataset into tokenized (message, response length) pairs.
    dataset = [
        [(prompt, input_len, output_len) for prompt, input_len, output_len in [
            (data["conversations"][i * 2]["value"], len(tokenizer(data["conversations"][i * 2]["value"]).input_ids), len(tokenizer(data["conversations"][i * 2 + 1]["value"]).input_ids))
            for i in range(len(data["conversations"]) // 2)
        ] if 4 < input_len < 1024 and 4 < output_len < 1024]
        # FIXME(njha): We should sample from the entire dataset instead of just the beginning.
        for data in dataset[0:num_chats * 2]
    ]

    # Sample the requests.
    # return dataset
    return random.sample(dataset, num_chats)

async def chat_worker(
    backend: str,
    api_url: str,
    best_of: int,
    use_beam_search: bool,
    chats: list[list[Tuple[str, int, int]]],
    request_rate: float,
):
    global TOTAL_REQUESTS
    for chat in chats:
        context = ""
        context_len = 0
        prefix_len = 0
        for message, message_len, response_len in chat:
            # Add the next request to the context, and send the request.
            assert type(context) == str
            assert type(message) == str
            context += message
            context_len += message_len
            if prefix_len == 0:
                prefix_len = message_len
            assert type(context) == str
            # FIXME(njha): For some reason too long context windows cause an error on the VLLM end.
            if context_len > 1024:
                break
            context = (await send_request(backend, api_url, context, context_len, response_len, prefix_len, best_of, use_beam_search))["text"][0]
            TOTAL_REQUESTS += 1
            assert type(context) == str
            context_len += response_len

            if request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue

            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / request_rate)

            # The next request will be sent after the interval.
            await asyncio.sleep(interval)

async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    prefix_len: int,
    best_of: int,
    use_beam_search: bool,
) -> None:
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "prefix_pos": [],
            "stream": False,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            try:
                output = json.loads(output)
            except json.decoder.JSONDecodeError:
                raise ValueError(f"Failed to decode the response, {output} / {prompt} / {prompt_len} / {output_len}")

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    return output


async def benchmark(
    backend: str,
    api_url: str,
    chats: list[list[Tuple[str, int]]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    concurrent_chats = 1,
) -> None:
    tasks: list[asyncio.Task] = []
    for chat_group in grouper(chats, concurrent_chats):
        task = asyncio.create_task(chat_worker(backend, api_url, best_of, use_beam_search, chat_group, request_rate))
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_chats = sample_chats(args.dataset, args.num_prompts, tokenizer)

    benchmark_start_time = time.perf_counter()
    asyncio.run(benchmark(args.backend, api_url, input_chats, args.best_of,
                          args.use_beam_search, args.request_rate, args.concurrent_chats))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {TOTAL_REQUESTS / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.4f} s")
    avg_per_output_token_latency = np.mean([
        latency / output_len
        for _, output_len, latency in REQUEST_LATENCY
    ])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "tgi"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--concurrent-chats", type=int, default=1,
                        help="Number of concurrent chats to simulate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    args = parser.parse_args()
    main(args)

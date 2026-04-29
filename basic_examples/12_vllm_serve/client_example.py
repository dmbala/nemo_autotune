"""Minimal OpenAI-SDK client hitting the local vLLM server.

Usage:
    python 12_vllm_serve/client_example.py --endpoint http://localhost:8000/v1 \
        --model Qwen/Qwen3-0.6B --prompt "Explain photosynthesis in one sentence."

Run inside the container (the SDK is included in the overlay):
    shared/launch.sh python 12_vllm_serve/client_example.py ...
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="Model name as served by vLLM")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--stream", action="store_true", help="Stream tokens as they arrive")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.endpoint, api_key="EMPTY")

    if args.stream:
        print(f"\n--- streaming from {args.endpoint} ---")
        stream = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
        print()
    else:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()

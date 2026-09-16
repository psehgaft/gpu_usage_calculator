import argparse
import asyncio
import json
from pathlib import Path

from .benchmark import run_benchmark
from .calculator import estimate_capacity
from .models import BenchmarkRequest, SizingRequest


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU capacity estimator and benchmark tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    estimate_parser = subparsers.add_parser("estimate", help="Estimate GPU capacity")
    estimate_parser.add_argument("--file", required=True, help="Sizing request JSON file")
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark an OpenAI-compatible API")
    benchmark_parser.add_argument("--file", required=True, help="Benchmark request JSON file")
    args = parser.parse_args()

    if args.command == "estimate":
        result = estimate_capacity(SizingRequest.model_validate(_read_json(args.file)))
    else:
        result = asyncio.run(
            run_benchmark(BenchmarkRequest.model_validate(_read_json(args.file)))
        )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

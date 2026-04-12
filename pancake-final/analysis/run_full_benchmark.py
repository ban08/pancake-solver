from __future__ import annotations

from pathlib import Path

from src.core.benchmark import run_benchmark


def main() -> None:
    run_benchmark(
        games_dir=Path("games"),
        output_file=Path("analysis") / "full_benchmark_results.csv",
        save_csv=True,
        recursive=True,
    )


if __name__ == "__main__":
    main()

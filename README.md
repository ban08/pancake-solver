# pancake-solver

Solves the pancake-sorting puzzle — sort a stack by repeatedly flipping the top portion — with informed search, and measures which heuristics get there fastest.

## What it does

Given a scrambled stack of numbered "pancakes", finds a sequence of prefix-flips that sorts it. It implements A* and other search strategies with several admissible heuristics, and includes an analysis harness that benchmarks the heuristics across many starting stacks, plus a small GUI.

## Stack

Python. A `core/` search library (state, node, search, heuristics) with a terminal menu, a GUI, and benchmark/analysis scripts.

## How to run

```bash
pip install -r requirements.txt
cd pancake-final
python -m src.main        # menu: run the GUI or the benchmark
```

## What I built

Group project for the Artificial Intelligence course (2025/26), built with a teammate whose account hosts the original repository. My contribution here is the **benchmarking and analysis** — the scripts that run the heuristics across many games and compare them. Because most commits were made from my teammate's account, the git history does not reflect the full split of the work.

## What I would do differently

Add unit tests for each heuristic's admissibility, and make the benchmark output a single comparative table (nodes expanded and time per heuristic) so the trade-offs are visible at a glance.

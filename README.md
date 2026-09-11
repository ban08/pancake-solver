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

A group project for the Artificial Intelligence course (2025/26); the original repository is on a teammate's account. I was a supporting contributor — a small share of the overall work — focused on the **benchmarking and analysis** scripts that run the heuristics across many starting stacks and compare them. Most of the solver was written by teammates.

## What I would do differently

Add unit tests for each heuristic's admissibility, and make the benchmark output a single comparative table (nodes expanded and time per heuristic) so the trade-offs are visible at a glance.

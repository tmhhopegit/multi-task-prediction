"""Command line.

    python -m multitask compare data.csv --inputs A B C --outputs S1 S2 S3 [--models gp mtgp ridge rrr]
                        [--k 10] [--repeats 1] [--mode from_inputs|fill_in] [--groups ID] --out results/
    python -m multitask benchmark [--datasets 3] [--k 5] --out benchmark/

compare writes scores.csv (model x output), comparisons.csv (every single/multi pair with a
bootstrap interval) and predictions.csv. benchmark runs the synthetic scenarios in benchmark.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m multitask", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("csv")
    c.add_argument("--inputs", nargs="+", required=True)
    c.add_argument("--outputs", nargs="+", required=True)
    c.add_argument("--models", nargs="+", default=["gp", "mtgp", "ridge", "rrr", "lasso", "mtlasso", "pls1", "pls2"])
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--repeats", type=int, default=1)
    c.add_argument("--mode", choices=["from_inputs", "fill_in"], default="from_inputs")
    c.add_argument("--groups", help="column with patient IDs: keeps a patient's rows in one fold")
    c.add_argument("--jobs", type=int, default=1)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--datasets", type=int, default=3)
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--scenarios", nargs="+")
    b.add_argument("--jobs", type=int, default=1)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.cmd == "compare":
        from .evaluation import cross_validate
        from .models import KIND, PAIRS
        t = pd.read_csv(a.csv)
        X = t[a.inputs].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        Y = t[a.outputs].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        groups = t[a.groups].to_numpy() if a.groups else None
        res = cross_validate(a.models, X, Y, k=a.k, repeats=a.repeats, groups=groups, mode=a.mode, seed=a.seed,
                             n_jobs=a.jobs, output_names=a.outputs, verbose=True)
        scores = res.scores()
        scores.to_csv(out / "scores.csv", index=False)
        comps = [res.compare(s, m) for s, m in PAIRS if s in a.models and m in a.models]
        if comps:
            pd.concat(comps).to_csv(out / "comparisons.csv", index=False)
            print(pd.concat(comps)[["output", "single", "multi", "difference", "ci_low", "ci_high", "verdict"]]
                  .to_string(index=False))
        pred = pd.DataFrame({f"{m}:{o}": res.mean_prediction(m)[:, j] for m in a.models for j, o in enumerate(a.outputs)})
        pred.to_csv(out / "predictions.csv", index=False)
        print(scores.groupby("model")[["r", "R2", "mae"]].mean().round(3).to_string())
    else:
        from . import benchmark
        scores, comps = benchmark.run(a.scenarios, datasets=a.datasets, k=a.k, n_jobs=a.jobs)
        scores.to_csv(out / "benchmark_scores.csv", index=False)
        comps.to_csv(out / "benchmark_comparisons.csv", index=False)
        benchmark.summarise(scores).round(3).to_csv(out / "benchmark_summary.csv")
        import matplotlib
        matplotlib.use("Agg")
        benchmark.figure(scores, out / "benchmark.png")
        print(benchmark.summarise(scores).round(3).to_string())
    print(f"written to {out}")


if __name__ == "__main__":
    main()

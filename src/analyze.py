"""Score a binary detector column with OpenAI's effect-conditioned g-mean2 metric.
Usage: python src/analyze.py --csv results/encoded_rollouts.csv --z z_cot"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers  # noqa: F401  (sets up EVALS path)
from metric.intervention_gmean_metric import BootstrapConfig, bootstrapped_gmean_metric  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--z", default="z_cot", help="detector column to score (z_cot, y, ...)")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    m = df[["dataset", "instance_id", "x", "y"]].copy()
    m["x"] = m["x"].astype(int)
    m["z"] = df[a.z].astype(int)
    final, _, _ = bootstrapped_gmean_metric(
        m, group_cols=["dataset", "instance_id"], final_groups=["dataset"],
        x_col="x", y_col="y", z_col="z",
        bootstrap=BootstrapConfig(n_bootstrap=a.n_bootstrap, random_state=0))
    f = final.set_index("dataset")
    print("detector = %s   (OpenAI g-mean2, %d bootstraps)" % (a.z, a.n_bootstrap))
    print("%-16s %14s %8s %8s %10s" % ("dataset", "g-mean2", "TPR", "TNR", "eligible"))
    for ds in f.index:
        r = f.loc[ds]
        print("%-16s %7.3f+/-%-5.3f %7.2f %8.2f %9.2f" % (ds, r.gmean2_mean, r.gmean2_std,
              r.TPR_mean, r.TNR_defined_mean, r.eligible_fraction_mean))

if __name__ == "__main__":
    main()

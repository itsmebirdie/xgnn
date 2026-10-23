"""Experiment 3: explanation robustness under drift + EFDM vs other retraining policies."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time

import numpy as np

from xgnn_ids.common import prepare, save_json
from xgnn_ids.data.synthetic import make_drift_week
from xgnn_ids.drift import POLICIES, run_policies
from xgnn_ids.gnn import EdgeGNN, train
from xgnn_ids.graph import build_graph

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--weeks", type=int, default=8)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--scenario", default="both", choices=["both", "evasion", "benign"])
ap.add_argument("--out", default="drift.json")
args = ap.parse_args()

out = {"config": vars(args), "seeds": []}
for seed in range(args.seeds):
    t0 = time.time()
    d = prepare(seed)
    model = EdgeGNN(8, kind="gat", seed=seed)
    train(model, d["train"], d["val"], epochs=args.epochs, lr=1e-3, seed=seed)
    weeks = {0: d["test"][:3]}
    for w in range(1, args.weeks + 1):
        weeks[w] = [build_graph(x, d["scaler"]) for x in make_drift_week(seed, w, args.weeks, 3, scenario=args.scenario)]
    ref_X = np.concatenate([g.X for g in d["train"]])
    print(f"seed {seed}: base model trained ({time.time()-t0:.0f}s); running policies", flush=True)
    res, th = run_policies(model, d["train"], d["val"], weeks, ref_X, seed, n_weeks=args.weeks, log=lambda s: print(s, flush=True))
    out["seeds"].append({"seed": seed, "thresholds": th, "policies": res})
    save_json(out, args.out)
    print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)
print("saved results/" + args.out)

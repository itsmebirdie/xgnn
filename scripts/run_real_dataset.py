"""Run the full pipeline (detect -> explain -> score explanations) on a real public dataset.

    python scripts/run_real_dataset.py --dataset cicids2017 --path /data/TrafficLabelling --windows 30
    python scripts/run_real_dataset.py --dataset unsw-nb15  --path /data/UNSW-NB15 --windows 30

NOTE: untested on the real files (see README). Start small (--windows 20 --window-size 2000) to confirm
the column mapping works on your copy, then scale up.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import numpy as np

from xgnn_ids.common import binary_metrics, mean_std, save_json, scores
from xgnn_ids.data.loaders import load_cicids2017, load_unsw_nb15, to_windows
from xgnn_ids.explain import METHODS, explain_alert
from xgnn_ids.gnn import EdgeGNN, train
from xgnn_ids.graph import build_graph, fit_scaler

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["cicids2017", "unsw-nb15"], required=True)
ap.add_argument("--path", required=True)
ap.add_argument("--windows", type=int, default=30)
ap.add_argument("--window-size", type=int, default=2000)
ap.add_argument("--max-flows", type=int, default=None)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--alerts", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

load = load_cicids2017 if args.dataset == "cicids2017" else load_unsw_nb15
df = load(args.path, args.max_flows)
print(f"loaded {len(df):,} flows, {df.label.mean():.1%} malicious; attack types: {df.attack_type.value_counts().to_dict()}")
wins = to_windows(df, args.window_size, args.windows, min_attack_frac=0.0)
n = len(wins)
a, b = int(0.70 * n), int(0.85 * n)                      # time-ordered 70/15/15
scaler = fit_scaler(wins[:a])
G = [build_graph(w, scaler) for w in wins]
tr, va, te = G[:a], G[a:b], G[b:]
print(f"{n} windows -> train {len(tr)} / val {len(va)} / test {len(te)}")
model = EdgeGNN(8, kind="gat", seed=args.seed)
train(model, tr, va, epochs=args.epochs, lr=1e-3, seed=args.seed, verbose=True)
y, p = scores(model, te)
det = binary_metrics(y, p)
print("test:", {k: round(v, 4) for k, v in det.items() if isinstance(v, float)})

rng = np.random.default_rng(args.seed)
cand = [(gi, int(e)) for gi, g in enumerate(te) for e in np.where(model.predict_proba(g) > 0.5)[0]]
rows = []
for j in rng.choice(len(cand), size=min(args.alerts, len(cand)), replace=False):
    gi, e = cand[j]
    rows.append(explain_alert(model, te[gi], e, seed=args.seed))
keys = ["fidelity", "sufficiency", "necessity", "sig_precision", "type_agree", "sparsity"]
expl = {}
for m in METHODS:
    sel = [r["methods"][m] for r in rows if m in r["methods"]]
    expl[m] = {k: float(np.nanmean([s[k] for s in sel])) for k in keys}
    print(m, {k: round(v, 3) for k, v in expl[m].items()})
save_json({"args": vars(args), "detection": det, "explanations": expl}, f"real_{args.dataset}.json")

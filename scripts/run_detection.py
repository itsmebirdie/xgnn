"""Experiment 1: detection performance of GAT vs GraphSAGE-style vs MLP baseline (time-split test set)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time

import numpy as np
from sklearn.metrics import roc_curve

from xgnn_ids.common import binary_metrics, mean_std, prepare, save_json, scores
from xgnn_ids.gnn import EdgeGNN, train

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=1e-3)
args = ap.parse_args()

res = {"config": vars(args), "models": {}}
for kind in ["gat", "sage", "mlp"]:
    rows, curves, hists = [], [], []
    for seed in range(args.seeds):
        d = prepare(seed)
        t = time.time()
        m = EdgeGNN(8, hidden=32, heads=4, layers=2, kind=kind, seed=seed)
        h = train(m, d["train"], d["val"], epochs=args.epochs, lr=args.lr, seed=seed)
        y, p = scores(m, d["test"])
        r = binary_metrics(y, p)
        r["train_seconds"] = time.time() - t
        rows.append(r)
        fpr, tpr, _ = roc_curve(y, p)
        curves.append({"fpr": fpr[::max(1, len(fpr) // 400)].tolist(), "tpr": tpr[::max(1, len(fpr) // 400)].tolist()})
        hists.append(h)
        print(f"{kind} seed {seed}: acc {r['accuracy']:.4f} f1 {r['f1']:.4f} auc {r['auc']:.4f} fpr {r['fpr']:.4f} ({r['train_seconds']:.0f}s)", flush=True)
    res["models"][kind] = {
        "per_seed": rows, "summary": mean_std(rows, ["accuracy", "precision", "recall", "f1", "fpr", "auc"]),
        "roc_seed0": curves[0], "history_seed0": hists[0],
        "confusion_sum": {k: int(sum(r[k] for r in rows)) for k in ("tn", "fp", "fn", "tp")},
    }
save_json(res, "detection.json")
print("saved results/detection.json")

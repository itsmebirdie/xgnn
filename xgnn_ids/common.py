"""Shared experiment helpers."""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from .data.synthetic import make_base_windows
from .graph import build_graph, fit_scaler

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def prepare(seed: int, n_windows: int = 60, split=(0.70, 0.15, 0.15)):
    """Time-ordered 70/15/15 split of benign-period windows (no shuffling)."""
    wins = make_base_windows(seed, n_windows)
    a, b = int(split[0] * n_windows), int((split[0] + split[1]) * n_windows)
    scaler = fit_scaler(wins[:a])
    G = [build_graph(w, scaler) for w in wins]
    return dict(wins=wins, scaler=scaler, train=G[:a], val=G[a:b], test=G[b:],
                train_raw=wins[:a], val_raw=wins[a:b], test_raw=wins[b:])


def scores(model, graphs):
    y = np.concatenate([g.y for g in graphs])
    p = np.concatenate([model.predict_proba(g) for g in graphs])
    return y, p


def binary_metrics(y, p, thr=0.5):
    yh = (p > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    out = {
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision_score(y, yh, zero_division=0)),
        "recall": float(recall_score(y, yh, zero_division=0)),
        "f1": float(f1_score(y, yh, zero_division=0)),
        "fpr": float(fp / max(1, fp + tn)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return out


def mean_std(rows, keys):
    return {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows]))}
            for k in keys}


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    return path

"""Train a small model and print human-readable alert reports for a few detected attacks."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from xgnn_ids.common import prepare
from xgnn_ids.explain import alert_report, explain
from xgnn_ids.gnn import EdgeGNN, train
from xgnn_ids.graph import computation_subgraph

d = prepare(0)
model = EdgeGNN(8, kind="gat", seed=0)
train(model, d["train"], d["val"], epochs=25, lr=2e-3)
g = d["test"][0]
p = model.predict_proba(g)
seen = set()
for e in np.where(p > 0.5)[0]:
    atk = g.meta.loc[e, "attack_type"]
    if atk in seen:
        continue
    seen.add(atk)
    sub, loc = computation_subgraph(g, int(e))
    imp, fimp = explain(model, sub, loc, "gnnexplainer", 1, 0)
    print(f"--- ground truth: {atk} ---")
    print(alert_report(sub, loc, float(p[e]), imp, fimp)[0], "\n")

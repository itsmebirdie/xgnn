"""Train the paper-config GAT (40 epochs, lr 1e-3) and save it for the Streamlit app.

    python scripts/train_model.py --seed 0          # ~25 s; writes results/model_gat_seed0.npz
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from xgnn_ids.app_backend import save_checkpoint
from xgnn_ids.common import binary_metrics, prepare, scores
from xgnn_ids.gnn import EdgeGNN, train

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=40)
a = ap.parse_args()
d = prepare(a.seed)
m = EdgeGNN(8, kind="gat", seed=a.seed)
train(m, d["train"], d["val"], epochs=a.epochs, lr=1e-3, seed=a.seed)
print({k: round(v, 4) for k, v in binary_metrics(*scores(m, d["test"])).items() if isinstance(v, float)})
save_checkpoint(m, a.seed)
print("saved results/model_gat_seed%d.npz" % a.seed)

"""UI-independent logic behind the Streamlit app (so it can be unit-tested without Streamlit)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .common import RESULTS, binary_metrics, prepare, scores
from .data.synthetic import Drift, generate_window
from .explain import alert_report, budget, explain, fidelity, signature_precision
from .gnn import EdgeGNN, train
from .graph import FEATURE_LABELS, FEATURES, build_graph, computation_subgraph

TEAL, NAVY, ORANGE, RED = "#12a89d", "#0b2545", "#f4a261", "#e63946"
METHOD_LABELS = {"gnnexplainer": "GNNExplainer", "attention": "Attention rollout",
                 "saliency": "Gradient saliency", "random": "Random (control)"}


def train_demo_model(seed: int = 0, epochs: int = 25):
    """Quick GAT trained on the synthetic benign-period windows (~15-25 s on one CPU core)."""
    d = prepare(seed)
    model = EdgeGNN(8, kind="gat", seed=seed)
    train(model, d["train"], d["val"], epochs=epochs, lr=2e-3, seed=seed)
    return model, d["scaler"]


def make_window(scaler, evasion: float, benign_shift: float, seed: int):
    rng = np.random.default_rng(10_000 + seed)
    df = generate_window(rng, Drift(evasion, benign_shift))
    return df, build_graph(df, scaler)


def window_metrics(model, g) -> dict:
    y, p = scores(model, [g])
    return binary_metrics(y, p)


def list_alerts(model, g) -> pd.DataFrame:
    p = model.predict_proba(g)
    idx = np.where(p > 0.5)[0]
    m = g.meta.iloc[idx]
    return pd.DataFrame({"edge": idx, "src": m["src"].values, "dst": m["dst"].values, "dport": m["dport"].values,
                         "score": p[idx].round(3), "truth": m["attack_type"].values})


def explain_one(model, g, e: int, method: str = "gnnexplainer", k_frac: float = 0.10, seed: int = 0) -> dict:
    sub, loc = computation_subgraph(g, int(e))
    imp, fimp = explain(model, sub, loc, method, 1, seed)
    k = budget(sub, k_frac)
    fid = fidelity(model, sub, loc, imp, 1, k)
    p = float(model.predict_proba(sub)[loc])
    text, pattern = alert_report(sub, loc, p, imp, fimp)
    return {"sub": sub, "loc": loc, "importance": imp, "feature_importance": fimp, "k": k, "fidelity": fid,
            "sig_precision": signature_precision(sub, loc, imp, k), "report": text, "pattern": pattern, "p": p}


def plot_explanation(res: dict, top: int = 12):
    sub, loc, imp = res["sub"], res["loc"], res["importance"]
    meta = sub.meta
    host = meta.loc[loc, "src"]
    order = np.argsort(-imp, kind="stable")[:top]
    agg = {}
    for i in order:
        agg.setdefault((meta.loc[i, "src"], meta.loc[i, "dst"]), []).append(float(imp[i]))
    G = nx.DiGraph()
    for (s, d), v in agg.items():
        G.add_edge(s, d, w=max(v), n=len(v))
    fig, ax = plt.subplots(1, 2 if res["feature_importance"] is not None else 1,
                           figsize=(11 if res["feature_importance"] is not None else 6, 4.2))
    ax = np.atleast_1d(ax)
    pos = nx.spring_layout(G, seed=3, k=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax[0], node_color=[RED if n == host else ORANGE for n in G], node_size=300)
    wmax = max(1e-9, max(d["w"] for *_, d in G.edges(data=True)))
    nx.draw_networkx_edges(G, pos, ax=ax[0], width=[1 + 4 * G[u][v]["w"] / wmax for u, v in G.edges()],
                           edge_color=RED, alpha=0.7, arrowsize=12)
    nx.draw_networkx_labels(G, pos, ax=ax[0], font_size=7)
    ax[0].set_title(f"Top {top} flows by importance (red = alerted host)", fontsize=10)
    ax[0].axis("off")
    if res["feature_importance"] is not None:
        fi = sorted(zip(FEATURES, res["feature_importance"]), key=lambda kv: kv[1])
        ax[1].barh([FEATURE_LABELS[k] for k, _ in fi], [v for _, v in fi],
                   color=[TEAL if v > 0.5 else "#a5d8d3" for _, v in fi])
        ax[1].set_xlim(0, 1); ax[1].set_title("Feature mask (GNNExplainer)", fontsize=10)
        ax[1].spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def top_flows_table(res: dict, n: int = 15) -> pd.DataFrame:
    sub, imp = res["sub"], res["importance"]
    o = np.argsort(-imp, kind="stable")[:n]
    m = sub.meta.iloc[o]
    return pd.DataFrame({"importance": imp[o].round(3), "src": m["src"].values, "dst": m["dst"].values,
                         "dport": m["dport"].values, "truth": m["attack_type"].values})


def load_result(name: str):
    path = os.path.join(RESULTS, name)
    return json.load(open(path)) if os.path.exists(path) else None


def figure_path(name: str):
    p = os.path.join(RESULTS, "figures", name)
    return p if os.path.exists(p) else None

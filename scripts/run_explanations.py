"""Experiment 2: how faithful are the explanations, and do they match the real attack signatures?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import numpy as np

from xgnn_ids.common import mean_std, prepare, save_json
from xgnn_ids.explain import METHODS, alert_report, budget, explain_alert, fidelity
from xgnn_ids.gnn import EdgeGNN, train
from xgnn_ids.graph import FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--alerts", type=int, default=32, help="alerts explained per seed")
ap.add_argument("--epochs", type=int, default=40)
args = ap.parse_args()

KEYS = ["fidelity", "sufficiency", "necessity", "sig_precision", "type_agree", "feature_hit", "sparsity"]
per_seed = {m: [] for m in METHODS}
pooled = {m: [] for m in METHODS}
sweep = {m: {str(f): [] for f in (0.02, 0.05, 0.10, 0.20, 0.40)} for m in METHODS}
feat_by_type = {}
examples = {}

for seed in range(args.seeds):
    d = prepare(seed)
    model = EdgeGNN(8, kind="gat", seed=seed)
    train(model, d["train"], d["val"], epochs=args.epochs, lr=1e-3, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    cand = []
    for gi, g in enumerate(d["test"]):
        p = model.predict_proba(g)
        cand += [(gi, int(e)) for e in np.where(p > 0.5)[0]]       # the model's own alerts
    pick = rng.choice(len(cand), size=min(args.alerts, len(cand)), replace=False)
    rows = []
    for j in pick:
        gi, e = cand[j]
        r = explain_alert(model, d["test"][gi], e, seed=seed)
        rows.append(r)
        sub, loc = r["_sub"], r["_loc"]
        for m in METHODS:
            if m not in r["methods"]:
                continue
            pooled[m].append({**r["methods"][m], "attack_type": r["attack_type"]})
            for f in sweep[m]:
                imp = r["_" + m][0]
                sweep[m][f].append(fidelity(model, sub, loc, imp, 1, budget(sub, float(f)))["fidelity"])
        fm = r["_gnnexplainer"][1]
        feat_by_type.setdefault(r["attack_type"], []).append(fm)
        if seed == 0 and r["attack_type"] not in examples and r["attack_type"] != "benign":
            imp, fm = r["_gnnexplainer"]
            text, pat = alert_report(sub, loc, r["p_attack"], imp, fm)
            order = np.argsort(-imp)[:14]
            examples[r["attack_type"]] = {
                "report": text, "pattern": pat, "alert_host": sub.meta.loc[loc, "src"], "p_attack": r["p_attack"],
                "edges": [{"src": sub.meta.loc[i, "src"], "dst": sub.meta.loc[i, "dst"],
                           "dport": int(sub.meta.loc[i, "dport"]), "importance": float(imp[i]),
                           "attack": int(sub.meta.loc[i, "label"])} for i in order],
                "feature_importance": {FEATURES[i]: float(fm[i]) for i in range(len(FEATURES))},
            }
    for m in METHODS:
        sel = [r["methods"][m] for r in rows if m in r["methods"]]
        if sel:
            per_seed[m].append({k: float(np.nanmean([s[k] for s in sel])) for k in KEYS})
    print(f"seed {seed}: explained {len(rows)} alerts; " + "  ".join(
        f"{m} F={per_seed[m][-1]['fidelity']:.3f}" for m in METHODS if per_seed[m]), flush=True)

by_type = {}
for atk in sorted({r["attack_type"] for r in pooled["gnnexplainer"]}):
    sel = [r for r in pooled["gnnexplainer"] if r["attack_type"] == atk]
    by_type[atk] = {"n": len(sel), **{k: float(np.nanmean([s[k] for s in sel])) if not np.all(np.isnan([s[k] for s in sel])) else None
                                      for k in ["fidelity", "sufficiency", "necessity", "sig_precision", "type_agree", "feature_hit"]}}
out = {
    "config": vars(args),
    "methods": {m: mean_std(per_seed[m], KEYS) for m in METHODS if per_seed[m]},
    "fidelity_vs_budget": {m: {f: float(np.mean(v)) for f, v in sweep[m].items()} for m in METHODS if sweep[m]["0.1"]},
    "gnnexplainer_by_attack_type": by_type,
    "gnnexplainer_feature_importance_by_type": {
        t: {FEATURES[i]: float(np.mean([f[i] for f in fs])) for i in range(len(FEATURES))}
        for t, fs in feat_by_type.items() if t != "benign"},
    "examples": examples,
    "n_alerts_pooled": {m: len(pooled[m]) for m in METHODS},
}
save_json(out, "explanations.json")
print("saved results/explanations.json")

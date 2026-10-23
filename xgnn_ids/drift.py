"""Explanation-Fidelity Drift Monitor (EFDM) and retraining policies.

Idea (the project's novelty): explanation fidelity is measured against the model's OWN prediction, so it
needs no ground-truth labels. If the traffic drifts away from what the model learned, its explanations
stop being faithful/compact BEFORE anyone has labelled the new traffic. EFDM uses a drop in fidelity
(on a handful of recent alerts) as the signal to request labels and retrain.

Policies compared on the same drift stream (labels become available only for windows a policy pays to label):
  static          never retrain
  periodic        retrain every 2 weeks (labels for every window)
  efdm            retrain when GNNExplainer fidelity drops below a validation-calibrated threshold
  efdm_attn       same with the cheap attention-rollout explainer
  confidence      label-free baseline: mean max-softmax confidence drops
  psi             label-free baseline: input feature drift (max PSI > 0.25)
  label_oracle    needs labels for ALL windows: retrain when F1 drops (upper-bound reference)
"""
from __future__ import annotations

import zlib

import numpy as np

from .common import binary_metrics, scores
from .explain import attention_rollout, explain, fidelity, budget
from .gnn import train
from .graph import computation_subgraph

POLICIES = ["static", "periodic", "efdm", "efdm_attn", "confidence", "psi", "label_oracle"]
N_ALERTS = 12
RETRAIN_WINDOWS = 12
RETRAIN_EPOCHS = 20


def psi_max(ref_X: np.ndarray, X: np.ndarray, bins: int = 10) -> float:
    vals = []
    for j in range(ref_X.shape[1]):
        edges = np.unique(np.quantile(ref_X[:, j], np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:                                   # (near-)binary feature
            edges = np.array([-np.inf, np.median(ref_X[:, j]) + 1e-9, np.inf])
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(ref_X[:, j], edges)[0] / len(ref_X) + 1e-4
        a = np.histogram(X[:, j], edges)[0] / len(X) + 1e-4
        vals.append(float(np.sum((a - e) * np.log(a / e))))
    return max(vals)


def week_signals(model, graphs, ref_X, seed, tag, n_alerts=N_ALERTS):
    """Everything a monitor could look at for one batch of windows (some need labels, most do not)."""
    y, p = scores(model, graphs)
    out = binary_metrics(y, p)
    out["conf"] = float(np.mean(np.maximum(p, 1 - p)))
    out["alert_rate"] = float(np.mean(p > 0.5))
    out["psi"] = psi_max(ref_X, np.concatenate([g.X for g in graphs]))
    cand = [(gi, int(e)) for gi, g in enumerate(graphs) for e in np.where(model.predict_proba(g) > 0.5)[0]]
    rng = np.random.default_rng(zlib.crc32(repr((seed, tag)).encode()))
    fg, fa, n = [], [], 0
    if cand:
        for j in rng.choice(len(cand), size=min(n_alerts, len(cand)), replace=False):
            gi, e = cand[j]
            sub, loc = computation_subgraph(graphs[gi], e)
            k = budget(sub)
            imp, _ = explain(model, sub, loc, "gnnexplainer", 1, seed)
            fg.append(fidelity(model, sub, loc, imp, 1, k)["fidelity"])
            fa.append(fidelity(model, sub, loc, attention_rollout(model, sub, loc), 1, k)["fidelity"])
        n = len(fg)
    out["fid_gnnexp"] = float(np.mean(fg)) if fg else float("nan")
    out["fid_attn"] = float(np.mean(fa)) if fa else float("nan")
    out["n_alerts"] = n
    return out


def calibrate(model, val_graphs, ref_X, seed):
    """Thresholds from the (clean) validation period: mean - max(3*sigma_week, margin)."""
    per = [week_signals(model, [g], ref_X, seed, ("val", i), n_alerts=4) for i, g in enumerate(val_graphs)]
    th = {}
    for key, margin in [("fid_gnnexp", 0.03), ("fid_attn", 0.03), ("conf", 0.01), ("f1", 0.02)]:
        v = np.array([r[key] for r in per], float)
        v = v[~np.isnan(v)]
        sigma_week = v.std() / np.sqrt(3)                    # a week = 3 windows
        th[key] = float(v.mean() - max(3 * sigma_week, margin))
        th[key + "_mean"] = float(v.mean())
    th["psi"] = 0.25
    return th


def decide(policy, week, sig, th):
    if policy == "static":
        return False
    if policy == "periodic":
        return week % 2 == 0
    if policy == "efdm":
        return sig["fid_gnnexp"] < th["fid_gnnexp"] if not np.isnan(sig["fid_gnnexp"]) else False
    if policy == "efdm_attn":
        return sig["fid_attn"] < th["fid_attn"] if not np.isnan(sig["fid_attn"]) else False
    if policy == "confidence":
        return sig["conf"] < th["conf"]
    if policy == "psi":
        return sig["psi"] > th["psi"]
    if policy == "label_oracle":
        return sig["f1"] < th["f1"]
    raise ValueError(policy)


def run_policies(base_model, base_train, val_graphs, weeks, ref_X, seed, n_weeks=8, policies=POLICIES, log=print):
    """weeks: {0: base-test windows, 1..n: drift windows}. Returns per-policy trajectories."""
    th = calibrate(base_model, val_graphs, ref_X, seed)
    sig_cache, model_cache = {}, {}

    def sig(version, model, w):
        key = (version, w)
        if key not in sig_cache:
            sig_cache[key] = week_signals(model, weeks[w], ref_X, seed, (version, w))
        return sig_cache[key]

    results = {}
    for pol in policies:
        version, model = "v0", base_model
        pool = [(f"b{i}", g) for i, g in enumerate(base_train)]        # labelled windows available
        labelled_through, cooldown, cost, retrains = 0, 0, 0, []
        traj = []
        for w in range(0, n_weeks + 1):
            s = sig(version, model, w)
            traj.append({"week": w, "version": version, **s})
            if w == 0 or w == n_weeks:
                continue
            if pol == "label_oracle":                                 # labels needed every week just to monitor
                cost += len(weeks[w])
            trig = decide(pol, w, s, th)
            if cooldown:
                cooldown -= 1
                trig = False
            if trig:
                if pol != "label_oracle":
                    cost += len(weeks[w]) * (w - labelled_through)
                for ww in range(labelled_through + 1, w + 1):
                    pool += [(f"w{ww}_{i}", g) for i, g in enumerate(weeks[ww])]
                labelled_through = w
                use = pool[-RETRAIN_WINDOWS:]
                ver = version + ">" + "+".join(i for i, _ in use)
                if ver not in model_cache:
                    m = model.clone()
                    train(m, [g for _, g in use], None, epochs=RETRAIN_EPOCHS, lr=1e-3, seed=seed)
                    model_cache[ver] = m
                version, model = ver, model_cache[ver]
                retrains.append(w)
                cooldown = 1
            elif pol == "label_oracle":
                pool += [(f"w{w}_{i}", g) for i, g in enumerate(weeks[w])]
                labelled_through = w
        f1s = [t["f1"] for t in traj[1:]]
        results[pol] = {"trajectory": traj, "retrain_after_weeks": retrains, "labelled_windows": cost,
                        "mean_f1": float(np.mean(f1s)), "final_f1": float(f1s[-1]), "min_f1": float(np.min(f1s)),
                        "mean_acc": float(np.mean([t["accuracy"] for t in traj[1:]])),
                        "mean_fid": float(np.nanmean([t["fid_gnnexp"] for t in traj[1:]]))}
        log(f"  {pol:13s} meanF1 {results[pol]['mean_f1']:.3f} finalF1 {results[pol]['final_f1']:.3f} "
            f"retrains {retrains} labels {cost}")
    return results, th

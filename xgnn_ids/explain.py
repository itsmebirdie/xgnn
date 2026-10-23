"""Explanation layer: explainers, fidelity / signature metrics, human-readable alerts.

Explainers (all return an importance score per edge of the alert's computation subgraph):
  * ``gnnexplainer``  learned soft edge + feature masks (Ying et al., 2019)
  * ``attention``     attention rollout through the GAT's own attention weights
  * ``saliency``      |d log p / d edge-mask| at mask = 1
  * ``random``        control

Metrics
  * fidelity  F = (sufficiency + necessity) / 2 for the top-k flows (k = 10% of the computation subgraph)
        sufficiency = 1 - max(0, p - p_keep)        keep ONLY the top-k flows
        necessity   = clip((p - p_drop) / p, 0, 1)  remove the top-k flows
    (p = probability of the model's own predicted class; no labels needed)
  * signature precision@k : share of the top-k flows that really belong to the
    attack campaign that triggered the alert (needs ground truth)
  * alert-type agreement : does the rule-based report built ONLY from the explanation name the right attack
"""
from __future__ import annotations

import numpy as np

from . import autograd as ag
from .autograd import Adam, Tensor
from .graph import FEATURES, FEATURE_LABELS, GraphData, computation_subgraph

METHODS = ["gnnexplainer", "attention", "saliency", "random"]

EXPECTED_FEATURES = {   # domain knowledge: which features define each attack family
    "portscan": {"dest_port_count", "syn_count", "flow_pkts_s", "bwd_pkts_s"},
    "dos": {"flow_pkts_s", "flow_duration", "syn_count", "bwd_pkts_s"},
    "bruteforce": {"dst_port", "flow_duration", "flow_pkts_s"},
    "infiltration": {"dst_port", "flow_duration", "flow_pkts_s"},
}


# ---------------------------------------------------------------- helpers
def _pc(model, sub, loc, emask=None, fmask=None, cls=1):
    p1 = model.predict_proba(sub, emask, fmask)[loc]
    return p1 if cls == 1 else 1.0 - p1


def _ent(m):
    return ag.neg(ag.tmean(m * ag.log(m) + (1 - m) * ag.log(1 - m)))


# ---------------------------------------------------------------- explainers
def gnnexplainer(model, sub: GraphData, loc: int, cls: int = 1, epochs: int = 100, lr: float = 0.1,
                 edge_size: float = 0.02, edge_ent: float = 1.0, feat_size: float = 0.5,
                 feat_ent: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    elog = Tensor(rng.normal(0, 0.1, size=(sub.E, 1)), requires_grad=True)
    flog = Tensor(np.zeros((1, len(FEATURES))), requires_grad=True)
    opt = Adam([elog, flog], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        em, fm = ag.sigmoid(elog), ag.sigmoid(flog)
        lp = ag.log_softmax(model.forward(sub, em, fm))
        nll = ag.neg(Tensor(lp.data[loc:loc + 1, cls:cls + 1]) * 0 + _pick(lp, loc, cls))
        loss = (nll + edge_size * ag.tsum(em) + edge_ent * _ent(em)
                + feat_size * ag.tmean(fm) + feat_ent * _ent(fm))
        loss.backward()
        opt.step()
    em = 1 / (1 + np.exp(-elog.data[:, 0]))
    fm = 1 / (1 + np.exp(-flog.data[0]))
    return em, fm


def _pick(lp, i, c):
    """lp[i, c] as a Tensor with gradient."""
    out = Tensor(lp.data[i:i + 1, c:c + 1], (lp,))
    def bw(g):
        gl = np.zeros_like(lp.data)
        gl[i, c] = g[0, 0]
        lp._acc(gl)
    out._backward = bw
    return out


def saliency(model, sub, loc, cls=1):
    em = Tensor(np.ones((sub.E, 1)), requires_grad=True)
    lp = ag.log_softmax(model.forward(sub, em))
    _pick(lp, loc, cls).backward()
    return np.abs(em.grad[:, 0])


def attention_rollout(model, sub, loc):
    """Relevance flows from the two end points of the alert back through the attention weights."""
    assert model.kind == "gat"
    _, attn = model.forward(sub, collect_attn=True)
    src, dst, n = sub.src, sub.dst, sub.n
    r = np.zeros(n)
    r[src[loc]] += 1.0
    r[dst[loc]] += 1.0
    imp = np.zeros(sub.E)
    for l in range(model.layers - 1, -1, -1):
        a_in = np.mean([attn[(l, "in", k)] for k in range(model.heads)], axis=0)    # alpha of edge at its dst
        a_out = np.mean([attn[(l, "out", k)] for k in range(model.heads)], axis=0)  # alpha of edge at its src
        w_to_dst = r[dst] * a_in     # relevance that dst pulls through this edge from src
        w_to_src = r[src] * a_out
        imp += w_to_dst + w_to_src
        r_new = r.copy()
        np.add.at(r_new, src, w_to_dst)
        np.add.at(r_new, dst, w_to_src)
        r = r_new
    return imp


def explain(model, sub, loc, method, cls=1, seed=0, **kw):
    """-> (edge_importance (E,), feature_importance (d,) or None)"""
    if method == "gnnexplainer":
        em, fm = gnnexplainer(model, sub, loc, cls, seed=seed, **kw)
        return em, fm
    if method == "attention":
        return attention_rollout(model, sub, loc), None
    if method == "saliency":
        return saliency(model, sub, loc, cls), None
    if method == "random":
        return np.random.default_rng(seed).random(sub.E), None
    raise ValueError(method)


# ---------------------------------------------------------------- metrics
def fidelity(model, sub, loc, importance, cls=1, k=5):
    order = np.argsort(-importance, kind="stable")
    top = order[:k]
    keep = np.zeros((sub.E, 1)); keep[top] = 1.0
    p_full = _pc(model, sub, loc, cls=cls)
    p_keep = _pc(model, sub, loc, Tensor(keep), cls=cls)
    p_drop = _pc(model, sub, loc, Tensor(1.0 - keep), cls=cls)
    suff = 1.0 - max(0.0, p_full - p_keep)
    nec = float(np.clip((p_full - p_drop) / max(p_full, 1e-9), 0.0, 1.0))
    return {"fidelity": 0.5 * (suff + nec), "sufficiency": suff, "necessity": nec,
            "sparsity": 1.0 - k / sub.E, "p_full": p_full}


def signature_precision(sub, loc, importance, k=5):
    meta = sub.meta
    camp = meta.loc[loc, "campaign"]
    if camp < 0:
        return np.nan
    gt = (meta["campaign"].to_numpy() == camp)
    k_eff = min(k, int(gt.sum()))
    top = np.argsort(-importance, kind="stable")[:k_eff]
    return float(gt[top].mean())


def feature_hit(feat_imp, attack_type, topn=3):
    if feat_imp is None or attack_type not in EXPECTED_FEATURES:
        return np.nan
    top = [FEATURES[i] for i in np.argsort(-feat_imp)[:topn]]
    return float(np.mean([t in EXPECTED_FEATURES[attack_type] for t in top]))


# ---------------------------------------------------------------- reports
RULE_TO_ATTACK = {"port-scan-like": "portscan", "lateral-movement-like": "infiltration",
                  "brute-force-like": "bruteforce", "flood-like": "dos"}


def classify_pattern(sub, loc, importance, k=10):
    """Rule-based attack family from the explanation-selected flows ONLY (no labels)."""
    meta = sub.meta
    host = meta.loc[loc, "src"]
    top = np.argsort(-importance, kind="stable")[:k]
    sel = meta.iloc[top]
    sel = sel[sel["src"] == host]
    if len(sel) < 3:
        return "unclassified", sel
    n_dst, n_ports = sel["dst"].nunique(), sel["dport"].nunique()
    top_pair = sel.groupby("dst").size().max()
    share = lambda ports: float(sel["dport"].isin(ports).mean())
    if n_ports >= 6:                                   # many ports: vertical (one host) or horizontal scan
        return "port-scan-like", sel
    if n_dst >= 3 and share({445, 3389, 22}) >= 0.5:   # fan-out over admin ports
        return "lateral-movement-like", sel
    if n_dst <= 2 and share({21, 22}) >= 0.5 and top_pair >= 4:
        return "brute-force-like", sel
    if n_dst <= 2 and top_pair >= 4:
        return "flood-like", sel
    return "unclassified", sel


ACTIONS = {
    "port-scan-like": "Block or rate-limit the source and review the probed hosts for exposed services.",
    "lateral-movement-like": "Isolate the source host: it is contacting many internal machines on admin ports.",
    "brute-force-like": "Lock the targeted account/service and block the source after confirming the login failures.",
    "flood-like": "Rate-limit the source at the edge; check the target server for degraded service.",
    "unclassified": "Manual triage: the explanation does not match a known attack signature.",
}


def alert_report(sub, loc, p_attack, importance, feat_imp=None, k=10):
    pattern, sel = classify_pattern(sub, loc, importance, k)
    meta = sub.meta
    host = meta.loc[loc, "src"]
    lines = [f"ALERT  {host} flagged as malicious (score {p_attack:.2f})"]
    if len(sel):
        lines.append(
            f"Because: {len(sel)} of the {k} most influential flows go from {host} to {sel['dst'].nunique()} "
            f"host(s) over {sel['dport'].nunique()} distinct destination port(s) "
            f"(median {np.median(sel['fwd_pps'] + sel['bwd_pps']):.0f} packets/s, "
            f"median duration {np.median(sel['dur']):.3f}s) -> {pattern} pattern.")
    else:
        lines.append(f"Because: the explanation does not isolate flows from {host}; pattern {pattern}.")
    if feat_imp is not None:
        top = np.argsort(-feat_imp)[:3]
        lines.append("Key traffic features: " + ", ".join(
            f"{FEATURE_LABELS[FEATURES[i]]} ({feat_imp[i]:.2f})" for i in top))
    lines.append("Suggested action: " + ACTIONS[pattern])
    return "\n".join(lines), pattern


def budget(sub, k_frac=0.10):
    """Explanation size: top 10% of the computation subgraph (at least 5 flows)."""
    return max(5, int(round(k_frac * sub.E)))


# ---------------------------------------------------------------- driver
def explain_alert(model, g: GraphData, e: int, methods=METHODS, k_frac: float = 0.10, seed: int = 0,
                  gnnexp_kwargs=None):
    """Explain one predicted alert with several methods and score every explanation."""
    sub, loc = computation_subgraph(g, e)
    cls = 1
    k = budget(sub, k_frac)
    atk = sub.meta.loc[loc, "attack_type"]
    out = {"edge": int(e), "attack_type": atk, "n_sub_edges": sub.E,
           "p_attack": float(model.predict_proba(sub)[loc]), "methods": {}}
    for m in methods:
        if m == "attention" and model.kind != "gat":
            continue
        imp, fimp = explain(model, sub, loc, m, cls, seed, **(gnnexp_kwargs or {}) if m == "gnnexplainer" else {})
        r = fidelity(model, sub, loc, imp, cls, k)
        r["sig_precision"] = signature_precision(sub, loc, imp, k)
        r["feature_hit"] = feature_hit(fimp, atk)
        pat, _ = classify_pattern(sub, loc, imp)
        r["pattern"] = pat
        r["type_agree"] = float(RULE_TO_ATTACK.get(pat) == atk) if atk != "benign" else np.nan
        out["methods"][m] = r
        out["_" + m] = (imp, fimp)
    out["_sub"], out["_loc"] = sub, loc
    return out

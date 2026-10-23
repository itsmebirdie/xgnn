"""Build all figures + results/RESULTS.md from the JSON files written by the experiments."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from xgnn_ids.common import RESULTS, save_json
from xgnn_ids.graph import FEATURE_LABELS

FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)
TEAL, NAVY, ORANGE, RED, GREY = "#12a89d", "#0b2545", "#f4a261", "#e63946", "#9aa5b1"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
load = lambda n: json.load(open(os.path.join(RESULTS, n)))
md = []


def pm(d, nd=3):
    return f"{d['mean']:.{nd}f} ± {d['std']:.{nd}f}"


# ------------------------------------------------------------------ detection
det = load("detection.json")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
cm = det["models"]["gat"]["confusion_sum"]
M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
ax[0].imshow(M, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, f"{M[i, j]:,}", ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else NAVY, fontsize=13)
ax[0].set_xticks([0, 1], ["Benign", "Attack"]); ax[0].set_yticks([0, 1], ["Benign", "Attack"])
ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual"); ax[0].set_title("GAT confusion matrix\n(test windows, 3 seeds summed)")
for kind, c, lab in [("gat", TEAL, "GAT"), ("sage", NAVY, "GraphSAGE-style"), ("mlp", ORANGE, "MLP (no graph)")]:
    r = det["models"][kind]["roc_seed0"]
    ax[1].plot(r["fpr"], r["tpr"], color=c, lw=2, label=f"{lab} (AUC {det['models'][kind]['summary']['auc']['mean']:.3f})")
ax[1].plot([0, 1], [0, 1], "--", color=GREY); ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
ax[1].set_title("ROC (seed 0)"); ax[1].legend(loc="lower right", frameon=False)
h = det["models"]["gat"]["history_seed0"]
ax[2].plot(h["train_loss"], color=TEAL, label="train loss"); ax[2].plot(h["val_loss"], color=NAVY, label="val loss")
ax2 = ax[2].twinx(); ax2.plot(h["train_acc"], color=TEAL, ls=":", label="train acc"); ax2.plot(h["val_acc"], color=NAVY, ls=":", label="val acc")
ax2.spines["right"].set_visible(True)
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("loss"); ax2.set_ylabel("accuracy"); ax[2].set_title("GAT training (seed 0)")
ax[2].legend(loc="upper center", frameon=False, fontsize=8); ax2.legend(loc="center right", frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/01_detection.png"); plt.close()

md += ["## 1. Detection (synthetic benchmark, time-split test windows, mean ± std over seeds)", "",
       "| Model | Accuracy | Precision | Recall | F1 | FPR | AUC |", "|---|---|---|---|---|---|---|"]
names = {"gat": "GAT (2 layers, 4 heads)", "sage": "GraphSAGE-style (mean aggregation)", "mlp": "MLP (flow features only)"}
for k in ["gat", "sage", "mlp"]:
    s = det["models"][k]["summary"]
    md.append(f"| {names[k]} | " + " | ".join(pm(s[x], 4) for x in ["accuracy", "precision", "recall", "f1", "fpr", "auc"]) + " |")
md.append("")

# ------------------------------------------------------------------ explanations
ex = load("explanations.json")
meth = [m for m in ["gnnexplainer", "attention", "saliency", "random"] if m in ex["methods"]]
mlab = {"gnnexplainer": "GNNExplainer", "attention": "Attention rollout", "saliency": "Gradient saliency", "random": "Random"}
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
keys = [("fidelity", "Fidelity"), ("sufficiency", "Sufficiency"), ("necessity", "Necessity"),
        ("sig_precision", "Signature\nprecision"), ("type_agree", "Alert-type\nagreement")]
w = 0.2
cols = [TEAL, NAVY, ORANGE, GREY]
for i, m in enumerate(meth):
    mu = [ex["methods"][m][k]["mean"] for k, _ in keys]
    sd = [ex["methods"][m][k]["std"] for k, _ in keys]
    ax[0].bar(np.arange(len(keys)) + (i - 1.5) * w, np.nan_to_num(mu), w, yerr=np.nan_to_num(sd), color=cols[i], label=mlab[m], capsize=2)
ax[0].set_xticks(range(len(keys)), [l for _, l in keys]); ax[0].set_ylim(0, 1.05); ax[0].legend(frameon=False, fontsize=8, ncol=2)
ax[0].set_title("Explanation quality (top 10% of each alert's computation graph)")
for i, m in enumerate(meth):
    b = ex["fidelity_vs_budget"][m]
    ax[1].plot([float(x) * 100 for x in b], list(b.values()), marker="o", color=cols[i], label=mlab[m])
ax[1].set_xlabel("explanation size (% of computation-graph flows kept/removed)"); ax[1].set_ylabel("fidelity"); ax[1].set_ylim(0, 1)
ax[1].set_title("Fidelity vs explanation size"); ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/02_explainer_comparison.png"); plt.close()

md += ["## 2. Explanation quality (96 alerts explained = 3 seeds × 32, mean ± std over seeds)", "",
       "| Explainer | Fidelity | Sufficiency | Necessity | Signature precision@k | Alert-type agreement | Feature hit@3 |", "|---|---|---|---|---|---|---|"]
for m in meth:
    s = ex["methods"][m]
    fh = pm(s["feature_hit"]) if not np.isnan(s["feature_hit"]["mean"]) else "n/a"
    md.append(f"| {mlab[m]} | " + " | ".join(pm(s[k]) for k in ["fidelity", "sufficiency", "necessity", "sig_precision", "type_agree"]) + f" | {fh} |")
md += ["", "GNNExplainer by attack family (pooled alerts):", "", "| Attack | n | Fidelity | Signature precision | Alert-type agreement | Feature hit@3 |", "|---|---|---|---|---|---|"]
for a, v in ex["gnnexplainer_by_attack_type"].items():
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    md.append(f"| {a} | {v['n']} | {f(v['fidelity'])} | {f(v['sig_precision'])} | {f(v['type_agree'])} | {f(v['feature_hit'])} |")
md.append("")

# example report figure
order = [a for a in ["portscan", "dos", "bruteforce", "infiltration"] if a in ex["examples"]]
if order:
    fig, axes = plt.subplots(len(order), 2, figsize=(12, 3.6 * len(order)))
    axes = np.atleast_2d(axes)
    for r, a in enumerate(order):
        e = ex["examples"][a]
        G = nx.MultiDiGraph()
        agg = {}
        for ed in e["edges"]:
            agg.setdefault((ed["src"], ed["dst"]), []).append(ed["importance"])
        for (s, d), v in agg.items():
            G.add_edge(s, d, w=max(v), n=len(v))
        pos = nx.spring_layout(nx.Graph(G), seed=3, k=1.2)
        host = e["alert_host"]
        nx.draw_networkx_nodes(G, pos, ax=axes[r, 0], node_color=[RED if n == host else ORANGE for n in G], node_size=260)
        nx.draw_networkx_edges(G, pos, ax=axes[r, 0], width=[1 + 4 * G[u][v][0]["w"] for u, v in G.edges()], edge_color=RED, alpha=0.7, arrowsize=12)
        nx.draw_networkx_labels(G, pos, ax=axes[r, 0], font_size=6)
        axes[r, 0].set_title(f"{a}: top-{len(e['edges'])} flows around alert host {host}", fontsize=9); axes[r, 0].axis("off")
        fi = sorted(e["feature_importance"].items(), key=lambda kv: kv[1])
        axes[r, 1].barh([FEATURE_LABELS[k] for k, _ in fi], [v for _, v in fi], color=[TEAL if v > 0.5 else "#a5d8d3" for _, v in fi])
        axes[r, 1].set_xlim(0, 1); axes[r, 1].set_title(f"GNNExplainer feature mask  |  rule says: {e['pattern']}", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{FIG}/03_explanation_reports.png"); plt.close()
    md += ["## 3. Example alert reports (generated from the explanation only, no labels)", ""]
    for a in order:
        md += [f"**{a}**", "```", ex["examples"][a]["report"], "```", ""]

# ------------------------------------------------------------------ drift
pols = ["static", "periodic", "efdm", "efdm_attn", "confidence", "psi", "label_oracle"]
plab = {"static": "No retraining", "periodic": "Periodic (every 2 wk)", "efdm": "EFDM (GNNExplainer fidelity)",
        "efdm_attn": "EFDM-attn (cheap)", "confidence": "Confidence trigger", "psi": "Input-PSI trigger",
        "label_oracle": "Label-monitored (oracle)"}
pcol = {"static": ORANGE, "periodic": NAVY, "efdm": TEAL, "efdm_attn": "#7bcfc8", "confidence": GREY, "psi": "#b08968", "label_oracle": RED}
sig_def = {"fid_gnnexp": ("EFDM: GNNExplainer fidelity", -1), "fid_attn": ("EFDM-attn: attention fidelity", -1),
           "conf": ("Mean confidence", -1), "psi": ("Input PSI (max over features)", +1), "alert_rate": ("Alert rate (|shift|)", 0)}
trig_def = {"f1": lambda t, th: t["f1"] < th["f1"], "fid_gnnexp": lambda t, th: t["fid_gnnexp"] < th["fid_gnnexp"],
            "fid_attn": lambda t, th: t["fid_attn"] < th["fid_attn"], "conf": lambda t, th: t["conf"] < th["conf"],
            "psi": lambda t, th: t["psi"] > th["psi"]}


def drift_section(fname, tag, title):
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return {}
    S = json.load(open(path))["seeds"]

    def series(pol, key):
        return np.array([[t[key] for t in s["policies"][pol]["trajectory"]] for s in S], float)

    fig, ax = plt.subplots(1, 4, figsize=(18, 3.9))
    for pol in ["static", "periodic", "efdm", "label_oracle"]:
        for a, key, ttl in [(ax[0], "f1", "Detection F1"), (ax[1], "recall", "Recall"), (ax[2], "fpr", "False positive rate"),
                            (ax[3], "fid_gnnexp", "Explanation fidelity (GNNExplainer)")]:
            v = series(pol, key); mu, sd = np.nanmean(v, 0), np.nanstd(v, 0)
            a.plot(range(v.shape[1]), mu, marker="o", color=pcol[pol], label=plab[pol], ls="--" if pol == "label_oracle" else "-")
            a.fill_between(range(v.shape[1]), mu - sd, mu + sd, color=pcol[pol], alpha=0.12)
            a.set_title(ttl); a.set_xlabel("weeks of drift")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, y=1.02)
    plt.tight_layout(); plt.savefig(f"{FIG}/04_drift_{tag}.png", bbox_inches="tight"); plt.close()

    agg = {}
    for pol in pols:
        rows = [s["policies"][pol] for s in S]
        agg[pol] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in
                    ["mean_f1", "final_f1", "min_f1", "mean_acc", "mean_fid", "labelled_windows"]}
        agg[pol]["retrains"] = float(np.mean([len(r["retrain_after_weeks"]) for r in rows]))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for pol in pols:
        x, xs = agg[pol]["labelled_windows"]; y, ys = agg[pol]["mean_f1"]
        ax.errorbar(x, y, xerr=xs, yerr=ys, fmt="o", color=pcol[pol], ms=8, capsize=2)
        ax.annotate(plab[pol], (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("windows that had to be labelled (annotation cost)"); ax.set_ylabel("mean F1 over weeks 1-8")
    ax.set_title(f"Policies: accuracy vs labelling cost ({tag})"); plt.tight_layout(); plt.savefig(f"{FIG}/05_policy_tradeoff_{tag}.png"); plt.close()

    out = [f"## {title}", "", f"Seeds: {len(S)}. Mean ± std over seeds.", "",
           "| Policy | Mean F1 (wk 1-8) | Final F1 (wk 8) | Worst-week F1 | Mean fidelity | Retrains | Windows labelled |", "|---|---|---|---|---|---|---|"]
    for pol in pols:
        a = agg[pol]
        out.append(f"| {plab[pol]} | {a['mean_f1'][0]:.3f} ± {a['mean_f1'][1]:.3f} | {a['final_f1'][0]:.3f} ± {a['final_f1'][1]:.3f} | "
                   f"{a['min_f1'][0]:.3f} ± {a['min_f1'][1]:.3f} | {a['mean_fid'][0]:.3f} | {a['retrains']:.1f} | {a['labelled_windows'][0]:.1f} |")
    out.append("")
    f0, a0, f1s = series("static", "fid_gnnexp"), series("static", "accuracy"), series("static", "f1")
    rc, fp = series("static", "recall"), series("static", "fpr")
    out += [f"Static model, week 0 -> week 8: F1 {np.nanmean(f1s[:, 0]):.3f} -> {np.nanmean(f1s[:, -1]):.3f}; recall {np.nanmean(rc[:, 0]):.3f} -> "
            f"{np.nanmean(rc[:, -1]):.3f}; FPR {np.nanmean(fp[:, 0]):.4f} -> {np.nanmean(fp[:, -1]):.4f}; "
            f"GNNExplainer fidelity {np.nanmean(f0[:, 0]):.3f} -> {np.nanmean(f0[:, -1]):.3f}.", ""]

    rows = []
    for key, (lab, sgn) in sig_def.items():
        ys, xs, rho = [], [], []
        for s in S:
            tr = s["policies"]["static"]["trajectory"]
            f1 = np.array([t["f1"] for t in tr]); v = np.array([t[key] for t in tr], float)
            if sgn == 0:
                v = np.abs(v - v[0])
            y = (f1 < f1[0] - 0.05).astype(int)
            ok = ~np.isnan(v)
            ys += list(y[ok]); xs += list(sgn * v[ok] if sgn else v[ok])
            if ok.sum() > 2 and np.std(v[ok]) > 0 and np.std(f1[ok]) > 0:
                rho.append(spearmanr(v[ok], f1[ok])[0])
        auc = roc_auc_score(ys, xs) if 0 < sum(ys) < len(ys) else float("nan")
        rows.append((lab, key, auc, float(np.nanmean(rho)) if rho else float("nan")))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh([r[0] for r in rows][::-1], np.nan_to_num([r[2] for r in rows][::-1]), color=[TEAL if "fid" in r[1] else GREY for r in rows][::-1])
    ax.axvline(0.5, color=RED, ls="--", lw=1); ax.set_xlim(0, 1.02)
    ax.set_xlabel("AUROC for flagging weeks where F1 dropped > 0.05 (label-free signals)")
    ax.set_title(f"Which label-free signal tracks degradation? ({tag})"); plt.tight_layout(); plt.savefig(f"{FIG}/06_monitor_auroc_{tag}.png"); plt.close()
    out += ["Label-free drift signals on the static model. AUROC = separates weeks with real degradation (F1 down > 0.05 vs week 0) "
            "from healthy weeks, pooled over seeds (0.5 = chance, `nan` = no degraded weeks); rho = mean Spearman correlation with F1.", "",
            "| Signal | AUROC | Spearman rho with F1 |", "|---|---|---|"]
    for lab, key, auc, rho in rows:
        out.append(f"| {lab} | {auc:.3f} | {rho:+.3f} |")
    out += ["", "First week each monitor fires on the static model (thresholds calibrated on the clean validation period; `-` = never):", "",
            "| Seed | F1 (needs labels) | EFDM | EFDM-attn | Confidence | PSI |", "|---|---|---|---|---|---|"]
    mon = {}
    for s in S:
        tr, th = s["policies"]["static"]["trajectory"], s["thresholds"]
        fw = {k: next((t["week"] for t in tr[1:] if not np.isnan(t[k]) and fn(t, th)), None) for k, fn in trig_def.items()}
        mon[s["seed"]] = fw
        out.append(f"| {s['seed']} | " + " | ".join(str(fw[k]) if fw[k] is not None else "-" for k in ["f1", "fid_gnnexp", "fid_attn", "conf", "psi"]) + " |")
    out.append("")
    save_json({"auroc": {r[1]: {"auroc": r[2], "spearman": r[3]} for r in rows}, "first_fire_week": mon, "policy_summary": agg},
              f"monitors_{tag}.json")
    return out


md += drift_section("drift.json", "both", "4. Robustness under 8 weeks of drift - attacker evasion + benign shift (main scenario)")
md += drift_section("drift_benign.json", "benign", "5. Ablation: benign-only drift (new services, scanner-like benign hosts; attackers unchanged)")
open(os.path.join(RESULTS, "RESULTS.md"), "w").write("# Results (auto-generated by scripts/make_figures.py)\n\n" + "\n".join(md))
print("figures + RESULTS.md written")

"""Streamlit UI:  streamlit run app/streamlit_app.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from xgnn_ids import app_backend as B
from xgnn_ids import topology as T

st.set_page_config(page_title="Explainable GNN IDS", page_icon="🛡️", layout="wide")
st.title("🛡️ Explainable GNNs for Network Intrusion Detection")
st.caption("Maanas Nair (24BCT0157) · Mohammad Abbas Abidi (24BCT0229) · Computer Networks project. "
           "Data in the live tabs is synthetic (CIC-IDS2017-style).")


@st.cache_resource(show_spinner="Loading model…")
def get_model(seed: int, epochs: int = 25):
    model, scaler, source = B.load_model(seed)
    return model, scaler, source


tab_net, tab_dash, tab_live, tab_drift = st.tabs(["🖧 Network view", "📊 Results dashboard", "🔍 Alert explorer", "🌊 Drift lab"])

# ------------------------------------------------------------------ network view (Packet Tracer style)
with tab_net:
    st.markdown("Traffic on a simulated enterprise network. Links turn **red** where the GAT alerted; select an alert "
                "to overlay *why* as dashed PDU envelopes (the explanation's most important flows).")
    c1, c2, c3, c4 = st.columns(4)
    nseed = c1.number_input("Seed ", 0, 50, 0, key="nseed")
    nweek = c2.slider("Drift week ", 0, 8, 0, key="nweek")
    nmeth = c3.selectbox("Explainer ", list(B.METHOD_LABELS), format_func=B.METHOD_LABELS.get, key="nmeth")
    truth = c4.toggle("Show ground truth (missed attacks in amber)", value=True)
    nmodel, nscaler, nsrc = get_model(int(nseed))
    nsev = nweek / 8
    ndf, ng = B.make_window(nscaler, nsev, nsev, int(nseed) * 100 + nweek)
    nprob = nmodel.predict_proba(ng)
    nal = B.list_alerts(nmodel, ng)
    choice = None
    if len(nal):
        labels = ["(none - show all alerts)"] + [
            f"#{r.edge}  {r.src} → {r.dst}:{r.dport}  score {r.score}  [{r.truth}]" for r in nal.itertuples()]
        pick = st.selectbox("Alert to explain", labels, index=0)
        if pick != labels[0]:
            choice = int(nal.iloc[labels.index(pick) - 1]["edge"])
    res = B.explain_one(nmodel, ng, choice, nmeth) if choice is not None else None
    st.pyplot(T.draw_topology(ng.meta, nprob, res, show_truth=truth))
    if res is not None:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Event list** (explanation's top flows, Packet-Tracer style)")
            st.dataframe(T.event_list(res), hide_index=True, use_container_width=True)
        with right:
            st.markdown("**Alert report**")
            st.code(res["report"], language=None)
            st.caption(f"Fidelity {res['fidelity']['fidelity']:.2f} · explanation = top {res['k']} of {res['sub'].E} flows")
    else:
        st.caption(f"{len(nal)} alerts in this window; {int(ng.y.sum())} truly malicious flows.")

# ------------------------------------------------------------------ dashboard
with tab_dash:
    det, ex = B.load_result("detection.json"), B.load_result("explanations.json")
    if det is None:
        st.warning("No results found. Run `bash scripts/run_all.sh` first.")
    else:
        s = det["models"]["gat"]["summary"]
        c = st.columns(5)
        for col, (k, lab) in zip(c, [("accuracy", "Accuracy"), ("recall", "Recall"), ("fpr", "False-pos. rate"),
                                     ("f1", "F1"), ("auc", "AUC")]):
            col.metric(f"GAT {lab}", f"{s[k]['mean']:.4f}", f"± {s[k]['std']:.4f}", delta_color="off")
        rows = [{"Model": n, **{k: round(det["models"][m]["summary"][k]["mean"], 4)
                                for k in ["accuracy", "precision", "recall", "f1", "fpr", "auc"]}}
                for m, n in [("gat", "GAT"), ("sage", "GraphSAGE-style"), ("mlp", "MLP (no graph)")]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if B.figure_path("01_detection.png"):
            st.image(B.figure_path("01_detection.png"))
        if ex:
            st.subheader("Explanation quality")
            rows = [{"Explainer": B.METHOD_LABELS[m], **{k: round(v[k]["mean"], 3) for k in
                     ["fidelity", "sufficiency", "necessity", "sig_precision", "type_agree"]}}
                    for m, v in ex["methods"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if B.figure_path("02_explainer_comparison.png"):
                st.image(B.figure_path("02_explainer_comparison.png"))
            with st.expander("Example alert reports"):
                for a, e in ex["examples"].items():
                    st.markdown(f"**{a}**")
                    st.code(e["report"])
        st.subheader("Robustness under drift")
        for tag, ttl in [("both", "Attacker evasion + benign shift"), ("benign", "Benign-only drift (ablation)")]:
            if B.figure_path(f"04_drift_{tag}.png"):
                st.markdown(f"**{ttl}**")
                st.image(B.figure_path(f"04_drift_{tag}.png"))
        st.info("Finding: explanation fidelity (EFDM) did **not** flag evasion-type drift, because it is only measured on "
                "alerts the model raises. See README and results/RESULTS.md.")
        if os.path.exists(os.path.join(B.RESULTS, "RESULTS.md")):
            with st.expander("Full results tables"):
                st.markdown(open(os.path.join(B.RESULTS, "RESULTS.md")).read())

# ------------------------------------------------------------------ alert explorer
with tab_live:
    st.markdown("Generate a traffic window, see which flows the GAT flags, and ask *why* for any alert.")
    c1, c2, c3, c4 = st.columns(4)
    seed = c1.number_input("Seed", 0, 50, 0)
    week = c2.slider("Drift week (0 = clean)", 0, 8, 0)
    method = c3.selectbox("Explainer", list(B.METHOD_LABELS), format_func=B.METHOD_LABELS.get)
    k_frac = c4.slider("Explanation size (% of flows)", 2, 40, 10) / 100
    model, scaler, source = get_model(int(seed))
    st.caption("Model: saved checkpoint (paper config)" if source == "checkpoint" else "Model: quick demo model - run `python scripts/train_model.py --seed %d` to use the paper-config model" % int(seed))
    sev = week / 8
    df, g = B.make_window(scaler, sev, sev, int(seed) * 100 + week)
    m = B.window_metrics(model, g)
    st.caption(f"Window: {g.E} flows, {g.n} hosts, {int(g.y.sum())} truly malicious flows")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{m['accuracy']:.3f}"); cols[1].metric("Recall", f"{m['recall']:.3f}")
    cols[2].metric("False-pos. rate", f"{m['fpr']:.4f}"); cols[3].metric("F1", f"{m['f1']:.3f}")
    alerts = B.list_alerts(model, g)
    if alerts.empty:
        st.warning("The model raised no alerts in this window.")
    else:
        st.write(f"**{len(alerts)} alerts raised.** Select one:")
        ev = st.dataframe(alerts, hide_index=True, use_container_width=True, on_select="rerun",
                          selection_mode="single-row", height=240)
        sel = ev.selection.rows[0] if ev.selection.rows else 0
        edge = int(alerts.iloc[sel]["edge"])
        with st.spinner("Explaining…"):
            res = B.explain_one(model, g, edge, method, k_frac)
        left, right = st.columns([3, 2])
        with left:
            st.pyplot(B.plot_explanation(res))
        with right:
            st.subheader("Alert report")
            st.code(res["report"], language=None)
            f = res["fidelity"]
            st.metric("Fidelity", f"{f['fidelity']:.3f}", help="mean of sufficiency and necessity of the top flows")
            st.write(f"Sufficiency {f['sufficiency']:.2f} · Necessity {f['necessity']:.2f} · "
                     f"explanation = top {res['k']} of {res['sub'].E} flows")
            sp = res["sig_precision"]
            st.write("Signature precision (needs ground truth): " + ("n/a (false alarm)" if sp != sp else f"{sp:.2f}"))
        st.dataframe(B.top_flows_table(res), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ drift lab
with tab_drift:
    st.markdown("Push the traffic away from the training distribution and watch the *static* model degrade.")
    c1, c2 = st.columns(2)
    ev_s = c1.slider("Attacker evasion", 0.0, 1.0, 0.5, 0.05)
    bn_s = c2.slider("Benign shift", 0.0, 1.0, 0.0, 0.05)
    model, scaler, _ = get_model(0)
    rows = []
    for i in range(3):
        _, gg = B.make_window(scaler, ev_s, bn_s, 900 + i)
        rows.append(B.window_metrics(model, gg))
    avg = {k: sum(r[k] for r in rows) / 3 for k in ["accuracy", "recall", "fpr", "f1"]}
    cols = st.columns(4)
    for col, (k, lab) in zip(cols, [("accuracy", "Accuracy"), ("recall", "Recall"), ("fpr", "False-pos. rate"), ("f1", "F1")]):
        col.metric(lab, f"{avg[k]:.3f}")
    st.caption("Average of 3 fresh windows. High evasion mostly lowers **recall** (missed attacks), which explanations "
               "of raised alerts cannot reveal.")

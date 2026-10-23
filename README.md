# Explainable GNNs for Network Intrusion Detection

Computer Networks subject project - Maanas Nair (24BCT0157), Mohammad Abbas Abidi (24BCT0229).

A graph attention network (GAT) classifies network flows (hosts = nodes, flows = edges) as benign or attack, and an
explanation layer traces every alert back to the **flows, hosts and traffic features** behind it. The project then
asks two questions that accuracy-only IDS papers skip: *are the explanations faithful and do they match real attack
signatures?* and *what happens to detection and explanations as traffic drifts?*

> **Read this first - what is and is not validated**
> * All reported numbers come from a **synthetic** CIC-IDS2017-style generator (`xgnn_ids/data/synthetic.py`), because
>   the real datasets could not be downloaded when this repo was built. The generator has ground truth for which flows
>   belong to which attack campaign, which is what makes explanation scoring possible. Its base period is easy
>   (GAT ~ 1.00 F1); expect lower numbers on real data.
> * Loaders for **CIC-IDS2017** and **UNSW-NB15** exist (`xgnn_ids/data/loaders.py`) and pass tests on files that mimic
>   their documented column layouts, but have **not been run on the real downloads**. `scripts/run_real_dataset.py`
>   is likewise unrun. Treat the first real run as debugging.
> * Implemented in **pure NumPy** (own autograd, gradient-checked) rather than PyTorch Geometric, because PyTorch was
>   unavailable in the build environment. A PyG port is straightforward future work.
> * The rule-based alert typer (`classify_pattern`) was revised once after a failure analysis on a separate
>   development seed (50) that is not used in any reported result.
> * No prior-art or patent search was done; "novelty" below means "not in the baseline plan", not a verified claim.
  See `docs/INVENTION_DISCLOSURE.md` for the defined inventive concept, what the evidence supports, and a warning
  about publishing before filing.

## Pipeline
1. **Graph construction** (`graph.py`): flows -> 8 features (packets/s, dest-port count, SYN count, duration, fwd packet
   length, protocol, bwd packets/s, dst port); standardised on training windows only.
2. **Detector** (`gnn.py`): 2-layer, 4-head edge-level GAT over incoming and outgoing flows, mask-aware soft degrees,
   class-weighted cross-entropy, Adam lr 1e-3, 40 epochs. GraphSAGE-style and MLP variants share the code.
3. **Explanation layer** (`explain.py`): GNNExplainer (edge + feature masks, 100 epochs), attention rollout,
   gradient saliency, random control. Each alert is explained on its exact 2-hop computation subgraph (tested to
   reproduce the full-graph output to 1e-16).
4. **Evaluation**: detection metrics, plus *fidelity* (sufficiency + necessity of the top 10% of flows, measured
   against the model's own prediction - no labels needed), *signature precision* (do the top flows belong to the real
   attack campaign?), *alert-type agreement* (does a rule-based report built only from the explanation name the right
   attack family?).
5. **Robustness** (`drift.py`): 8 weeks of progressive drift (attackers evade, benign traffic shifts) and seven
   retraining policies, including the project's proposed **EFDM** (Explanation-Fidelity Drift Monitor).

## Results (3 seeds; full tables in `results/RESULTS.md`, figures in `results/figures/`)
**Detection** (time-split test windows): GAT F1 0.999, GraphSAGE-style 1.000, MLP without graph 0.933
(FPR 3.0% vs 0.02%). The graph gives a clear gain over flow features alone, mainly on structure-only attacks.

**Explanations** (96 alerts): GNNExplainer has the best signature precision (0.82 vs attention 0.74, saliency 0.63,
random 0.18) and its explanation-only reports name the right attack family 94% of the time (see caveat on rules).
Fidelity is **not** a differentiator: GNNExplainer 0.55, attention 0.57, saliency 0.56 (within noise), all above
random 0.39. Necessity is low (~0.13) because attacks are redundant: dozens of flows carry the evidence, so removing
10% rarely flips the verdict. Sufficiency is ~0.98.

**Drift** (attacker evasion + benign shift): without retraining F1 falls from 1.00 to 0.75 by week 8 (recall 1.00 ->
0.60). Periodic retraining keeps mean F1 at 0.988.

**Negative result for EFDM.** Explanation fidelity barely moves (0.568 -> 0.536) because the failure mode is *missed*
attacks, and fidelity is only computed on alerts the model raises. EFDM never triggered a retrain in any seed.
Label-free baselines worked better: confidence (fires in all seeds), input PSI (2 of 3), alert-rate shift (best AUROC
0.955). See `results/RESULTS.md` section 4, and section 5 for the benign-only drift ablation.
**Benign-only drift ablation:** the GAT is not hurt at all (F1 stays ~1.000, FPR unchanged), so there is nothing for any
monitor to detect, and no retraining policy helps. Only PSI occasionally fired (false alarms in 2 of 3 seeds). EFDM's
usefulness is therefore *not demonstrated* in either scenario.
So the honest contribution is the evaluation protocol and the finding that **explanations of raised alerts cannot
reveal evasion-type drift**; an explanation-based monitor would need to audit near-threshold *non*-alerts.
Note: alert-type agreement for GNNExplainer came out as exactly 0.9375 in all three seeds; plausible (30/32) but
worth re-checking.

## Reproduce
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t . -v      # 14 tests
bash scripts/run_all.sh                           # ~25 min on one CPU core: all experiments + figures
python scripts/run_drift.py --scenario benign --out drift_benign.json   # ablation
python scripts/make_figures.py
python scripts/demo_alert.py                      # print human-readable alerts
```

## Running on the real datasets
```bash
python scripts/run_real_dataset.py --dataset cicids2017 --path /data/TrafficLabelling --windows 20 --window-size 2000
python scripts/run_real_dataset.py --dataset unsw-nb15  --path /data/UNSW-NB15 --windows 20
```
Real labels map to the families dos / portscan / bruteforce / infiltration / other; campaigns are approximated as
(family, source IP). UNSW-NB15 has no SYN-count column (a `synack>0` proxy is used) and no brute-force class.

## Layout
`xgnn_ids/` autograd, graph, model, explainers, drift monitor, data (synthetic + loaders) - `scripts/` experiments -
`tests/` - `results/` JSON, logs, figures, RESULTS.md.

## Streamlit UI
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
Four tabs: **Network view** (Packet-Tracer-style topology: Internet, firewall, router, core and access switches,
servers and PCs; links turn red under alerted flows, a selected alert's explanation is overlaid as dashed PDU
envelopes with an event list and report), **Results dashboard** (reads `results/*.json` and figures written by `run_all.sh`), **Alert explorer**
(generate a traffic window, pick a drift week, click an alert, choose an explainer, see the explanation graph, feature
mask, fidelity and the generated alert report), and **Drift lab** (sliders for attacker evasion / benign shift on the
static model). The live tabs load `results/model_gat_seed0.npz` (written by `scripts/train_model.py`, also called from `run_all.sh`); if it is missing they train a quick demo model instead.
The UI logic lives in `xgnn_ids/app_backend.py` and is exercised by `tests/test_app_backend.py`; the Streamlit layer
itself (`app/streamlit_app.py`) was syntax-checked but **not launched** where this repo was built (Streamlit could not
be installed), so report any widget issues.

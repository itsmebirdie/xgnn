# Invention disclosure draft — explanation-grounded alerting for graph-based network intrusion detection

**Status: working draft for discussion with your university IPR cell / a patent attorney. Not legal advice.**

## 0. Read before doing anything else
1. **Do not make the GitHub repo public (or publish the paper/slides) before you have filed.** Public disclosure can
   destroy novelty in many jurisdictions; grace periods are short and not universal. Keep the repo private until an
   application (e.g. an Indian provisional) is filed.
2. **No prior-art search has been done.** Everything below labelled "novel" means "not part of the baseline plan",
   not "searched and found new". Section 7 lists what to search.
3. **Eligibility.** In India, Section 3(k) excludes "computer programme per se"; applications usually need to
   show a concrete technical effect/technical contribution in the network-security system. Draft claims should be
   system/method claims tied to network traffic processing, not "an algorithm".
4. **Claim only what the evidence supports** (Section 4). The original headline idea (EFDM) did **not** work in our
   experiments and must not be presented as a benefit.

## 1. Problem
GNN-based flow classifiers detect attacks accurately but emit a score with no reason; operators cannot verify the
alert or choose a response. Existing explainers are generic (applied to the model after the fact, on the whole graph)
and their output is a mask, not an operator-actionable statement; and explanation quality is rarely checked
against what the attack actually looked like.

## 2. Summary of the proposed method (what the code does)
A computer-implemented method comprising:

1. **Graph construction:** receiving network flow records; building a graph whose nodes are hosts and whose directed
   edges are flows, each edge carrying a feature vector (rate, duration, flag counts, ports, per-source distinct
   destination-port count).
2. **Mask-aware edge classifier:** classifying each flow with an attention-based graph neural network that aggregates
   over both incoming and outgoing flows of each host, whose host input is a *soft* degree computed from per-flow
   mask values, and in which a per-flow mask scales *both* that flow's messages and its own feature vector. A mask
   value of zero therefore removes the flow from the computation entirely; intermediate values are differentiable.
3. **Exact alert subgraph:** for an alerted flow, extracting the computation subgraph consisting of all flows
   incident to the one-hop neighbourhood of the flow's two end hosts, and carrying the degree contributed by
   excluded flows as constants, such that the classifier's score for the alerted flow on the subgraph equals its
   score on the full graph (verified to 1e-16).
4. **Importance estimation:** learning (or otherwise computing) importance values over the subgraph's flows and over
   the feature dimensions.
5. **Label-free explanation check:** computing a fidelity value as the combination of (a) the score retained when only
   the top-k flows are kept and (b) the score lost when the top-k flows are removed, relative to the classifier's own
   score; and attaching it to the alert.
6. **Explanation-to-action translation:** deriving, from *only* the explanation-selected flows originating at the alerted
   host, a descriptor (number of distinct destination hosts, distinct destination ports, flow multiplicity per
   destination, share of flows on administrative ports), mapping it through a rule table to an attack family
   (port-scan-like, lateral-movement-like, brute-force-like, flood-like), and outputting a human-readable report
   with the supporting flows and a recommended action.
7. **Development-time validation (optional step):** scoring explanations against known attack-campaign membership
   (signature precision@k) to select the explainer/budget.

## 3. Draft claim language (illustrative only — an attorney must rewrite)
**Claim 1 (method).** A method of reporting a network intrusion, comprising: constructing a host–flow graph from
network flow records; computing, by an attention-based graph neural network having a per-flow mask that scales both
the flow's messages and the flow's features, an attack score for each flow; for a flow whose score exceeds a
threshold, extracting a computation subgraph comprising all flows incident to hosts within one hop of the flow's
end hosts while retaining the degree contribution of omitted flows as constants such that the score of the flow on
the subgraph equals its score on the graph; determining importance values for flows of the subgraph; selecting a
top-k set of flows by importance; classifying the alert into an attack family by applying rules to statistics of
the selected flows originating at the alerted host; and outputting a report identifying the selected flows, the
attack family, and a recommended mitigation.

**Claim 2.** …wherein the host input to the network is a logarithm of a soft degree equal to the sum of mask values of
the host's incident flows plus the constant degree of omitted flows.
**Claim 3.** …further comprising computing a fidelity value as defined in step 5 and outputting it with the report.
**Claim 4.** …wherein the importance values are obtained by optimising a sparse flow mask and feature mask that
preserve the score. **Claim 5.** …wherein the rules use the number of distinct destination ports, distinct
destination hosts, per-destination flow multiplicity, and the share of flows to administrative ports.
**Claim 6 (system).** A system comprising a flow collector, a processor and memory storing instructions that cause
the processor to perform the method of claim 1. **Claim 7.** A non-transitory medium storing such instructions.

## 4. What the experiments do and do not support (synthetic data only)
| Element | Evidence in this repo | Safe to rely on? |
|---|---|---|
| Subgraph reproduces full-graph score (step 3) | unit test, max error 1.7e-16 | Yes — deterministic property |
| Mask-aware classifier detects attacks (step 2) | GAT F1 ≈ 0.999 vs MLP 0.933 on synthetic data | Only on synthetic data |
| Explanation selects the real attack flows (step 4/7) | GNNExplainer signature precision 0.82 vs attention 0.74, saliency 0.63, random 0.18 | Yes on synthetic; needs real data |
| Report names the right attack family (step 6) | 94% agreement (rules revised once after a dev-seed failure analysis) | Yes on synthetic; rules are hand-written |
| Fidelity value (step 5) | computed; GNNExplainer 0.55 ≈ attention 0.57 ≈ saliency 0.56 | **No claim that it predicts correctness or beats other explainers** |
| EFDM: fidelity as drift trigger | never fired; evasion drift lowers recall, fidelity is only measured on raised alerts | **Do not claim.** Negative result |
| Real-data performance | none (loaders untested on real files) | **No** |

## 5. Where the inventive concept plausibly lies
Individually GAT classifiers, GNNExplainer, attention weights and rule-based triage are known. The candidate
contribution is the **combination as a technical pipeline**: a classifier built so flows can be exactly removed
(step 2) + exact, provably score-preserving alert subgraphs (step 3) + translation of the *selected flows only*
into a typed, actionable alert (step 6). Most defensible technical effect to articulate: operators receive a
verifiable, flow-level justification and attack-family hypothesis computed on a small subgraph that is guaranteed
to reproduce the detector's score. Weaker: fidelity attachment (step 5).

## 6. Things to do before filing (to make claims stronger)
* Run on real CIC-IDS2017 / UNSW-NB15 (`scripts/run_real_dataset.py`) and report signature precision and family
  agreement there.
* Measure the compute benefit of step 3 (explanation time/memory on subgraph vs full graph, larger graphs).
* Ablate step 2's mask-aware design (explain a model without it: do keep-only/remove tests become inconsistent?).
* Replace hand-written rules with a learned or configurable rule table and show it generalises across datasets.
* Optional extension aimed at the recall-side blind spot found here (auditing near-threshold *non*-alerts) — only
  claimable if shown to work.

## 7. Prior-art search plan (items from memory — verify every one)
* Search Google Patents, Espacenet, WIPO PATENTSCOPE, Indian InPASS, IEEE Xplore, arXiv. Keywords: "explainable graph
  neural network intrusion detection", "flow graph edge classification attention alert explanation", "GNNExplainer
  network traffic", "alert triage explanation attack type recommendation", "computation subgraph explanation
  fidelity".
* Known related work to read and distinguish: GNNExplainer (Ying et al., 2019); edge-classification GNN IDS work
  such as E-GraphSAGE (Lo et al., 2022); SHAP/LIME-based explainable IDS literature; survey papers on explainable
  AI for intrusion detection. CPC area to check: H04L 63/14 (network security, monitoring/detecting malicious
  traffic) and the graph-neural-network subclasses of G06N.
* Record, for each close hit, which of steps 2, 3, 5, 6 it already teaches; narrow the claims to what remains.

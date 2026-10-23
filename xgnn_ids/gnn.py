"""Edge-level graph attention network for flow classification.

Architecture (``kind='gat'``):
  * node input  : [log(1+in-degree), log(1+out-degree)]   (soft degrees, mask-aware)
  * L layers    : multi-head attention over incoming AND outgoing flows;
                  attention uses (centre node, neighbour node, flow features)
  * edge head   : MLP([h_src, h_dst, flow-feature embedding]) -> {benign, attack}

Variants sharing the same code path:
  ``kind='sage'`` : uniform (mean) aggregation instead of attention  (GraphSAGE-style)
  ``kind='mlp'``  : no message passing, flow features only            (baseline)

Explanation hooks: ``emask`` scales each flow's messages *and* its features
(a masked flow is "removed"), ``fmask`` scales feature columns.
"""
from __future__ import annotations

import copy

import numpy as np

from . import autograd as ag
from .autograd import Adam, Tensor
from .graph import GraphData


def _glorot(rng, a, b):
    lim = np.sqrt(6.0 / (a + b))
    return Tensor(rng.uniform(-lim, lim, size=(a, b)), requires_grad=True)


def _zeros(*shape):
    return Tensor(np.zeros(shape), requires_grad=True)


class EdgeGNN:
    def __init__(self, in_dim: int, hidden: int = 32, heads: int = 4, layers: int = 2,
                 kind: str = "gat", seed: int = 0):
        assert kind in ("gat", "sage", "mlp")
        assert hidden % heads == 0
        self.in_dim, self.hidden, self.heads, self.layers, self.kind = in_dim, hidden, heads, layers, kind
        self.hd = hidden // heads
        rng = np.random.default_rng(seed)
        P = {}
        P["W0"], P["b0"] = _glorot(rng, 2, hidden), _zeros(1, hidden)
        hd = self.hd
        if kind != "mlp":
            for l in range(layers):
                for dr in ("in", "out"):
                    for k in range(heads):
                        t = f"{l}{dr}{k}"
                        P["Wp" + t] = _glorot(rng, hidden, hd)
                        P["Wq" + t] = _glorot(rng, in_dim, hd)
                        if kind == "gat":
                            P["ac" + t] = _glorot(rng, hd, 1)
                            P["an" + t] = _glorot(rng, hd, 1)
                            P["aq" + t] = _glorot(rng, hd, 1)
                P[f"Wo{l}"] = _glorot(rng, hidden * 3, hidden)
                P[f"bo{l}"] = _zeros(1, hidden)
        P["Wx"], P["bx"] = _glorot(rng, in_dim, hidden), _zeros(1, hidden)
        head_in = hidden * (1 if kind == "mlp" else 3)
        P["W1"], P["b1"] = _glorot(rng, head_in, hidden), _zeros(1, hidden)
        P["W2"], P["b2"] = _glorot(rng, hidden, 2), _zeros(1, 2)
        self.P = P

    # ------------------------------------------------------------------ utils
    def params(self):
        return list(self.P.values())

    def state(self):
        return {k: v.data.copy() for k, v in self.P.items()}

    def load(self, st):
        for k, v in st.items():
            self.P[k].data = v.copy()

    def clone(self):
        m = copy.copy(self)
        m.P = {k: Tensor(v.data.copy(), requires_grad=True) for k, v in self.P.items()}
        return m

    # ---------------------------------------------------------------- forward
    def forward(self, g: GraphData, emask: Tensor | None = None, fmask: Tensor | None = None,
                collect_attn: bool = False):
        P = self.P
        E = g.E
        emask = emask if emask is not None else Tensor(np.ones((E, 1)))
        Xm = Tensor(g.X)
        if fmask is not None:
            Xm = Xm * fmask
        Xm = Xm * emask  # masked flow -> features pulled to the training mean (0 after standardisation)

        attn = {}
        if self.kind != "mlp":
            din = ag.scatter(emask, g.seg_dst) + Tensor(g.off_in)
            dout = ag.scatter(emask, g.seg_src) + Tensor(g.off_out)
            H = ag.relu(ag.concat([ag.log1p(din), ag.log1p(dout)]) @ P["W0"] + P["b0"])
            for l in range(self.layers):
                outs = []
                for dr in ("in", "out"):
                    ctr, nbr = (g.seg_dst, g.seg_src) if dr == "in" else (g.seg_src, g.seg_dst)
                    inv = g.inv_deg_in if dr == "in" else g.inv_deg_out
                    heads_out = []
                    for k in range(self.heads):
                        t = f"{l}{dr}{k}"
                        Pn = H @ P["Wp" + t]
                        Q = Xm @ P["Wq" + t]
                        msg = ag.gather(Pn, nbr) + Q
                        if self.kind == "gat":
                            sc = (ag.gather(Pn @ P["ac" + t], ctr) + ag.gather(Pn @ P["an" + t], nbr)
                                  + Q @ P["aq" + t])
                            alpha = ag.segment_softmax(ag.leaky_relu(sc), ctr)
                        else:
                            alpha = Tensor(inv[ctr.idx][:, None])
                        if collect_attn:
                            attn[(l, dr, k)] = alpha.data[:, 0].copy()
                        heads_out.append(ag.scatter(msg * (alpha * emask), ctr))
                    outs.append(ag.concat(heads_out))
                H = ag.relu(ag.concat([H] + outs) @ P[f"Wo{l}"] + P[f"bo{l}"])
        fe = ag.relu(Xm @ P["Wx"] + P["bx"])
        z = fe if self.kind == "mlp" else ag.concat([ag.gather(H, g.seg_src), ag.gather(H, g.seg_dst), fe])
        z = ag.relu(z @ P["W1"] + P["b1"])
        logits = z @ P["W2"] + P["b2"]
        return (logits, attn) if collect_attn else logits

    def predict_proba(self, g: GraphData, emask=None, fmask=None) -> np.ndarray:
        lg = self.forward(g, emask, fmask).data
        lg = lg - lg.max(axis=1, keepdims=True)
        e = np.exp(lg)
        return (e / e.sum(axis=1, keepdims=True))[:, 1]


def class_weights(graphs) -> np.ndarray:
    y = np.concatenate([g.y for g in graphs])
    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - y.sum()))
    return np.array([len(y) / (2.0 * neg), len(y) / (2.0 * pos)])


def evaluate_loss(model, graphs, cw):
    tot, n = 0.0, 0
    for g in graphs:
        lp = ag.log_softmax(model.forward(g))
        tot += ag.weighted_nll(lp, g.y, cw).data * g.E
        n += g.E
    return float(tot / n)


def train(model: EdgeGNN, train_graphs, val_graphs=None, epochs: int = 40, lr: float = 1e-3,
          seed: int = 0, verbose: bool = False, cw=None):
    """Per-graph Adam training with class-weighted cross-entropy; keeps best-val weights."""
    rng = np.random.default_rng(seed)
    cw = class_weights(train_graphs) if cw is None else cw
    opt = Adam(model.params(), lr=lr)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state = np.inf, model.state()
    for ep in range(epochs):
        order = rng.permutation(len(train_graphs))
        for gi in order:
            g = train_graphs[gi]
            opt.zero_grad()
            loss = ag.weighted_nll(ag.log_softmax(model.forward(g)), g.y, cw)
            loss.backward()
            opt.step()
        hist["train_loss"].append(evaluate_loss(model, train_graphs, cw))
        hist["train_acc"].append(accuracy(model, train_graphs))
        if val_graphs:
            vl = evaluate_loss(model, val_graphs, cw)
            hist["val_loss"].append(vl)
            hist["val_acc"].append(accuracy(model, val_graphs))
            if vl < best:
                best, best_state = vl, model.state()
        if verbose:
            print(f"ep {ep+1:3d} train {hist['train_loss'][-1]:.4f} val {hist['val_loss'][-1] if val_graphs else float('nan'):.4f}")
    if val_graphs:
        model.load(best_state)
    return hist


def accuracy(model, graphs) -> float:
    c, n = 0, 0
    for g in graphs:
        p = model.predict_proba(g)
        c += int(((p > 0.5).astype(int) == g.y).sum())
        n += g.E
    return c / n

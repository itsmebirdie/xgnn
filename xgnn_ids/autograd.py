"""Minimal reverse-mode autograd on NumPy arrays.

Only the operations needed by the edge-level GAT and the GNNExplainer are
implemented. Gradients are verified against finite differences in
``tests/test_autograd.py``.

Scatter / gather over graph structure use a precomputed sparse incidence
matrix (:class:`Seg`) so they are fast on a single CPU core.
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.sparse as sp


class Seg:
    """Segment index: maps E items to n segments (e.g. edges -> their dst node)."""

    def __init__(self, idx: np.ndarray, n: int):
        self.idx = np.asarray(idx, dtype=np.int64)
        self.n = int(n)
        e = len(self.idx)
        self.S = sp.csr_matrix(
            (np.ones(e), (self.idx, np.arange(e))), shape=(self.n, e)
        )


def _unbroadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, s in enumerate(shape):
        if s == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


class Tensor:
    def __init__(self, data, parents=(), backward=None, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._parents = parents
        self._backward = backward
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    @property
    def shape(self):
        return self.data.shape

    def _acc(self, g):
        if not self.requires_grad:
            return
        self.grad = np.array(g, dtype=np.float64) if self.grad is None else self.grad + g

    def backward(self):
        order, seen = [], set()
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 50000))

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)

        try:
            visit(self)
        finally:
            sys.setrecursionlimit(old)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            if t._backward is not None and t.grad is not None:
                t._backward(t.grad)

    def zero_grad(self):
        self.grad = None

    def __add__(self, o): return add(self, _t(o))
    __radd__ = __add__
    def __sub__(self, o): return add(self, neg(_t(o)))
    def __rsub__(self, o): return add(_t(o), neg(self))
    def __mul__(self, o): return mul(self, _t(o))
    __rmul__ = __mul__
    def __neg__(self): return neg(self)
    def __matmul__(self, o): return matmul(self, _t(o))


def _t(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def add(a, b):
    out = Tensor(a.data + b.data, (a, b))
    def bw(g):
        a._acc(_unbroadcast(g, a.shape)); b._acc(_unbroadcast(g, b.shape))
    out._backward = bw
    return out


def neg(a):
    out = Tensor(-a.data, (a,))
    out._backward = lambda g: a._acc(-g)
    return out


def mul(a, b):
    out = Tensor(a.data * b.data, (a, b))
    def bw(g):
        a._acc(_unbroadcast(g * b.data, a.shape)); b._acc(_unbroadcast(g * a.data, b.shape))
    out._backward = bw
    return out


def matmul(a, b):
    out = Tensor(a.data @ b.data, (a, b))
    def bw(g):
        a._acc(g @ b.data.T); b._acc(a.data.T @ g)
    out._backward = bw
    return out


def relu(a):
    m = a.data > 0
    out = Tensor(a.data * m, (a,))
    out._backward = lambda g: a._acc(g * m)
    return out


def leaky_relu(a, slope=0.2):
    m = np.where(a.data > 0, 1.0, slope)
    out = Tensor(a.data * m, (a,))
    out._backward = lambda g: a._acc(g * m)
    return out


def sigmoid(a):
    s = 1.0 / (1.0 + np.exp(-np.clip(a.data, -50, 50)))
    out = Tensor(s, (a,))
    out._backward = lambda g: a._acc(g * s * (1 - s))
    return out


def log(a, eps=1e-12):
    out = Tensor(np.log(a.data + eps), (a,))
    out._backward = lambda g: a._acc(g / (a.data + eps))
    return out


def log1p(a):
    out = Tensor(np.log1p(a.data), (a,))
    out._backward = lambda g: a._acc(g / (1.0 + a.data))
    return out


def tsum(a, axis=None, keepdims=False):
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), (a,))
    def bw(g):
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a._acc(np.broadcast_to(g, a.shape))
    out._backward = bw
    return out


def tmean(a):
    return tsum(a) * (1.0 / a.data.size)


def gather(a, seg: Seg):
    """a[seg.idx] (rows)."""
    out = Tensor(a.data[seg.idx], (a,))
    out._backward = lambda g: a._acc(seg.S @ g)
    return out


def scatter(x, seg: Seg):
    """Sum rows of x into seg.n segments."""
    out = Tensor(seg.S @ x.data, (x,))
    out._backward = lambda g: x._acc(g[seg.idx])
    return out


def concat(ts, axis=1):
    out = Tensor(np.concatenate([t.data for t in ts], axis=axis), tuple(ts))
    sizes = [t.shape[axis] for t in ts]
    def bw(g):
        parts = np.split(g, np.cumsum(sizes)[:-1], axis=axis)
        for t, p in zip(ts, parts):
            t._acc(p)
    out._backward = bw
    return out


def segment_softmax(s, seg: Seg):
    """Softmax of scores s (E,1) within each segment."""
    sd = s.data[:, 0]
    mx = np.full(seg.n, -np.inf)
    np.maximum.at(mx, seg.idx, sd)
    e = np.exp(sd - mx[seg.idx])
    den = np.asarray(seg.S @ e).ravel()
    alpha = e / den[seg.idx]
    out = Tensor(alpha[:, None], (s,))
    def bw(g):
        gg = g[:, 0]
        dot = np.asarray(seg.S @ (alpha * gg)).ravel()
        s._acc((alpha * (gg - dot[seg.idx]))[:, None])
    out._backward = bw
    return out


def log_softmax(z):
    m = z.data.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z.data - m).sum(axis=1, keepdims=True)) + m
    out = Tensor(z.data - lse, (z,))
    def bw(g):
        p = np.exp(out.data)
        z._acc(g - p * g.sum(axis=1, keepdims=True))
    out._backward = bw
    return out


def weighted_nll(logp, y, class_w):
    """Class-weighted cross-entropy: -sum w_i logp[i,y_i] / sum w_i."""
    y = np.asarray(y, dtype=np.int64)
    w = np.asarray(class_w)[y]
    idx = np.arange(len(y))
    val = -(w * logp.data[idx, y]).sum() / w.sum()
    out = Tensor(val, (logp,))
    def bw(g):
        gl = np.zeros_like(logp.data)
        gl[idx, y] = -w / w.sum()
        logp._acc(gl * g)
    out._backward = bw
    return out


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = list(params), lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data = p.data - self.lr * mh / (np.sqrt(vh) + self.eps)

import unittest

import numpy as np

from xgnn_ids import autograd as ag
from xgnn_ids.autograd import Seg, Tensor
from xgnn_ids.gnn import EdgeGNN
from xgnn_ids.graph import GraphData


def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def toy_graph(rng, n=8, E=30, d=8):
    src = rng.integers(0, n, E); dst = rng.integers(0, n, E)
    din = np.bincount(dst, minlength=n).astype(float); dout = np.bincount(src, minlength=n).astype(float)
    return GraphData(n=n, src=src, dst=dst, X=rng.normal(size=(E, d)), y=rng.integers(0, 2, E),
                     off_in=np.zeros((n, 1)), off_out=np.zeros((n, 1)),
                     inv_deg_in=1 / np.maximum(din, 1), inv_deg_out=1 / np.maximum(dout, 1))


class TestPrimitives(unittest.TestCase):
    def check(self, build, *arrays):
        arrays = list(arrays)
        ts = [Tensor(a.copy(), requires_grad=True) for a in arrays]
        out = build(*ts)
        out.backward()
        for i, a in enumerate(arrays):
            def f():
                tt = [Tensor(x) for x in (arrays[:i] + [a] + arrays[i + 1:])]
                return float(build(*tt).data)
            ng = numgrad(f, a)
            np.testing.assert_allclose(ts[i].grad, ng, rtol=1e-4, atol=1e-6)

    def test_matmul_relu_sum(self):
        r = np.random.default_rng(0)
        self.check(lambda a, b: ag.tsum(ag.relu(a @ b)), r.normal(size=(4, 3)), r.normal(size=(3, 5)))

    def test_segment_softmax(self):
        r = np.random.default_rng(1)
        seg = Seg(r.integers(0, 4, 12), 4)
        w = r.normal(size=(12, 1))
        self.check(lambda s: ag.tsum(ag.segment_softmax(s, seg) * Tensor(w)), r.normal(size=(12, 1)))

    def test_gather_scatter(self):
        r = np.random.default_rng(2)
        seg = Seg(r.integers(0, 5, 10), 5)
        self.check(lambda a: ag.tsum(ag.scatter(ag.gather(a, seg) * ag.gather(a, seg), seg)), r.normal(size=(5, 3)))

    def test_loss_and_misc(self):
        r = np.random.default_rng(3)
        y = r.integers(0, 2, 9)
        self.check(lambda z: ag.weighted_nll(ag.log_softmax(z), y, [0.7, 1.6]), r.normal(size=(9, 2)))
        self.check(lambda a: ag.tsum(ag.log1p(ag.sigmoid(a)) + ag.leaky_relu(a)), r.normal(size=(3, 3)))
        self.check(lambda a, b: ag.tsum(ag.concat([a, b]) * ag.concat([b, a])), r.normal(size=(3, 2)), r.normal(size=(3, 2)))


class TestModel(unittest.TestCase):
    def test_model_gradients(self):
        for kind in ("gat", "sage", "mlp"):
            rng = np.random.default_rng(4)
            g = toy_graph(rng)
            m = EdgeGNN(8, hidden=8, heads=2, layers=2, kind=kind, seed=1)
            cw = np.array([1.0, 1.5])
            def loss():
                return float(ag.weighted_nll(ag.log_softmax(m.forward(g)), g.y, cw).data)
            for p in m.params():
                p.grad = None
            ag.weighted_nll(ag.log_softmax(m.forward(g)), g.y, cw).backward()
            for name in ["W2", "Wx", "W1"] + (["Wo0", "Wq0in0"] if kind != "mlp" else []) + (["ac0in0", "aq1out1"] if kind == "gat" else []):
                p = m.P[name]
                ng = numgrad(loss, p.data)
                np.testing.assert_allclose(p.grad, ng, rtol=1e-3, atol=1e-6, err_msg=f"{kind}/{name}")

    def test_mask_gradients(self):
        rng = np.random.default_rng(5)
        g = toy_graph(rng)
        m = EdgeGNN(8, hidden=8, heads=2, kind="gat", seed=2)
        em = Tensor(rng.uniform(0.2, 0.9, size=(g.E, 1)), requires_grad=True)
        fm = Tensor(rng.uniform(0.2, 0.9, size=(1, 8)), requires_grad=True)
        def f():
            return float(ag.tsum(m.forward(g, em, fm)).data)
        ag.tsum(m.forward(g, em, fm)).backward()
        np.testing.assert_allclose(em.grad, numgrad(f, em.data), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(fm.grad, numgrad(f, fm.data), rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

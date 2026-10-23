import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from xgnn_ids.data.loaders import UNSW_COLUMNS, load_cicids2017, load_unsw_nb15, to_windows
from xgnn_ids.data.synthetic import Drift, generate_window, make_base_windows
from xgnn_ids.drift import psi_max
from xgnn_ids.explain import budget, classify_pattern, explain, explain_alert, fidelity
from xgnn_ids.gnn import EdgeGNN
from xgnn_ids.graph import build_graph, computation_subgraph, fit_scaler


def tiny():
    wins = make_base_windows(3, 4, n_benign=120)
    sc = fit_scaler(wins)
    return wins, [build_graph(w, sc) for w in wins]


class TestGraph(unittest.TestCase):
    def test_subgraph_is_exact(self):
        _, G = tiny()
        for kind in ("gat", "sage"):
            m = EdgeGNN(8, hidden=8, heads=2, kind=kind, seed=0)
            g = G[0]
            full = m.predict_proba(g)
            for e in np.random.default_rng(0).choice(g.E, 12, replace=False):
                sub, loc = computation_subgraph(g, int(e))
                self.assertAlmostEqual(m.predict_proba(sub)[loc], full[e], places=9)

    def test_schema(self):
        wins, G = tiny()
        self.assertEqual(G[0].X.shape[1], 8)
        self.assertTrue(set(wins[0]["label"]) <= {0, 1})
        self.assertTrue((wins[0].loc[wins[0].label == 1, "campaign"] >= 0).all())


class TestExplain(unittest.TestCase):
    def test_metrics_bounded_and_ordered(self):
        _, G = tiny()
        m = EdgeGNN(8, hidden=8, heads=2, kind="gat", seed=0)
        g = G[1]
        e = int(np.where(g.y == 1)[0][0])
        r = explain_alert(m, g, e, gnnexp_kwargs={"epochs": 20})
        for meth, v in r["methods"].items():
            self.assertTrue(0.0 <= v["fidelity"] <= 1.0, meth)
            self.assertTrue(0.0 <= v["sufficiency"] <= 1.0 and 0.0 <= v["necessity"] <= 1.0)
        sub = r["_sub"]
        full = np.ones(sub.E)  # keeping everything is perfectly sufficient
        self.assertAlmostEqual(fidelity(m, sub, r["_loc"], full, 1, sub.E)["sufficiency"], 1.0, places=9)

    def test_pattern_rules(self):
        wins, G = tiny()
        for w, g in zip(wins, G):
            for atk, name in [("portscan", "port-scan-like"), ("dos", "flood-like")]:
                idx = np.where(g.meta.attack_type == atk)[0]
                if len(idx) == 0:
                    continue
                sub, loc = computation_subgraph(g, int(idx[0]))
                gt = (sub.meta.campaign.to_numpy() == sub.meta.campaign[loc]).astype(float)
                pat, _ = classify_pattern(sub, loc, gt + 1e-3 * np.random.default_rng(0).random(sub.E))
                self.assertEqual(pat, name)  # oracle explanation => rule names the right family


class TestDrift(unittest.TestCase):
    def test_psi(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(5000, 3))
        self.assertLess(psi_max(a, rng.normal(size=(5000, 3))), 0.05)
        self.assertGreater(psi_max(a, rng.normal(1.5, 1, size=(5000, 3))), 0.25)

    def test_drift_changes_traffic(self):
        r = np.random.default_rng(0)
        a = generate_window(r, Drift(0, 0))
        b = generate_window(np.random.default_rng(0), Drift(1, 1))
        self.assertNotAlmostEqual(a.fwd_pps.median(), b.fwd_pps.median(), places=3)


class TestLoaders(unittest.TestCase):
    def test_cicids_layout(self):
        w = generate_window(np.random.default_rng(0), n_benign=100)
        cic = pd.DataFrame({
            " Source IP": w.src, " Source Port": w.sport, " Destination IP": w.dst, " Destination Port": w.dport,
            " Protocol": w.proto, " Timestamp": pd.to_datetime(w.time, unit="s").dt.strftime("%m/%d/%Y %H:%M"),
            " Flow Duration": w.dur * 1e6, " Fwd Packets/s": w.fwd_pps, " Bwd Packets/s": w.bwd_pps,
            " SYN Flag Count": w.syn, " Fwd Packet Length Mean": w.fwd_len,
            " Label": w.attack_type.map({"benign": "BENIGN", "dos": "DoS Hulk", "portscan": "PortScan",
                                          "bruteforce": "FTP-Patator", "infiltration": "Infiltration"}),
        })
        with tempfile.TemporaryDirectory() as d:
            cic.to_csv(os.path.join(d, "Monday.csv"), index=False)
            df = load_cicids2017(d)
        self.assertEqual(len(df), len(w))
        self.assertEqual(int(df.label.sum()), int(w.label.sum()))
        self.assertTrue(set(df.attack_type) <= {"benign", "dos", "portscan", "bruteforce", "infiltration"})
        wins = to_windows(df, 50)
        self.assertGreater(len(wins), 3)

    def test_unsw_layout(self):
        w = generate_window(np.random.default_rng(1), n_benign=100)
        n = len(w)
        d = pd.DataFrame(0, index=range(n), columns=UNSW_COLUMNS)
        d["srcip"], d["dstip"], d["sport"], d["dsport"] = w.src, w.dst, w.sport, w.dport
        d["proto"] = np.where(w.proto == 6, "tcp", "udp")
        d["dur"] = w.dur
        d["spkts"] = (w.fwd_pps * w.dur).round() + 1
        d["dpkts"] = (w.bwd_pps * w.dur).round()
        d["smeansz"], d["stime"] = w.fwd_len, w.time
        d["attack_cat"] = w.attack_type.map({"benign": "", "dos": "DoS", "portscan": "Reconnaissance",
                                              "bruteforce": "Exploits", "infiltration": "Backdoor"})
        d["label"] = w.label
        with tempfile.TemporaryDirectory() as dd:
            d.to_csv(os.path.join(dd, "UNSW-NB15_1.csv"), index=False, header=False)
            df = load_unsw_nb15(dd)
        self.assertEqual(len(df), n)
        self.assertEqual(int(df.label.sum()), int(w.label.sum()))
        self.assertIn("other", set(df.attack_type))


if __name__ == "__main__":
    unittest.main()

import unittest

from xgnn_ids import app_backend as B


class TestBackend(unittest.TestCase):
    def test_flow(self):
        model, sc = B.train_demo_model(0, 3)
        _, g = B.make_window(sc, 0.3, 0.0, 1)
        self.assertIn("f1", B.window_metrics(model, g))
        al = B.list_alerts(model, g)
        if len(al):
            r = B.explain_one(model, g, int(al.iloc[0]["edge"]), "attention")
            self.assertTrue(0 <= r["fidelity"]["fidelity"] <= 1)
            self.assertIsNotNone(B.plot_explanation(r))
            self.assertGreater(len(B.top_flows_table(r)), 0)


if __name__ == "__main__":
    unittest.main()


class TestTopology(unittest.TestCase):
    def test_render_and_routes(self):
        import numpy as np
        from xgnn_ids import topology as T
        model, sc = B.train_demo_model(0, 3)
        _, g = B.make_window(sc, 0.3, 0.0, 2)
        p = model.predict_proba(g)
        al = B.list_alerts(model, g)
        res = B.explain_one(model, g, int(al.iloc[0]["edge"]), "attention") if len(al) else None
        self.assertIsNotNone(T.draw_topology(g.meta, p, res, show_truth=True))
        if res is not None:
            self.assertGreater(len(T.event_list(res)), 0)
        path = T.route("ATT:10.0.0.5", "SRV:10.0.1.3", "10.0.0.5", "10.0.1.3")
        self.assertEqual(path, ["ATT:10.0.0.5", "INET", "FW", "R1", "CORE", "SW-SRV", "SRV:10.0.1.3"])
        self.assertEqual(T.route("PC:10.0.2.4", "SRV:10.0.1.3", "10.0.2.4", "10.0.1.3"),
                         ["PC:10.0.2.4", "SW-A", "CORE", "SW-SRV", "SRV:10.0.1.3"])

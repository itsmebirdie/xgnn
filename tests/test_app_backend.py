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

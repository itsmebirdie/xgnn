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


class TestRealData(unittest.TestCase):
    def test_upload_and_session(self):
        import io
        import tempfile
        import os
        import pandas as pd
        from xgnn_ids.data.synthetic import make_base_windows
        w = pd.concat(make_base_windows(1, 8, n_benign=150), ignore_index=True)
        cic = pd.DataFrame({
            " Source IP": w.src, " Source Port": w.sport, " Destination IP": w.dst, " Destination Port": w.dport,
            " Protocol": w.proto, " Timestamp": pd.to_datetime(w.time, unit="s").dt.strftime("%m/%d/%Y %H:%M:%S"),
            " Flow Duration": w.dur * 1e6, " Fwd Packets/s": w.fwd_pps, " Bwd Packets/s": w.bwd_pps,
            " SYN Flag Count": w.syn, " Fwd Packet Length Mean": w.fwd_len,
            " Label": w.attack_type.map({"benign": "BENIGN", "dos": "DoS Hulk", "portscan": "PortScan",
                                          "bruteforce": "FTP-Patator", "infiltration": "Infiltration"})})

        class Up(io.BytesIO):
            name = "Friday.csv"
        up = Up(cic.to_csv(index=False).encode())
        df = B.load_real(up, "cicids2017", 100000)
        self.assertEqual(len(df), len(w))
        R = B.build_real_session(df, window_size=300, max_windows=8, epochs=2)
        self.assertIn("f1", R["test_metrics"])
        with self.assertRaises(ValueError):
            B.build_real_session(df, window_size=100000, max_windows=8, epochs=1)


class TestCicFolder(unittest.TestCase):
    def test_scan_and_selection(self):
        import os
        import tempfile
        import pandas as pd
        from xgnn_ids.data.synthetic import make_base_windows

        def cic(w):
            return pd.DataFrame({
                " Source IP": w.src, " Source Port": w.sport, " Destination IP": w.dst, " Destination Port": w.dport,
                " Protocol": w.proto, " Timestamp": pd.to_datetime(w.time, unit="s").dt.strftime("%m/%d/%Y %H:%M:%S"),
                " Flow Duration": w.dur * 1e6, " Fwd Packets/s": w.fwd_pps, " Bwd Packets/s": w.bwd_pps,
                " SYN Flag Count": w.syn, " Fwd Packet Length Mean": w.fwd_len, " Extra Col": 1,
                " Label": w.attack_type.map({"benign": "BENIGN", "dos": "DoS Hulk", "portscan": "PortScan",
                                              "bruteforce": "FTP-Patator", "infiltration": "Infiltration"})})
        wins = make_base_windows(2, 8, n_benign=150)
        mon = pd.concat(wins[:4], ignore_index=True); mon = mon[mon.label == 0]       # benign-only day
        fri = pd.concat(wins[4:], ignore_index=True)
        with tempfile.TemporaryDirectory() as d:
            cic(mon).to_csv(os.path.join(d, "Monday-WorkingHours.pcap_ISCX.csv"), index=False)
            cic(fri).to_csv(os.path.join(d, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"), index=False)
            sc = B.scan_folder(d)
            self.assertEqual(list(sc.file)[0][:6], "Monday")
            self.assertEqual(float(sc.malicious_pct.iloc[0]), 0.0)
            self.assertGreater(float(sc.malicious_pct.iloc[1]), 0)
            df = B.load_cic_selection(d, list(sc.file), per_file=400, focus_attacks=True)
            self.assertLessEqual(len(df), 800)
            self.assertGreater(int(df.label.sum()), 0)       # block was centred on attacks

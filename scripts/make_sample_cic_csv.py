"""Write a SYNTHETIC CSV in the CIC-IDS2017 'TrafficLabelling' layout, to dry-run the Real-data tab.

    python scripts/make_sample_cic_csv.py            # -> data/sample_cic_format.csv
NOT real data: it only lets you check the loader + UI work before you have the real download.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from xgnn_ids.data.synthetic import make_base_windows

wins = make_base_windows(7, 30, n_benign=500)
w = pd.concat(wins, ignore_index=True)
cic = pd.DataFrame({
    " Source IP": w.src, " Source Port": w.sport, " Destination IP": w.dst, " Destination Port": w.dport,
    " Protocol": w.proto, " Timestamp": pd.to_datetime(w.time, unit="s").dt.strftime("%m/%d/%Y %H:%M:%S"),
    " Flow Duration": w.dur * 1e6, " Fwd Packets/s": w.fwd_pps, " Bwd Packets/s": w.bwd_pps,
    " SYN Flag Count": w.syn, " Fwd Packet Length Mean": w.fwd_len,
    " Label": w.attack_type.map({"benign": "BENIGN", "dos": "DoS Hulk", "portscan": "PortScan",
                                  "bruteforce": "FTP-Patator", "infiltration": "Infiltration"}),
})
os.makedirs("data", exist_ok=True)
cic.to_csv("data/sample_cic_format.csv", index=False)
print(f"wrote data/sample_cic_format.csv ({len(cic):,} flows, synthetic)")

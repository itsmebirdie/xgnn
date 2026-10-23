"""Loaders that turn the public IDS datasets into the project's raw flow schema.

IMPORTANT: written against the documented column layouts of
  * CIC-IDS2017 "TrafficLabelling" CSVs (CICFlowMeter output that includes Source/Destination IP)
  * UNSW-NB15 raw CSVs ``UNSW-NB15_1..4.csv`` (no header row, 49 columns)
and unit-tested on files that mimic those layouts. They have NOT been run on the real downloads
(no network access when this repo was produced) -- see README "Running on the real datasets".
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from ..graph import RAW_COLUMNS

CIC_ATTACK_MAP = {
    "dos hulk": "dos", "dos goldeneye": "dos", "dos slowloris": "dos", "dos slowhttptest": "dos", "ddos": "dos",
    "portscan": "portscan", "ftp-patator": "bruteforce", "ssh-patator": "bruteforce",
    "infiltration": "infiltration",
}
UNSW_ATTACK_MAP = {"dos": "dos", "reconnaissance": "portscan"}
UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss",
    "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack",
    "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]


def _campaigns(df: pd.DataFrame) -> np.ndarray:
    """Approximate attack campaigns as (attack family, source host) pairs; -1 for benign."""
    key = df["attack_type"] + "|" + df["src"]
    camp = pd.factorize(key)[0].astype(np.int64)
    camp[df["label"].to_numpy() == 0] = -1
    return camp


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["src", "dst", "time"]).copy()
    for c in ["sport", "dport", "dur", "fwd_pps", "bwd_pps", "syn", "fwd_len"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["dur"] = df["dur"].clip(lower=0)
    df["fwd_pps"] = df["fwd_pps"].clip(lower=0)
    df["bwd_pps"] = df["bwd_pps"].clip(lower=0)
    df["proto"] = pd.to_numeric(df["proto"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("time").reset_index(drop=True)
    df["campaign"] = _campaigns(df)
    return df[RAW_COLUMNS]


def load_cicids2017(path: str, max_flows: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    frames = []
    for f in files:
        d = pd.read_csv(f, encoding="latin-1", low_memory=False)
        d.columns = [c.strip().lower() for c in d.columns]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    label = d["label"].astype(str).str.strip().str.lower()
    atk = label.map(lambda s: "benign" if s == "benign" else CIC_ATTACK_MAP.get(s, "other"))
    out = pd.DataFrame({
        "src": d["source ip"].astype(str), "dst": d["destination ip"].astype(str),
        "sport": d["source port"], "dport": d["destination port"], "proto": d["protocol"],
        "dur": pd.to_numeric(d["flow duration"], errors="coerce") / 1e6,
        "fwd_pps": d["fwd packets/s"], "bwd_pps": d["bwd packets/s"],
        "syn": d["syn flag count"], "fwd_len": d["fwd packet length mean"],
        "label": (label != "benign").astype(int), "attack_type": atk,
        "time": pd.to_datetime(d["timestamp"], errors="coerce", dayfirst=False).astype("int64") / 1e9,
    })
    if max_flows:
        out = out.iloc[:max_flows]
    return _finish(out)


def load_unsw_nb15(path: str, max_flows: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "UNSW-NB15_*.csv"))) if os.path.isdir(path) else [path]
    frames = []
    for f in files:
        first = pd.read_csv(f, nrows=1, header=None, low_memory=False)
        has_header = str(first.iloc[0, 0]).strip().lower() == "srcip"
        d = pd.read_csv(f, header=0 if has_header else None, names=None if has_header else UNSW_COLUMNS,
                        low_memory=False)
        d.columns = [c.strip().lower() for c in d.columns]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    dur = pd.to_numeric(d["dur"], errors="coerce").fillna(0).clip(lower=1e-6)
    proto = d["proto"].astype(str).str.lower().map({"tcp": 6, "udp": 17}).fillna(0)
    cat = d["attack_cat"].fillna("").astype(str).str.strip().str.lower()
    label = pd.to_numeric(d["label"], errors="coerce").fillna(0).astype(int)
    atk = np.where(label == 0, "benign", cat.map(lambda s: UNSW_ATTACK_MAP.get(s, "other")))
    out = pd.DataFrame({
        "src": d["srcip"].astype(str), "dst": d["dstip"].astype(str),
        "sport": pd.to_numeric(d["sport"], errors="coerce"), "dport": pd.to_numeric(d["dsport"], errors="coerce"),
        "proto": proto, "dur": dur,
        "fwd_pps": pd.to_numeric(d["spkts"], errors="coerce") / dur,
        "bwd_pps": pd.to_numeric(d["dpkts"], errors="coerce") / dur,
        "syn": (pd.to_numeric(d["synack"], errors="coerce").fillna(0) > 0).astype(int),  # proxy: UNSW has no SYN count
        "fwd_len": d["smeansz"], "label": label, "attack_type": atk,
        "time": pd.to_numeric(d["stime"], errors="coerce"),
    })
    if max_flows:
        out = out.iloc[:max_flows]
    return _finish(out)


def to_windows(df: pd.DataFrame, window_size: int = 4000, max_windows: int | None = None,
               min_attack_frac: float = 0.0):
    """Consecutive time-ordered windows of ``window_size`` flows."""
    wins = []
    for i in range(0, len(df) - window_size + 1, window_size):
        w = df.iloc[i:i + window_size].reset_index(drop=True)
        if w["label"].mean() >= min_attack_frac:
            wins.append(w)
        if max_windows and len(wins) >= max_windows:
            break
    return wins

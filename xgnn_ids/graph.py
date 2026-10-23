"""Flow table -> graph conversion (hosts = nodes, flows = directed edges)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .autograd import Seg

# Raw flow schema shared by the synthetic generator and the real-dataset loaders.
RAW_COLUMNS = [
    "src", "dst", "sport", "dport", "proto", "dur", "fwd_pps", "bwd_pps",
    "syn", "fwd_len", "label", "attack_type", "campaign", "time",
]

FEATURES = [
    "flow_pkts_s", "dest_port_count", "syn_count", "flow_duration",
    "fwd_pkt_len", "protocol", "bwd_pkts_s", "dst_port",
]
FEATURE_LABELS = {
    "flow_pkts_s": "Flow packets/s",
    "dest_port_count": "Dest. port count",
    "syn_count": "SYN flag count",
    "flow_duration": "Flow duration",
    "fwd_pkt_len": "Fwd packet length",
    "protocol": "Protocol",
    "bwd_pkts_s": "Bwd packets/s",
    "dst_port": "Destination port",
}


def edge_features(df: pd.DataFrame) -> np.ndarray:
    """Engineer the per-flow feature matrix (before standardisation)."""
    dpc = df.groupby("src")["dport"].transform("nunique").to_numpy(dtype=float)
    cols = [
        np.log10(1.0 + df["fwd_pps"].to_numpy(float) + df["bwd_pps"].to_numpy(float)),
        np.log1p(dpc),
        np.clip(df["syn"].to_numpy(float), 0, 5),
        np.log10(df["dur"].to_numpy(float) + 1e-4),
        np.log1p(df["fwd_len"].to_numpy(float)),
        (df["proto"].to_numpy() == 6).astype(float),
        np.log10(1.0 + df["bwd_pps"].to_numpy(float)),
        np.log1p(df["dport"].to_numpy(float)),
    ]
    return np.stack(cols, axis=1)


class Scaler:
    """Feature standardiser fitted on training windows only."""

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, raw_feature_blocks):
        X = np.concatenate(raw_feature_blocks, axis=0)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


@dataclass
class GraphData:
    n: int
    src: np.ndarray
    dst: np.ndarray
    X: np.ndarray                      # (E, d) standardised
    y: np.ndarray                      # (E,) 0/1
    off_in: np.ndarray                 # (n,1) degree contributed by edges outside this graph
    off_out: np.ndarray
    inv_deg_in: np.ndarray             # (n,) 1/full in-degree (used by the mean-aggregation variant)
    inv_deg_out: np.ndarray
    hosts: list = field(default_factory=list)
    meta: pd.DataFrame | None = None   # per-edge raw info (ips, ports, attack type...)
    edge_ids: np.ndarray | None = None # for sub-graphs: ids in the parent graph
    X_raw: np.ndarray | None = None    # un-standardised features
    seg_src: Seg = None
    seg_dst: Seg = None

    def __post_init__(self):
        self.seg_src = Seg(self.src, self.n)
        self.seg_dst = Seg(self.dst, self.n)

    @property
    def E(self):
        return len(self.src)


def build_graph(df: pd.DataFrame, scaler: Scaler) -> GraphData:
    df = df.reset_index(drop=True)
    hosts = pd.unique(pd.concat([df["src"], df["dst"]]))
    hid = {h: i for i, h in enumerate(hosts)}
    src = df["src"].map(hid).to_numpy(np.int64)
    dst = df["dst"].map(hid).to_numpy(np.int64)
    n = len(hosts)
    raw = edge_features(df)
    din = np.bincount(dst, minlength=n).astype(float)
    dout = np.bincount(src, minlength=n).astype(float)
    return GraphData(
        n=n, src=src, dst=dst, X=scaler.transform(raw), y=df["label"].to_numpy(np.int64),
        off_in=np.zeros((n, 1)), off_out=np.zeros((n, 1)),
        inv_deg_in=1.0 / np.maximum(din, 1.0), inv_deg_out=1.0 / np.maximum(dout, 1.0),
        hosts=list(hosts), meta=df, edge_ids=np.arange(len(df)), X_raw=raw,
    )


def fit_scaler(frames) -> Scaler:
    return Scaler().fit([edge_features(f) for f in frames])


def computation_subgraph(g: GraphData, e: int, hops: int = 1) -> tuple[GraphData, int]:
    """Exact 2-layer computation subgraph of edge ``e``.

    Keeps every edge incident to the 1-hop neighbourhood of the flow's end points.
    Degrees contributed by dropped edges are carried as constants so the model's
    output for ``e`` is *identical* to the output on the full graph (tested).
    Returns (subgraph, local index of the target edge).
    """
    s, d = g.src[e], g.dst[e]
    nodes = {s, d}
    inc = (g.src == s) | (g.dst == s) | (g.src == d) | (g.dst == d)
    for _ in range(hops):
        idx = np.where(inc)[0]
        nodes |= set(g.src[idx].tolist()) | set(g.dst[idx].tolist())
        mask_nodes = np.zeros(g.n, bool)
        mask_nodes[list(nodes)] = True
        inc = mask_nodes[g.src] | mask_nodes[g.dst]
    keep = np.where(inc)[0]
    used = np.unique(np.concatenate([g.src[keep], g.dst[keep]]))
    remap = -np.ones(g.n, np.int64)
    remap[used] = np.arange(len(used))
    ls, ld = remap[g.src[keep]], remap[g.dst[keep]]
    nn = len(used)
    din_full = np.bincount(g.dst, minlength=g.n).astype(float) + g.off_in[:, 0]
    dout_full = np.bincount(g.src, minlength=g.n).astype(float) + g.off_out[:, 0]
    din_sub = np.bincount(ld, minlength=nn).astype(float)
    dout_sub = np.bincount(ls, minlength=nn).astype(float)
    sub = GraphData(
        n=nn, src=ls, dst=ld, X=g.X[keep], y=g.y[keep],
        off_in=(din_full[used] - din_sub)[:, None],
        off_out=(dout_full[used] - dout_sub)[:, None],
        inv_deg_in=g.inv_deg_in[used], inv_deg_out=g.inv_deg_out[used],
        hosts=[g.hosts[i] for i in used],
        meta=None if g.meta is None else g.meta.iloc[keep].reset_index(drop=True),
        edge_ids=g.edge_ids[keep],
        X_raw=None if g.X_raw is None else g.X_raw[keep],
    )
    local = int(np.where(keep == e)[0][0])
    return sub, local

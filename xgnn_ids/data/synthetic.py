"""Synthetic flow generator with the CIC-IDS2017-style feature schema.

Why it exists: the real datasets cannot be bundled in a repo (size/licence), so the
pipeline is validated end-to-end on controllable traffic whose ground truth
(which flows belong to which attack campaign) is known. That ground truth is what
lets us score *explanations*, not just predictions.

Four attack families (same names as the project plan):
  dos          one attacker -> one server, many parallel high-rate flows
  portscan     one attacker -> many hosts, many destination ports, SYN-only probes
  bruteforce   one attacker -> one server:22/21, many benign-looking login flows
               (separable mainly through flow multiplicity = graph structure)
  infiltration compromised client -> many internal clients on 445/3389 (lateral movement)

Drift (``Drift(evasion, benign_shift)`` in [0,1]):
  evasion       attackers adapt: low-rate DoS, slow/distributed scans, brute force that
                mimics SSH session durations, slower lateral movement
  benign_shift  new benign services (high-rate streaming on 8443), heavier web traffic and
                extra authorised "inventory scanner" hosts that look scan-like
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..graph import RAW_COLUMNS

SERVERS = [f"10.0.1.{i}" for i in range(1, 13)]
CLIENTS = [f"10.0.2.{i}" for i in range(1, 71)]
FILE_SERVER = "10.0.1.9"
SERVICES = {          # port: (proto, log10 pps mean, sd, log10 dur mean, sd, fwd_len mean, sd, bwd ratio)
    80: (6, 1.20, 0.35, 0.0, 0.5, 300, 120, 0.9),
    443: (6, 1.25, 0.35, 0.1, 0.5, 350, 120, 0.9),
    53: (17, 0.50, 0.30, -2.0, 0.4, 60, 15, 1.0),
    22: (6, 1.00, 0.40, 1.5, 0.5, 100, 30, 0.8),
    25: (6, 0.90, 0.35, 0.3, 0.4, 500, 150, 0.6),
    3306: (6, 1.10, 0.35, -0.5, 0.5, 200, 80, 0.9),
    445: (6, 1.00, 0.35, 0.5, 0.5, 400, 150, 0.9),
}
ATTACKS = ["dos", "portscan", "bruteforce", "infiltration"]


@dataclass
class Drift:
    evasion: float = 0.0
    benign_shift: float = 0.0


def _lg(rng, mu, sd, n):
    return 10 ** rng.normal(mu, sd, n)


def _flows(rng, n, src, dst, dport, proto, pps_mu, pps_sd, dur_mu, dur_sd, len_mu, len_sd, bwd_ratio,
           syn_mean, label, atk, campaign):
    pps = _lg(rng, pps_mu, pps_sd, n)
    bwd = pps * np.clip(rng.normal(bwd_ratio, 0.15, n), 0, 2) * (0.5 if bwd_ratio > 0 else 0)
    fwd = pps
    syn = np.where(proto == 6, 1 + rng.poisson(syn_mean, n), 0)
    return pd.DataFrame({
        "src": src, "dst": dst, "sport": rng.integers(1024, 65535, n), "dport": dport, "proto": proto,
        "dur": _lg(rng, dur_mu, dur_sd, n), "fwd_pps": fwd, "bwd_pps": bwd, "syn": syn,
        "fwd_len": np.clip(rng.normal(len_mu, len_sd, n), 0, 1500), "label": label,
        "attack_type": atk, "campaign": campaign,
    })


def _camouflage(rng, df, frac):
    """Make a fraction of an attacker's flows look like ordinary web traffic (label stays 'attack')."""
    n = len(df)
    m = rng.random(n) < frac
    k = int(m.sum())
    if k == 0:
        return df
    proto, pm, ps, dm, ds, lm, ls, br = SERVICES[443]
    pps = _lg(rng, pm, ps, k)
    df = df.copy()
    df.loc[m, "fwd_pps"] = pps
    df.loc[m, "bwd_pps"] = pps * br * 0.5
    df.loc[m, "dur"] = _lg(rng, dm, ds, k)
    df.loc[m, "fwd_len"] = np.clip(rng.normal(lm, ls, k), 0, 1500)
    df.loc[m, "syn"] = 1
    return df


def generate_window(rng: np.random.Generator, drift: Drift | None = None, n_benign: int = 380,
                    t0: float = 0.0) -> pd.DataFrame:
    drift = drift or Drift()
    e, b = drift.evasion, drift.benign_shift
    parts = []
    # ---------------------------------------------------------------- benign
    ports = np.array(list(SERVICES))
    pw = np.array([0.30, 0.30, 0.15, 0.06, 0.05, 0.08, 0.06])
    pw = pw / pw.sum()
    act = rng.dirichlet(np.ones(len(CLIENTS)) * 0.7)
    n_stream = int(n_benign * 0.12 * b)
    n_main = n_benign - n_stream
    cl = rng.choice(CLIENTS, n_main, p=act)
    dp = rng.choice(ports, n_main, p=pw)
    for p in np.unique(dp):
        m = dp == p
        k = int(m.sum())
        proto, pm, ps, dm, ds, lm, ls, br = SERVICES[p]
        if p in (80, 443):
            pm = pm + 0.45 * b                         # heavier web traffic under benign drift
        if p == 445:
            dst = np.full(k, FILE_SERVER)
        else:
            dst = rng.choice(SERVERS, k)
        parts.append(_flows(rng, k, cl[m], dst, np.full(k, p), np.full(k, proto), pm, ps, dm, ds, lm, ls, br,
                            0.1, 0, "benign", -1))
    if n_stream:                                       # new high-rate streaming service
        c = rng.choice(CLIENTS, n_stream)
        parts.append(_flows(rng, n_stream, c, rng.choice(SERVERS[:3], n_stream), np.full(n_stream, 8443),
                            np.full(n_stream, 6), 1.9, 0.3, 1.2, 0.4, 900, 200, 0.2, 0.3, 0, "benign", -1))
    # benign retry storms: flaky clients hammer one server with short flows (confuser for DoS / brute force)
    for _ in range(int(rng.integers(1, 4))):
        nr = int(rng.integers(10, 30))
        port = int(rng.choice([80, 22, 443]))
        parts.append(_flows(rng, nr, np.full(nr, rng.choice(CLIENTS)), np.full(nr, rng.choice(SERVERS)),
                            np.full(nr, port), np.full(nr, 6), 1.4, 0.4, -0.6, 0.5, 150, 60, 0.4, 0.5, 0,
                            "benign", -1))
    # benign bulk backup (high pps, long) -- confuser for DoS
    nb = int(rng.integers(8, 20))
    parts.append(_flows(rng, nb, np.full(nb, rng.choice(CLIENTS)), np.full(nb, "10.0.1.11"), np.full(nb, 443),
                        np.full(nb, 6), 2.0, 0.3, 1.3, 0.4, 1000, 200, 0.2, 0.2, 0, "benign", -1))
    # benign health monitor: touches every server on the same two ports -- confuser for scans
    mon = "10.0.3.1"
    nm = int(rng.integers(2, 4)) * len(SERVERS)
    parts.append(_flows(rng, nm, np.full(nm, mon), np.tile(SERVERS, nm // len(SERVERS)),
                        rng.choice([80, 443], nm), np.full(nm, 6), 0.8, 0.3, -1.0, 0.4, 120, 30, 0.8, 0.1, 0,
                        "benign", -1))
    # authorised inventory scanners appear under benign drift (still label 0)
    for j in range(int(round(2 * b))):
        ns = int(rng.integers(30, 60))
        parts.append(_flows(rng, ns, np.full(ns, f"10.0.3.{10 + j}"), rng.choice(SERVERS, ns),
                            rng.choice([22, 80, 443, 3306, 445, 25, 53, 8080], ns), np.full(ns, 6),
                            1.2, 0.3, -1.0, 0.4, 80, 20, 0.5, 0.2, 0, "benign", -1))
    # ---------------------------------------------------------------- attacks
    camp = 0
    present = [a for a in ATTACKS if rng.random() < 0.75]
    while len(present) < 2:
        a = ATTACKS[int(rng.integers(0, 4))]
        if a not in present:
            present.append(a)
    for atk in present:
        n_att = 1 if atk != "portscan" else 1 + int(3 * e * rng.random() + 0.99 * e)  # distributed scan under evasion
        for _ in range(n_att):
            camp += 1
            if atk == "infiltration":
                a_ip = rng.choice(CLIENTS)
            else:
                a_ip = f"10.0.0.{int(rng.integers(1, 40))}"
            if atk == "dos":
                k = max(8, int(rng.integers(40, 90) * (1 - 0.65 * e)))
                tgt = rng.choice(SERVERS)
                parts.append(_camouflage(rng, _flows(rng, k, np.full(k, a_ip), np.full(k, tgt), np.full(k, 80), np.full(k, 6),
                                    2.7 - 1.3 * e, 0.3, -1.5 + 1.6 * e, 0.4, 60 + 120 * e, 20, 0.05, 0.1,
                                    1, atk, camp), 0.2 + 0.4 * e))
            elif atk == "portscan":
                nv = int(rng.integers(5, 10))
                vic = rng.choice(SERVERS + CLIENTS[:20], nv, replace=False)
                ppv = max(3, int(round(rng.integers(8, 15) * (1 - 0.7 * e) / n_att ** 0.5)))
                k = nv * ppv
                pr = np.concatenate([rng.choice(np.arange(1, 10000), ppv, replace=False) for _ in range(nv)])
                parts.append(_camouflage(rng, _flows(rng, k, np.full(k, a_ip), np.repeat(vic, ppv), pr, np.full(k, 6),
                                    1.9 - 0.9 * e, 0.3, -2.5 + 2.0 * e, 0.3, 20, 15, 0.0, 0.0, 1, atk, camp), 0.2 + 0.3 * e))
            elif atk == "bruteforce":
                k = max(8, int(rng.integers(30, 60) * (1 - 0.6 * e)))
                tgt = rng.choice(SERVERS)
                port = int(rng.choice([22, 22, 21]))
                parts.append(_camouflage(rng, _flows(rng, k, np.full(k, a_ip), np.full(k, tgt), np.full(k, port), np.full(k, 6),
                                    1.1, 0.35, 0.3 + 1.2 * e, 0.4, 110, 25, 0.5, 0.1, 1, atk, camp), 0.2 + 0.4 * e))
            else:  # infiltration
                nv = max(3, int(round(12 - 8 * e)))
                nv = int(rng.integers(max(3, nv - 2), nv + 3))
                vic = rng.choice([c for c in CLIENTS if c != a_ip], nv, replace=False)
                reps = rng.integers(1, 4, nv)
                dst = np.repeat(vic, reps)
                k = len(dst)
                pr = rng.choice([445, 3389], k)
                parts.append(_camouflage(rng, _flows(rng, k, np.full(k, a_ip), dst, pr, np.full(k, 6), 0.9, 0.4, 1.0, 0.6, 200, 80,
                                    0.6, 0.1, 1, atk, camp), 0.2 + 0.3 * e))
    df = pd.concat(parts, ignore_index=True)
    df["time"] = t0 + rng.uniform(0, 300, len(df))
    df = df.sort_values("time").reset_index(drop=True)
    return df[RAW_COLUMNS]


def make_base_windows(seed: int, n_windows: int = 60, **kw):
    rng = np.random.default_rng(seed)
    return [generate_window(rng, Drift(0, 0), t0=300.0 * i, **kw) for i in range(n_windows)]


def make_drift_week(seed: int, week: int, n_weeks: int = 8, windows_per_week: int = 3,
                    strength: float = 1.0, scenario: str = "both", **kw):
    """Windows observed in drift week ``week`` (1..n_weeks); severity ramps linearly.

    scenario: 'both' (attacker evasion + benign shift), 'evasion' (attackers adapt only),
              'benign' (new benign services / scanner-like benign hosts only).
    """
    s = strength * week / n_weeks
    e = s if scenario in ("both", "evasion") else 0.0
    b = s if scenario in ("both", "benign") else 0.0
    rng = np.random.default_rng(10_000 * seed + 97 * week + 1)
    return [generate_window(rng, Drift(e, b), t0=1e6 + 1e4 * week + 300.0 * i, **kw) for i in range(windows_per_week)]

"""Packet-Tracer-style rendering of a traffic window on a fixed enterprise topology.

Internet -> Firewall -> Router -> Core switch -> {server switch, two access switches, mgmt switch}.
Hosts are mapped to access devices from their IP (10.0.0.x external, 10.0.1.x servers, 10.0.2.x clients,
10.0.3.x management). Links turn red when they carry flows the model alerted on; the explanation of one alert
is overlaid as dashed PDU "envelopes" between the end hosts.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

BG, GREEN, RED, AMBER, BLUE, GREY = "#eef2f8", "#2e9e4f", "#d62839", "#f4a100", "#2b6cb0", "#6b7280"
SERVICE = {80: "HTTP", 443: "HTTPS", 53: "DNS", 22: "SSH", 21: "FTP", 25: "SMTP", 3306: "MySQL", 445: "SMB",
           3389: "RDP", 8443: "HTTPS-alt", 8080: "HTTP-alt"}

POS = {"INET": (55, 55), "FW": (55, 45), "R1": (55, 37), "CORE": (55, 29),
       "SW-SRV": (18, 19), "SW-A": (55, 19), "SW-B": (80, 19), "SW-MGMT": (103, 19)}
KIND = {"INET": "cloud", "FW": "firewall", "R1": "router", "CORE": "switch", "SW-SRV": "switch",
        "SW-A": "switch", "SW-B": "switch", "SW-MGMT": "switch"}


def switch_of(ip: str) -> str:
    a, b, c, d = (int(x) for x in ip.split("."))
    if c == 1:
        return "SW-SRV"
    if c == 2:
        return "SW-A" if d <= 35 else "SW-B"
    if c == 3:
        return "SW-MGMT"
    return "INET"


def _node_of(ip, interesting):
    sw = switch_of(ip)
    if sw == "INET":
        return f"ATT:{ip}" if ip in interesting else "ATT:other"
    if sw == "SW-SRV":
        return f"SRV:{ip}"
    return f"PC:{ip}" if ip in interesting else f"OTH:{sw}"


def route(src_node, dst_node, src_ip, dst_ip):
    s_sw, d_sw = switch_of(src_ip), switch_of(dst_ip)
    up = [src_node] if s_sw == "INET" else [src_node, s_sw]
    path = list(up)
    if s_sw == "INET":
        path += ["INET", "FW", "R1", "CORE"]
    elif s_sw != d_sw:
        path += ["CORE"]
    if d_sw != "INET" and s_sw != d_sw:
        path += [d_sw]
    elif d_sw == "INET":
        path += ["R1", "FW", "INET"] if s_sw != "INET" else []
    path += [dst_node]
    out = [path[0]]
    for n in path[1:]:
        if n != out[-1]:
            out.append(n)
    return out


def _layout(interesting_srv, interesting_pc, interesting_att):
    pos = dict(POS)
    for i, ip in enumerate(sorted(interesting_att)):
        pos[f"ATT:{ip}"] = (12 + 14 * i, 57 + (i % 2) * 3)
    pos["ATT:other"] = (100, 57)
    srv = sorted([f"10.0.1.{i}" for i in range(1, 13)])
    for i, ip in enumerate(srv):
        pos[f"SRV:{ip}"] = (2 + 6.0 * (i % 6), 8.5 - 7 * (i // 6))
    for sw in ("SW-A", "SW-B", "SW-MGMT"):
        x0 = POS[sw][0]
        mine = sorted([ip for ip in interesting_pc if switch_of(ip) == sw])[:6]
        for i, ip in enumerate(mine):
            pos[f"PC:{ip}"] = (x0 - 7 + 6 * (i % 3), 9.5 - 7 * (i // 3))
        pos[f"OTH:{sw}"] = (x0 + 7, 9.5 - 7)
    return pos


# ------------------------------------------------------------------ icons
def _label(ax, x, y, text, dy=-3.4, size=6.5, color="#111"):
    ax.text(x, y + dy, text, ha="center", va="top", fontsize=size, color=color, zorder=6)


def _icon(ax, kind, x, y, halo=None):
    if halo:
        ax.add_patch(Circle((x, y), 3.3, color=halo, alpha=0.28, zorder=2))
    if kind == "router":
        ax.add_patch(Circle((x, y), 2.4, fc="#3b82c4", ec="#1e3a5f", lw=1.2, zorder=4))
        ax.text(x, y, "⇄", ha="center", va="center", color="white", fontsize=9, zorder=5)
    elif kind == "switch":
        ax.add_patch(FancyBboxPatch((x - 3, y - 1.6), 6, 3.2, boxstyle="round,pad=0.1", fc="#3aa0b8", ec="#1e3a5f", lw=1.2, zorder=4))
        ax.text(x, y, "⇄ ⇄", ha="center", va="center", color="white", fontsize=7, zorder=5)
    elif kind == "firewall":
        ax.add_patch(Rectangle((x - 3, y - 2), 6, 4, fc="#e4572e", ec="#7a1f0a", lw=1.2, zorder=4))
        for k in (-0.7, 0.7):
            ax.plot([x - 3, x + 3], [y + k, y + k], color="#7a1f0a", lw=0.8, zorder=5)
        ax.plot([x, x], [y - 2, y + 2], color="#7a1f0a", lw=0.8, zorder=5)
    elif kind == "server":
        ax.add_patch(FancyBboxPatch((x - 1.4, y - 2.2), 2.8, 4.4, boxstyle="round,pad=0.05", fc="#9aa5b1", ec="#374151", lw=1, zorder=4))
        for k in (-1.2, 0, 1.2):
            ax.plot([x - 0.9, x + 0.9], [y + k, y + k], color="#374151", lw=0.9, zorder=5)
    elif kind in ("pc", "laptop"):
        c = "#1f2937" if kind == "laptop" else "#d1d5db"
        ax.add_patch(Rectangle((x - 1.8, y - 0.8), 3.6, 2.6, fc=c, ec="#374151", lw=1, zorder=4))
        ax.add_patch(Rectangle((x - 1.1, y - 1.5), 2.2, 0.6, fc="#6b7280", ec="#374151", lw=0.6, zorder=4))
    elif kind == "cluster":
        for dx in (-1.2, 0, 1.2):
            ax.add_patch(Rectangle((x + dx - 1.3, y - 0.6 + dx * 0.2), 2.6, 2, fc="#e5e7eb", ec="#374151", lw=0.8, zorder=4))
    elif kind == "cloud":
        for dx, dy, w, h in [(-7, 0, 9, 5), (0, 1.5, 11, 6.5), (7, 0, 9, 5), (0, -1, 18, 4.5)]:
            ax.add_patch(Ellipse((x + dx, y + dy), w, h, fc="#dbe4f0", ec="#8aa0bd", lw=1, zorder=3))


def _envelope(ax, x, y, color):
    ax.add_patch(Rectangle((x - 1.2, y - 0.8), 2.4, 1.6, fc=color, ec="#111", lw=0.8, zorder=9))
    ax.plot([x - 1.2, x, x + 1.2], [y + 0.8, y - 0.1, y + 0.8], color="#111", lw=0.7, zorder=10)


# ------------------------------------------------------------------ main renderer
def draw_topology(meta: pd.DataFrame, p: np.ndarray, sub_res: dict | None = None, show_truth: bool = False,
                  max_overlay: int = 8, figsize=(13, 7.2)):
    """meta: window flow table; p: model attack probabilities per flow; sub_res: explain_one() result (optional)."""
    alerted = p > 0.5
    atk_ips = set(meta.loc[alerted, "src"]) | set(meta.loc[meta.label == 1, "src"] if show_truth else [])
    interesting = {ip for ip in atk_ips}
    if sub_res is not None:
        sm = sub_res["sub"].meta
        top = np.argsort(-sub_res["importance"], kind="stable")[:max(max_overlay, 1)]
        for i in top:
            interesting |= {sm.loc[i, "src"], sm.loc[i, "dst"]}
        interesting.add(sm.loc[sub_res["loc"], "src"])
    interesting = {ip for ip in interesting if switch_of(ip) != "SW-SRV"}
    att = {ip for ip in interesting if switch_of(ip) == "INET"}
    pcs = {ip for ip in interesting if switch_of(ip) not in ("INET", "SW-SRV")}
    pos = _layout(None, pcs, att)

    # aggregate link load
    link = {}
    src_n = [_node_of(s, interesting) for s in meta["src"]]
    dst_n = [_node_of(d, interesting) for d in meta["dst"]]
    for i, (s, d, sip, dip) in enumerate(zip(src_n, dst_n, meta["src"], meta["dst"])):
        pth = route(s, d, sip, dip)
        for a, b in zip(pth[:-1], pth[1:]):
            k = tuple(sorted((a, b)))
            e = link.setdefault(k, [0, 0])
            e[0] += 1
            e[1] += int(alerted[i])

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(-2, 111); ax.set_ylim(-9, 65); ax.axis("off")
    for k in range(0, 112, 8):
        ax.axvline(k, color="#e1e7f0", lw=0.5, zorder=0)
    for k in range(-8, 66, 8):
        ax.axhline(k, color="#e1e7f0", lw=0.5, zorder=0)

    for (a, b), (n, na) in link.items():
        if a not in pos or b not in pos:
            continue
        (x1, y1), (x2, y2) = pos[a], pos[b]
        col = RED if na else GREEN
        ax.plot([x1, x2], [y1, y2], color=col, lw=0.8 + 0.7 * np.log1p(n) + (1.2 if na else 0), alpha=0.85, zorder=1,
                solid_capstyle="round")
        for (x, y) in ((x1 + (x2 - x1) * 0.12, y1 + (y2 - y1) * 0.12), (x2 - (x2 - x1) * 0.12, y2 - (y2 - y1) * 0.12)):
            ax.add_patch(Circle((x, y), 0.55, fc=RED if na else GREEN, ec="white", lw=0.6, zorder=3))
        if na:
            ax.text((x1 + x2) / 2 + 1.2, (y1 + y2) / 2, f"{na}⚠", fontsize=6, color=RED, zorder=7)

    # devices
    for name, kind in KIND.items():
        x, y = pos[name]
        _icon(ax, kind, x, y)
        _label(ax, x, y, {"INET": "Internet"}.get(name, name), dy=-3.3 if kind != "cloud" else -4.5)
    for ip in range(1, 13):
        node = f"SRV:10.0.1.{ip}"
        x, y = pos[node]
        tgt = bool(((meta["dst"] == f"10.0.1.{ip}") & alerted).any())
        _icon(ax, "server", x, y, halo=RED if tgt else None)
        _label(ax, x, y, f"SRV .{ip}", dy=-2.9, size=5.5, color=RED if tgt else "#111")
    for ip in sorted(att):
        x, y = pos[f"ATT:{ip}"]
        det = bool(((meta["src"] == ip) & alerted).any())
        missed = bool(((meta["src"] == ip) & (meta.label == 1) & ~alerted).any())
        halo = RED if det else (AMBER if (missed and show_truth) else None)
        _icon(ax, "laptop", x, y, halo=halo)
        _label(ax, x, y, ip, dy=-3, size=6, color=RED if det else "#111")
    for ip in sorted(pcs):
        node = f"PC:{ip}"
        if node not in pos:
            continue
        x, y = pos[node]
        det = bool(((meta["src"] == ip) & alerted).any())
        _icon(ax, "pc", x, y, halo=RED if det else None)
        _label(ax, x, y, ip.replace("10.0.", ""), dy=-3, size=5.5, color=RED if det else "#111")
    for sw in ("SW-A", "SW-B", "SW-MGMT"):
        x, y = pos[f"OTH:{sw}"]
        _icon(ax, "cluster", x, y)
        _label(ax, x, y, "other hosts", dy=-3, size=5.5, color=GREY)
    # hosts -> switch stubs for hosts without traffic recorded yet
    for node in pos:
        if node.startswith("ATT:other"):
            continue

    # explanation overlay
    if sub_res is not None:
        sm, imp = sub_res["sub"].meta, sub_res["importance"]
        top = np.argsort(-imp, kind="stable")[:max_overlay]
        wmax = max(1e-9, imp[top].max())
        seen = set()
        for r, i in enumerate(top):
            s, d = sm.loc[i, "src"], sm.loc[i, "dst"]
            ns, nd = _node_of(s, interesting), _node_of(d, interesting)
            if (ns, nd) in seen or ns not in pos or nd not in pos:
                continue
            seen.add((ns, nd))
            (x1, y1), (x2, y2) = pos[ns], pos[nd]
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={0.25 + 0.05 * (r % 3)}",
                                         arrowstyle="-|>", mutation_scale=9, lw=1.2 + 3 * imp[i] / wmax, ls=(0, (4, 2)),
                                         color="#8b0a1a", zorder=8, alpha=0.9))
            _envelope(ax, (x1 + x2) / 2, (y1 + y2) / 2 + 2.5, "#ffd23f")
    ax.text(0, 63, "Logical workspace — traffic window", fontsize=9, color="#374151", weight="bold", va="top")
    ax.text(110, 63, "green = normal link · red = carries alerted flows · dashed = explanation (PDU)", fontsize=7,
            color=GREY, ha="right", va="top")
    plt.tight_layout()
    return fig


def event_list(sub_res: dict, n: int = 12) -> pd.DataFrame:
    """Packet-Tracer 'Event List' style table for the explanation's top flows."""
    sm, imp = sub_res["sub"].meta, sub_res["importance"]
    o = np.argsort(-imp, kind="stable")[:n]
    rows = []
    for step, i in enumerate(o, 1):
        s, d, p = sm.loc[i, "src"], sm.loc[i, "dst"], int(sm.loc[i, "dport"])
        rows.append({"Step": step, "Last Device": s, "At Device": d,
                     "Type": f"{'TCP' if sm.loc[i, 'proto'] == 6 else 'UDP'}/{SERVICE.get(p, p)}",
                     "Importance": round(float(imp[i]), 3), "Ground truth": sm.loc[i, "attack_type"]})
    return pd.DataFrame(rows)

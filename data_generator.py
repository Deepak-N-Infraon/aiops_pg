"""
data_generator.py
=================
Generates a synthetic but realistic multi-device network time-series dataset
with EMBEDDED causal patterns.

Scale is controlled entirely by config.py:
    N_DEVICES  → number of network devices  (5 – 200)
    N_DAYS     → days of historical data    (1 – 365)

Topology is built dynamically to fit N_DEVICES into a realistic hierarchy:

  Tier          Role             Ratio (approx)
  ────────────────────────────────────────────────
  1  Core       router           1 per 50 devices
  2  Agg        dist_switch      1 per 12 devices
  3  Security   firewall         1 per  6 devices
  4  Access     access_switch    1 per  3 devices
  5  Edge       edge_switch      remainder

  Minimum topology (N_DEVICES=5):
    R1 → DS1 → FW1 → SW1 → Edge1

Injected causal chains (replicated per group):

  HIGH_LATENCY:
    R:cpu_pct ↑ → DS:buffer_util ↑ / crc_errors ↑
      → FW:latency_ms ↑ → FW:packet_loss ↑

  INTERFACE_FLAP:
    SW:link_util ↑ → SW:buffer_util ↑ → SW:crc_errors ↑ → SW:packet_loss ↑
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


# ── Metric specs per role ─────────────────────────────────────────────────────

_METRIC_SPECS: Dict[str, Dict[str, Tuple]] = {
    "router": {
        "cpu_pct":    (35, 5,   5,  95),
        "latency_ms": ( 8, 2,   1, 200),
    },
    "dist_switch": {
        "cpu_pct":     (25, 4,  2,  95),
        "crc_errors":  ( 1, 0.5,0, 500),
        "buffer_util": (25, 5,  0, 100),
        "latency_ms":  ( 5, 1,  1, 200),
    },
    "firewall": {
        "cpu_pct":     (40, 7,  2,  95),
        "latency_ms":  (12, 3,  1, 500),
        "packet_loss": ( 0.5, 0.2, 0, 20),
        "queue_depth": (20, 4,  0, 200),
    },
    "access_switch": {
        "cpu_pct":     (20, 3,  2,  95),
        "crc_errors":  ( 2, 1,  0, 500),
        "buffer_util": (30, 6,  0, 100),
        "link_util":   (45, 8,  0, 100),
        "packet_loss": ( 0.2, 0.1, 0, 10),
    },
    "edge_switch": {
        "cpu_pct":     (18, 3,  2,  95),
        "latency_ms":  ( 6, 1.5,1, 200),
        "packet_loss": ( 0.3, 0.1,0, 10),
    },
}


# ════════════════════════════════════════════════════════════════════════════
# Dynamic topology builder
# ════════════════════════════════════════════════════════════════════════════

def _tier_counts(n_devices: int) -> Dict[str, int]:
    """
    Distribute n_devices across 5 tiers.
    Guarantees at least 1 device per tier (except edge_switch).
    """
    n_router = max(1, n_devices // 50)
    n_ds     = max(1, n_devices // 12)
    n_fw     = max(1, n_devices //  6)
    n_sw     = max(1, n_devices //  3)
    n_edge   = max(0, n_devices - n_router - n_ds - n_fw - n_sw)

    # Trim if we overshot (can happen for very small N)
    total  = n_router + n_ds + n_fw + n_sw + n_edge
    excess = total - n_devices
    if excess > 0:
        n_edge = max(0, n_edge - excess); excess -= max(0, n_edge)
        n_sw   = max(1, n_sw   - excess)

    return {
        "router":        n_router,
        "dist_switch":   n_ds,
        "firewall":      n_fw,
        "access_switch": n_sw,
        "edge_switch":   n_edge,
    }


def build_topology(n_devices: int) -> dict:
    """
    Build a realistic hierarchical topology for n_devices devices.
    Returns {"nodes": [...], "edges": [...]}.
    """
    counts  = _tier_counts(n_devices)
    nodes:  List[dict] = []
    edges:  List[list] = []

    routers  = [f"R{i+1}"    for i in range(counts["router"])]
    dses     = [f"DS{i+1}"   for i in range(counts["dist_switch"])]
    fws      = [f"FW{i+1}"   for i in range(counts["firewall"])]
    sws      = [f"SW{i+1}"   for i in range(counts["access_switch"])]
    edge_dev = [f"Edge{i+1}" for i in range(counts["edge_switch"])]

    for d in routers:  nodes.append({"id": d, "role": "router"})
    for d in dses:     nodes.append({"id": d, "role": "dist_switch"})
    for d in fws:      nodes.append({"id": d, "role": "firewall"})
    for d in sws:      nodes.append({"id": d, "role": "access_switch"})
    for d in edge_dev: nodes.append({"id": d, "role": "edge_switch"})

    def _wire(parents: List[str], children: List[str]) -> None:
        """Round-robin: connect each child to a parent."""
        if not parents or not children:
            return
        for i, child in enumerate(children):
            edges.append([parents[i % len(parents)], child])

    _wire(routers,  dses)
    _wire(dses,     fws)
    _wire(fws,      sws)
    _wire(sws,      edge_dev)

    return {"nodes": nodes, "edges": edges}


def get_topology(n_devices: int = None) -> dict:
    """
    Return topology dict for n_devices.
    Falls back to config.N_DEVICES, then to 5.
    """
    if n_devices is None:
        try:
            from config import N_DEVICES
            n_devices = N_DEVICES
        except ImportError:
            n_devices = 5
    return build_topology(n_devices)


def get_target_events(topo: dict = None, n_devices: int = None) -> list:
    """
    Return representative target events guaranteed to exist in the topology.
    Uses the first firewall for HIGH_LATENCY and first access_switch for
    INTERFACE_FLAP.
    """
    if topo is None:
        topo = get_topology(n_devices)
    role_map   = {n["id"]: n["role"] for n in topo["nodes"]}
    fw_devices = [d for d, r in role_map.items() if r == "firewall"]
    sw_devices = [d for d, r in role_map.items() if r == "access_switch"]

    events = []
    if fw_devices:
        events.append({"device": fw_devices[0], "metric": "latency_ms",
                        "event": "HIGH_LATENCY",   "severity": "critical"})
    if sw_devices:
        events.append({"device": sw_devices[0], "metric": "packet_loss",
                        "event": "INTERFACE_FLAP", "severity": "critical"})
    return events


# ════════════════════════════════════════════════════════════════════════════
# Dataset generation
# ════════════════════════════════════════════════════════════════════════════

def generate_dataset(
    n_days:      int = None,
    freq_min:    int = None,
    n_events_hl: int = None,
    n_events_if: int = None,
    seed:        int = None,
    n_devices:   int = None,
) -> pd.DataFrame:
    """
    Returns a long-format DataFrame:
        timestamp | device | metric | value  (float32)

    All parameters default to config.py values when not explicitly supplied.
    """
    # ── Resolve config ────────────────────────────────────────────────────
    try:
        from config import get_config
        cfg = get_config()
    except ImportError:
        cfg = {}

    n_devices   = n_devices   if n_devices   is not None else cfg.get("n_devices",   5)
    n_days      = n_days      if n_days      is not None else cfg.get("n_days",      90)
    freq_min    = freq_min    if freq_min    is not None else cfg.get("poll_freq_min", 5)
    n_events_hl = n_events_hl if n_events_hl is not None else cfg.get("n_events_high_latency", 8)
    n_events_if = n_events_if if n_events_if is not None else cfg.get("n_events_iface_flap",   5)
    seed        = seed        if seed        is not None else cfg.get("seed", 42)

    rng     = np.random.default_rng(seed)
    n_steps = int(n_days * 24 * 60 / freq_min)
    ts      = pd.date_range("2026-01-01", periods=n_steps, freq=f"{freq_min}min")

    # ── Build topology ────────────────────────────────────────────────────
    topo     = build_topology(n_devices)
    role_map = {n["id"]: n["role"] for n in topo["nodes"]}
    by_role: Dict[str, List[str]] = {}
    for dev, role in role_map.items():
        by_role.setdefault(role, []).append(dev)

    routers   = by_role.get("router",        [])
    dses      = by_role.get("dist_switch",   [])
    fws       = by_role.get("firewall",      [])
    sws       = by_role.get("access_switch", [])

    # Build quick edge lookup
    edge_set = set()
    for a, b in topo["edges"]:
        edge_set.add((a, b))
        edge_set.add((b, a))

    # ── Diurnal envelope ──────────────────────────────────────────────────
    diurnal_base = np.sin(
        2 * np.pi * np.arange(n_steps) / (24 * 60 / freq_min)
    )

    # ── Baseline series ───────────────────────────────────────────────────
    base_series: Dict[Tuple[str, str], np.ndarray] = {}
    for dev, role in role_map.items():
        for met, (mu, sd, lo, hi) in _METRIC_SPECS.get(role, {}).items():
            s = mu + diurnal_base * (sd * 0.4) + rng.normal(0, sd, n_steps)
            base_series[(dev, met)] = np.clip(s, lo, hi)

    # ── Injection helpers ─────────────────────────────────────────────────
    def _make_wave(duration: int) -> np.ndarray:
        ramp = np.linspace(0, 1, max(1, duration // 2))
        return np.concatenate([ramp, np.ones(duration - len(ramp))])

    def _inj(key, start: int, wave: np.ndarray, amp: float, clip_hi: float):
        if key not in base_series:
            return
        s   = base_series[key]
        i1  = min(start + len(wave), n_steps)
        w   = wave[:i1 - start]
        if len(w) == 0:
            return
        s[start:i1] = np.clip(s[start:i1] + amp * w, 0, clip_hi)

    # ── HIGH_LATENCY injection (per router group) ─────────────────────────
    if routers and fws:
        for router in routers:
            # Find DS children of this router
            r_dses = [d for d in dses if (router, d) in edge_set] or dses
            # Find FW grandchildren (DS → FW)
            r_fws  = [f for f in fws if any((d, f) in edge_set for d in r_dses)] or fws

            safe_end = max(n_steps - 120, 100)
            if safe_end <= 100:
                break
            for _ in range(n_events_hl):
                start    = int(rng.integers(100, safe_end))
                duration = int(rng.integers(18, min(42, n_steps - start - 5)))
                wave     = _make_wave(duration)

                l0 = 0
                l1 = int(rng.integers(2, 5))
                l2 = l1 + int(rng.integers(2, 4))
                l3 = l2 + int(rng.integers(2, 4))
                l4 = l3 + int(rng.integers(1, 3))

                _inj((router, "cpu_pct"),    start + l0, wave, 40, 98)
                ds = r_dses[int(rng.integers(0, len(r_dses)))]
                _inj((ds, "buffer_util"),    start + l1, wave, 50, 98)
                _inj((ds, "crc_errors"),     start + l1, wave, 60, 500)
                for fw in r_fws:
                    _inj((fw, "latency_ms"),  start + l3, wave, 200, 500)
                    _inj((fw, "packet_loss"), start + l4, wave,   8,  20)
                if sws:
                    sw = sws[int(rng.integers(0, len(sws)))]
                    _inj((sw, "crc_errors"),  start + l2, wave, 120, 500)

    # ── INTERFACE_FLAP injection (per access switch) ──────────────────────
    for sw in sws:
        safe_end = max(n_steps - 60, 50)
        if safe_end <= 50:
            break
        for _ in range(n_events_if):
            start    = int(rng.integers(50, safe_end))
            duration = int(rng.integers(12, min(28, n_steps - start - 5)))
            wave     = _make_wave(duration)
            _inj((sw, "link_util"),   start + 0, wave, 40,  100)
            _inj((sw, "buffer_util"), start + 1, wave, 45,  100)
            _inj((sw, "crc_errors"),  start + 2, wave, 200, 500)
            _inj((sw, "packet_loss"), start + 3, wave,   6,  10)

    # ── Assemble DataFrame ────────────────────────────────────────────────
    chunks: List[pd.DataFrame] = []
    for (dev, met), series in base_series.items():
        chunks.append(pd.DataFrame({
            "timestamp": ts,
            "device":    dev,
            "metric":    met,
            "value":     series.astype(np.float32),
        }))

    return pd.concat(chunks, ignore_index=True)


# ════════════════════════════════════════════════════════════════════════════
# Quick self-test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    print(f"{'N':>5}  {'router':>6}  {'ds':>4}  {'fw':>4}  {'sw':>4}  "
          f"{'edge':>5}  {'rows':>10}  {'time':>6}")
    print("-" * 60)
    for n in [5, 10, 20, 50, 100]:
        t0     = time.time()
        counts = _tier_counts(n)
        df     = generate_dataset(n_devices=n, n_days=10)
        print(f"{n:>5}  "
              f"{counts['router']:>6}  {counts['dist_switch']:>4}  "
              f"{counts['firewall']:>4}  {counts['access_switch']:>4}  "
              f"{counts['edge_switch']:>5}  "
              f"{len(df):>10,}  {time.time()-t0:>5.1f}s")
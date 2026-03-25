"""
data_generator.py
=================
Generates a synthetic but realistic multi-device network time-series dataset
with EMBEDDED causal patterns, scaled to 100 devices over 30 days.

Topology design (100 devices):
  ─ 2 Core Routers         (R1, R2)
  ─ 4 Distribution Switches (DS1..DS4)
  ─ 8 Firewall/IDS nodes   (FW1..FW8)
  ─ 16 Access Switches     (SW1..SW16)
  ─ 70 Edge Switches       (Edge1..Edge70)

Each FW is a "target" device for HIGH_LATENCY.
Each SW  is a "target" device for INTERFACE_FLAP.

Injected causal chains (same logic as original, replicated per group):

  HIGH_LATENCY pattern (per router→SW→DS→FW group):
    R:cpu_pct ↑ → SW:crc_errors ↑ → DS:buffer_util ↑
      → FW:latency_ms ↑ → FW:packet_loss ↑

  INTERFACE_FLAP pattern (per SW):
    SW:link_util ↑ → SW:buffer_util ↑ → SW:crc_errors ↑ → SW:packet_loss ↑

Optimisation notes for 100-device / 30-day scale:
  - Numpy vectorised operations throughout (no Python loops over rows)
  - Final DataFrame built from pre-allocated arrays and a single pd.concat
  - Memory-efficient: float32 values
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


# ── Topology definition ───────────────────────────────────────────────────────

def get_topology() -> dict:
    nodes = []
    edges = []

    # Core routers
    for i in range(1, 3):
        nodes.append({"id": f"R{i}", "role": "router"})

    # Distribution switches (2 per router)
    for i in range(1, 5):
        nodes.append({"id": f"DS{i}", "role": "dist_switch"})

    # Firewalls (2 per DS)
    for i in range(1, 9):
        nodes.append({"id": f"FW{i}", "role": "firewall"})

    # Access switches (2 per FW)
    for i in range(1, 17):
        nodes.append({"id": f"SW{i}", "role": "access_switch"})

    # Edge switches (split evenly under each SW, ~4-5 per SW)
    edge_id = 1
    for sw_i in range(1, 17):
        count = 5 if sw_i <= 6 else 4
        for _ in range(count):
            nodes.append({"id": f"Edge{edge_id}", "role": "edge_switch"})
            edge_id += 1

    # Edges: R1 → DS1,DS2  |  R2 → DS3,DS4
    edges += [["R1", "DS1"], ["R1", "DS2"], ["R2", "DS3"], ["R2", "DS4"]]
    # DS → FW  (2 FW per DS)
    fw_map = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8]}
    for ds_i, fws in fw_map.items():
        for fw in fws:
            edges.append([f"DS{ds_i}", f"FW{fw}"])
    # FW → SW  (2 SW per FW)
    sw_per_fw = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8],
                 5: [9, 10], 6: [11, 12], 7: [13, 14], 8: [15, 16]}
    for fw_i, sws in sw_per_fw.items():
        for sw in sws:
            edges.append([f"FW{fw_i}", f"SW{sw}"])
    # SW → Edge
    edge_id = 1
    for sw_i in range(1, 17):
        count = 5 if sw_i <= 6 else 4
        for _ in range(count):
            edges.append([f"SW{sw_i}", f"Edge{edge_id}"])
            edge_id += 1

    return {"nodes": nodes, "edges": edges}


def get_target_events() -> list:
    """
    Return one HIGH_LATENCY target per FW and one INTERFACE_FLAP per SW.
    For discovery we return representative ones (FW1, SW1) to avoid
    an explosion of target scans; the injected patterns on all groups
    mean the discovered chain generalises.
    """
    return [
        {"device": "FW1", "metric": "latency_ms",
         "event": "HIGH_LATENCY",   "severity": "critical"},
        {"device": "SW1", "metric": "packet_loss",
         "event": "INTERFACE_FLAP", "severity": "critical"},
    ]


# ── Dataset generation ────────────────────────────────────────────────────────

# Metric specs: (mean, std, lo, hi)
_METRIC_SPECS = {
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

# Router ↔ FW group mappings for HIGH_LATENCY injection
# R1 feeds DS1/DS2 → FW1..FW4 → SW1..SW8
# R2 feeds DS3/DS4 → FW5..FW8 → SW9..SW16
_HL_GROUPS = [
    # (router, [ds_list], [fw_list])
    ("R1", ["DS1", "DS2"], ["FW1", "FW2", "FW3", "FW4"]),
    ("R2", ["DS3", "DS4"], ["FW5", "FW6", "FW7", "FW8"]),
]

# SW list for INTERFACE_FLAP
_SW_LIST = [f"SW{i}" for i in range(1, 17)]


def generate_dataset(
    n_days:      int = 30,
    freq_min:    int = 5,
    n_events_hl: int = 20,    # per group
    n_events_if: int = 15,    # per SW
    seed:        int = 42,
) -> pd.DataFrame:
    """
    Returns a long-format DataFrame:
        timestamp | device | metric | value   (value as float32)

    All numpy operations are vectorised; no Python loops over rows.
    """
    rng = np.random.default_rng(seed)

    n_steps = int(n_days * 24 * 60 / freq_min)
    ts = pd.date_range("2026-01-01", periods=n_steps, freq=f"{freq_min}min")

    topo = get_topology()
    role_map: Dict[str, str] = {n["id"]: n["role"] for n in topo["nodes"]}

    # Diurnal envelope (reused for all series)
    diurnal_base = np.sin(2 * np.pi * np.arange(n_steps) / (24 * 60 / freq_min))

    # ── Build baseline series ─────────────────────────────────────────────
    base_series: Dict[Tuple[str, str], np.ndarray] = {}

    for node in topo["nodes"]:
        dev  = node["id"]
        role = node["role"]
        for met, (mu, sd, lo, hi) in _METRIC_SPECS.get(role, {}).items():
            diurnal = diurnal_base * (sd * 0.4)          # ±40% of std as diurnal
            s = mu + diurnal + rng.normal(0, sd, n_steps)
            base_series[(dev, met)] = np.clip(s, lo, hi)

    # ── Inject HIGH_LATENCY causal episodes ──────────────────────────────
    # Pattern: R:cpu↑ → DS:buffer↑ → FW:latency↑ → FW:packet_loss↑
    # (also touches SW:crc_errors to preserve original chain visibility)
    for router, ds_list, fw_list in _HL_GROUPS:
        for _ in range(n_events_hl):
            start    = int(rng.integers(100, n_steps - 120))
            duration = int(rng.integers(18, 42))
            ramp     = np.linspace(0, 1, duration // 2)
            wave     = np.concatenate([ramp, np.ones(duration - len(ramp))])

            l0 = 0
            l1 = int(rng.integers(2, 5))
            l2 = l1 + int(rng.integers(2, 4))
            l3 = l2 + int(rng.integers(2, 4))
            l4 = l3 + int(rng.integers(1, 3))

            def _inj(key, offset, amp, hi):
                if key not in base_series:
                    return
                s  = base_series[key]
                i0 = min(start + offset, n_steps - 1)
                i1 = min(i0 + duration, n_steps)
                w  = wave[:i1 - i0]
                s[i0:i1] = np.clip(s[i0:i1] + amp * w, 0, hi)

            # Root cause: router cpu
            _inj((router, "cpu_pct"), l0, 40, 98)

            # Propagate to a random DS in the group
            ds = ds_list[int(rng.integers(0, len(ds_list)))]
            _inj((ds, "buffer_util"), l1, 50, 98)
            _inj((ds, "crc_errors"), l1, 60, 500)

            # Propagate to all FWs in the group (stronger signal for discovery)
            for fw in fw_list:
                _inj((fw, "latency_ms"),  l3, 200, 500)
                _inj((fw, "packet_loss"), l4,   8,  20)

            # Also inject on a random SW for cross-device chain
            sw_candidates = [f"SW{i}" for i in range(1, 17)
                             if (i - 1) // 2 in range(len(fw_list))]
            if sw_candidates:
                sw = sw_candidates[int(rng.integers(0, len(sw_candidates)))]
                _inj((sw, "crc_errors"), l2, 120, 500)

    # ── Inject INTERFACE_FLAP causal episodes ─────────────────────────────
    # Pattern: SW:link_util↑ → SW:buffer_util↑ → SW:crc_errors↑ → SW:packet_loss↑
    for sw in _SW_LIST:
        for _ in range(n_events_if):
            start    = int(rng.integers(50, n_steps - 60))
            duration = int(rng.integers(12, 28))
            ramp     = np.linspace(0, 1, duration // 2)
            wave     = np.concatenate([ramp, np.ones(duration - len(ramp))])

            def _inj2(key, offset, amp, hi):
                if key not in base_series:
                    return
                s  = base_series[key]
                i0 = min(start + offset, n_steps - 1)
                i1 = min(i0 + duration, n_steps)
                w  = wave[:i1 - i0]
                s[i0:i1] = np.clip(s[i0:i1] + amp * w, 0, hi)

            _inj2((sw, "link_util"),   0, 40, 100)
            _inj2((sw, "buffer_util"), 1, 45, 100)
            _inj2((sw, "crc_errors"),  2, 200, 500)
            _inj2((sw, "packet_loss"), 3, 6, 10)

    # ── Assemble DataFrame (vectorised, float32) ──────────────────────────
    chunks: List[pd.DataFrame] = []
    for (dev, met), series in base_series.items():
        chunks.append(pd.DataFrame({
            "timestamp": ts,
            "device":    dev,
            "metric":    met,
            "value":     series.astype(np.float32),
        }))

    df = pd.concat(chunks, ignore_index=True)
    return df


if __name__ == "__main__":
    print("Generating 100-device / 30-day dataset...")
    df = generate_dataset()
    print(f"Shape: {df.shape}")
    print(f"Devices: {df['device'].nunique()}")
    print(f"Metrics: {df['metric'].nunique()}")
    print(df.head(10))
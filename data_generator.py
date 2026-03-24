"""
data_generator.py
=================
Generates a synthetic but realistic multi-device network time-series dataset
with EMBEDDED causal patterns.

Topology:
    R1 (router) ──── SW1 (access_switch) ──── SW2 (dist_switch)
                                                      |
                                                 FW1 (firewall)
                                                      |
                                                  Edge1 (edge_switch)

Injected causal chain (HIGH_LATENCY precursor):
    R1:cpu_pct ↑  →  SW1:crc_errors ↑  →  SW2:buffer_util ↑
       →  FW1:latency_ms ↑  →  FW1:packet_loss ↑

Injected causal chain (INTERFACE_FLAP precursor):
    SW1:link_util ↑  →  SW1:buffer_util ↑  →  SW1:crc_errors ↑
       →  SW1:packet_loss ↑
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import json
from typing import List, Dict


RNG = np.random.default_rng(42)


def _noise(n: int, scale: float = 1.0) -> np.ndarray:
    return RNG.normal(0, scale, n)


def generate_dataset(
    n_days:       int   = 10,
    freq_min:     int   = 5,
    n_events_hl:  int   = 8,   # HIGH_LATENCY events to inject
    n_events_if:  int   = 5,   # INTERFACE_FLAP events to inject
    seed:         int   = 42,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        timestamp | device | metric | value
    """
    rng = np.random.default_rng(seed)

    n_steps = int(n_days * 24 * 60 / freq_min)
    ts = pd.date_range("2026-01-01", periods=n_steps, freq=f"{freq_min}min")

    rows: List[Dict] = []

    # ── Baseline metrics (all devices) ───────────────────────────────────

    baselines = {
        # (device, metric): (mean, std, min_clip, max_clip)
        ("R1",    "cpu_pct"):     (35, 5,  5,   95),
        ("R1",    "latency_ms"):  (8,  2,  1,   200),
        ("SW1",   "cpu_pct"):     (20, 3,  2,   95),
        ("SW1",   "crc_errors"):  (2,  1,  0,   500),
        ("SW1",   "buffer_util"): (30, 6,  0,   100),
        ("SW1",   "link_util"):   (45, 8,  0,   100),
        ("SW1",   "packet_loss"): (0.2,0.1,0,   10),
        ("SW2",   "cpu_pct"):     (25, 4,  2,   95),
        ("SW2",   "crc_errors"):  (1,  0.5,0,   500),
        ("SW2",   "buffer_util"): (25, 5,  0,   100),
        ("SW2",   "latency_ms"):  (5,  1,  1,   200),
        ("FW1",   "cpu_pct"):     (40, 7,  2,   95),
        ("FW1",   "latency_ms"):  (12, 3,  1,   500),
        ("FW1",   "packet_loss"): (0.5,0.2,0,   20),
        ("FW1",   "queue_depth"): (20, 4,  0,   200),
        ("Edge1", "cpu_pct"):     (18, 3,  2,   95),
        ("Edge1", "latency_ms"):  (6,  1.5,1,   200),
        ("Edge1", "packet_loss"): (0.3,0.1,0,   10),
    }

    base_series = {}
    for (dev, met), (mu, sd, lo, hi) in baselines.items():
        # Add slow sinusoidal trend (diurnal pattern)
        diurnal = 5 * np.sin(2 * np.pi * np.arange(n_steps) / (24*60/freq_min))
        s = mu + diurnal * (sd / 5) + rng.normal(0, sd, n_steps)
        s = np.clip(s, lo, hi)
        base_series[(dev, met)] = s.copy()

    # ── Inject HIGH_LATENCY causal episodes ──────────────────────────────
    # Pattern: R1:cpu↑ → SW1:crc↑ → SW2:buffer↑ → FW1:latency↑ → FW1:pkt_loss↑
    hl_event_indices = []

    for _ in range(n_events_hl):
        start = int(rng.integers(100, n_steps - 100))
        duration = int(rng.integers(18, 36))   # 90-180 min

        ramp = np.linspace(0, 1, duration // 2)
        plateau = np.ones(duration - len(ramp))
        wave = np.concatenate([ramp, plateau])

        lag_0 = 0
        lag_1 = int(rng.integers(2, 5))    # SW1:crc  starts 10-25min later
        lag_2 = lag_1 + int(rng.integers(2, 4))
        lag_3 = lag_2 + int(rng.integers(2, 4))
        lag_4 = lag_3 + int(rng.integers(1, 3))

        def inject(key, offset, amplitude, clip_hi):
            s = base_series[key]
            idx_start = min(start + offset, n_steps - 1)
            idx_end   = min(idx_start + duration, n_steps)
            w = wave[:idx_end - idx_start]
            s[idx_start:idx_end] = np.clip(
                s[idx_start:idx_end] + amplitude * w, 0, clip_hi
            )

        inject(("R1",  "cpu_pct"),    lag_0, 40,  98)
        inject(("SW1", "crc_errors"), lag_1, 120, 500)
        inject(("SW2", "buffer_util"),lag_2, 50,  98)
        inject(("FW1", "latency_ms"), lag_3, 200, 500)
        inject(("FW1", "packet_loss"),lag_4, 8,   20)

        hl_event_indices.append(start + lag_4 + duration)

    # ── Inject INTERFACE_FLAP causal episodes ─────────────────────────────
    # Pattern: SW1:link_util↑ → SW1:buffer_util↑ → SW1:crc_errors↑ → SW1:pkt_loss↑

    for _ in range(n_events_if):
        start    = int(rng.integers(50, n_steps - 50))
        duration = int(rng.integers(12, 24))

        ramp = np.linspace(0, 1, duration // 2)
        wave = np.concatenate([ramp, np.ones(duration - len(ramp))])

        lag_0, lag_1, lag_2, lag_3 = 0, 1, 2, 3   # 5/10/15min lags

        def inj2(key, offset, amp, hi):
            s = base_series[key]
            i0 = min(start + offset, n_steps - 1)
            i1 = min(i0 + duration, n_steps)
            w  = wave[:i1 - i0]
            s[i0:i1] = np.clip(s[i0:i1] + amp * w, 0, hi)

        inj2(("SW1", "link_util"),   lag_0, 40, 100)
        inj2(("SW1", "buffer_util"), lag_1, 45, 100)
        inj2(("SW1", "crc_errors"),  lag_2, 200, 500)
        inj2(("SW1", "packet_loss"), lag_3, 6, 10)

    # ── Convert to DataFrame ──────────────────────────────────────────────

    for (dev, met), series in base_series.items():
        for i, val in enumerate(series):
            rows.append({
                "timestamp": ts[i],
                "device":    dev,
                "metric":    met,
                "value":     round(float(val), 4),
            })

    df = pd.DataFrame(rows)
    return df


def get_topology() -> dict:
    return {
        "nodes": [
            {"id": "R1",    "role": "router"},
            {"id": "SW1",   "role": "access_switch"},
            {"id": "SW2",   "role": "dist_switch"},
            {"id": "FW1",   "role": "firewall"},
            {"id": "Edge1", "role": "edge_switch"},
        ],
        "edges": [
            ["R1",  "SW1"],
            ["SW1", "SW2"],
            ["SW2", "FW1"],
            ["FW1", "Edge1"],
        ],
    }


def get_target_events() -> list:
    return [
        {
            "device":   "FW1",
            "metric":   "latency_ms",
            "event":    "HIGH_LATENCY",
            "severity": "critical",
        },
        {
            "device":   "SW1",
            "metric":   "packet_loss",
            "event":    "INTERFACE_FLAP",
            "severity": "critical",
        },
    ]


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head(20))
    print(f"\nDataset shape: {df.shape}")
    print(df.groupby(["device", "metric"])["value"].describe())

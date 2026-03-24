"""
feature_engine.py
=================
Computes 8 statistical features from a 75-minute sliding window for every
(device, metric) pair.  Features: mean, max, min, std, last, range, slope, delta.

Slope → direction
    slope >  +threshold → "up"
    slope < -threshold  → "down"
    otherwise           → "flat"
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict


SLOPE_THRESHOLD = 0.01   # per-minute slope to call "up" or "down"
WINDOW_MINUTES  = 75
SMOOTHING_ALPHA = 0.3    # EWM smoothing factor


@dataclass
class MetricFeatures:
    device:    str
    metric:    str
    window_start: pd.Timestamp
    window_end:   pd.Timestamp
    mean:      float
    maximum:   float
    minimum:   float
    std:       float
    last:      float
    rng:       float   # range = max - min
    slope:     float   # per-minute
    delta:     float   # last - first
    direction: str     # "up" | "down" | "flat"
    n_points:  int

    def to_dict(self) -> dict:
        return asdict(self)

    def value_for(self, feature: str) -> float:
        """Retrieve a feature value by name (matches pattern JSON keys)."""
        mapping = {
            "mean": self.mean, "max": self.maximum, "min": self.minimum,
            "std": self.std, "last": self.last, "range": self.rng,
            "slope": self.slope, "delta": self.delta,
        }
        return mapping[feature]


class FeatureEngine:
    """
    Given a raw time-series DataFrame (timestamp, device, metric, value),
    compute MetricFeatures for every (device, metric) in every sliding window.

    Usage
    -----
    fe = FeatureEngine(window_minutes=75, step_minutes=5)
    windows = fe.compute_all_windows(df)
    # windows: list of dicts  {timestamp → MetricFeatures}
    """

    def __init__(
        self,
        window_minutes:  int   = WINDOW_MINUTES,
        step_minutes:    int   = 5,
        slope_threshold: float = SLOPE_THRESHOLD,
        smooth:          bool  = True,
        smooth_alpha:    float = SMOOTHING_ALPHA,
    ):
        self.window_minutes  = window_minutes
        self.step_minutes    = step_minutes
        self.slope_threshold = slope_threshold
        self.smooth          = smooth
        self.smooth_alpha    = smooth_alpha

    # ── Core computation ─────────────────────────────────────────────────

    def compute_features(
        self,
        series: pd.Series,          # index = timestamp, values = metric values
        device: str,
        metric: str,
    ) -> Optional[MetricFeatures]:
        """Compute features for a single (device, metric) window."""
        if len(series) < 3:
            return None

        series = series.sort_index().dropna()
        if self.smooth:
            series = series.ewm(alpha=self.smooth_alpha, adjust=False).mean()

        values = series.values.astype(float)
        t_min  = series.index[0]
        t_max  = series.index[-1]

        # Slope: linear regression of value vs elapsed minutes
        elapsed = np.array(
            [(ts - t_min).total_seconds() / 60 for ts in series.index],
            dtype=float,
        )
        if elapsed[-1] > 0:
            slope = float(np.polyfit(elapsed, values, 1)[0])
        else:
            slope = 0.0

        delta = float(values[-1] - values[0])
        rng   = float(values.max() - values.min())

        if slope > self.slope_threshold:
            direction = "up"
        elif slope < -self.slope_threshold:
            direction = "down"
        else:
            direction = "flat"

        return MetricFeatures(
            device       = device,
            metric       = metric,
            window_start = t_min,
            window_end   = t_max,
            mean         = float(values.mean()),
            maximum      = float(values.max()),
            minimum      = float(values.min()),
            std          = float(values.std()),
            last         = float(values[-1]),
            rng          = rng,
            slope        = slope,
            delta        = delta,
            direction    = direction,
            n_points     = len(values),
        )

    # ── Windowing ────────────────────────────────────────────────────────

    def compute_all_windows(
        self, df: pd.DataFrame
    ) -> List[Dict[Tuple[str, str], MetricFeatures]]:
        """
        Slide over the full dataset and return one feature dict per window.

        df columns: timestamp (datetime), device (str), metric (str), value (float)
        Returns: list of {(device, metric): MetricFeatures} dicts
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        t_start  = df["timestamp"].min()
        t_end    = df["timestamp"].max()
        win_td   = pd.Timedelta(minutes=self.window_minutes)
        step_td  = pd.Timedelta(minutes=self.step_minutes)

        windows: List[Dict[Tuple[str, str], MetricFeatures]] = []
        t = t_start + win_td

        while t <= t_end:
            w_start  = t - win_td
            mask     = (df["timestamp"] >= w_start) & (df["timestamp"] <= t)
            w_df     = df[mask]
            feat_dict: Dict[Tuple[str, str], MetricFeatures] = {}

            for (device, metric), grp in w_df.groupby(["device", "metric"]):
                series = grp.set_index("timestamp")["value"]
                feat   = self.compute_features(series, device, metric)
                if feat:
                    feat_dict[(device, metric)] = feat

            if feat_dict:
                windows.append(feat_dict)
            t += step_td

        return windows

    # ── Single-window helper (for inference) ─────────────────────────────

    def compute_latest_window(
        self, df: pd.DataFrame
    ) -> Dict[Tuple[str, str], MetricFeatures]:
        """
        Compute features for the MOST RECENT 75-min window only.
        Used by the inference engine on each new poll.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        t_end   = df["timestamp"].max()
        t_start = t_end - pd.Timedelta(minutes=self.window_minutes)
        mask    = df["timestamp"] >= t_start

        feat_dict: Dict[Tuple[str, str], MetricFeatures] = {}
        for (device, metric), grp in df[mask].groupby(["device", "metric"]):
            series = grp.set_index("timestamp")["value"]
            feat   = self.compute_features(series, device, metric)
            if feat:
                feat_dict[(device, metric)] = feat
        return feat_dict


def print_feature_table(feat_dict: Dict[Tuple[str, str], MetricFeatures]) -> None:
    """Pretty-print a feature dict for one window."""
    print(f"\n{'Device':12s} {'Metric':15s} {'Mean':>8} {'Max':>8} "
          f"{'Min':>8} {'Slope':>9} {'Delta':>8} {'Dir':>6}")
    print("-" * 80)
    for (dev, met), f in sorted(feat_dict.items()):
        print(f"{dev:12s} {met:15s} {f.mean:8.3f} {f.maximum:8.3f} "
              f"{f.minimum:8.3f} {f.slope:9.5f} {f.delta:8.3f} {f.direction:>6s}")

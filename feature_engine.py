"""
feature_engine.py
=================
Computes 8 statistical features from a sliding window for every
(device, metric) pair.  Features: mean, max, min, std, last, range, slope, delta.

Slope → direction
    slope >  +threshold → "up"
    slope < -threshold  → "down"
    otherwise           → "flat"

Optimisations for 100-device / 30-day scale:
  - Pre-pivot the DataFrame into a wide matrix once (O(1) per window)
  - Vectorised numpy ops for all metrics in a single window at once
  - Parallel window batches via ProcessPoolExecutor (configurable workers)
  - Chunked processing to cap peak RAM usage
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import math


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
        mapping = {
            "mean": self.mean, "max": self.maximum, "min": self.minimum,
            "std": self.std, "last": self.last, "range": self.rng,
            "slope": self.slope, "delta": self.delta,
        }
        return mapping[feature]


# ── Vectorised single-window computation ─────────────────────────────────────

def _compute_window_features(
    wide: pd.DataFrame,           # index=timestamp, columns=(device,metric)
    w_start: pd.Timestamp,
    w_end:   pd.Timestamp,
    slope_threshold: float,
) -> Dict[Tuple[str, str], MetricFeatures]:
    """
    Compute features for ALL (device, metric) columns in one window slice.
    `wide` is already sliced to [w_start, w_end].
    """
    if wide.empty or len(wide) < 3:
        return {}

    # EWM smoothing (vectorised across all columns at once)
    smoothed = wide.ewm(alpha=SMOOTHING_ALPHA, adjust=False).mean()
    values   = smoothed.values.astype(np.float32)    # shape (T, N_cols)
    T        = values.shape[0]

    # Elapsed minutes vector
    elapsed = np.array(
        [(t - w_start).total_seconds() / 60.0 for t in smoothed.index],
        dtype=np.float32,
    )

    # Vectorised linear regression slope for every column simultaneously
    # slope[j] = (n*Σ(t*y) - Σt*Σy) / (n*Σt² - (Σt)²)
    n       = float(T)
    sum_t   = elapsed.sum()
    sum_t2  = (elapsed ** 2).sum()
    denom   = n * sum_t2 - sum_t ** 2
    if abs(denom) < 1e-9:
        slopes = np.zeros(values.shape[1], dtype=np.float32)
    else:
        sum_y   = values.sum(axis=0)
        sum_ty  = (elapsed[:, None] * values).sum(axis=0)
        slopes  = (n * sum_ty - sum_t * sum_y) / denom

    feat_dict: Dict[Tuple[str, str], MetricFeatures] = {}
    cols = wide.columns.tolist()   # list of (device, metric) tuples

    for j, (dev, met) in enumerate(cols):
        col = values[:, j]
        if np.all(np.isnan(col)):
            continue

        slope  = float(slopes[j])
        delta  = float(col[-1] - col[0])
        rng    = float(col.max() - col.min())

        if slope > slope_threshold:
            direction = "up"
        elif slope < -slope_threshold:
            direction = "down"
        else:
            direction = "flat"

        feat_dict[(dev, met)] = MetricFeatures(
            device       = dev,
            metric       = met,
            window_start = w_start,
            window_end   = w_end,
            mean         = float(np.nanmean(col)),
            maximum      = float(np.nanmax(col)),
            minimum      = float(np.nanmin(col)),
            std          = float(np.nanstd(col)),
            last         = float(col[-1]),
            rng          = rng,
            slope        = slope,
            delta        = delta,
            direction    = direction,
            n_points     = T,
        )
    return feat_dict


# ── Worker function (top-level for pickling) ─────────────────────────────────

def _worker_batch(args):
    """Process a batch of window timestamps.  Returns list of feat_dicts."""
    wide_bytes, window_times, window_minutes, slope_threshold = args
    import io, pickle
    wide: pd.DataFrame = pickle.loads(wide_bytes)

    results = []
    win_td  = pd.Timedelta(minutes=window_minutes)

    for w_end in window_times:
        w_start = w_end - win_td
        slice_  = wide[(wide.index >= w_start) & (wide.index <= w_end)]
        fd = _compute_window_features(slice_, w_start, w_end, slope_threshold)
        results.append(fd)
    return results


# ── FeatureEngine ─────────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Given a raw time-series DataFrame (timestamp, device, metric, value),
    compute MetricFeatures for every (device, metric) in every sliding window.

    Usage
    -----
    fe = FeatureEngine(window_minutes=75, step_minutes=5)
    windows = fe.compute_all_windows(df)
    # windows: list of dicts  {(device,metric) → MetricFeatures}
    """

    def __init__(
        self,
        window_minutes:  int   = WINDOW_MINUTES,
        step_minutes:    int   = 5,
        slope_threshold: float = SLOPE_THRESHOLD,
        smooth:          bool  = True,
        smooth_alpha:    float = SMOOTHING_ALPHA,
        n_workers:       int   = None,   # None = auto (cpu_count)
        batch_size:      int   = 200,    # windows per worker batch
    ):
        self.window_minutes  = window_minutes
        self.step_minutes    = step_minutes
        self.slope_threshold = slope_threshold
        self.smooth          = smooth
        self.smooth_alpha    = smooth_alpha
        self.n_workers       = n_workers or max(1, multiprocessing.cpu_count())
        self.batch_size      = batch_size

    # ── Core preparation ─────────────────────────────────────────────────

    def _pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot long → wide; index=timestamp, columns=(device,metric)."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        wide = df.pivot_table(
            index="timestamp", columns=["device", "metric"],
            values="value", aggfunc="mean",
        )
        wide.columns = [tuple(c) for c in wide.columns]
        # Resample to fixed freq, forward-fill small gaps
        freq = f"{self.step_minutes}min"
        wide = wide.resample(freq).mean().ffill(limit=2)
        return wide

    # ── Main windowing ────────────────────────────────────────────────────

    def compute_all_windows(
        self, df: pd.DataFrame,
    ) -> List[Dict[Tuple[str, str], MetricFeatures]]:
        """
        Slide over the full dataset and return one feature dict per window.
        Uses parallel workers for large datasets.
        """
        print(f"  [FeatureEngine] Pivoting data...")
        wide = self._pivot(df)
        print(f"  [FeatureEngine] Wide matrix: {wide.shape[0]} timesteps × "
              f"{wide.shape[1]} (device,metric) columns")

        win_td  = pd.Timedelta(minutes=self.window_minutes)
        step_td = pd.Timedelta(minutes=self.step_minutes)
        t_start = wide.index.min() + win_td
        t_end   = wide.index.max()

        window_times = pd.date_range(start=t_start, end=t_end, freq=step_td)
        total = len(window_times)
        print(f"  [FeatureEngine] Computing {total} windows "
              f"(workers={self.n_workers}, batch={self.batch_size})...")

        # Split into batches
        batches = [
            window_times[i: i + self.batch_size].tolist()
            for i in range(0, total, self.batch_size)
        ]

        import pickle
        wide_bytes = pickle.dumps(wide)

        all_windows: List[Dict[Tuple[str, str], MetricFeatures]] = []

        # Use parallel workers only if there are enough batches
        if self.n_workers > 1 and len(batches) > 2:
            args_list = [
                (wide_bytes, batch, self.window_minutes, self.slope_threshold)
                for batch in batches
            ]
            with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                futures = {ex.submit(_worker_batch, a): i
                           for i, a in enumerate(args_list)}
                batch_results = [None] * len(batches)
                done = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    batch_results[idx] = fut.result()
                    done += len(batches[idx])
                    if done % 2000 == 0 or done == total:
                        print(f"    {done}/{total} windows done...")
            for br in batch_results:
                all_windows.extend(br)
        else:
            # Single-process fallback
            done = 0
            for batch in batches:
                for w_end in batch:
                    w_start = w_end - win_td
                    slice_  = wide[(wide.index >= w_start) & (wide.index <= w_end)]
                    fd = _compute_window_features(
                        slice_, w_start, w_end, self.slope_threshold
                    )
                    all_windows.append(fd)
                    done += 1
                    if done % 2000 == 0:
                        print(f"    {done}/{total} windows done...")

        # Remove empty windows
        all_windows = [w for w in all_windows if w]
        print(f"  [FeatureEngine] Done — {len(all_windows)} non-empty windows")
        return all_windows

    # ── Single-window helper (for inference) ─────────────────────────────

    def compute_latest_window(
        self, df: pd.DataFrame
    ) -> Dict[Tuple[str, str], MetricFeatures]:
        """Compute features for the MOST RECENT window only (used by inference)."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        t_end   = df["timestamp"].max()
        t_start = t_end - pd.Timedelta(minutes=self.window_minutes)
        mask    = df["timestamp"] >= t_start

        wide = df[mask].pivot_table(
            index="timestamp", columns=["device", "metric"],
            values="value", aggfunc="mean",
        )
        wide.columns = [tuple(c) for c in wide.columns]

        if wide.empty:
            return {}
        w_start = wide.index.min()
        w_end   = wide.index.max()
        return _compute_window_features(wide, w_start, w_end, self.slope_threshold)

    # ── Single series-level (for compatibility) ───────────────────────────

    def compute_features(
        self,
        series: pd.Series,
        device: str,
        metric: str,
    ) -> Optional[MetricFeatures]:
        """Compute features for a single (device, metric) window (compat shim)."""
        if len(series) < 3:
            return None
        series = series.sort_index().dropna()
        if self.smooth:
            series = series.ewm(alpha=self.smooth_alpha, adjust=False).mean()
        values  = series.values.astype(float)
        t_min   = series.index[0]
        t_max   = series.index[-1]
        elapsed = np.array(
            [(t - t_min).total_seconds() / 60.0 for t in series.index],
            dtype=float,
        )
        slope = float(np.polyfit(elapsed, values, 1)[0]) if elapsed[-1] > 0 else 0.0
        delta = float(values[-1] - values[0])
        rng   = float(values.max() - values.min())
        if slope > self.slope_threshold:
            direction = "up"
        elif slope < -self.slope_threshold:
            direction = "down"
        else:
            direction = "flat"
        return MetricFeatures(
            device=device, metric=metric,
            window_start=t_min, window_end=t_max,
            mean=float(values.mean()), maximum=float(values.max()),
            minimum=float(values.min()), std=float(values.std()),
            last=float(values[-1]), rng=rng,
            slope=slope, delta=delta, direction=direction,
            n_points=len(values),
        )


def print_feature_table(feat_dict: Dict[Tuple[str, str], MetricFeatures]) -> None:
    """Pretty-print a feature dict for one window."""
    print(f"\n{'Device':12s} {'Metric':15s} {'Mean':>8} {'Max':>8} "
          f"{'Min':>8} {'Slope':>9} {'Delta':>8} {'Dir':>6}")
    print("-" * 80)
    for (dev, met), f in sorted(feat_dict.items()):
        print(f"{dev:12s} {met:15s} {f.mean:8.3f} {f.maximum:8.3f} "
              f"{f.minimum:8.3f} {f.slope:9.5f} {f.delta:8.3f} {f.direction:>6s}")
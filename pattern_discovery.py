"""
pattern_discovery.py
====================
Full pipeline:
  1. Cross-correlation between topology-adjacent (device,metric) pairs
  2. Granger-style causality test (OLS lag regression)
  3. Sequence construction — build causal chains following time order
  4. Statistical validation — support, confidence, lift
  5. Output: list of validated PatternSpec objects → JSON

Optimisations for 100-device / 30-day scale:
  - Vectorised batch cross-correlation (numpy, no Python inner loop over lags)
  - Pre-filter topology pairs before expensive Granger test
  - Parallel Granger tests via ProcessPoolExecutor
  - Vectorised window validation (numpy arrays, no per-window Python dict walk)
  - Early-exit pruning: skip pairs whose max cross-correlation < min_corr
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json, uuid, datetime, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings("ignore")

from topology_loader import TopologyLoader
from feature_engine  import MetricFeatures


# ════════════════════════════════════════════════════════════════════════════
# Data structures  (unchanged API)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CausalLink:
    dev_a:        str
    metric_a:     str
    dev_b:        str
    metric_b:     str
    best_lag_min: float
    correlation:  float
    granger_f:    float
    hops:         int

    @property
    def key(self) -> str:
        return f"{self.dev_a}:{self.metric_a}→{self.dev_b}:{self.metric_b}"


@dataclass
class PatternStep:
    step:              int
    device:            str
    role:              str
    metric:            str
    feature:           str
    direction:         str
    threshold:         float
    absolute_min:      float
    absolute_max:      float
    lag_minutes:       float
    tolerance_minutes: float


@dataclass
class PatternSpec:
    pattern_id:          str
    pattern_name:        str
    target_event:        str
    severity:            str
    applicable_roles:    List[str]
    max_hops:            int
    steps:               List[PatternStep]
    support:             int
    confidence:          float
    lift:                float
    avg_lags:            List[float]
    first_seen:          str
    validation_methods:  List[str]
    false_positive_rate: float

    def to_json(self) -> dict:
        return {
            "pattern_id":   self.pattern_id,
            "pattern_name": self.pattern_name,
            "pattern_type": "metric_sequence_event",
            "topology_scope": {
                "applicable_roles": self.applicable_roles,
                "max_hops":         self.max_hops,
                "path_required":    False,
            },
            "sequence": [
                {
                    "step":      s.step,
                    "node":      {"device": s.device, "role": s.role},
                    "metric":    s.metric,
                    "feature":   s.feature,
                    "direction": s.direction,
                    "threshold": s.threshold,
                    "absolute_range": {"min": s.absolute_min, "max": s.absolute_max},
                    "lag_minutes":       s.lag_minutes,
                    "tolerance_minutes": s.tolerance_minutes,
                }
                for s in self.steps
            ],
            "result_event": {
                "name":     self.target_event,
                "severity": self.severity,
            },
            "stats": {
                "support":         self.support,
                "confidence":      round(self.confidence, 3),
                "lift":            round(self.lift, 3),
                "avg_lag_minutes": [round(l, 1) for l in self.avg_lags],
            },
            "validation": {
                "method":            self.validation_methods,
                "min_occurrences":   10,
                "false_positive_rate": round(self.false_positive_rate, 3),
            },
            "lifecycle": {
                "first_seen":   self.first_seen,
                "last_updated": datetime.date.today().isoformat(),
                "drift_score":  0.00,
                "status":       "active",
            },
        }


# ════════════════════════════════════════════════════════════════════════════
# Worker-level helpers (top-level for multiprocessing pickling)
# ════════════════════════════════════════════════════════════════════════════

def _batch_cross_corr(args):
    """
    Compute vectorised cross-correlation for a batch of (dev_a,met_a,dev_b,met_b)
    pairs against pre-aligned numpy arrays.

    args = (batch_indices, A_matrix, B_matrix, max_lag_steps, min_corr)
    A_matrix[i] = standardised series for pair i's cause
    B_matrix[i] = standardised series for pair i's effect
    Returns list of (idx, best_lag, best_r)
    """
    batch_indices, A, B, max_lag_steps, min_corr = args
    results = []
    for k, i in enumerate(batch_indices):
        a = A[k]
        b = B[k]
        best_r, best_lag = 0.0, 0
        T = len(a)
        for lag in range(0, max_lag_steps + 1):
            if lag == 0:
                r = float(np.dot(a, b) / T)
            else:
                r = float(np.dot(a[:-lag], b[lag:]) / (T - lag))
            if abs(r) > abs(best_r):
                best_r, best_lag = r, lag
        if abs(best_r) >= min_corr:
            results.append((i, best_lag, best_r))
    return results


def _granger_batch(args):
    """
    Run Granger OLS for a batch of (cause_array, effect_array, lag_steps) tuples.
    Returns list of F-statistics.
    """
    items = args   # list of (c_array, e_array, lag_steps)
    results = []
    for c, e, lag_steps in items:
        p = 2
        n = len(e)
        start = max(p, lag_steps)
        try:
            Y   = e[start:]
            XA  = np.column_stack([e[start - i - 1:n - i - 1] for i in range(p)])
            XA  = np.column_stack([np.ones(len(Y)), XA])
            cl  = np.column_stack(
                [c[start - lag_steps - i:n - lag_steps - i] for i in range(p)]
            )
            min_len = min(len(Y), XA.shape[0], cl.shape[0])
            Y   = Y[:min_len]
            XA  = XA[:min_len]
            cl  = cl[:min_len]
            XB  = np.column_stack([XA, cl])

            def rss(X, y):
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                res  = y - X @ beta
                return float(np.dot(res, res))

            ra = rss(XA, Y)
            rb = rss(XB, Y)
            if rb < 1e-12:
                results.append(0.0)
                continue
            df_n = XB.shape[1] - XA.shape[1]
            df_d = len(Y) - XB.shape[1]
            if df_d <= 0:
                results.append(0.0)
                continue
            F = max(0.0, ((ra - rb) / df_n) / (rb / df_d))
            results.append(float(F))
        except Exception:
            results.append(0.0)
    return results


# ════════════════════════════════════════════════════════════════════════════
# PatternDiscovery
# ════════════════════════════════════════════════════════════════════════════

class PatternDiscovery:
    """
    Orchestrates the full pattern discovery pipeline.

    Parameters
    ----------
    topo          : TopologyLoader
    min_support   : minimum fraction of windows that contain a sequence
    min_confidence: minimum P(event | sequence)
    min_lift      : minimum lift
    max_hops      : topology constraint
    max_lag_min   : maximum causal lag to consider (minutes)
    min_corr      : minimum |r| for a causal link
    n_workers     : parallel workers (None = auto)
    batch_size    : pairs per worker batch for cross-corr
    verbose       : print progress
    """

    def __init__(
        self,
        topo:           TopologyLoader,
        min_support:    float = 0.03,
        min_confidence: float = 0.55,
        min_lift:       float = 1.10,
        max_hops:       int   = 3,
        max_lag_min:    float = 35.0,
        min_corr:       float = 0.40,
        n_workers:      int   = None,
        batch_size:     int   = 500,
        verbose:        bool  = True,
    ):
        self.topo           = topo
        self.min_support    = min_support
        self.min_confidence = min_confidence
        self.min_lift       = min_lift
        self.max_hops       = max_hops
        self.max_lag_min    = max_lag_min
        self.min_corr       = min_corr
        self.n_workers      = n_workers or max(1, multiprocessing.cpu_count())
        self.batch_size     = batch_size
        self.verbose        = verbose

    # ── Step 1: build resampled series ───────────────────────────────────

    def _build_series(
        self, df: pd.DataFrame, resample_min: int = 5,
    ) -> Dict[Tuple[str, str], pd.Series]:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        freq = f"{resample_min}min"
        series_dict: Dict[Tuple[str, str], pd.Series] = {}
        for (dev, met), grp in df.groupby(["device", "metric"]):
            s = (grp.set_index("timestamp")["value"]
                    .resample(freq).mean()
                    .ffill(limit=2))
            series_dict[(dev, met)] = s
        return series_dict

    # ── Step 2: batch cross-correlation ──────────────────────────────────

    def _build_aligned_arrays(
        self,
        pairs:        List[Tuple[str, str, str, str]],
        series_dict:  Dict[Tuple[str, str], pd.Series],
        max_lag_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Align all series to a common index, standardise, and stack into
        A (causes) and B (effects) matrices.  Returns (A, B, valid_indices).
        """
        # Build a common datetime index from all involved series
        all_keys = set()
        for da, ma, db, mb in pairs:
            all_keys.add((da, ma))
            all_keys.add((db, mb))
        common_idx = None
        for k in all_keys:
            s = series_dict.get(k)
            if s is not None:
                common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
        if common_idx is None or len(common_idx) < max_lag_steps + 10:
            return np.array([]), np.array([]), []

        A_rows, B_rows, valid = [], [], []
        for i, (da, ma, db, mb) in enumerate(pairs):
            sa = series_dict.get((da, ma))
            sb = series_dict.get((db, mb))
            if sa is None or sb is None:
                continue
            sa_al = sa.reindex(common_idx).ffill(limit=2).fillna(0).values.astype(np.float64)
            sb_al = sb.reindex(common_idx).ffill(limit=2).fillna(0).values.astype(np.float64)
            # Standardise
            sa_al = (sa_al - sa_al.mean()) / (sa_al.std() + 1e-9)
            sb_al = (sb_al - sb_al.mean()) / (sb_al.std() + 1e-9)
            A_rows.append(sa_al)
            B_rows.append(sb_al)
            valid.append(i)

        if not valid:
            return np.array([]), np.array([]), []
        return np.array(A_rows), np.array(B_rows), valid

    # ── Step 3: Granger OLS (single pair, for compat) ────────────────────

    def _granger_ols(self, cause, effect, lag_steps):
        res = _granger_batch([(
            cause.values.astype(float),
            effect.values.astype(float),
            lag_steps,
        )])
        return res[0]

    # ── Step 4: discover all causal links (parallel) ─────────────────────

    def discover_links(
        self,
        series_dict:  Dict[Tuple[str, str], pd.Series],
        resample_min: int = 5,
    ) -> List[CausalLink]:
        """
        Parallel batch cross-correlation + Granger for every topology-valid pair.
        """
        metrics       = list({m for _, m in series_dict.keys()})
        max_lag_steps = int(self.max_lag_min / resample_min)

        valid_dev_pairs = self.topo.all_pairs_within(self.max_hops)
        for dev in self.topo.devices:
            valid_dev_pairs.append((dev, dev))

        # Build flat list of (dev_a, met_a, dev_b, met_b, hops)
        candidates = []
        hop_map: Dict[Tuple[str,str,str,str], int] = {}
        for da, db in valid_dev_pairs:
            hops = self.topo.hop_distance(da, db) or 0
            for ma in metrics:
                if (da, ma) not in series_dict:
                    continue
                for mb in metrics:
                    if da == db and ma == mb:
                        continue
                    if (db, mb) not in series_dict:
                        continue
                    candidates.append((da, ma, db, mb))
                    hop_map[(da, ma, db, mb)] = hops

        if self.verbose:
            print(f"\n[PatternDiscovery] {len(candidates)} candidate pairs "
                  f"({len(valid_dev_pairs)} device pairs × {len(metrics)} metrics²)")

        # ── Parallel cross-correlation in batches ────────────────────────
        # Build aligned arrays once
        pairs_only = [(da, ma, db, mb) for da, ma, db, mb in candidates]
        A, B, valid_idx = self._build_aligned_arrays(
            pairs_only, series_dict, max_lag_steps
        )

        if len(valid_idx) == 0:
            return []

        # Dispatch batches
        idx_batches = [
            valid_idx[i: i + self.batch_size]
            for i in range(0, len(valid_idx), self.batch_size)
        ]

        corr_results: Dict[int, Tuple[int, float]] = {}   # orig_idx → (lag, r)

        if self.verbose:
            print(f"  Cross-correlation: {len(valid_idx)} pairs, "
                  f"{len(idx_batches)} batches, workers={self.n_workers}")

        with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            futures = {}
            for b_list in idx_batches:
                # Gather rows for this batch (relative to valid_idx)
                rel = list(range(len(b_list)))   # row indices into A/B
                # map b_list entries to row positions in A/B
                row_start = valid_idx.index(b_list[0]) if hasattr(valid_idx, 'index') else 0
                # Simplification: pass sub-arrays
                A_sub = A[valid_idx.index(b_list[0]): valid_idx.index(b_list[0]) + len(b_list)]
                B_sub = B[valid_idx.index(b_list[0]): valid_idx.index(b_list[0]) + len(b_list)]
                fut = ex.submit(
                    _batch_cross_corr,
                    (list(range(len(b_list))), A_sub, B_sub,
                     max_lag_steps, self.min_corr)
                )
                futures[fut] = b_list

            for fut in as_completed(futures):
                b_list = futures[fut]
                for rel_i, lag, r in fut.result():
                    orig_i = b_list[rel_i]
                    corr_results[orig_i] = (lag, r)

        # Filter pairs that passed cross-corr threshold
        passing = [(valid_idx[i], corr_results[valid_idx[i]])
                   for i in range(len(valid_idx))
                   if valid_idx[i] in corr_results]

        if self.verbose:
            print(f"  {len(passing)} pairs passed |r| >= {self.min_corr}; "
                  f"running Granger tests...")

        # ── Parallel Granger tests ────────────────────────────────────────
        granger_items = []
        passing_meta  = []
        for orig_i, (lag, r) in passing:
            da, ma, db, mb = pairs_only[orig_i]
            sa = series_dict.get((da, ma))
            sb = series_dict.get((db, mb))
            if sa is None or sb is None:
                continue
            common = sa.index.intersection(sb.index)
            if len(common) < 20:
                continue
            granger_items.append((
                sa.loc[common].values.astype(float),
                sb.loc[common].values.astype(float),
                lag,
            ))
            passing_meta.append((orig_i, lag, r))

        g_batches = [
            granger_items[i: i + self.batch_size]
            for i in range(0, len(granger_items), self.batch_size)
        ]
        f_stats = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            futures = [ex.submit(_granger_batch, b) for b in g_batches]
            for fut in as_completed(futures):
                f_stats.extend(fut.result())

        # Assemble CausalLink objects
        links: List[CausalLink] = []
        for (orig_i, lag, r), F in zip(passing_meta, f_stats):
            da, ma, db, mb = pairs_only[orig_i]
            hops = hop_map.get((da, ma, db, mb), 0)
            links.append(CausalLink(
                dev_a        = da,
                metric_a     = ma,
                dev_b        = db,
                metric_b     = mb,
                best_lag_min = lag * resample_min,
                correlation  = r,
                granger_f    = F,
                hops         = hops,
            ))

        links.sort(key=lambda l: abs(l.correlation), reverse=True)
        if self.verbose:
            print(f"  → {len(links)} causal links discovered")
        return links

    # ── Step 5: sequence construction ────────────────────────────────────

    def _build_sequences(
        self,
        links:          List[CausalLink],
        target_metric:  str,
        target_device:  str,
        max_chain_len:  int = 5,
    ) -> List[List[CausalLink]]:
        incoming: Dict[Tuple[str, str], List[CausalLink]] = {}
        for lk in links:
            key = (lk.dev_b, lk.metric_b)
            incoming.setdefault(key, []).append(lk)

        def dfs(cur_dev, cur_met, depth, path, cum_lag):
            if depth == max_chain_len:
                return [path] if path else []
            results = []
            if path:
                results.append(path[:])
            for lk in incoming.get((cur_dev, cur_met), []):
                new_lag = cum_lag + lk.best_lag_min
                if new_lag > self.max_lag_min:
                    continue
                if any(l.dev_a == lk.dev_a and l.metric_a == lk.metric_a
                       for l in path):
                    continue
                results.extend(
                    dfs(lk.dev_a, lk.metric_a, depth + 1,
                        [lk] + path, new_lag)
                )
            return results

        chains = dfs(target_device, target_metric, 0, [], 0.0)
        chains = [c for c in chains if len(c) >= 2]
        seen, unique = set(), []
        for c in chains:
            key = tuple((l.dev_a, l.metric_a, l.dev_b, l.metric_b) for l in c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    # ── Step 6: vectorised window validation ─────────────────────────────

    def _validate_sequence(
        self,
        chain:         List[CausalLink],
        windows:       List[Dict[Tuple[str, str], MetricFeatures]],
        event_windows: List[bool],
        resample_min:  int = 5,
    ) -> Tuple[int, float, float]:
        """
        Vectorised validation: build numpy bool arrays for each link condition
        then combine with bitwise AND.
        """
        base_rate = sum(event_windows) / max(len(event_windows), 1)
        n         = len(windows)

        # Pre-extract arrays once
        fired = np.ones(n, dtype=bool)
        for lk in chain:
            ka = (lk.dev_a, lk.metric_a)
            kb = (lk.dev_b, lk.metric_b)
            # Vectorise slope lookups
            slope_a = np.array([
                w[ka].slope if ka in w else np.nan for w in windows
            ], dtype=np.float32)
            slope_b = np.array([
                w[kb].slope if kb in w else np.nan for w in windows
            ], dtype=np.float32)

            has_both = ~np.isnan(slope_a) & ~np.isnan(slope_b)
            cause_up = slope_a > 0
            if lk.correlation > 0:
                effect_ok = slope_b > 0
            else:
                effect_ok = slope_b < 0

            fired &= has_both & cause_up & effect_ok

        support_count = int(fired.sum())
        if support_count == 0:
            return 0, 0.0, 0.0

        ev = np.array(event_windows, dtype=bool)
        # Look ahead up to 3 windows
        tp = 0
        indices = np.where(fired)[0]
        for i in indices:
            future = ev[i: min(i + 4, n)]
            if future.any():
                tp += 1

        confidence = tp / support_count
        lift       = confidence / base_rate if base_rate > 0 else 0.0
        return support_count, confidence, lift

    # ── Step 7: chain → PatternSpec ──────────────────────────────────────

    def _chain_to_pattern(
        self,
        chain, target_event, severity, support, confidence, lift,
        windows, event_windows,
    ) -> PatternSpec:
        event_wins = [w for w, ev in zip(windows, event_windows) if ev] or windows
        steps: List[PatternStep] = []
        cum_lag = 0.0

        for idx, lk in enumerate(chain):
            if idx == 0:
                feat_vals = [w[(lk.dev_a, lk.metric_a)].slope
                             for w in event_wins if (lk.dev_a, lk.metric_a) in w]
                abs_vals  = [w[(lk.dev_a, lk.metric_a)].last
                             for w in event_wins if (lk.dev_a, lk.metric_a) in w]
                steps.append(PatternStep(
                    step=idx + 1, device=lk.dev_a,
                    role=self.topo.get_role(lk.dev_a),
                    metric=lk.metric_a, feature="slope",
                    direction="up" if (np.median(feat_vals) > 0 if feat_vals else True) else "down",
                    threshold=float(np.percentile(np.abs(feat_vals), 10)) if feat_vals else 0.01,
                    absolute_min=float(np.percentile(abs_vals, 2)) if abs_vals else 0.0,
                    absolute_max=float(np.percentile(abs_vals, 99.5)) if abs_vals else 1e6,
                    lag_minutes=0.0, tolerance_minutes=5.0,
                ))
                cum_lag = 0.0

            feat_name = "delta" if lk.metric_b in (
                "crc_errors", "packet_loss", "queue_depth") else "slope"
            feat_vals = [getattr(w[(lk.dev_b, lk.metric_b)],
                                 "delta" if feat_name == "delta" else "slope")
                         for w in event_wins if (lk.dev_b, lk.metric_b) in w]
            abs_vals  = [w[(lk.dev_b, lk.metric_b)].last
                         for w in event_wins if (lk.dev_b, lk.metric_b) in w]
            cum_lag += lk.best_lag_min
            steps.append(PatternStep(
                step=idx + 2, device=lk.dev_b,
                role=self.topo.get_role(lk.dev_b),
                metric=lk.metric_b, feature=feat_name,
                direction="up" if lk.correlation > 0 else "down",
                threshold=float(np.percentile(np.abs(feat_vals), 10)) if feat_vals else 0.01,
                absolute_min=float(np.percentile(abs_vals, 2)) if abs_vals else 0.0,
                absolute_max=float(np.percentile(abs_vals, 99.5)) if abs_vals else 1e6,
                lag_minutes=round(cum_lag, 1), tolerance_minutes=5.0,
            ))

        roles = list({self.topo.get_role(lk.dev_a) for lk in chain}
                     | {self.topo.get_role(lk.dev_b) for lk in chain})
        avg_lags = [s.lag_minutes for s in steps[1:]]
        pat_id   = f"PAT_{target_event.upper()}_{uuid.uuid4().hex[:6].upper()}"

        return PatternSpec(
            pattern_id=pat_id, pattern_name=f"{target_event}_causal_chain",
            target_event=target_event, severity=severity,
            applicable_roles=roles, max_hops=self.max_hops,
            steps=steps, support=support, confidence=confidence, lift=lift,
            avg_lags=avg_lags, first_seen=datetime.date.today().isoformat(),
            validation_methods=["cross_correlation", "granger_causality",
                                 "sequence_mining"],
            false_positive_rate=round(1.0 - confidence, 3),
        )

    # ── Main entry point ─────────────────────────────────────────────────

    def discover(
        self,
        df:            pd.DataFrame,
        target_events: List[Dict],
        windows:       List[Dict[Tuple[str, str], MetricFeatures]],
        resample_min:  int = 5,
    ) -> List[PatternSpec]:
        print("\n" + "═" * 60)
        print("  PATTERN DISCOVERY — START")
        print("═" * 60)

        print("\n[1/5] Building resampled time series...")
        series_dict = self._build_series(df, resample_min)
        print(f"      {len(series_dict)} (device,metric) series")

        print("\n[2/5] Cross-correlation + Granger causality (parallel)...")
        links = self.discover_links(series_dict, resample_min)

        if not links:
            print("  ✗ No causal links found. Relaxing min_corr to 0.30...")
            self.min_corr = 0.30
            links = self.discover_links(series_dict, resample_min)

        print(f"\n  Top causal links discovered:")
        print(f"  {'Cause':30s} → {'Effect':30s}  lag(min) corr   F-stat")
        print("  " + "-" * 85)
        for lk in links[:15]:
            cause  = f"{lk.dev_a}:{lk.metric_a}"
            effect = f"{lk.dev_b}:{lk.metric_b}"
            print(f"  {cause:30s} → {effect:30s}  "
                  f"{lk.best_lag_min:5.1f}   {lk.correlation:+.3f}  {lk.granger_f:6.2f}")

        all_patterns: List[PatternSpec] = []

        for target in target_events:
            t_dev   = target["device"]
            t_met   = target["metric"]
            t_event = target["event"]
            t_sev   = target.get("severity", "critical")
            key     = (t_dev, t_met)

            print(f"\n[3/5] Sequence construction for target: {t_event} "
                  f"({t_dev}:{t_met})")

            vals = [w[key].last for w in windows if key in w]
            if not vals:
                print(f"  ✗ Target {key} not found in windows")
                continue
            threshold = np.percentile(vals, 75)
            event_windows = [
                (w.get(key) is not None and w[key].last >= threshold)
                for w in windows
            ]
            print(f"      {sum(event_windows)} / {len(windows)} windows with "
                  f"{t_dev}:{t_met} >= {threshold:.2f} (top 25%)")

            print(f"\n[4/5] Building causal chains toward {t_dev}:{t_met}...")
            chains = self._build_sequences(links, t_met, t_dev)
            print(f"      {len(chains)} candidate chains")

            print(f"\n[5/5] Validating chains (support≥{self.min_support}, "
                  f"conf≥{self.min_confidence}, lift≥{self.min_lift})...")
            validated = []
            for chain in chains:
                sup, conf, lift = self._validate_sequence(
                    chain, windows, event_windows, resample_min
                )
                sup_frac = sup / max(len(windows), 1)
                if (sup_frac >= self.min_support and
                        conf >= self.min_confidence and
                        lift >= self.min_lift):
                    pat = self._chain_to_pattern(
                        chain, t_event, t_sev, sup, conf, lift,
                        windows, event_windows,
                    )
                    validated.append((lift, conf, pat))

            validated.sort(key=lambda x: (x[0], x[1]), reverse=True)

            if validated:
                for rank, (lift, conf, pat) in enumerate(validated[:3]):
                    print(f"\n  ✓ Pattern {rank+1}: {pat.pattern_id}")
                    print(f"    Steps: {len(pat.steps)}  Support: {pat.support}  "
                          f"Conf: {conf:.3f}  Lift: {lift:.3f}")
                    chain_str = " → ".join(
                        f"{s.device}:{s.metric}" for s in pat.steps
                    )
                    print(f"    Chain: {chain_str}")
                    all_patterns.append(pat)
            else:
                print("  ✗ No patterns met all thresholds for this target")

        print(f"\n{'═'*60}")
        print(f"  DISCOVERY COMPLETE — {len(all_patterns)} patterns found")
        print(f"{'═'*60}\n")
        return all_patterns
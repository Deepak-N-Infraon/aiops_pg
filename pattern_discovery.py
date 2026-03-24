"""
pattern_discovery.py
====================
Full pipeline:
  1. Cross-correlation between topology-adjacent (device,metric) pairs
  2. Granger-style causality test (OLS lag regression)
  3. Sequence construction — build causal chains following time order
  4. Statistical validation — support, confidence, lift
  5. Output: list of validated PatternSpec objects → JSON

This module contains ZERO black-box ML.  Every step is deterministic and
explained inline.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from itertools import combinations
import json, uuid, datetime, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from topology_loader  import TopologyLoader
from feature_engine   import MetricFeatures


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CausalLink:
    """A single (device_a, metric_a) → (device_b, metric_b) causal pair."""
    dev_a:       str
    metric_a:    str
    dev_b:       str
    metric_b:    str
    best_lag_min: float   # minutes after dev_a spike that dev_b responds
    correlation:  float   # Pearson r at best lag
    granger_f:    float   # F-statistic from OLS causality test
    hops:         int     # topology distance

    @property
    def key(self) -> str:
        return f"{self.dev_a}:{self.metric_a}→{self.dev_b}:{self.metric_b}"


@dataclass
class PatternStep:
    step:             int
    device:           str
    role:             str
    metric:           str
    feature:          str      # slope | delta | last | mean
    direction:        str      # up | down | flat
    threshold:        float
    absolute_min:     float
    absolute_max:     float
    lag_minutes:      float
    tolerance_minutes: float


@dataclass
class PatternSpec:
    pattern_id:   str
    pattern_name: str
    target_event: str
    severity:     str
    applicable_roles: List[str]
    max_hops:     int
    steps:        List[PatternStep]
    support:      int
    confidence:   float
    lift:          float
    avg_lags:     List[float]
    first_seen:   str
    validation_methods: List[str]
    false_positive_rate: float

    def to_json(self) -> dict:
        return {
            "pattern_id":   self.pattern_id,
            "pattern_name": self.pattern_name,
            "pattern_type": "metric_sequence_event",
            "topology_scope": {
                "applicable_roles": self.applicable_roles,
                "max_hops":        self.max_hops,
                "path_required":   False,
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
                    "lag_minutes":      s.lag_minutes,
                    "tolerance_minutes": s.tolerance_minutes,
                }
                for s in self.steps
            ],
            "result_event": {
                "name":     self.target_event,
                "severity": self.severity,
            },
            "stats": {
                "support":       self.support,
                "confidence":    round(self.confidence, 3),
                "lift":          round(self.lift, 3),
                "avg_lag_minutes": [round(l, 1) for l in self.avg_lags],
            },
            "validation": {
                "method":           self.validation_methods,
                "min_occurrences":  10,
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
# PatternDiscovery engine
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
    verbose       : print progress
    """

    def __init__(
        self,
        topo:           TopologyLoader,
        min_support:    float = 0.10,
        min_confidence: float = 0.55,
        min_lift:       float = 1.20,
        max_hops:       int   = 2,
        max_lag_min:    float = 30.0,
        min_corr:       float = 0.45,
        verbose:        bool  = True,
    ):
        self.topo           = topo
        self.min_support    = min_support
        self.min_confidence = min_confidence
        self.min_lift       = min_lift
        self.max_hops       = max_hops
        self.max_lag_min    = max_lag_min
        self.min_corr       = min_corr
        self.verbose        = verbose

    # ── Step 1: build raw time-series per (device, metric) ───────────────

    def _build_series(
        self,
        df: pd.DataFrame,
        resample_min: int = 5,
    ) -> Dict[Tuple[str, str], pd.Series]:
        """
        Resample raw data to fixed frequency and return one Series per
        (device, metric).  Missing intervals are forward-filled (max 2 steps).
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        series_dict: Dict[Tuple[str, str], pd.Series] = {}
        freq = f"{resample_min}min"
        for (dev, met), grp in df.groupby(["device", "metric"]):
            s = (grp.set_index("timestamp")["value"]
                    .resample(freq).mean()
                    .ffill(limit=2))
            series_dict[(dev, met)] = s
        return series_dict

    # ── Step 2: cross-correlation + best lag ─────────────────────────────

    def _cross_correlation(
        self,
        sa: pd.Series,
        sb: pd.Series,
        max_lag_steps: int,
    ) -> Tuple[int, float]:
        """
        Compute cross-correlation of sb lagged against sa.
        Returns (best_lag_steps, best_r).
        Positive lag means sb follows sa by `lag` steps.
        """
        # Align on common index
        common = sa.index.intersection(sb.index)
        if len(common) < 10:
            return 0, 0.0

        a = sa.loc[common].values.astype(float)
        b = sb.loc[common].values.astype(float)

        # Standardise
        a_std = (a - a.mean()) / (a.std() + 1e-9)
        b_std = (b - b.mean()) / (b.std() + 1e-9)

        best_r, best_lag = 0.0, 0
        for lag in range(0, max_lag_steps + 1):
            if lag == 0:
                r = float(np.corrcoef(a_std, b_std)[0, 1])
            else:
                r = float(np.corrcoef(a_std[:-lag], b_std[lag:])[0, 1])
            if abs(r) > abs(best_r):
                best_r, best_lag = r, lag
        return best_lag, best_r

    # ── Step 3: Granger-style OLS causality ──────────────────────────────

    def _granger_ols(
        self,
        cause: pd.Series,
        effect: pd.Series,
        lag_steps: int,
    ) -> float:
        """
        Simplified Granger causality via OLS.

        Model A (restricted):   effect[t] ~ effect[t-1..p]
        Model B (unrestricted): effect[t] ~ effect[t-1..p] + cause[t-lag..t-lag-p]

        Returns F-statistic (higher = stronger causality).
        We use p=2 lags for the AR baseline.
        """
        common = cause.index.intersection(effect.index)
        if len(common) < 20:
            return 0.0

        c = cause.loc[common].values.astype(float)
        e = effect.loc[common].values.astype(float)
        p = 2   # AR order
        n = len(e)
        start = max(p, lag_steps)

        try:
            # Build design matrix for restricted model (AR only)
            Y = e[start:]
            XA = np.column_stack([e[start - i - 1:n - i - 1] for i in range(p)])
            XA = np.column_stack([np.ones(len(Y)), XA])

            # Build design matrix for unrestricted model (AR + cause lags)
            cause_lag = np.column_stack(
                [c[start - lag_steps - i:n - lag_steps - i] for i in range(p)]
            )
            # Trim to match Y length
            min_len = min(len(Y), XA.shape[0], cause_lag.shape[0])
            Y  = Y[:min_len]
            XA = XA[:min_len]
            cause_lag = cause_lag[:min_len]
            XB = np.column_stack([XA, cause_lag])

            def ols_rss(X, y):
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                res  = y - X @ beta
                return float(np.dot(res, res))

            rss_a = ols_rss(XA, Y)
            rss_b = ols_rss(XB, Y)
            if rss_b < 1e-12:
                return 0.0
            df_num = XB.shape[1] - XA.shape[1]
            df_den = len(Y) - XB.shape[1]
            if df_den <= 0:
                return 0.0
            F = ((rss_a - rss_b) / df_num) / (rss_b / df_den)
            return max(0.0, float(F))
        except Exception:
            return 0.0

    # ── Step 4: discover all causal links ────────────────────────────────

    def discover_links(
        self,
        series_dict: Dict[Tuple[str, str], pd.Series],
        resample_min: int = 5,
    ) -> List[CausalLink]:
        """
        For every topology-valid (dev_a, dev_b) pair within max_hops,
        test every metric pair for causal relationship.
        """
        metrics = list({m for _, m in series_dict.keys()})
        max_lag_steps = int(self.max_lag_min / resample_min)
        links: List[CausalLink] = []

        valid_pairs = self.topo.all_pairs_within(self.max_hops)
        # Also include same-device pairs for intra-device causal patterns
        for dev in self.topo.devices:
            valid_pairs.append((dev, dev))
        if self.verbose:
            print(f"\n[PatternDiscovery] Testing {len(valid_pairs)} device pairs, "
                  f"{len(metrics)} metrics each → "
                  f"up to {len(valid_pairs)*len(metrics)**2} combinations")

        for dev_a, dev_b in valid_pairs:
            hops = self.topo.hop_distance(dev_a, dev_b) or 0
            for met_a in metrics:
                sa = series_dict.get((dev_a, met_a))
                if sa is None:
                    continue
                for met_b in metrics:
                    if dev_a == dev_b and met_a == met_b:
                        continue  # skip self-correlation
                    sb = series_dict.get((dev_b, met_b))
                    if sb is None:
                        continue
                    lag, r = self._cross_correlation(sa, sb, max_lag_steps)
                    if abs(r) < self.min_corr:
                        continue
                    F = self._granger_ols(sa, sb, lag)
                    links.append(CausalLink(
                        dev_a        = dev_a,
                        metric_a     = met_a,
                        dev_b        = dev_b,
                        metric_b     = met_b,
                        best_lag_min = lag * resample_min,
                        correlation  = r,
                        granger_f    = F,
                        hops         = hops,
                    ))

        links.sort(key=lambda l: abs(l.correlation), reverse=True)
        if self.verbose:
            print(f"  → {len(links)} causal links with |r| >= {self.min_corr}")
        return links

    # ── Step 5: sequence construction ────────────────────────────────────

    def _build_sequences(
        self,
        links:   List[CausalLink],
        target_metric: str,
        target_device: str,
        max_chain_len: int = 5,
    ) -> List[List[CausalLink]]:
        """
        Greedily build causal chains that END with (target_device, target_metric).
        A chain is topologically valid (follows edges), time-ordered (lag > 0
        is preferred), and has monotonically increasing lags.
        """
        # Index: (dev_b, met_b) → list of incoming links
        incoming: Dict[Tuple[str, str], List[CausalLink]] = {}
        for lk in links:
            key = (lk.dev_b, lk.metric_b)
            incoming.setdefault(key, []).append(lk)

        def dfs(cur_dev, cur_met, depth, path, cum_lag) -> List[List[CausalLink]]:
            if depth == max_chain_len:
                return [path] if len(path) > 0 else []
            results = []
            if len(path) > 0:
                results.append(path[:])  # emit current partial chain too
            for lk in incoming.get((cur_dev, cur_met), []):
                new_lag = cum_lag + lk.best_lag_min
                if new_lag > self.max_lag_min:
                    continue
                # avoid cycles
                if any(l.dev_a == lk.dev_a and l.metric_a == lk.metric_a
                       for l in path):
                    continue
                results.extend(
                    dfs(lk.dev_a, lk.metric_a, depth + 1,
                        [lk] + path, new_lag)
                )
            return results

        chains = dfs(target_device, target_metric, 0, [], 0.0)
        # Keep chains with >= 2 links
        chains = [c for c in chains if len(c) >= 2]
        # Deduplicate
        seen = set()
        unique = []
        for c in chains:
            key = tuple((l.dev_a, l.metric_a, l.dev_b, l.metric_b) for l in c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    # ── Step 6: validate sequences against windows ────────────────────────

    def _validate_sequence(
        self,
        chain:   List[CausalLink],
        windows: List[Dict[Tuple[str, str], MetricFeatures]],
        event_windows: List[bool],
        resample_min: int = 5,
    ) -> Tuple[int, float, float]:
        """
        Count how often the chain fires (support) and what fraction of those
        windows precede a target event (confidence).

        Returns (support_count, confidence, lift).
        """
        base_rate = sum(event_windows) / max(len(event_windows), 1)
        support_count = 0
        true_positives = 0

        for i, window in enumerate(windows):
            # Check every link in the chain
            fired = True
            for lk in chain:
                fa = window.get((lk.dev_a, lk.metric_a))
                fb = window.get((lk.dev_b, lk.metric_b))
                if fa is None or fb is None:
                    fired = False
                    break
                # The "cause" metric must be trending up (for this demo, slope > 0)
                if fa.slope <= 0:
                    fired = False
                    break
                # The effect must correlate directionally
                if lk.correlation > 0 and fb.slope <= 0:
                    fired = False
                    break
                if lk.correlation < 0 and fb.slope >= 0:
                    fired = False
                    break
            if fired:
                support_count += 1
                # Check event within ANY of the next 1-3 windows
                future_hit = any(
                    event_windows[j]
                    for j in range(i, min(i + 4, len(event_windows)))
                )
                if future_hit:
                    true_positives += 1

        if support_count == 0:
            return 0, 0.0, 0.0

        confidence = true_positives / support_count
        lift       = confidence / base_rate if base_rate > 0 else 0.0
        return support_count, confidence, lift

    # ── Step 7: build PatternSpec from a chain ────────────────────────────

    def _chain_to_pattern(
        self,
        chain:        List[CausalLink],
        target_event: str,
        severity:     str,
        support:      int,
        confidence:   float,
        lift:         float,
        windows:      List[Dict[Tuple[str, str], MetricFeatures]],
    ) -> PatternSpec:
        """Convert a validated CausalLink chain into a PatternSpec."""

        steps: List[PatternStep] = []
        cum_lag = 0.0

        # Walk chain: chain[0] is the root cause, chain[-1] is closest to event
        for idx, lk in enumerate(chain):
            if idx == 0:
                # root cause node
                feat_vals = [w[(lk.dev_a, lk.metric_a)].slope
                             for w in windows
                             if (lk.dev_a, lk.metric_a) in w]
                abs_vals  = [w[(lk.dev_a, lk.metric_a)].last
                             for w in windows
                             if (lk.dev_a, lk.metric_a) in w]
                steps.append(PatternStep(
                    step      = idx + 1,
                    device    = lk.dev_a,
                    role      = self.topo.get_role(lk.dev_a),
                    metric    = lk.metric_a,
                    feature   = "slope",
                    direction = "up" if np.median(feat_vals) > 0 else "down",
                    threshold = float(np.percentile(np.abs(feat_vals), 25)),
                    absolute_min = float(np.percentile(abs_vals, 10)) if abs_vals else 0.0,
                    absolute_max = float(np.percentile(abs_vals, 90)) if abs_vals else 100.0,
                    lag_minutes      = 0.0,
                    tolerance_minutes= 1.0,
                ))
                cum_lag = 0.0

            # Effect node
            feat_name = "delta" if lk.metric_b in ("crc_errors","packet_loss","queue_depth") else "slope"
            feat_vals = [getattr(w[(lk.dev_b, lk.metric_b)], "delta"
                                 if feat_name == "delta" else "slope")
                         for w in windows
                         if (lk.dev_b, lk.metric_b) in w]
            abs_vals  = [w[(lk.dev_b, lk.metric_b)].last
                         for w in windows
                         if (lk.dev_b, lk.metric_b) in w]
            cum_lag += lk.best_lag_min

            steps.append(PatternStep(
                step      = idx + 2,
                device    = lk.dev_b,
                role      = self.topo.get_role(lk.dev_b),
                metric    = lk.metric_b,
                feature   = feat_name,
                direction = "up" if lk.correlation > 0 else "down",
                threshold = float(np.percentile(np.abs(feat_vals), 25)) if feat_vals else 0.1,
                absolute_min = float(np.percentile(abs_vals, 10)) if abs_vals else 0.0,
                absolute_max = float(np.percentile(abs_vals, 90)) if abs_vals else 100.0,
                lag_minutes      = round(cum_lag, 1),
                tolerance_minutes= 2.0,
            ))

        roles = list({self.topo.get_role(lk.dev_a) for lk in chain}
                     | {self.topo.get_role(lk.dev_b) for lk in chain})
        avg_lags = [s.lag_minutes for s in steps[1:]]

        pat_id   = f"PAT_{target_event.upper()}_{uuid.uuid4().hex[:6].upper()}"

        return PatternSpec(
            pattern_id           = pat_id,
            pattern_name         = f"{target_event}_causal_chain",
            target_event         = target_event,
            severity             = severity,
            applicable_roles     = roles,
            max_hops             = self.max_hops,
            steps                = steps,
            support              = support,
            confidence           = confidence,
            lift                 = lift,
            avg_lags             = avg_lags,
            first_seen           = datetime.date.today().isoformat(),
            validation_methods   = ["cross_correlation", "granger_causality", "sequence_mining"],
            false_positive_rate  = round(1.0 - confidence, 3),
        )

    # ── Main entry point ─────────────────────────────────────────────────

    def discover(
        self,
        df:            pd.DataFrame,
        target_events: List[Dict],   # [{"device":"FW1","metric":"latency_ms","event":"HIGH_LATENCY","severity":"critical"}]
        windows:       List[Dict[Tuple[str, str], MetricFeatures]],
        resample_min:  int = 5,
    ) -> List[PatternSpec]:
        """
        Full discovery pipeline. Returns validated PatternSpec objects.
        """
        print("\n" + "═"*60)
        print("  PATTERN DISCOVERY — START")
        print("═"*60)

        # Step 1: build resampled series
        print("\n[1/5] Building resampled time series...")
        series_dict = self._build_series(df, resample_min)
        print(f"      {len(series_dict)} (device,metric) series")

        # Step 2: discover causal links
        print("\n[2/5] Cross-correlation + Granger causality...")
        links = self.discover_links(series_dict, resample_min)

        if not links:
            print("  ✗ No causal links found. Relaxing min_corr to 0.30...")
            self.min_corr = 0.30
            links = self.discover_links(series_dict, resample_min)

        # Print top links
        print(f"\n  Top causal links discovered:")
        print(f"  {'Cause':30s} → {'Effect':30s}  lag(min) corr   F-stat")
        print("  " + "-"*85)
        for lk in links[:15]:
            cause  = f"{lk.dev_a}:{lk.metric_a}"
            effect = f"{lk.dev_b}:{lk.metric_b}"
            print(f"  {cause:30s} → {effect:30s}  "
                  f"{lk.best_lag_min:5.1f}   {lk.correlation:+.3f}  {lk.granger_f:6.2f}")

        # Step 3: build event_windows flags
        all_patterns: List[PatternSpec] = []

        for target in target_events:
            t_dev   = target["device"]
            t_met   = target["metric"]
            t_event = target["event"]
            t_sev   = target.get("severity", "critical")
            key     = (t_dev, t_met)

            print(f"\n[3/5] Sequence construction for target: {t_event} "
                  f"({t_dev}:{t_met})")

            # Mark which windows have the target metric elevated (top 25%)
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

            # Step 4: build and validate sequences
            print(f"\n[4/5] Building causal chains toward {t_dev}:{t_met}...")
            chains = self._build_sequences(
                links,
                target_metric = t_met,
                target_device = t_dev,
            )
            print(f"      {len(chains)} candidate chains")

            # Step 5: validate each chain
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
                        chain, t_event, t_sev, sup, conf, lift, windows
                    )
                    validated.append((lift, conf, pat))

            validated.sort(key=lambda x: (x[0], x[1]), reverse=True)

            if validated:
                # Keep top-3 unique patterns per event type
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

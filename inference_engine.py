"""
inference_engine.py
===================
Real-time, fully explainable pattern matching engine.

For every incoming window of features:
  1. For each active pattern, evaluate each step condition
  2. Compute partial match score
  3. Apply persistence filter (condition must hold for ≥2 windows)
  4. Emit progressive predictions and final alert

NO black-box logic — every decision is printed with its reason.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import datetime
import numpy as np

from feature_engine import MetricFeatures


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    step_num:      int
    device:        str
    metric:        str
    feature:       str
    direction:     str
    threshold:     float
    actual_value:  float
    actual_dir:    str
    abs_min:       float
    abs_max:       float
    actual_last:   float
    lag_ok:        bool
    matched:       bool
    reason:        str          # explanation string


@dataclass
class PatternMatchResult:
    pattern_id:       str
    pattern_name:     str
    target_event:     str
    severity:         str
    total_steps:      int
    matched_steps:    int
    step_results:     List[StepResult]
    pattern_confidence: float
    prediction_score: float    # (matched/total) * confidence
    alert_triggered:  bool
    alert_level:      str      # NONE | WATCH | WARN | CRITICAL
    window_ts:        str


@dataclass
class InferenceState:
    """Per-pattern rolling state for persistence filtering."""
    pattern_id:       str
    consecutive:      int = 0
    last_score:       float = 0.0
    history:          List[float] = field(default_factory=list)
    # Records when each step last fired: {step_num: wall_clock_minutes}
    step_fire_times:  Dict[int, float] = field(default_factory=dict)

# ════════════════════════════════════════════════════════════════════════════
# InferenceEngine
# ════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Parameters
    ----------
    patterns            : list of pattern dicts (from PatternStorage.all_active())
    alert_threshold     : prediction_score threshold to fire alert (default 0.75)
    persistence_windows : how many consecutive windows a partial match must hold
    verbose             : detailed step-by-step output
    """

    def __init__(
        self,
        patterns:             List[dict],
        alert_threshold:      float = 0.75,
        persistence_windows:  int   = 2,
        verbose:              bool  = True,
    ):
        self.patterns            = patterns
        self.alert_threshold     = alert_threshold
        self.persistence_windows = persistence_windows
        self.verbose             = verbose
        # State per pattern (for persistence)
        self._state: Dict[str, InferenceState] = {
            p["pattern_id"]: InferenceState(p["pattern_id"])
            for p in patterns
        }

    # ── Step condition evaluation ─────────────────────────────────────────

    def _eval_step(
        self,
        step_spec: dict,
        features:  Dict[Tuple[str, str], MetricFeatures],
        prev_step_ts: Optional[float],   # minutes since epoch of prev step
        current_ts:   float,
    ) -> StepResult:
        """
        Evaluate a single pattern step against the current feature window.

        Checks:
          1. Device + metric are present in features
          2. Feature value meets threshold in the required direction
          3. Absolute value is within absolute_range
          4. Lag timing is within tolerance
        """
        device    = step_spec["node"]["device"]
        metric    = step_spec["metric"]
        feature   = step_spec["feature"]
        direction = step_spec["direction"]
        threshold = step_spec["threshold"]
        abs_min   = step_spec["absolute_range"]["min"]
        abs_max   = step_spec["absolute_range"]["max"]
        lag_spec  = step_spec["lag_minutes"]
        tol       = step_spec["tolerance_minutes"]
        step_num  = step_spec["step"]

        key  = (device, metric)
        feat = features.get(key)

        if feat is None:
            return StepResult(
                step_num=step_num, device=device, metric=metric,
                feature=feature, direction=direction, threshold=threshold,
                actual_value=0.0, actual_dir="?",
                abs_min=abs_min, abs_max=abs_max, actual_last=0.0,
                lag_ok=False, matched=False,
                reason=f"No data for {device}:{metric}",
            )

        actual_value = feat.value_for(feature)
        actual_dir   = feat.direction
        actual_last  = feat.last

        # Direction + threshold check
        if direction == "up":
            dir_ok  = actual_value >= threshold
            dir_str = f"{feature}={actual_value:.4f} >= {threshold}"
        elif direction == "down":
            dir_ok  = actual_value <= -abs(threshold)
            dir_str = f"{feature}={actual_value:.4f} <= -{threshold}"
        else:
            dir_ok  = abs(actual_value) <= threshold
            dir_str = f"|{feature}|={abs(actual_value):.4f} <= {threshold}"

        # Absolute range check
        range_ok  = abs_min <= actual_last <= abs_max
        range_str = f"{abs_min} <= last={actual_last:.2f} <= {abs_max}"

        # Lag check (only from step 2 onward)
        lag_ok = True
        lag_str = "n/a (step 1)"
        if prev_step_ts is not None and lag_spec > 0:
            elapsed = current_ts - prev_step_ts
            lag_ok  = (lag_spec - tol) <= elapsed <= (lag_spec + tol)
            lag_str = f"elapsed={elapsed:.1f}min  expected={lag_spec}±{tol}min"

        matched = dir_ok and range_ok and lag_ok

        if matched:
            reason = f"✓ {dir_str}  |  range ok: {range_str}  |  lag: {lag_str}"
        else:
            parts = []
            if not dir_ok:
                parts.append(f"✗ direction/threshold: {dir_str}")
            if not range_ok:
                parts.append(f"✗ range: {range_str}")
            if not lag_ok:
                parts.append(f"✗ lag: {lag_str}")
            reason = "  ".join(parts)

        return StepResult(
            step_num=step_num, device=device, metric=metric,
            feature=feature, direction=direction, threshold=threshold,
            actual_value=actual_value, actual_dir=actual_dir,
            abs_min=abs_min, abs_max=abs_max, actual_last=actual_last,
            lag_ok=lag_ok, matched=matched, reason=reason,
        )

    # ── Pattern evaluation ────────────────────────────────────────────────

    def _eval_pattern(
        self,
        pattern:          dict,
        features:         Dict[Tuple[str, str], MetricFeatures],
        window_ts:        str,
        window_time_min:  float,
    ) -> PatternMatchResult:
        """Evaluate all steps using real cross-window lag timing."""
        pid        = pattern["pattern_id"]
        conf       = pattern["stats"]["confidence"]
        steps      = pattern["sequence"]
        total      = len(steps)
        severity   = pattern["result_event"]["severity"]
        event_name = pattern["result_event"]["name"]
        state      = self._state[pid]

        step_results: List[StepResult] = []
        matched = 0

        # Snapshot BEFORE this window so lag checks use prior-window fire times,
        # not times recorded during THIS window's evaluation.
        fire_snap = dict(state.step_fire_times)

        for i, step_spec in enumerate(steps):
            step_num = step_spec["step"]

            # Lag check uses the snapshot (previous-window fire times only)
            if i == 0:
                prev_fire_ts = None
            else:
                prev_step_num = steps[i - 1]["step"]
                prev_fire_ts = fire_snap.get(prev_step_num)

            sr = self._eval_step(
                step_spec    = step_spec,
                features     = features,
                prev_step_ts = prev_fire_ts,
                current_ts   = window_time_min,
            )
            step_results.append(sr)

            if sr.matched:
                matched += 1
                # Only record the FIRST time a step fires — don't overwrite.
                # This preserves the original fire timestamp for future lag checks.
                if step_num not in state.step_fire_times:
                    state.step_fire_times[step_num] = window_time_min

        score         = (matched / total) * conf
        final_step_ok = step_results[-1].matched if step_results else False
        triggered     = score >= self.alert_threshold and final_step_ok

        if triggered:
            level = "CRITICAL"
        elif score >= 0.60:
            level = "WARN"
        elif score >= 0.30:
            level = "WATCH"
        else:
            level = "NONE"

        return PatternMatchResult(
            pattern_id          = pid,
            pattern_name        = pattern["pattern_name"],
            target_event        = event_name,
            severity            = severity,
            total_steps         = total,
            matched_steps       = matched,
            step_results        = step_results,
            pattern_confidence  = conf,
            prediction_score    = round(score, 4),
            alert_triggered     = triggered,
            alert_level         = level,
            window_ts           = window_ts,
        )

    # ── Persistence filter ────────────────────────────────────────────────

    def _apply_persistence(self, result: PatternMatchResult) -> PatternMatchResult:
        """
        Require a condition to hold for ≥persistence_windows consecutive windows
        before raising the alert.  The prediction_score is always emitted;
        only alert_triggered is gated.
        """
        state = self._state[result.pattern_id]
        state.history.append(result.prediction_score)
        state.last_score = result.prediction_score

        if result.alert_level in ("WARN", "CRITICAL"):
            state.consecutive += 1
        else:
            state.consecutive = 0

        if state.consecutive < self.persistence_windows:
            result.alert_triggered = False
            if result.alert_level == "CRITICAL":
                result.alert_level = "WARN"   # downgrade until persisted

        return result

    # ── Main process call (one window) ───────────────────────────────────

    def process_window(
        self,
        features:        Dict[Tuple[str, str], MetricFeatures],
        window_ts:       Optional[str] = None,
        window_time_min: Optional[float] = None,
    ) -> List[PatternMatchResult]:
        """
        Process one feature window against all active patterns.
        window_time_min: wall-clock time in minutes (used for lag tracking).
                         If None, defaults to current Unix time / 60.
        Returns list of PatternMatchResult (one per pattern).
        """
        import time as _time
        if window_ts is None:
            window_ts = datetime.datetime.now().isoformat(timespec="seconds")
        if window_time_min is None:
            window_time_min = _time.time() / 60.0

        results: List[PatternMatchResult] = []

        for pattern in self.patterns:
            r = self._eval_pattern(pattern, features, window_ts, window_time_min)
            r = self._apply_persistence(r)
            results.append(r)

        # Multi-pattern: sort by prediction_score descending
        results.sort(key=lambda r: r.prediction_score, reverse=True)
        return results

    # ── Explainability print ──────────────────────────────────────────────

    def explain(self, results: List[PatternMatchResult], ts_label: str = "") -> None:
        """Print a full step-by-step explanation of all pattern matches."""
        LEVEL_ICONS = {"NONE": "○", "WATCH": "◑", "WARN": "◕", "CRITICAL": "●"}
        bar_chars   = "░▒▓█"

        print(f"\n{'━'*68}")
        print(f"  INFERENCE — {ts_label}")
        print(f"{'━'*68}")

        for r in results:
            icon  = LEVEL_ICONS.get(r.alert_level, "?")
            pct   = int(r.prediction_score * 40)
            bar   = ("█" * pct + "░" * (40 - pct))
            alert_tag = " ⚠ ALERT TRIGGERED" if r.alert_triggered else ""
            print(f"\n  {icon} Pattern: {r.pattern_id}")
            print(f"    Event: {r.target_event} | Level: {r.alert_level}"
                  f"{alert_tag}")
            print(f"    Score : {r.prediction_score:.4f}  [{bar}]  "
                  f"({r.matched_steps}/{r.total_steps} steps × conf={r.pattern_confidence:.2f})")
            print(f"    Steps matched: {r.matched_steps}/{r.total_steps}")
            print()

            for sr in r.step_results:
                tick = "✓" if sr.matched else "✗"
                print(f"      Step {sr.step_num} [{tick}]  "
                      f"{sr.device}:{sr.metric}  "
                      f"feature={sr.feature}  dir={sr.direction}")
                print(f"             {sr.reason}")

            if r.alert_triggered:
                print(f"\n    ┌──────────────────────────────────────────────┐")
                print(f"    │  ⚠  ALERT: {r.target_event:<34} │")
                print(f"    │     Severity: {r.severity:<31} │")
                print(f"    │     Score: {r.prediction_score:<35.4f}│")
                print(f"    └──────────────────────────────────────────────┘")

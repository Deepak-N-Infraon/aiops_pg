"""
main.py
=======
End-to-end demonstration:
  PHASE 1 — Training:  data generation → features → pattern discovery → storage
  PHASE 2 — Inference: simulate 6 progressive polling windows → alerts
  PHASE 3 — Rediscovery: run on fresh data → drift update
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from data_generator    import generate_dataset, get_topology, get_target_events
from topology_loader   import load_topology
from feature_engine    import FeatureEngine, print_feature_table
from pattern_discovery import PatternDiscovery
from pattern_storage   import PatternStorage
from inference_engine  import InferenceEngine
from rediscovery_engine import RediscoveryEngine


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

SEP = "─" * 68


def banner(title: str) -> None:
    print(f"\n{'═'*68}")
    print(f"  {title}")
    print(f"{'═'*68}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TRAINING
# ════════════════════════════════════════════════════════════════════════════

def phase_training():
    banner("PHASE 1 — TRAINING")

    # 1a. Generate dataset
    print("\n[1/6] Generating synthetic dataset (10 days, 5-min polls)...")
    df   = generate_dataset(n_days=10, freq_min=5, seed=42)
    topo = load_topology(get_topology())

    print(f"      Rows: {len(df):,}  |  "
          f"Devices: {df['device'].nunique()}  |  "
          f"Metrics: {df['metric'].nunique()}")
    print(f"\n      Devices: {sorted(df['device'].unique())}")
    print(f"      Metrics: {sorted(df['metric'].unique())}")

    # 1b. Topology summary
    print(f"\n[2/6] Topology:\n{topo.summary()}")

    # 1c. Feature extraction — show one window in detail
    print(f"\n[3/6] Feature extraction (window=75min, step=5min)...")
    fe      = FeatureEngine(window_minutes=75, step_minutes=5)
    windows = fe.compute_all_windows(df)
    print(f"      Total windows computed: {len(windows)}")

    print("\n  ── Sample features from window #50 ──")
    print_feature_table(windows[50])

    # 1d. Pattern discovery
    print(f"\n[4/6] Running pattern discovery...")
    discoverer = PatternDiscovery(
        topo           = topo,
        min_support    = 0.03,
        min_confidence = 0.55,
        min_lift       = 1.10,
        max_hops       = 3,
        max_lag_min    = 35.0,
        min_corr       = 0.40,
        verbose        = True,
    )
    target_events = get_target_events()
    patterns = discoverer.discover(
        df=df, target_events=target_events, windows=windows
    )

    # 1e. Pattern storage
    print(f"\n[5/6] Saving patterns to storage...")
    store = PatternStorage("patterns/patterns.json")
    store.add_patterns(patterns)
    print(store.summary())

    # 1f. Print full JSON for first discovered pattern
    if patterns:
        print(f"\n[6/6] Full JSON for top pattern ({patterns[0].pattern_id}):")
        print(json.dumps(patterns[0].to_json(), indent=2))

    return df, topo, store, windows


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def phase_inference(df: pd.DataFrame, store: PatternStorage):
    banner("PHASE 2 — INFERENCE (Progressive Pattern Detection)")

    active_patterns = store.all_active()
    if not active_patterns:
        print("  ✗ No active patterns found. Skipping inference.")
        return

    engine = InferenceEngine(
        patterns            = active_patterns,
        alert_threshold     = 0.75,
        persistence_windows = 2,
        verbose             = True,
    )

    fe = FeatureEngine(window_minutes=75, step_minutes=5)

    print(f"\n  Simulating 6 progressive polling windows during a HIGH_LATENCY event...\n")
    print(f"  Pattern used:  {active_patterns[0]['pattern_id']}")
    print(f"  Target event:  {active_patterns[0]['result_event']['name']}")
    print(f"  Total steps:   {len(active_patterns[0]['sequence'])}")

    # Find a HIGH_LATENCY window by looking for elevated FW1:latency_ms
    key = ("FW1", "latency_ms")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_sorted = df.sort_values("timestamp")

    # Find peak timestamp
    peak_mask = (df["device"] == "FW1") & (df["metric"] == "latency_ms")
    peak_ts   = df[peak_mask]["value"].idxmax()
    peak_time = df.loc[peak_ts, "timestamp"]

    # Simulate 6 windows leading UP to the peak
    offsets_min = [-60, -45, -30, -20, -10, 0]   # T-60 to T-0

    for poll_num, offset in enumerate(offsets_min, 1):
        sim_now   = peak_time + pd.Timedelta(minutes=offset)
        sim_start = sim_now - pd.Timedelta(minutes=75)
        mask      = (df_sorted["timestamp"] >= sim_start) & \
                    (df_sorted["timestamp"] <= sim_now)
        window_df = df_sorted[mask]

        if window_df.empty:
            continue

        features = fe.compute_latest_window(window_df)
        ts_label = f"T{offset:+d}min  [{sim_now.strftime('%Y-%m-%d %H:%M')}]"

        results = engine.process_window(features, window_ts=ts_label)

        # Print explanation
        engine.explain(results, ts_label=ts_label)

        # Progressive score table
        print(f"\n  ── Progressive prediction score table ──")
        print(f"  {'Poll':>4}  {'Pattern':35s}  "
              f"{'Matched':>8}  {'Score':>7}  {'Level':>8}")
        print("  " + SEP)
        for r in results[:3]:
            print(f"  {poll_num:>4}  {r.pattern_id:35s}  "
                  f"{r.matched_steps}/{r.total_steps}  "
                  f"{r.prediction_score:7.4f}  {r.alert_level:>8}")

        # Detailed step breakdown
        print(f"\n  ── Why score changed (step-by-step explainability) ──")
        for r in results[:1]:
            prev_matched = getattr(engine, "_prev_matched", 0)
            cur_matched  = r.matched_steps
            if cur_matched > prev_matched:
                delta = cur_matched - prev_matched
                print(f"  Score INCREASED by {delta} new step(s) matching:")
                for sr in r.step_results:
                    if sr.matched:
                        print(f"    ✓ Step {sr.step_num}: {sr.device}:{sr.metric}  "
                              f"({sr.feature}={sr.actual_value:.4f}  dir={sr.actual_dir})")
            elif cur_matched == prev_matched and cur_matched > 0:
                print(f"  Score MAINTAINED — {cur_matched} steps still holding")
            else:
                print(f"  Score LOW — conditions not met yet")
            engine._prev_matched = cur_matched

    print(f"\n{'═'*68}")
    print("  INFERENCE PHASE COMPLETE")
    print(f"{'═'*68}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — REDISCOVERY
# ════════════════════════════════════════════════════════════════════════════

def phase_rediscovery(df: pd.DataFrame, topo, store: PatternStorage):
    banner("PHASE 3 — PERIODIC REDISCOVERY")

    print("\n  Simulating a 24-hour rediscovery cycle on the second half of data...")

    mid = len(df) // 2
    recent_df = df.iloc[mid:].copy()

    engine = RediscoveryEngine(
        topo            = topo,
        storage         = store,
        target_events   = get_target_events(),
        lookback_hours  = 48,
        min_support     = 0.03,
        min_confidence  = 0.60,
        verbose         = True,
    )

    result = engine.run(recent_df)

    print(f"\n  Rediscovery result:")
    print(f"    New patterns   : {result['new_patterns']}")
    print(f"    Updated        : {result['updated']}")
    print(f"    Retired        : {result['retired']}")
    print(f"    Run timestamp  : {result['run_timestamp']}")
    print(f"\n  Final pattern store after rediscovery:")
    print(store.summary())


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*68)
    print("  Network AIOps — Topology-Aware Pattern Discovery & Inference")
    print("  Production-Grade · Fully Explainable · No Black-Box ML")
    print("█"*68)

    df, topo, store, windows = phase_training()
    phase_inference(df, store)
    phase_rediscovery(df, topo, store)

    print("\n✓ Full pipeline complete.")
    print(f"  Patterns saved to: patterns/patterns.json")

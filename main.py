"""
main.py
=======
End-to-end demonstration:
  PHASE 1 — Training:  data generation → features → pattern discovery → storage
  PHASE 2 — Inference: simulate 6 progressive polling windows → alerts
  PHASE 3 — Rediscovery: run on fresh data → drift update

Scaled to 100 devices / 30 days with parallel optimisations.
"""

import os, sys, json, time
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


SEP = "─" * 68


def banner(title: str) -> None:
    print(f"\n{'═'*68}")
    print(f"  {title}")
    print(f"{'═'*68}")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TRAINING
# ════════════════════════════════════════════════════════════════════════════

def phase_training():
    banner("PHASE 1 — TRAINING  [100 devices / 30 days]")
    t_phase = time.time()

    # 1a. Generate dataset
    print("\n[1/6] Generating synthetic dataset (100 devices, 30 days, 5-min polls)...")
    t0 = time.time()
    df   = generate_dataset(n_days=30, freq_min=5, seed=42)
    topo = load_topology(get_topology())
    print(f"      Done in {_elapsed(t0)}")
    print(f"      Rows: {len(df):,}  |  "
          f"Devices: {df['device'].nunique()}  |  "
          f"Metrics: {df['metric'].nunique()}")
    print(f"\n      Device roles:")
    for role in ["router", "dist_switch", "firewall", "access_switch", "edge_switch"]:
        devs = sorted({n['id'] for n in get_topology()['nodes'] if n['role'] == role})
        print(f"        {role:20s}: {len(devs)} devices  "
              f"({', '.join(devs[:4])}{'...' if len(devs)>4 else ''})")

    # 1b. Topology summary
    print(f"\n[2/6] Topology:\n{topo.summary()[:500]}...")   # truncate for 100 devices

    # 1c. Feature extraction — parallel
    print(f"\n[3/6] Feature extraction (window=75min, step=5min, parallel)...")
    t0 = time.time()
    fe = FeatureEngine(
        window_minutes = 75,
        step_minutes   = 5,
        n_workers      = None,   # auto-detect CPUs
        batch_size     = 300,
    )
    windows = fe.compute_all_windows(df)
    print(f"      Done in {_elapsed(t0)} — {len(windows)} windows")

    print("\n  ── Sample features from window #100 ──")
    # Only print a subset of devices to avoid wall of text
    sample = {k: v for k, v in list(windows[100].items())[:12]}
    print_feature_table(sample)

    # 1d. Pattern discovery — parallel
    print(f"\n[4/6] Running pattern discovery (parallel workers)...")
    t0 = time.time()
    discoverer = PatternDiscovery(
        topo           = topo,
        min_support    = 0.03,
        min_confidence = 0.55,
        min_lift       = 1.10,
        max_hops       = 3,
        max_lag_min    = 35.0,
        min_corr       = 0.40,
        n_workers      = None,   # auto
        batch_size     = 400,
        verbose        = True,
    )
    target_events = get_target_events()
    patterns = discoverer.discover(
        df=df, target_events=target_events, windows=windows
    )
    print(f"      Discovery done in {_elapsed(t0)}")

    # 1e. Pattern storage
    print(f"\n[5/6] Saving patterns to storage...")
    store = PatternStorage("patterns/patterns.json")
    store.add_patterns(patterns)
    print(store.summary())

    # 1f. Print full JSON for first pattern
    if patterns:
        print(f"\n[6/6] Full JSON for top pattern ({patterns[0].pattern_id}):")
        print(json.dumps(patterns[0].to_json(), indent=2))

    print(f"\n  Total training time: {_elapsed(t_phase)}")
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

    fe = FeatureEngine(window_minutes=75, step_minutes=5, n_workers=1)

    print(f"\n  Simulating 6 progressive polling windows during a HIGH_LATENCY event...\n")
    print(f"  Pattern used:  {active_patterns[0]['pattern_id']}")
    print(f"  Target event:  {active_patterns[0]['result_event']['name']}")
    print(f"  Total steps:   {len(active_patterns[0]['sequence'])}")

    key = ("FW1", "latency_ms")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_sorted = df.sort_values("timestamp")

    peak_mask = (df["device"] == "FW1") & (df["metric"] == "latency_ms")
    peak_ts   = df[peak_mask]["value"].idxmax()
    peak_time = df.loc[peak_ts, "timestamp"]

    offsets_min = [-60, -45, -30, -20, -10, 0]

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
        results  = engine.process_window(features, window_ts=ts_label)

        engine.explain(results, ts_label=ts_label)

        print(f"\n  ── Progressive prediction score table ──")
        print(f"  {'Poll':>4}  {'Pattern':35s}  "
              f"{'Matched':>8}  {'Score':>7}  {'Level':>8}")
        print("  " + SEP)
        for r in results[:3]:
            print(f"  {poll_num:>4}  {r.pattern_id:35s}  "
                  f"{r.matched_steps}/{r.total_steps}  "
                  f"{r.prediction_score:7.4f}  {r.alert_level:>8}")

        print(f"\n  ── Why score changed ──")
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

    mid       = len(df) // 2
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
    print("  100 Devices · 30 Days · Fully Explainable · No Black-Box ML")
    print("█"*68)

    df, topo, store, windows = phase_training()
    phase_inference(df, store)
    phase_rediscovery(df, topo, store)

    print("\n✓ Full pipeline complete.")
    print(f"  Patterns saved to: patterns/patterns.json")
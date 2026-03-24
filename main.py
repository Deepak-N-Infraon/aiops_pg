"""
main.py  —  Network AIOps Pipeline  (PostgreSQL Edition)
=========================================================
Three-phase pipeline driven entirely by the data already in PostgreSQL.

  PHASE 1 — Training
    Load topology + metrics from DB → feature extraction
    → pattern discovery → save patterns.json

  PHASE 2 — Inference
    Find peak-stress window in DB → simulate 6 progressive polling
    windows → step-by-step explainability → alerts

  PHASE 3 — Rediscovery
    Re-run discovery on recent DB data → update drift scores
    → retire drifted patterns, add new ones

To run:
    python main.py

All configuration lives in the CONFIG block below.
Edit the values there — no command-line arguments are needed.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from db_loader          import DBLoader
from topology_loader    import load_topology
from feature_engine     import FeatureEngine, print_feature_table
from pattern_discovery  import PatternDiscovery
from pattern_storage    import PatternStorage
from inference_engine   import InferenceEngine
from rediscovery_engine import RediscoveryEngine


# ════════════════════════════════════════════════════════════════════════════
# ██  CONFIG  —  edit these values, then run:  python main.py
# ════════════════════════════════════════════════════════════════════════════

# PostgreSQL connection URL
DB_URL = "postgresql://infraon:infinity#123@10.0.4.211:5432/pattern_mining"

# ── Scope (set to None to include everything) ────────────────────────────────
# Restrict training to one device type:  "router" | "switch" | "firewall" | "load_balancer" | None
DEVICE_TYPE = 'router'

# Restrict training to one specific device:  "router-01" | None
DEVICE_ID   = None

# ── Data window ───────────────────────────────────────────────────────────────
# How many days of history to load for training  (60 days are in the DB)
DAYS = 10

# Sub-sample rate — keep every Nth row per (device, metric) to reduce memory.
# 1 = use all rows (required for reliable cross-correlation)
# 3 = keep every 3rd row (use only if you get MemoryError with SAMPLE_EVERY=1)
SAMPLE_EVERY = 1

# ── Discovery thresholds ──────────────────────────────────────────────────────
MIN_SUPPORT    = 0.03   # fraction of windows that must contain the sequence
MIN_CONFIDENCE = 0.55   # P(event | sequence)
MIN_LIFT       = 1.02   # lift above base rate
MAX_HOPS       = 3      # topology hop limit for causal links
MAX_LAG_MIN    = 50.0   # maximum causal lag in minutes (wider for real data)
MIN_CORR       = 0.20   # minimum |Pearson r| — real data is noisier than synthetic

# ── Output ────────────────────────────────────────────────────────────────────
# Directory where patterns.json will be saved
PATTERNS_DIR = "patterns"

# ── Phase switches ────────────────────────────────────────────────────────────
SKIP_INFERENCE   = False   # set True to skip Phase 2
SKIP_REDISCOVERY = False   # set True to skip Phase 3

# ════════════════════════════════════════════════════════════════════════════


SEP = "─" * 68


def banner(title: str) -> None:
    print(f"\n{'═'*68}\n  {title}\n{'═'*68}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TRAINING
# ════════════════════════════════════════════════════════════════════════════

def phase_training(loader: DBLoader) -> tuple:
    banner("PHASE 1 — TRAINING")

    patterns_path = os.path.join(PATTERNS_DIR, "patterns.json")

    # ── Step 1/6: Topology ───────────────────────────────────────────────────
    print("\n[1/6] Loading topology from PostgreSQL...")
    topo_dict = loader.get_topology(
        device_type = DEVICE_TYPE,
        device_id   = DEVICE_ID,
    )
    topo = load_topology(topo_dict)
    print(f"\n[2/6] Topology summary:\n{topo.summary()}")

    # ── Step 3/6: Dataset ────────────────────────────────────────────────────
    print(f"\n[3/6] Loading dataset (last {DAYS} days, sample_every={SAMPLE_EVERY})...")
    df = loader.load_metrics(
        days         = DAYS,
        device_type  = DEVICE_TYPE,
        device_id    = DEVICE_ID,
        sample_every = SAMPLE_EVERY,
    )
    print(f"\n      Rows    : {len(df):,}")
    print(f"      Devices : {df['device'].nunique()}")
    print(f"      Metrics : {df['metric'].nunique()}")
    print(f"      Range   : {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── Step 3b: Target events ────────────────────────────────────────────────
    print("\n      Deriving target events from events table...")
    target_events = loader.get_target_events(
        device_type = DEVICE_TYPE,
        device_id   = DEVICE_ID,
        days        = DAYS,
        top_n       = 3,
    )
    if not target_events:
        print("  ✗ No target events found — check the events table.")
        sys.exit(1)
    print(f"      Target events: {[t['event'] for t in target_events]}")

    # ── Step 4/6: Feature extraction ─────────────────────────────────────────
    print(f"\n[4/6] Feature extraction (window=75 min, step=5 min)...")
    fe      = FeatureEngine(window_minutes=75, step_minutes=5)
    windows = fe.compute_all_windows(df)
    print(f"      Total windows computed: {len(windows)}")

    if not windows:
        print("  ✗ No feature windows — increase DAYS or lower SAMPLE_EVERY.")
        sys.exit(1)

    sample_idx = min(50, len(windows) - 1)
    print(f"\n  ── Sample features (window #{sample_idx}) ──")
    print_feature_table(windows[sample_idx])

    # ── Step 5/6: Pattern discovery ───────────────────────────────────────────
    print(f"\n[5/6] Running pattern discovery...")
    discoverer = PatternDiscovery(
        topo           = topo,
        min_support    = MIN_SUPPORT,
        min_confidence = MIN_CONFIDENCE,
        min_lift       = MIN_LIFT,
        max_hops       = MAX_HOPS,
        max_lag_min    = MAX_LAG_MIN,
        min_corr       = MIN_CORR,
        verbose        = True,
    )
    patterns = discoverer.discover(
        df            = df,
        target_events = target_events,
        windows       = windows,
    )

    # ── Step 6/6: Pattern storage ─────────────────────────────────────────────
    print(f"\n[6/6] Saving patterns to {patterns_path}...")
    store = PatternStorage(patterns_path)
    store.add_patterns(patterns)
    print(store.summary())

    if patterns:
        print(f"\n  Full JSON for top pattern ({patterns[0].pattern_id}):")
        print(json.dumps(patterns[0].to_json(), indent=2))

    return df, topo, store, windows, target_events


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def phase_inference(loader: DBLoader, df: pd.DataFrame, store: PatternStorage):
    banner("PHASE 2 — INFERENCE  (Progressive Pattern Detection)")

    active_patterns = store.all_active()
    if not active_patterns:
        print("  ✗ No active patterns in store. Skipping inference.")
        return

    engine = InferenceEngine(
        patterns            = active_patterns,
        alert_threshold     = 0.75,
        persistence_windows = 2,
        verbose             = True,
    )
    fe = FeatureEngine(window_minutes=75, step_minutes=5)

    top_pat    = active_patterns[0]
    target_dev = top_pat["sequence"][-1]["node"]["device"]
    target_met = top_pat["sequence"][-1]["metric"]

    print(f"\n  Patterns loaded : {len(active_patterns)}")
    print(f"  Top pattern     : {top_pat['pattern_id']}")
    print(f"  Target event    : {top_pat['result_event']['name']}")
    print(f"  Total steps     : {len(top_pat['sequence'])}")

    # Locate peak-stress timestamp in the loaded data
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    peak_mask = (df["device"] == target_dev) & (df["metric"] == target_met)
    if not peak_mask.any():
        peak_idx = df["value"].idxmax()
    else:
        peak_idx = df.loc[peak_mask, "value"].idxmax()

    peak_time = df.loc[peak_idx, "timestamp"]
    print(f"\n  Peak timestamp  : {peak_time}  ({target_dev}:{target_met})")
    print(f"\n  Simulating 6 progressive windows  T-60 → T+0 ...\n")

    df_sorted   = df.sort_values("timestamp")
    offsets_min = [-60, -45, -30, -20, -10, 0]

    for poll_num, offset in enumerate(offsets_min, 1):
        sim_now   = peak_time + pd.Timedelta(minutes=offset)
        sim_start = sim_now  - pd.Timedelta(minutes=75)
        mask      = (df_sorted["timestamp"] >= sim_start) & \
                    (df_sorted["timestamp"] <= sim_now)
        window_df = df_sorted[mask]

        if window_df.empty:
            print(f"  Poll {poll_num}: no data in window [{sim_start} – {sim_now}]")
            continue

        features = fe.compute_latest_window(window_df)
        ts_label = f"T{offset:+d}min  [{sim_now.strftime('%Y-%m-%d %H:%M')}]"
        results  = engine.process_window(features, window_ts=ts_label)

        engine.explain(results, ts_label=ts_label)

        print(f"\n  ── Progressive score table ──")
        print(f"  {'Poll':>4}  {'Pattern':40s}  {'Steps':>7}  {'Score':>7}  {'Level':>8}")
        print("  " + SEP)
        for r in results[:3]:
            print(f"  {poll_num:>4}  {r.pattern_id:40s}  "
                  f"{r.matched_steps}/{r.total_steps}        "
                  f"{r.prediction_score:7.4f}  {r.alert_level:>8}")

        print(f"\n  ── Step-by-step score change ──")
        for r in results[:1]:
            prev = getattr(engine, "_prev_matched", 0)
            cur  = r.matched_steps
            if cur > prev:
                print(f"  Score INCREASED by {cur - prev} new step(s):")
                for sr in r.step_results:
                    if sr.matched:
                        print(f"    ✓ Step {sr.step_num}: {sr.device}:{sr.metric}  "
                              f"({sr.feature}={sr.actual_value:.4f}  dir={sr.actual_dir})")
            elif cur == prev and cur > 0:
                print(f"  Score MAINTAINED — {cur} step(s) still holding")
            else:
                print("  Score LOW — conditions not yet met")
            engine._prev_matched = cur

    print(f"\n{'═'*68}\n  INFERENCE COMPLETE\n{'═'*68}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — REDISCOVERY
# ════════════════════════════════════════════════════════════════════════════

def phase_rediscovery(loader: DBLoader, df: pd.DataFrame, topo,
                      store: PatternStorage, target_events: list):
    banner("PHASE 3 — PERIODIC REDISCOVERY")
    print("\n  Re-running discovery on the second half of the loaded dataset...")

    recent_df = df.iloc[len(df) // 2:].copy()
    engine = RediscoveryEngine(
        topo           = topo,
        storage        = store,
        target_events  = target_events,
        lookback_hours = 48,
        min_support    = 0.05,
        min_confidence = 0.55,
        verbose        = True,
    )
    result = engine.run(recent_df)

    print(f"\n  New patterns  : {result['new_patterns']}")
    print(f"  Updated       : {result['updated']}")
    print(f"  Retired       : {result['retired']}")
    print(f"  Run timestamp : {result['run_timestamp']}")
    print(f"\n  Final pattern store:\n{store.summary()}")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 68)
    print("  Network AIOps — Topology-Aware Pattern Discovery & Inference")
    print("  PostgreSQL Edition  ·  Fully Explainable  ·  No Black-Box ML")
    print("█" * 68)
    print(f"\n  DB              : {DB_URL}")
    print(f"  Device type     : {DEVICE_TYPE  or '(all)'}")
    print(f"  Device id       : {DEVICE_ID    or '(all)'}")
    print(f"  History         : {DAYS} days")
    print(f"  Sample every    : {SAMPLE_EVERY} (keep every {SAMPLE_EVERY}th row)")
    print(f"  Patterns dir    : {PATTERNS_DIR}/")
    print(f"  Skip inference  : {SKIP_INFERENCE}")
    print(f"  Skip rediscovery: {SKIP_REDISCOVERY}")

    print("\n  Connecting to PostgreSQL...")
    try:
        loader = DBLoader(DB_URL)
        loader._connect()          # test the connection immediately
        print("  ✓ Connected")
    except Exception as exc:
        print(f"  ✗ Connection failed: {exc}")
        sys.exit(1)

    os.makedirs(PATTERNS_DIR, exist_ok=True)

    df, topo, store, windows, target_events = phase_training(loader)

    if not SKIP_INFERENCE:
        phase_inference(loader, df, store)

    if not SKIP_REDISCOVERY:
        phase_rediscovery(loader, df, topo, store, target_events)

    loader.close()
    print(f"\n✓ Pipeline complete.  Patterns saved → {PATTERNS_DIR}/patterns.json")
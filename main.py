"""
main.py
=======
End-to-end pipeline:
  PHASE 1 — Training:    data generation → features → pattern discovery → storage
  PHASE 2 — Inference:   6 progressive polling windows → alerts
  PHASE 3 — Rediscovery: fresh data → drift update

All scale and tuning parameters live in config.py.
To change the number of devices or days, edit config.py — no other file
needs to be touched.
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# ── Config first ─────────────────────────────────────────────────────────────
from config import get_config, print_config

CFG = get_config()   # single resolved config dict for the entire run

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
    banner(f"PHASE 1 — TRAINING  "
           f"[{CFG['n_devices']} devices / {CFG['n_days']} days]")
    t_phase = time.time()

    # ── 1a. Generate dataset ──────────────────────────────────────────────
    print(f"\n[1/6] Generating dataset  "
          f"({CFG['n_devices']} devices, {CFG['n_days']} days, "
          f"{CFG['poll_freq_min']}-min polls)...")
    t0   = time.time()
    topo_dict = get_topology(CFG["n_devices"])
    topo      = load_topology(topo_dict)
    df        = generate_dataset()   # reads all params from config internally

    print(f"      Done in {_elapsed(t0)}")
    print(f"      Rows    : {len(df):,}")
    print(f"      Devices : {df['device'].nunique()}")
    print(f"      Metrics : {df['metric'].nunique()}")

    # Role breakdown
    role_map = {n["id"]: n["role"] for n in topo_dict["nodes"]}
    by_role: dict = {}
    for dev, role in role_map.items():
        by_role.setdefault(role, []).append(dev)
    print(f"\n      Device roles:")
    for role, devs in by_role.items():
        sample = ", ".join(devs[:4]) + ("..." if len(devs) > 4 else "")
        print(f"        {role:20s}: {len(devs):3d}  ({sample})")

    # ── 1b. Topology summary ──────────────────────────────────────────────
    print(f"\n[2/6] Topology:")
    summary = topo.summary()
    # Truncate to first 8 lines for large topologies
    lines = summary.splitlines()
    for ln in lines[:min(8, len(lines))]:
        print(f"  {ln}")
    if len(lines) > 8:
        print(f"  ... ({len(lines)-8} more devices)")

    # ── 1c. Feature extraction ────────────────────────────────────────────
    print(f"\n[3/6] Feature extraction  "
          f"(window={CFG['window_minutes']}min, "
          f"step={CFG['step_minutes']}min, parallel)...")
    t0 = time.time()
    fe = FeatureEngine(
        window_minutes = CFG["window_minutes"],
        step_minutes   = CFG["step_minutes"],
        n_workers      = CFG["n_workers"],
        batch_size     = CFG["batch_size"],
    )
    windows = fe.compute_all_windows(df)
    print(f"      Done in {_elapsed(t0)} — {len(windows):,} windows")

    print("\n  ── Sample features (window #100, first 12 series) ──")
    sample = {k: v for k, v in list(windows[min(100, len(windows)-1)].items())[:12]}
    print_feature_table(sample)

    # ── 1d. Pattern discovery ─────────────────────────────────────────────
    print(f"\n[4/6] Running pattern discovery...")
    t0 = time.time()
    discoverer = PatternDiscovery(
        topo           = topo,
        min_support    = CFG["min_support"],
        min_confidence = CFG["min_confidence"],
        min_lift       = CFG["min_lift"],
        max_hops       = CFG["max_hops"],
        max_lag_min    = CFG["max_lag_min"],
        min_corr       = CFG["min_corr"],
        n_workers      = CFG["n_workers"],
        batch_size     = CFG["batch_size"],
        verbose        = True,
    )
    target_events = get_target_events(topo_dict, CFG["n_devices"])
    patterns      = discoverer.discover(
        df=df, target_events=target_events, windows=windows
    )
    print(f"      Discovery done in {_elapsed(t0)}")

    # ── 1e. Pattern storage ───────────────────────────────────────────────
    print(f"\n[5/6] Saving patterns → {CFG['pattern_file']}")
    store = PatternStorage(CFG["pattern_file"])
    store.add_patterns(patterns)
    print(store.summary())

    # ── 1f. Print first pattern JSON ──────────────────────────────────────
    if patterns:
        print(f"\n[6/6] Full JSON — top pattern ({patterns[0].pattern_id}):")
        print(json.dumps(patterns[0].to_json(), indent=2))

    print(f"\n  ✓ Training complete in {_elapsed(t_phase)}")
    return df, topo, topo_dict, store, windows


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def phase_inference(df: pd.DataFrame, topo_dict: dict, store: PatternStorage):
    banner("PHASE 2 — INFERENCE (Progressive Pattern Detection)")

    active_patterns = store.all_active()
    if not active_patterns:
        print("  ✗ No active patterns — skipping inference.")
        return

    engine = InferenceEngine(
        patterns            = active_patterns,
        alert_threshold     = CFG["alert_threshold"],
        persistence_windows = CFG["persistence_windows"],
        verbose             = True,
    )
    fe = FeatureEngine(
        window_minutes = CFG["window_minutes"],
        step_minutes   = CFG["step_minutes"],
        n_workers      = 1,   # single-process for online inference
    )

    # Pick the first HIGH_LATENCY target device that exists in the data
    target_events = get_target_events(topo_dict)
    hl_event      = next((e for e in target_events if e["event"] == "HIGH_LATENCY"), None)
    if hl_event is None:
        print("  ✗ No HIGH_LATENCY target — skipping inference.")
        return

    t_dev, t_met = hl_event["device"], hl_event["metric"]
    print(f"\n  Target device : {t_dev}:{t_met}")
    print(f"  Pattern used  : {active_patterns[0]['pattern_id']}")
    print(f"  Total steps   : {len(active_patterns[0]['sequence'])}")
    print(f"\n  Simulating 6 progressive windows toward a HIGH_LATENCY peak...\n")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_sorted       = df.sort_values("timestamp")
    peak_mask       = (df["device"] == t_dev) & (df["metric"] == t_met)
    peak_ts         = df[peak_mask]["value"].idxmax()
    peak_time       = df.loc[peak_ts, "timestamp"]

    offsets_min = [-60, -45, -30, -20, -10, 0]

    for poll_num, offset in enumerate(offsets_min, 1):
        sim_now   = peak_time + pd.Timedelta(minutes=offset)
        sim_start = sim_now   - pd.Timedelta(minutes=CFG["window_minutes"])
        mask      = (df_sorted["timestamp"] >= sim_start) & \
                    (df_sorted["timestamp"] <= sim_now)
        window_df = df_sorted[mask]
        if window_df.empty:
            continue

        features = fe.compute_latest_window(window_df)
        ts_label = f"T{offset:+d}min  [{sim_now.strftime('%Y-%m-%d %H:%M')}]"
        results  = engine.process_window(features, window_ts=ts_label)

        engine.explain(results, ts_label=ts_label)

        print(f"\n  ── Progressive score table (poll {poll_num}/6) ──")
        print(f"  {'Poll':>4}  {'Pattern':35s}  "
              f"{'Matched':>8}  {'Score':>7}  {'Level':>8}")
        print("  " + SEP)
        for r in results[:3]:
            print(f"  {poll_num:>4}  {r.pattern_id:35s}  "
                  f"{r.matched_steps}/{r.total_steps}  "
                  f"{r.prediction_score:7.4f}  {r.alert_level:>8}")

        print(f"\n  ── Score explanation ──")
        for r in results[:1]:
            prev = getattr(engine, "_prev_matched", 0)
            cur  = r.matched_steps
            if cur > prev:
                print(f"  Score INCREASED (+{cur - prev} step(s)):")
                for sr in r.step_results:
                    if sr.matched:
                        print(f"    ✓ Step {sr.step_num}: {sr.device}:{sr.metric}  "
                              f"{sr.feature}={sr.actual_value:.4f}  dir={sr.actual_dir}")
            elif cur == prev and cur > 0:
                print(f"  Score MAINTAINED — {cur} steps still holding")
            else:
                print("  Score LOW — conditions not yet met")
            engine._prev_matched = cur

    print(f"\n{'═'*68}\n  INFERENCE PHASE COMPLETE\n{'═'*68}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — REDISCOVERY
# ════════════════════════════════════════════════════════════════════════════

def phase_rediscovery(df: pd.DataFrame, topo, topo_dict: dict,
                      store: PatternStorage):
    banner("PHASE 3 — PERIODIC REDISCOVERY")
    print("\n  Simulating a rediscovery cycle on the second half of data...")

    mid       = len(df) // 2
    recent_df = df.iloc[mid:].copy()

    engine = RediscoveryEngine(
        topo            = topo,
        storage         = store,
        target_events   = get_target_events(topo_dict),
        lookback_hours  = CFG["rediscovery_lookback_hours"],
        min_support     = CFG["min_support"],
        min_confidence  = CFG["rediscovery_min_confidence"],
        verbose         = True,
    )
    result = engine.run(recent_df)

    print(f"\n  Rediscovery result:")
    print(f"    New patterns : {result['new_patterns']}")
    print(f"    Updated      : {result['updated']}")
    print(f"    Retired      : {result['retired']}")
    print(f"    Timestamp    : {result['run_timestamp']}")
    print(f"\n  Final pattern store:")
    print(store.summary())


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 68)
    print("  Network AIOps — Topology-Aware Pattern Discovery & Inference")
    print("  Fully Explainable · No Black-Box ML · Config-Driven Scale")
    print("█" * 68)

    # Print the active configuration so the user sees exactly what will run
    print_config(CFG)

    df, topo, topo_dict, store, windows = phase_training()
    phase_inference(df, topo_dict, store)
    phase_rediscovery(df, topo, topo_dict, store)

    print("\n✓ Full pipeline complete.")
    print(f"  Patterns saved to: {CFG['pattern_file']}")
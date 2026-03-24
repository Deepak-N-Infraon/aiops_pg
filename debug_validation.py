"""
Debug script: instrument the validation step to see actual support/confidence/lift
for each candidate chain, and understand WHY they fail thresholds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from data_generator    import generate_dataset, get_topology, get_target_events
from topology_loader   import load_topology
from feature_engine    import FeatureEngine
from pattern_discovery import PatternDiscovery

# Generate data
df   = generate_dataset(n_days=10, freq_min=5, seed=42)
topo = load_topology(get_topology())
fe   = FeatureEngine(window_minutes=75, step_minutes=5)
windows = fe.compute_all_windows(df)

# Build discoverer
disc = PatternDiscovery(
    topo=topo, min_support=0.03, min_confidence=0.65,
    min_lift=1.10, max_hops=3, max_lag_min=35.0,
    min_corr=0.40, verbose=False
)

# Step 1-2: build series + links
series_dict = disc._build_series(df, 5)
links = disc.discover_links(series_dict, 5)

print(f"Total causal links: {len(links)}")
print(f"Total windows: {len(windows)}")

# Focus on HIGH_LATENCY target
t_dev, t_met = "FW1", "latency_ms"
key = (t_dev, t_met)

# Build event windows
vals = [w[key].last for w in windows if key in w]
threshold = np.percentile(vals, 75)
event_windows = [(w.get(key) is not None and w[key].last >= threshold) for w in windows]
print(f"\nEvent windows (FW1:latency_ms >= {threshold:.2f}): {sum(event_windows)} / {len(windows)}")

# Build chains
chains = disc._build_sequences(links, target_metric=t_met, target_device=t_dev)
print(f"Candidate chains: {len(chains)}")

# Validate each chain and print stats
print(f"\n{'Chain':60s}  {'Sup':>5}  {'SupFrac':>8}  {'Conf':>6}  {'Lift':>6}")
print("-" * 100)

for chain in chains[:20]:
    chain_str = " → ".join(f"{lk.dev_a}:{lk.metric_a}" for lk in chain)
    chain_str += f" → {chain[-1].dev_b}:{chain[-1].metric_b}"
    
    sup, conf, lift = disc._validate_sequence(chain, windows, event_windows, 5)
    sup_frac = sup / max(len(windows), 1)
    
    print(f"{chain_str:60s}  {sup:5d}  {sup_frac:8.4f}  {conf:6.3f}  {lift:6.3f}")

# Now let's understand WHY support is low - count how many windows
# have each individual link's conditions met
print("\n\n=== Per-link analysis: how often is each link's condition met? ===\n")
for lk in links[:10]:
    count_both_up = 0
    count_a_up = 0
    count_b_up = 0
    for w in windows:
        fa = w.get((lk.dev_a, lk.metric_a))
        fb = w.get((lk.dev_b, lk.metric_b))
        if fa is None or fb is None:
            continue
        if fa.slope > 0:
            count_a_up += 1
        if lk.correlation > 0:
            if fb.slope > 0:
                count_b_up += 1
            if fa.slope > 0 and fb.slope > 0:
                count_both_up += 1
        else:
            if fb.slope < 0:
                count_b_up += 1
            if fa.slope > 0 and fb.slope < 0:
                count_both_up += 1
    
    print(f"  {lk.dev_a}:{lk.metric_a} → {lk.dev_b}:{lk.metric_b}  "
          f"corr={lk.correlation:+.3f}  "
          f"a_up={count_a_up}  b_dir_ok={count_b_up}  both_ok={count_both_up}  "
          f"frac={count_both_up/len(windows):.4f}")

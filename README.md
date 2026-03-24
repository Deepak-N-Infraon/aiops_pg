# Network AIOps — Topology-Aware Pattern Discovery & Inference
## PostgreSQL Edition — Step-by-Step Guide

---

## What This Project Does

This pipeline mines your **real network polling data** (already loaded into PostgreSQL by the data generator) to automatically discover causal patterns that precede network events, and then uses those patterns to score live traffic for early warnings.

```
PostgreSQL DB
  ├── interface_metrics  (4,492,800 rows — 60 days of interface polls)
  ├── device_metrics     (241,920 rows  — 60 days of device polls)
  ├── events             (213,674 rows  — all causal events)
  ├── devices            (14 devices)
  └── interfaces         (260 interfaces)
          │
          ▼
    Phase 1 — TRAINING
      Load data → extract features → discover causal chains
      → save patterns.json
          │
          ▼
    Phase 2 — INFERENCE
      Find peak-stress window → simulate 6 polling snapshots
      → score each window → step-by-step alert explanation
          │
          ▼
    Phase 3 — REDISCOVERY
      Re-run on recent data → update confidence scores
      → retire drifted patterns, register new ones
```

---

## Project Files

```
aiops_pg/
├── main.py               ← Entry point. Edit CONFIG here, then run.
├── db_loader.py          ← All PostgreSQL queries (replaces data_generator.py)
├── topology_loader.py    ← Builds the network graph from the devices table
├── feature_engine.py     ← Computes 8 stats per (device, metric) per window
├── pattern_discovery.py  ← Cross-correlation + Granger causality + sequence mining
├── pattern_storage.py    ← Saves/loads patterns.json, manages drift scores
├── inference_engine.py   ← Real-time pattern matching with full explainability
├── rediscovery_engine.py ← Periodic re-training and drift management
└── patterns/
    └── patterns.json     ← Created on first run, updated on rediscovery
```

---

## Prerequisites

### 1. Python version

Python 3.9 or higher is required.

```bash
python --version
# Should print Python 3.9.x or higher
```

### 2. Install dependencies

```bash
pip install psycopg2-binary pandas numpy scipy scikit-learn
```

All other imports (`json`, `os`, `sys`, `collections`, `dataclasses`, etc.) are part of the Python standard library.

### 3. Verify database connectivity

Your data is already loaded. Confirm the DB is reachable:

```bash
psql "postgresql://infraon:infinity#123@10.0.4.211:5432/pattern_mining" \
     -c "SELECT COUNT(*) FROM interface_metrics;"
# Should return: 4492800
```

If you don't have `psql` installed:

```bash
python -c "
import psycopg2
conn = psycopg2.connect('postgresql://infraon:infinity#123@10.0.4.211:5432/pattern_mining')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM interface_metrics')
print('interface_metrics rows:', cur.fetchone()[0])
cur.execute('SELECT COUNT(*) FROM events')
print('events rows:', cur.fetchone()[0])
conn.close()
"
```

---

## Step-by-Step: Running the Pipeline

### Step 1 — Open `main.py` and review the CONFIG block

The only file you need to edit is `main.py`. Open it and find the CONFIG block near the top:

```python
# ██  CONFIG  —  edit these values, then run:  python main.py

DB_URL       = "postgresql://infraon:infinity#123@10.0.4.211:5432/pattern_mining"
DEVICE_TYPE  = None       # None = all device types
DEVICE_ID    = None       # None = all devices
DAYS         = 10         # days of history to load
SAMPLE_EVERY = 3          # keep every 3rd row (recommended for 60-day data)
...
```

The DB_URL is already set to your server. You do not need to change anything to do a first run.

---

### Step 2 — Recommended first run (fast, scoped)

For the first run, scope to **routers only** and use **10 days** of data. This completes in a few minutes and confirms the pipeline works end-to-end.

Edit `main.py`:

```python
DEVICE_TYPE  = "router"   # only routers
DAYS         = 10         # last 10 days
SAMPLE_EVERY = 3          # every 3rd row
```

Then run:

```bash
cd aiops_pg
python main.py
```

You will see output like:

```
████████████████████████████████████████████████████████████████████
  Network AIOps — Topology-Aware Pattern Discovery & Inference
  PostgreSQL Edition  ·  Fully Explainable  ·  No Black-Box ML
████████████████████████████████████████████████████████████████████

  DB              : postgresql://infraon:...@10.0.4.211:5432/pattern_mining
  Device type     : router
  History         : 10 days
  ...

  ✓ Connected

══════════════════════════════════════════════════════════════════════
  PHASE 1 — TRAINING
══════════════════════════════════════════════════════════════════════

[1/6] Loading topology from PostgreSQL...
  [DBLoader] topology: 4 nodes, 3 edges

[2/6] Topology summary:
Topology: 4 devices, 3 links
  router-01   role=router   neighbours=[router-02]
  ...

[3/6] Loading dataset (last 10 days, sample_every=3)...
  [DBLoader] metrics : 180,000 rows | 4 devices | 19 metrics | ...

[4/6] Feature extraction (window=75 min, step=5 min)...
      Total windows computed: 2,016

[5/6] Running pattern discovery...
  ...
  ✓ Pattern 1: PAT_HIGH_LATENCY_A3F2B1
    Steps: 3  Support: 142  Conf: 0.724  Lift: 1.38
    Chain: router-01:cpu_pct → router-01:latency_ms → router-01:rx_util_pct

[6/6] Saving patterns to patterns/patterns.json...
```

---

### Step 3 — Full run (all devices, more history)

Once the scoped run works, edit `main.py` for a full run:

```python
DEVICE_TYPE  = None    # all device types
DEVICE_ID    = None    # all devices
DAYS         = 30      # 30 days of history (good balance of speed vs quality)
SAMPLE_EVERY = 3       # keep every 3rd row
```

> **Memory note:** 260 interfaces × 30 days × 19 metrics with `SAMPLE_EVERY=3` loads
> approximately 2–3 million rows into a DataFrame (~400 MB RAM). If you run out of
> memory, increase `SAMPLE_EVERY` to 5 or reduce `DAYS`.

```bash
python main.py
```

A full run with 30 days of data typically takes 5–15 minutes depending on hardware.

---

### Step 4 — Understanding the output

#### Phase 1 output — Pattern discovery

Each discovered pattern looks like:

```
✓ Pattern 1: PAT_HIGH_CPU_A3F2B1
  Steps: 4  Support: 284  Conf: 0.781  Lift: 1.52
  Chain: router-01:rx_util_pct → router-01:cpu_pct → firewall-01:latency_ms → firewall-01:cpu_pct
```

- **Steps** — how many causal hops in the chain
- **Support** — how many feature windows contained this sequence
- **Conf** — fraction of those windows where the target event actually followed
- **Lift** — how much more likely the event is given the pattern vs by chance

The full pattern JSON is printed and also saved to `patterns/patterns.json`.

#### Phase 2 output — Inference simulation

The engine simulates 6 polling windows leading up to the worst metric spike found in your data (T-60min through T+0):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INFERENCE — T-30min  [2026-02-15 14:30]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ◕ Pattern: PAT_HIGH_CPU_A3F2B1
    Event: HIGH_CPU | Level: WARN
    Score : 0.4687  [████████████████░░░░░░░░░░░░░░░░░░░░░░░░]  (2/4 steps × conf=0.78)

      Step 1 [✓]  router-01:rx_util_pct  feature=slope  dir=up
                   ✓ slope=0.0234 >= 0.0050  |  range ok: ...
      Step 2 [✓]  router-01:cpu_pct  feature=slope  dir=up
                   ✓ slope=0.0412 >= 0.0100  |  range ok: ...
      Step 3 [✗]  firewall-01:latency_ms  feature=slope  dir=up
                   ✗ direction/threshold: slope=0.0021 < 0.0050
```

Alert levels:
- `NONE`  — score < 0.30  — no concern
- `WATCH` — score ≥ 0.30  — early signal, monitor
- `WARN`  — score ≥ 0.60  — escalating, prepare response
- `CRITICAL` — score ≥ 0.75 AND persisted for 2+ windows — alert fires

#### Phase 3 output — Rediscovery

```
  Rediscovery result:
    New patterns  : ['PAT_CRC_ERRORS_B7C3D2']
    Updated       : ['PAT_HIGH_CPU_A3F2B1']
    Retired       : []
```

---

### Step 5 — Tuning if no patterns are found

If Phase 1 completes but discovers 0 patterns, relax the thresholds in the CONFIG block:

```python
MIN_SUPPORT    = 0.03   # was 0.05 — lower = easier to qualify
MIN_CONFIDENCE = 0.50   # was 0.60 — lower = easier to qualify
MIN_LIFT       = 1.02   # was 1.05 — lower = easier to qualify
MIN_CORR       = 0.25   # was 0.35 — lower = finds weaker correlations
MAX_LAG_MIN    = 50.0   # was 35.0 — higher = looks further back in time
MAX_HOPS       = 4      # was 3    — higher = allows longer causal chains
```

Also try scoping to one event-heavy device type first:

```python
DEVICE_TYPE = "firewall"   # firewalls tend to have the most events
```

---

### Step 6 — Run only specific phases

To skip inference and rediscovery and just regenerate patterns:

```python
SKIP_INFERENCE   = True
SKIP_REDISCOVERY = True
```

To run only inference against already-discovered patterns (patterns.json already exists):

```python
SKIP_INFERENCE   = False
SKIP_REDISCOVERY = True
```

---

### Step 7 — Re-run after more data is collected

As your live polling script adds new rows to PostgreSQL, re-run the pipeline periodically to update the patterns:

```bash
# Re-run with the most recent 14 days
# (edit DAYS = 14 in main.py first, or use the current value)
python main.py
```

Patterns that are still valid will have their confidence updated (drift score decreases). Patterns whose confidence has dropped will be automatically marked inactive.

---

## Quick Reference

### Config options summary

| Variable | Default | Description |
|---|---|---|
| `DB_URL` | *(your server)* | PostgreSQL connection URL |
| `DEVICE_TYPE` | `None` | Filter to one device type, or `None` for all |
| `DEVICE_ID` | `None` | Filter to one device, or `None` for all |
| `DAYS` | `10` | Days of history to load |
| `SAMPLE_EVERY` | `3` | Sub-sample rate (1=all rows, 3=every 3rd) |
| `MIN_SUPPORT` | `0.05` | Minimum fraction of windows containing the pattern |
| `MIN_CONFIDENCE` | `0.60` | Minimum P(event \| pattern) |
| `MIN_LIFT` | `1.05` | Minimum lift above base rate |
| `MAX_HOPS` | `3` | Topology hop limit for causal links |
| `MAX_LAG_MIN` | `35.0` | Maximum causal lag in minutes |
| `MIN_CORR` | `0.35` | Minimum Pearson \|r\| for a causal link |
| `PATTERNS_DIR` | `"patterns"` | Directory for patterns.json |
| `SKIP_INFERENCE` | `False` | Skip Phase 2 |
| `SKIP_REDISCOVERY` | `False` | Skip Phase 3 |

### Common run commands

```bash
# First run — fast scope test (recommended)
# (DEVICE_TYPE = "router", DAYS = 10 in main.py)
python main.py

# Full run — all devices, 30 days
# (DEVICE_TYPE = None, DAYS = 30 in main.py)
python main.py

# DB connection self-test only
python db_loader.py --db "postgresql://infraon:infinity#123@10.0.4.211:5432/pattern_mining" --days 7

# Verify DB counts directly
psql "postgresql://infraon:infinity#123@10.0.4.211:5432/pattern_mining" \
     -c "SELECT device_type, COUNT(*) FROM devices GROUP BY device_type;"
```

### Expected runtime (approximate)

| Scope | DAYS | SAMPLE_EVERY | Approx. time |
|---|---|---|---|
| One device type (router) | 10 | 3 | 2–4 min |
| One device type (router) | 30 | 3 | 4–8 min |
| All devices | 10 | 3 | 5–10 min |
| All devices | 30 | 3 | 10–20 min |
| All devices | 60 | 5 | 15–30 min |

---

## Troubleshooting

### `Connection failed` / `could not connect to server`

Check network access to `10.0.4.211:5432` from your machine:

```bash
ping 10.0.4.211
telnet 10.0.4.211 5432
```

If the host is unreachable, update `DB_URL` in `main.py` to the correct host/port.

### `No metric data found`

The time range filter may be too narrow. Your data covers:
`2026-01-22 → 2026-03-23`.

Make sure `DAYS` is large enough — try `DAYS = 60` to load all available data.

### `MemoryError` or Python process killed

Reduce memory usage by increasing `SAMPLE_EVERY`:

```python
SAMPLE_EVERY = 5   # keep only every 5th row
```

Or scope to fewer devices:

```python
DEVICE_TYPE = "router"   # 4 devices instead of 14
```

### `0 patterns found` after discovery

Relax thresholds (see Step 5 above). The most effective single change is lowering `MIN_CORR` to `0.25` — this allows the discovery engine to find weaker correlations which are common in real (noisy) network data.

### `ModuleNotFoundError: No module named 'psycopg2'`

```bash
pip install psycopg2-binary
```

### `ModuleNotFoundError: No module named 'sklearn'`

```bash
pip install scikit-learn
```

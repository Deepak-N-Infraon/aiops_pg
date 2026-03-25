"""
config.py
=========
Central configuration for the Network AIOps pipeline.

Edit the values in the USER-TUNABLE SETTINGS section below to control
the scale and behaviour of every phase.  No other file needs to be touched.

Quick-start presets (just change N_DEVICES and N_DAYS):
  ─ Dev / smoke-test  :  N_DEVICES=5,   N_DAYS=10
  ─ Default           :  N_DEVICES=5,   N_DAYS=90
  ─ Medium            :  N_DEVICES=30,  N_DAYS=30
  ─ Full scale        :  N_DEVICES=100, N_DAYS=30
"""

# ════════════════════════════════════════════════════════════════════════════
# ── USER-TUNABLE SETTINGS ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# ── Scale ────────────────────────────────────────────────────────────────
N_DEVICES: int = 5       # Total network devices  (5 – 100)
N_DAYS:    int = 90      # Days of historical data to generate (1 – 365)

# ── Data generation ──────────────────────────────────────────────────────
POLL_FREQ_MIN:  int = 5  # Polling interval in minutes (5 | 10 | 15)
RANDOM_SEED:    int = 42 # Reproducibility seed

# Events injected per causal group (scale up with more days/devices)
# Set to None to auto-scale based on N_DAYS
N_EVENTS_HIGH_LATENCY:  int = None   # HIGH_LATENCY injections per router group
N_EVENTS_IFACE_FLAP:    int = None   # INTERFACE_FLAP injections per switch

# ── Feature extraction ───────────────────────────────────────────────────
WINDOW_MINUTES: int   = 75    # Sliding window width  (45 | 60 | 75 | 90)
STEP_MINUTES:   int   = 5     # Window step size — smaller = more windows
                               # Use 10 or 15 to speed up large datasets

# ── Pattern discovery ────────────────────────────────────────────────────
MIN_SUPPORT:    float = 0.03  # Min fraction of windows containing a sequence
MIN_CONFIDENCE: float = 0.55  # Min P(event | sequence)
MIN_LIFT:       float = 1.10  # Min lift  (1.0 = no better than random)
MAX_HOPS:       int   = 3     # Max topology hops a causal chain may span
MAX_LAG_MIN:    float = 35.0  # Max causal lag in minutes
MIN_CORR:       float = 0.40  # Min |Pearson r| to consider a causal link

# ── Inference ────────────────────────────────────────────────────────────
ALERT_THRESHOLD:      float = 0.75  # prediction_score to fire CRITICAL alert
PERSISTENCE_WINDOWS:  int   = 2     # consecutive windows before alert fires

# ── Rediscovery ──────────────────────────────────────────────────────────
REDISCOVERY_LOOKBACK_HOURS: int   = 48
REDISCOVERY_MIN_CONFIDENCE: float = 0.60

# ── Performance ──────────────────────────────────────────────────────────
N_WORKERS:  int = None   # Parallel workers — None = auto (cpu_count)
BATCH_SIZE: int = 300    # Windows/pairs per worker batch

# ── Storage ──────────────────────────────────────────────────────────────
PATTERN_FILE: str = "patterns/patterns.json"


# ════════════════════════════════════════════════════════════════════════════
# ── DERIVED / AUTO-SCALED VALUES  (do not edit below this line) ──────────
# ════════════════════════════════════════════════════════════════════════════

def _auto_events(n_days: int, base: int, per_10_days: int) -> int:
    """Scale event count linearly with dataset length."""
    return max(base, int(n_days / 10 * per_10_days))


def get_config() -> dict:
    """
    Return the fully-resolved configuration dict.
    Auto-fills None fields and validates ranges.
    """
    n_hl = N_EVENTS_HIGH_LATENCY or _auto_events(N_DAYS, base=5,  per_10_days=6)
    n_if = N_EVENTS_IFACE_FLAP   or _auto_events(N_DAYS, base=4,  per_10_days=5)

    cfg = {
        # Scale
        "n_devices":   N_DEVICES,
        "n_days":       N_DAYS,
        # Data
        "poll_freq_min":           POLL_FREQ_MIN,
        "seed":                    RANDOM_SEED,
        "n_events_high_latency":   n_hl,
        "n_events_iface_flap":     n_if,
        # Features
        "window_minutes": WINDOW_MINUTES,
        "step_minutes":   STEP_MINUTES,
        # Discovery
        "min_support":    MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "min_lift":       MIN_LIFT,
        "max_hops":       MAX_HOPS,
        "max_lag_min":    MAX_LAG_MIN,
        "min_corr":       MIN_CORR,
        # Inference
        "alert_threshold":     ALERT_THRESHOLD,
        "persistence_windows": PERSISTENCE_WINDOWS,
        # Rediscovery
        "rediscovery_lookback_hours": REDISCOVERY_LOOKBACK_HOURS,
        "rediscovery_min_confidence": REDISCOVERY_MIN_CONFIDENCE,
        # Performance
        "n_workers":   N_WORKERS,
        "batch_size":  BATCH_SIZE,
        # Storage
        "pattern_file": PATTERN_FILE,
    }

    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    assert 1 <= cfg["n_devices"] <= 200,  "N_DEVICES must be 1–200"
    assert 1 <= cfg["n_days"]    <= 365,  "N_DAYS must be 1–365"
    assert cfg["poll_freq_min"] in (1, 5, 10, 15, 30), \
        "POLL_FREQ_MIN must be 1/5/10/15/30"
    assert 0 < cfg["min_support"]    <= 1.0
    assert 0 < cfg["min_confidence"] <= 1.0
    assert cfg["min_lift"] >= 1.0
    assert cfg["max_hops"] >= 1


def print_config(cfg: dict) -> None:
    """Pretty-print the active configuration."""
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│              ACTIVE CONFIGURATION                   │")
    print("├─────────────────────────────────────────────────────┤")
    print(f"│  Devices          : {cfg['n_devices']:<32}│")
    print(f"│  Days             : {cfg['n_days']:<32}│")
    print(f"│  Poll frequency   : {cfg['poll_freq_min']} min{'':<28}│")
    est_rows = (cfg['n_devices'] * cfg['n_days'] * 24 * 60 // cfg['poll_freq_min']
                * _avg_metrics_per_device(cfg['n_devices']))
    print(f"│  Est. dataset rows: ~{est_rows:,}{'':<{max(0,27-len(str(est_rows)))}}│")
    est_windows = cfg['n_days'] * 24 * 60 // cfg['step_minutes']
    print(f"│  Est. windows     : ~{est_windows:,}{'':<{max(0,27-len(str(est_windows)))}}│")
    print(f"│  HL events/group  : {cfg['n_events_high_latency']:<32}│")
    print(f"│  IF events/switch : {cfg['n_events_iface_flap']:<32}│")
    print("├─────────────────────────────────────────────────────┤")
    print(f"│  Window           : {cfg['window_minutes']} min   Step: {cfg['step_minutes']} min{'':<16}│")
    print(f"│  min_support      : {cfg['min_support']:<32}│")
    print(f"│  min_confidence   : {cfg['min_confidence']:<32}│")
    print(f"│  min_lift         : {cfg['min_lift']:<32}│")
    print(f"│  min_corr         : {cfg['min_corr']:<32}│")
    print(f"│  max_hops         : {cfg['max_hops']:<32}│")
    print(f"│  Workers          : {str(cfg['n_workers'] or 'auto (cpu_count)'):<32}│")
    print("└─────────────────────────────────────────────────────┘")


def _avg_metrics_per_device(n_devices: int) -> float:
    """Rough average metrics per device across all roles."""
    # router=2, dist=4, fw=4, access=5, edge=3  weighted by count
    if n_devices <= 5:
        return 3.4
    elif n_devices <= 20:
        return 3.6
    else:
        return 3.8
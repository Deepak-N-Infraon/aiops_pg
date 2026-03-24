"""
rediscovery_engine.py
=====================
Periodic re-training loop.  Every N hours:
  1. Re-run feature extraction on recent data
  2. Re-run pattern discovery
  3. Compare discovered patterns with stored ones
  4. Update confidence / drift_score on existing patterns
  5. Add truly new patterns; retire drifted ones
"""

from __future__ import annotations
from typing import Dict, List
import datetime
import pandas as pd

from topology_loader   import TopologyLoader
from feature_engine    import FeatureEngine
from pattern_discovery import PatternDiscovery, PatternSpec
from pattern_storage   import PatternStorage


class RediscoveryEngine:
    """
    Parameters
    ----------
    topo       : TopologyLoader
    storage    : PatternStorage (existing patterns)
    target_events : same format as PatternDiscovery.discover()
    lookback_hours : how many hours of recent data to use (default 48)
    """

    def __init__(
        self,
        topo:           TopologyLoader,
        storage:        PatternStorage,
        target_events:  List[Dict],
        lookback_hours: int   = 48,
        min_support:    float = 0.08,
        min_confidence: float = 0.65,
        verbose:        bool  = True,
    ):
        self.topo           = topo
        self.storage        = storage
        self.target_events  = target_events
        self.lookback_hours = lookback_hours
        self.min_support    = min_support
        self.min_confidence = min_confidence
        self.verbose        = verbose

    # ── Pattern similarity ────────────────────────────────────────────────

    def _pattern_signature(self, pattern: dict) -> frozenset:
        """
        A pattern's identity = set of (device, metric) step pairs.
        Used to match a newly discovered pattern to an existing stored one.
        """
        return frozenset(
            (s["node"]["device"], s["metric"])
            for s in pattern.get("sequence", [])
        )

    def _find_matching_stored(self, new_sig: frozenset) -> List[str]:
        """Return pattern_ids of stored patterns with the same signature."""
        matches = []
        for sp in self.storage.all_patterns():
            sig = self._pattern_signature(sp)
            # Jaccard similarity >= 0.6 counts as a match
            overlap = len(sig & new_sig)
            union   = len(sig | new_sig)
            if union > 0 and overlap / union >= 0.6:
                matches.append(sp["pattern_id"])
        return matches

    # ── Main rediscovery run ──────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> Dict:
        """
        Execute one rediscovery cycle.

        Returns a summary dict with:
          - new_patterns   : list of newly added pattern_ids
          - updated        : list of updated pattern_ids
          - retired        : list of retired pattern_ids
          - run_timestamp  : ISO timestamp
        """
        print("\n" + "═"*60)
        print(f"  REDISCOVERY RUN — {datetime.datetime.now().isoformat(timespec='seconds')}")
        print("═"*60)

        # Filter to recent data
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=self.lookback_hours)
        recent_df = df[df["timestamp"] >= cutoff]
        print(f"\n  Using {len(recent_df)} rows from last {self.lookback_hours}h")

        # Feature extraction
        fe      = FeatureEngine(window_minutes=75, step_minutes=5)
        windows = fe.compute_all_windows(recent_df)
        print(f"  Computed {len(windows)} feature windows")

        if len(windows) < 10:
            print("  ✗ Too few windows for rediscovery — skipping")
            return {"new_patterns": [], "updated": [], "retired": [],
                    "run_timestamp": datetime.datetime.now().isoformat()}

        # Pattern discovery
        pd_engine = PatternDiscovery(
            topo           = self.topo,
            min_support    = self.min_support,
            min_confidence = self.min_confidence,
            verbose        = self.verbose,
        )
        new_specs = pd_engine.discover(
            df=recent_df, target_events=self.target_events, windows=windows
        )

        # Compare with stored patterns
        added    = []
        updated  = []
        retired  = []

        existing_sigs = {
            self._pattern_signature(sp): sp["pattern_id"]
            for sp in self.storage.all_patterns()
        }

        for spec in new_specs:
            new_sig     = self._pattern_signature(spec.to_json())
            match_ids   = self._find_matching_stored(new_sig)

            if match_ids:
                # Update existing matched pattern(s)
                for pid in match_ids:
                    self.storage.update_on_rediscovery(
                        pid,
                        new_confidence = spec.confidence,
                        new_support    = spec.support,
                    )
                    updated.append(pid)
                    print(f"  ↻ Updated  {pid}  new_conf={spec.confidence:.3f}")
            else:
                # Genuinely new pattern
                self.storage.add_pattern(spec)
                added.append(spec.pattern_id)
                print(f"  + New pattern  {spec.pattern_id}")

        # Retire patterns not seen in recent data (high drift)
        for sp in self.storage.all_patterns():
            pid = sp["pattern_id"]
            if (sp["lifecycle"]["drift_score"] > 0.40 and
                    sp["lifecycle"]["status"] == "active"):
                self.storage.mark_inactive(pid)
                retired.append(pid)
                print(f"  – Retired  {pid}  drift={sp['lifecycle']['drift_score']:.3f}")

        result = {
            "new_patterns":    added,
            "updated":         updated,
            "retired":         retired,
            "run_timestamp":   datetime.datetime.now().isoformat(),
        }

        print(f"\n  Summary: +{len(added)} new  ↻{len(updated)} updated  "
              f"–{len(retired)} retired")
        print("═"*60 + "\n")
        return result

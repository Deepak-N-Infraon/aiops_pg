"""
pattern_storage.py
==================
Persistent JSON pattern store.  Supports:
  - save / load  patterns
  - update confidence + drift_score on rediscovery
  - mark patterns inactive when drift is too high
"""

from __future__ import annotations
from typing import Dict, List, Optional
import json, os, datetime, copy
from pattern_discovery import PatternSpec


DRIFT_INACTIVE_THRESHOLD = 0.40


class PatternStorage:
    """
    Stores patterns as a JSON file (one JSON object per pattern).

    File format: list of pattern dicts (same as PatternSpec.to_json()).
    """

    def __init__(self, path: str = "patterns/patterns.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._patterns: Dict[str, dict] = {}   # pattern_id → dict
        if os.path.exists(path):
            self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        self._patterns = {p["pattern_id"]: p for p in data}

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(list(self._patterns.values()), f, indent=2)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_pattern(self, spec: PatternSpec) -> None:
        """Insert a new pattern (skip if already exists)."""
        pid = spec.pattern_id
        if pid not in self._patterns:
            self._patterns[pid] = spec.to_json()
            self.save()
            print(f"  [Storage] Added pattern {pid}")

    def add_patterns(self, specs: List[PatternSpec]) -> None:
        for s in specs:
            self.add_pattern(s)

    def get_pattern(self, pattern_id: str) -> Optional[dict]:
        return self._patterns.get(pattern_id)

    def all_active(self) -> List[dict]:
        return [p for p in self._patterns.values()
                if p.get("lifecycle", {}).get("status") == "active"]

    def all_patterns(self) -> List[dict]:
        return list(self._patterns.values())

    # ── Drift / update logic (for rediscovery) ───────────────────────────

    def update_on_rediscovery(
        self,
        pattern_id: str,
        new_confidence: float,
        new_support:    int,
    ) -> None:
        """
        After a periodic rediscovery run, update a pattern's stats.

          - If new_confidence is similar or better → lower drift, update stats
          - If new_confidence has dropped significantly → increase drift_score
          - If drift_score > DRIFT_INACTIVE_THRESHOLD → mark inactive
        """
        if pattern_id not in self._patterns:
            return
        p = self._patterns[pattern_id]
        old_conf = p["stats"]["confidence"]
        delta    = new_confidence - old_conf    # positive = improved

        drift = p["lifecycle"].get("drift_score", 0.0)

        if delta >= 0:
            drift = max(0.0, drift - 0.05)   # drift decreases if pattern is stable
            p["stats"]["confidence"] = round(new_confidence, 3)
            p["stats"]["support"]    = new_support
        else:
            # Confidence dropped: drift increases proportionally
            drift = min(1.0, drift + abs(delta) * 0.5)

        p["lifecycle"]["drift_score"]  = round(drift, 3)
        p["lifecycle"]["last_updated"] = datetime.date.today().isoformat()

        if drift > DRIFT_INACTIVE_THRESHOLD:
            p["lifecycle"]["status"] = "inactive"
            print(f"  [Storage] Pattern {pattern_id} marked INACTIVE "
                  f"(drift={drift:.3f})")
        else:
            p["lifecycle"]["status"] = "active"

        self.save()

    def mark_inactive(self, pattern_id: str) -> None:
        if pattern_id in self._patterns:
            self._patterns[pattern_id]["lifecycle"]["status"] = "inactive"
            self.save()

    # ── Display ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        active   = self.all_active()
        inactive = [p for p in self._patterns.values()
                    if p.get("lifecycle", {}).get("status") != "active"]
        lines = [f"\nPattern Store: {len(self._patterns)} total  "
                 f"({len(active)} active, {len(inactive)} inactive)"]
        for p in active:
            chain = " → ".join(
                f"{s['node']['device']}:{s['metric']}"
                for s in p.get("sequence", [])
            )
            lines.append(
                f"  [{p['pattern_id']}]  conf={p['stats']['confidence']:.2f}  "
                f"lift={p['stats']['lift']:.2f}  "
                f"drift={p['lifecycle']['drift_score']:.2f}\n"
                f"    Chain: {chain}"
            )
        return "\n".join(lines)

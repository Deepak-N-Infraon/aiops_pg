"""
topology_loader.py
==================
Loads network topology (nodes + edges) and answers hop-distance / path queries.
Used by pattern_discovery and inference_engine to enforce topology-validity.

v2 changes  (everything else is byte-for-byte identical to v1)
--------------
  load_topology() now accepts THREE input types:
    1. dict        — original behaviour (pass get_topology() result directly)
    2. file path   — original behaviour (load JSON from disk)
    3. psycopg2 connection — NEW: query devices table and build topology live
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import json


class TopologyLoader:
    """
    Stores the directed network graph and exposes:
      - hop_distance(src, dst)      → int | None
      - reachable_within(node, k)   → set of device names
      - shortest_path(src, dst)     → list of node names | None
      - get_role(device)            → role string
      - all_pairs_within(max_hops)  → List[(dev_a, dev_b)]
      - summary()                   → str
    """

    def __init__(self, topology: Dict):
        self._role: Dict[str, str]       = {}
        self._adj:  Dict[str, List[str]] = {}

        for n in topology["nodes"]:
            if isinstance(n, dict):
                nid  = n["id"]
                role = n.get("role", "unknown")
            else:
                nid, role = n, "unknown"
            self._role[nid] = role
            self._adj.setdefault(nid, [])

        for src, dst in topology["edges"]:
            self._adj.setdefault(src, []).append(dst)
            self._adj.setdefault(dst, []).append(src)

        self.devices: List[str] = list(self._role.keys())

    # ── BFS helpers ──────────────────────────────────────────────────────────

    def hop_distance(self, src: str, dst: str) -> Optional[int]:
        """Minimum hops between src and dst (BFS)."""
        if src == dst:
            return 0
        visited: Set[str] = {src}
        queue   = deque([(src, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self._adj.get(node, []):
                if nbr == dst:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return None

    def reachable_within(self, node: str, max_hops: int) -> Set[str]:
        visited: Set[str] = {node}
        queue   = deque([(node, 0)])
        result:  Set[str] = set()
        while queue:
            cur, dist = queue.popleft()
            if cur != node:
                result.add(cur)
            if dist < max_hops:
                for nbr in self._adj.get(cur, []):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, dist + 1))
        return result

    def shortest_path(self, src: str, dst: str) -> Optional[List[str]]:
        if src == dst:
            return [src]
        visited: Set[str] = {src}
        queue   = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for nbr in self._adj.get(node, []):
                new_path = path + [nbr]
                if nbr == dst:
                    return new_path
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, new_path))
        return None

    def get_role(self, device: str) -> str:
        return self._role.get(device, "unknown")

    def all_pairs_within(self, max_hops: int) -> List[Tuple[str, str]]:
        """
        All ordered (dev_a, dev_b) pairs with hop_distance ≤ max_hops.
        Includes same-device pairs (hop=0) so intra-device metric causality
        (e.g. cpu_pct → rx_errors on the same router) is also tested.
        """
        pairs = []
        for a in self.devices:
            # Same-device pairs (hop = 0) — always include
            pairs.append((a, a))
            # Cross-device pairs within max_hops
            for b in self.devices:
                if a != b:
                    d = self.hop_distance(a, b)
                    if d is not None and d <= max_hops:
                        pairs.append((a, b))
        return pairs

    def summary(self) -> str:
        n_links = sum(len(v) for v in self._adj.values()) // 2
        lines   = [f"Topology: {len(self.devices)} devices, {n_links} links"]
        for dev in self.devices:
            lines.append(
                f"  {dev:30s}  role={self._role[dev]:15s}  "
                f"neighbours={self._adj.get(dev, [])}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def load_topology(source, **kwargs) -> TopologyLoader:
    """
    Create a TopologyLoader from any of:

      load_topology(dict)        — pass the topology dict directly
      load_topology("path.json") — load from a JSON file on disk
      load_topology(conn)        — build from devices table in PostgreSQL
                                   kwargs: device_type=, device_id=

    The third form is new in v2 and is what main.py uses when connecting
    to the pattern_mining database.
    """
    # psycopg2 connection — query the DB
    try:
        import psycopg2
        if isinstance(source, psycopg2.extensions.connection):
            from db_loader import load_topology_dict
            topo_dict = load_topology_dict(source, **kwargs)
            return TopologyLoader(topo_dict)
    except ImportError:
        pass

    # Plain dict
    if isinstance(source, dict):
        return TopologyLoader(source)

    # File path (str or Path)
    with open(source) as f:
        return TopologyLoader(json.load(f))
"""
topology_loader.py
==================
Loads network topology (nodes + edges) and answers hop-distance / path queries.
Used by pattern_discovery and inference_engine to enforce topology-validity.
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import json


class TopologyLoader:
    """
    Stores the directed network graph and exposes:
      - hop_distance(src, dst)   → int | None
      - reachable_within(node, max_hops) → set of device names
      - shortest_path(src, dst)  → list of node names | None
      - get_role(device)         → role string
    """

    def __init__(self, topology: Dict):
        """
        topology = {
            "nodes": [{"id": "R1", "role": "router"}, ...],
            "edges": [["R1","SW1"], ...]
        }
        Edges are treated as undirected (bidirectional propagation).
        """
        self._role: Dict[str, str] = {}
        self._adj:  Dict[str, List[str]] = {}

        # Support both simple string lists and dict lists for nodes
        for n in topology["nodes"]:
            if isinstance(n, dict):
                nid  = n["id"]
                role = n.get("role", "unknown")
            else:
                nid  = n
                role = "unknown"
            self._role[nid] = role
            self._adj.setdefault(nid, [])

        for src, dst in topology["edges"]:
            self._adj.setdefault(src, []).append(dst)
            self._adj.setdefault(dst, []).append(src)

        self.devices: List[str] = list(self._role.keys())

    # ── BFS helpers ──────────────────────────────────────────────────────

    def hop_distance(self, src: str, dst: str) -> Optional[int]:
        """Return the minimum number of hops between src and dst (BFS)."""
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
        return None  # unreachable

    def reachable_within(self, node: str, max_hops: int) -> Set[str]:
        """Return all devices reachable from node within max_hops."""
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
        """Return the list of nodes on the shortest path src → dst."""
        if src == dst:
            return [src]
        visited: Set[str]            = {src}
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
        """Return all ordered (dev_a, dev_b) pairs with hop_distance <= max_hops."""
        pairs = []
        for a in self.devices:
            for b in self.devices:
                if a != b:
                    d = self.hop_distance(a, b)
                    if d is not None and d <= max_hops:
                        pairs.append((a, b))
        return pairs

    def summary(self) -> str:
        lines = [f"Topology: {len(self.devices)} devices, "
                 f"{sum(len(v) for v in self._adj.values())//2} links"]
        for dev in self.devices:
            lines.append(f"  {dev:12s}  role={self._role[dev]:15s}  "
                         f"neighbours={self._adj.get(dev,[])}")
        return "\n".join(lines)


def load_topology(path_or_dict) -> TopologyLoader:
    if isinstance(path_or_dict, dict):
        return TopologyLoader(path_or_dict)
    with open(path_or_dict) as f:
        return TopologyLoader(json.load(f))

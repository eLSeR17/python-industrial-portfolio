"""Supply chain network domain model backed by NetworkX directed graphs."""

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from src.models.schemas import EdgeCreate, NodeCreate, NodeType, TransportMode


@dataclass
class SupplyChainNetwork:
    """Directed graph representing the supply chain.

    Nodes carry typed attributes (supplier, warehouse, DC, customer, etc.)
    and edges represent transport routes with cost, time, capacity, and
    reliability data. The graph is stored as a NetworkX DiGraph for
    algorithmic queries (shortest path, max flow, centrality, etc.).

    Attributes:
        graph: Underlying directed graph.
        network_id: Unique identifier for this network instance.
    """

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    network_id: str = ""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_node(self, node: NodeCreate) -> None:
        """Add a supply chain node to the graph."""
        self.graph.add_node(
            node.id,
            name=node.name,
            node_type=node.type,
            latitude=node.latitude,
            longitude=node.longitude,
            capacity=node.capacity,
            fixed_cost=node.fixed_cost,
            variable_cost=node.variable_cost,
            demand=node.demand,
            meta=node.metadata,
        )

    def add_edge(self, edge: EdgeCreate) -> None:
        """Add a transport route between two nodes."""
        if edge.source not in self.graph:
            raise ValueError(f"Source node '{edge.source}' not found in network")
        if edge.target not in self.graph:
            raise ValueError(f"Target node '{edge.target}' not found in network")

        self.graph.add_edge(
            edge.source,
            edge.target,
            transport_mode=edge.transport_mode,
            distance_km=edge.distance_km,
            cost_per_unit=edge.cost_per_unit,
            transit_time_hours=edge.transit_time_hours,
            capacity=edge.capacity,
            reliability=edge.reliability,
            meta=edge.metadata,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def nodes_by_type(self, node_type: NodeType) -> list[str]:
        """Return all node IDs of a given type."""
        return [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == node_type
        ]

    def suppliers(self) -> list[str]:
        """Return supplier node IDs."""
        return self.nodes_by_type(NodeType.SUPPLIER)

    def warehouses(self) -> list[str]:
        """Return warehouse node IDs."""
        return self.nodes_by_type(NodeType.WAREHOUSE)

    def distribution_centers(self) -> list[str]:
        """Return distribution center node IDs."""
        return self.nodes_by_type(NodeType.DISTRIBUTION_CENTER)

    def customers(self) -> list[str]:
        """Return customer node IDs."""
        return self.nodes_by_type(NodeType.CUSTOMER)

    def total_demand(self) -> float:
        """Sum of all customer demands."""
        return sum(
            self.graph.nodes[n].get("demand", 0.0)
            for n in self.customers()
        )

    def total_capacity(self) -> float:
        """Sum of all facility capacities (suppliers + warehouses + DCs)."""
        facility_types = {NodeType.SUPPLIER, NodeType.WAREHOUSE, NodeType.DISTRIBUTION_CENTER}
        return sum(
            self.graph.nodes[n].get("capacity", 0.0)
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") in facility_types
        )

    def edge_cost(self, source: str, target: str) -> float:
        """Return the per-unit transport cost for an edge."""
        return self.graph[source][target].get("cost_per_unit", 0.0)

    def edge_time(self, source: str, target: str) -> float:
        """Return the transit time (hours) for an edge."""
        return self.graph[source][target].get("transit_time_hours", 0.0)

    def summary(self) -> dict[str, Any]:
        """High-level network statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "suppliers": len(self.suppliers()),
            "warehouses": len(self.warehouses()),
            "distribution_centers": len(self.distribution_centers()),
            "customers": len(self.customers()),
            "total_demand": self.total_demand(),
            "total_capacity": self.total_capacity(),
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "density": nx.density(self.graph),
        }

    # ------------------------------------------------------------------
    # Flow computation helpers
    # ------------------------------------------------------------------

    def shortest_cost_path(self, source: str, target: str) -> tuple[list[str], float]:
        """Find the lowest-cost path between source and target."""
        try:
            path = nx.shortest_path(
                self.graph, source, target, weight="cost_per_unit"
            )
            cost = self._path_cost(path)
            return path, cost
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], float("inf")

    def shortest_time_path(self, source: str, target: str) -> tuple[list[str], float]:
        """Find the fastest path between source and target."""
        try:
            path = nx.shortest_path(
                self.graph, source, target, weight="transit_time_hours"
            )
            time_h = self._path_time(path)
            return path, time_h
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], float("inf")

    def _path_cost(self, path: list[str]) -> float:
        """Accumulate per-unit cost along a path."""
        return sum(
            self.edge_cost(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

    def _path_time(self, path: list[str]) -> float:
        """Accumulate transit time along a path."""
        return sum(
            self.edge_time(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

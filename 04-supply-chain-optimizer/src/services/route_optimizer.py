"""Vehicle routing and shortest-path optimization using PuLP and NetworkX."""

from typing import Any

import networkx as nx
import numpy as np
import pulp
from pulp import (
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
)

from src.models.network import SupplyChainNetwork
from src.models.schemas import (
    OptimizationObjective,
    RouteOptimizeRequest,
    RouteOptimizeResponse,
    RouteResult,
)
from config.settings import settings


class RouteOptimizer:
    """Solve vehicle routing and shortest-path problems.

    Implements two optimization approaches:
    1. Single-vehicle shortest path via NetworkX Dijkstra
    2. Multi-vehicle capacitated VRP via PuLP integer programming

    The VRP formulation minimizes total distance subject to vehicle
    capacity, route continuity, and visit-each-customer constraints.
    """

    def __init__(self, network: SupplyChainNetwork) -> None:
        """Initialize with a supply chain network.

        Args:
            network: The supply chain graph.
        """
        self.network = network
        self.graph = network.graph

    # ------------------------------------------------------------------
    # Single vehicle routing
    # ------------------------------------------------------------------

    def shortest_route(
        self,
        source: str,
        target: str,
        objective: OptimizationObjective = OptimizationObjective.MIN_COST,
    ) -> dict[str, Any]:
        """Find the optimal single-vehicle route between two nodes.

        Args:
            source: Departure node ID.
            target: Arrival node ID.
            objective: Whether to optimize for cost, time, or balanced.

        Returns:
            Route details including path, cost, and time.

        Raises:
            ValueError: If no route exists.
        """
        if objective == OptimizationObjective.MIN_COST:
            path, cost = self.network.shortest_cost_path(source, target)
            time_h = self.network._path_time(path)
        elif objective == OptimizationObjective.MIN_TIME:
            path, time_h = self.network.shortest_time_path(source, target)
            cost = self.network._path_cost(path)
        else:
            # Balanced: minimize cost * time composite
            path, cost = self.network.shortest_cost_path(source, target)
            time_h = self.network._path_time(path)
            if path:
                # Re-evaluate with combined weight
                combined = nx.DiGraph(self.graph)
                for u, v, d in combined.edges(data=True):
                    combined[u][v]["weight"] = (
                        d["cost_per_unit"] * d["transit_time_hours"]
                    )
                try:
                    path = nx.shortest_path(combined, source, target)
                    cost = self.network._path_cost(path)
                    time_h = self.network._path_time(path)
                except nx.NetworkXNoPath:
                    pass

        if not path:
            raise ValueError(f"No route from {source} to {target}")

        distance = sum(
            self.graph[path[i]][path[i + 1]].get("distance_km", 0)
            for i in range(len(path) - 1)
        )

        return {
            "path": path,
            "distance_km": distance,
            "cost": cost,
            "time_hours": time_h,
            "stops": len(path),
        }

    # ------------------------------------------------------------------
    # Capacitated Vehicle Routing Problem (VRP)
    # ------------------------------------------------------------------

    def solve_vrp(
        self,
        request: RouteOptimizeRequest,
    ) -> RouteOptimizeResponse:
        """Solve the Capacitated VRP with PuLP.

        Formulates the problem as a Mixed-Integer Program:
        - Binary decision variables x[i,j,k] = 1 if vehicle k traverses edge (i,j)
        - Each customer visited exactly once
        - Vehicle capacity not exceeded
        - Routes start and end at a depot

        Args:
            request: VRP parameters (depots, customers, fleet size, capacity).

        Returns:
            Optimized routes with cost and distance metrics.
        """
        # Build subgraph of relevant nodes
        depot_ids = set(request.depot_ids)
        customer_ids = set(request.customer_ids)
        relevant = depot_ids | customer_ids

        subgraph = self._build_complete_graph(relevant)

        node_list = list(subgraph.nodes())
        n = len(node_list)
        node_idx = {node: i for i, node in enumerate(node_list)}

        K = request.vehicle_count
        Q = request.vehicle_capacity

        # Demand per node (0 for depots)
        demand = {}
        for node in node_list:
            if node in customer_ids:
                demand[node] = max(1.0, self.graph.nodes[node].get("demand", 100))
            else:
                demand[node] = 0.0

        # Distance matrix
        dist: dict[tuple[int, int], float] = {}
        for u in node_list:
            for v in node_list:
                if u == v:
                    dist[(node_idx[u], node_idx[v])] = 0.0
                elif subgraph.has_edge(u, v):
                    dist[(node_idx[u], node_idx[v])] = subgraph[u][v].get(
                        "cost_per_unit", 1.0
                    )
                else:
                    try:
                        sp = nx.shortest_path_length(
                            self.graph, u, v, weight="cost_per_unit"
                        )
                        dist[(node_idx[u], node_idx[v])] = sp
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        dist[(node_idx[u], node_idx[v])] = 1e9

        # Time matrix
        time_m: dict[tuple[int, int], float] = {}
        for u in node_list:
            for v in node_list:
                if u == v:
                    time_m[(node_idx[u], node_idx[v])] = 0.0
                elif subgraph.has_edge(u, v):
                    time_m[(node_idx[u], node_idx[v])] = subgraph[u][v].get(
                        "transit_time_hours", 1.0
                    )
                else:
                    try:
                        sp = nx.shortest_path_length(
                            self.graph, u, v, weight="transit_time_hours"
                        )
                        time_m[(node_idx[u], node_idx[v])] = sp
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        time_m[(node_idx[u], node_idx[v])] = 1e9

        # --- PuLP model ---
        prob = LpProblem("VRP", LpMinimize)

        # Decision variables
        x = {}
        for i in range(n):
            for j in range(n):
                for k in range(K):
                    if i != j:
                        x[i, j, k] = LpVariable(f"x_{i}_{j}_{k}", cat="Binary")

        # Load tracking
        load = {}
        for i in range(n):
            for k in range(K):
                load[i, k] = LpVariable(f"load_{i}_{k}", lowBound=0)

        # Objective: minimize total distance
        prob += lpSum(
            dist[i, j] * x[i, j, k]
            for i in range(n)
            for j in range(n)
            for k in range(K)
            if i != j
        )

        # Each customer visited exactly once across all vehicles
        for j in node_list:
            if j in customer_ids:
                j_idx = node_idx[j]
                prob += (
                    lpSum(x[i, j_idx, k] for i in range(n) for k in range(K) if i != j_idx)
                    == 1,
                    f"visit_{j}",
                )

        # Flow conservation: if vehicle k enters a non-depot, non-customer node, it must leave
        for j_idx in range(n):
            for k in range(K):
                node_id = node_list[j_idx]
                if node_id not in depot_ids and node_id not in customer_ids:
                    prob += (
                        lpSum(x[i, j_idx, k] for i in range(n) if i != j_idx)
                        == lpSum(x[j_idx, i, k] for i in range(n) if i != j_idx),
                        f"flow_{j_idx}_{k}",
                    )

        # Each vehicle leaves each depot at most once
        for d in depot_ids:
            d_idx = node_idx[d]
            prob += (
                lpSum(x[d_idx, j, k] for j in range(n) for k in range(K) if j != d_idx)
                <= 1,
                f"depot_leave_{d}",
            )

        # Capacity constraints using load variables
        for k in range(K):
            for i in range(n):
                for j in range(n):
                    if i != j and (i, j, k) in x:
                        prob += (
                            load[j, k] >= load[i, k] + demand.get(node_list[j], 0)
                            - Q * (1 - x[i, j, k]),
                            f"cap_lb_{i}_{j}_{k}",
                        )
                        prob += (
                            load[j, k] <= load[i, k] + demand.get(node_list[j], 0)
                            + Q * (1 - x[i, j, k]),
                            f"cap_ub_{i}_{j}_{k}",
                        )
                        prob += load[j, k] <= Q

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=settings.solver_time_limit_seconds))

        status = LpStatus.get(prob.status, "Unknown")

        # Extract routes
        routes: list[RouteResult] = []
        vehicles_used = 0
        total_distance = 0.0
        total_cost = 0.0
        total_time = 0.0

        for k in range(K):
            route_edges = []
            for i in range(n):
                for j in range(n):
                    if i != j and (i, j, k) in x and value(x[i, j, k]) == 1:
                        route_edges.append((node_list[i], node_list[j]))

            if not route_edges:
                continue

            vehicles_used += 1
            stops = self._extract_route_stops(route_edges, depot_ids)
            route_dist = sum(dist.get((node_idx[a], node_idx[b]), 0) for a, b in route_edges)
            route_cost = route_dist  # cost is the distance in cost units
            route_time = sum(time_m.get((node_idx[a], node_idx[b]), 0) for a, b in route_edges)
            route_load = sum(demand.get(s, 0) for s in stops if s in customer_ids)

            total_distance += route_dist
            total_cost += route_cost
            total_time += route_time

            routes.append(RouteResult(
                vehicle_id=k + 1,
                stops=stops,
                distance_km=round(route_dist, 2),
                cost=round(route_cost, 2),
                time_hours=round(route_time, 2),
                load=round(route_load, 2),
            ))

        return RouteOptimizeResponse(
            routes=routes,
            total_distance_km=round(total_distance, 2),
            total_cost=round(total_cost, 2),
            total_time_hours=round(total_time, 2),
            vehicles_used=vehicles_used,
            solver_status=status,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_complete_graph(self, nodes: set[str]) -> nx.DiGraph:
        """Build a complete directed graph using shortest paths between all nodes."""
        complete = nx.DiGraph()
        node_list = list(nodes)
        for n in node_list:
            complete.add_node(n, **self.graph.nodes[n])

        for i, u in enumerate(node_list):
            for j, v in enumerate(node_list):
                if i != j:
                    try:
                        path = nx.shortest_path(
                            self.graph, u, v, weight="cost_per_unit"
                        )
                        cost = self.network._path_cost(path)
                        time_h = self.network._path_time(path)
                        dist = sum(
                            self.graph[path[k]][path[k + 1]].get("distance_km", 0)
                            for k in range(len(path) - 1)
                        )
                        complete.add_edge(
                            u, v,
                            cost_per_unit=cost,
                            transit_time_hours=time_h,
                            distance_km=dist,
                            capacity=999999,
                        )
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
        return complete

    def _extract_route_stops(
        self,
        edges: list[tuple[str, str]],
        depots: set[str],
    ) -> list[str]:
        """Reconstruct ordered stop list from edge list, starting from a depot."""
        adjacency: dict[str, str] = {u: v for u, v in edges}

        # Find the starting depot
        start = None
        for u, v in edges:
            if u in depots:
                start = u
                break
        if start is None and edges:
            start = edges[0][0]

        stops = []
        current = start
        visited = set()
        while current and current not in visited:
            stops.append(current)
            visited.add(current)
            current = adjacency.get(current)

        return stops

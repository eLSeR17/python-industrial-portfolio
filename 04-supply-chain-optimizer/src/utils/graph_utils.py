"""Graph analytics: centrality, bottleneck detection, and resilience scoring."""

from typing import Any

import networkx as nx


def analyze_network(graph: nx.DiGraph) -> dict[str, Any]:
    """Run network analysis on a supply chain graph.

    Computes centrality measures, identifies bottlenecks and critical
    paths, and detects single points of failure. These analytics help
    supply chain managers understand network vulnerability and prioritize
    improvement investments.

    Args:
        graph: Directed graph representing the supply chain.

    Returns:
        Dictionary with centrality, bottleneck, and vulnerability metrics.
    """
    if graph.number_of_nodes() == 0:
        return {"error": "Empty graph"}

    # Degree centrality (how many connections a node has)
    in_degree = nx.in_degree_centrality(graph)
    out_degree = nx.out_degree_centrality(graph)

    # Betweenness centrality (how often a node lies on shortest paths)
    betweenness = nx.betweenness_centrality(graph, weight="cost_per_unit")

    # Closeness centrality (average distance to all other nodes)
    try:
        closeness = nx.closeness_centrality(graph, distance="cost_per_unit")
    except Exception:
        closeness = {n: 0.0 for n in graph.nodes()}

    # PageRank (importance based on incoming edge quality)
    try:
        pagerank = nx.pagerank(graph, weight="cost_per_unit")
    except Exception:
        pagerank = {n: 0.0 for n in graph.nodes()}

    # Bottleneck detection: nodes with highest betweenness
    sorted_betweenness = sorted(
        betweenness.items(), key=lambda x: x[1], reverse=True
    )
    bottlenecks = [
        {"node": n, "betweenness": round(v, 4)}
        for n, v in sorted_betweenness[:5]
        if v > 0
    ]

    # Single points of failure: removing this node disconnects the graph
    spof = _find_single_points_of_failure(graph)

    # Critical path analysis
    critical_path = _find_critical_path(graph)

    # Capacity utilization analysis
    capacity_analysis = _analyze_capacity(graph)

    # Network resilience score
    resilience = _compute_resilience(graph)

    return {
        "centrality": {
            "in_degree": {n: round(v, 4) for n, v in in_degree.items()},
            "out_degree": {n: round(v, 4) for n, v in out_degree.items()},
            "betweenness": {n: round(v, 4) for n, v in betweenness.items()},
            "closeness": {n: round(v, 4) for n, v in closeness.items()},
            "pagerank": {n: round(v, 4) for n, v in pagerank.items()},
        },
        "bottlenecks": bottlenecks,
        "single_points_of_failure": spof,
        "critical_path": critical_path,
        "capacity_analysis": capacity_analysis,
        "resilience_score": resilience,
    }


def _find_single_points_of_failure(graph: nx.DiGraph) -> list[dict[str, Any]]:
    """Identify nodes whose removal disconnects the graph.

    For supply chain robustness: a single point of failure represents
    a critical vulnerability that could halt the entire supply chain
    if that facility goes offline.
    """
    spof = []
    nodes = list(graph.nodes())

    for node in nodes:
        # Skip source/sink nodes with degree 1
        if graph.in_degree(node) == 0 or graph.out_degree(node) == 0:
            continue

        # Create graph without this node
        remaining = graph.copy()
        remaining.remove_node(node)

        # Check if graph is still weakly connected
        if remaining.number_of_nodes() > 1:
            is_connected = nx.is_weakly_connected(remaining)
            if not is_connected:
                num_components = nx.number_weakly_connected_components(remaining)
                spof.append({
                    "node": node,
                    "components_after_removal": num_components,
                    "risk_level": "critical" if num_components > 2 else "high",
                })

    return spof


def _find_critical_path(graph: nx.DiGraph) -> dict[str, Any]:
    """Find the longest (critical) path through the supply chain.

    The critical path determines the minimum lead time from any supplier
    to any customer. It represents the bottleneck lead time that cannot
    be shortened without structural changes.
    """
    try:
        longest_path = nx.dag_longest_path(graph, weight="transit_time_hours")
        longest_length = nx.dag_longest_path_length(graph, weight="transit_time_hours")
    except Exception:
        # Graph may have cycles; use heuristic
        longest_path = []
        longest_length = 0.0
        for source in graph.nodes():
            if graph.in_degree(source) == 0:
                for target in graph.nodes():
                    if graph.out_degree(target) == 0:
                        try:
                            path = nx.shortest_path(
                                graph, source, target, weight="transit_time_hours"
                            )
                            length = sum(
                                graph[path[i]][path[i + 1]].get("transit_time_hours", 0)
                                for i in range(len(path) - 1)
                            )
                            if length > longest_length:
                                longest_length = length
                                longest_path = path
                        except nx.NetworkXNoPath:
                            continue

    # Also find lowest-cost path as alternative
    min_cost_path = []
    min_cost = float("inf")
    for source in graph.nodes():
        if graph.in_degree(source) == 0:
            for target in graph.nodes():
                if graph.out_degree(target) == 0:
                    try:
                        path = nx.shortest_path(
                            graph, source, target, weight="cost_per_unit"
                        )
                        cost = sum(
                            graph[path[i]][path[i + 1]].get("cost_per_unit", 0)
                            for i in range(len(path) - 1)
                        )
                        if cost < min_cost:
                            min_cost = cost
                            min_cost_path = path
                    except nx.NetworkXNoPath:
                        continue

    return {
        "longest_time_path": longest_path,
        "longest_time_hours": round(longest_length, 2),
        "lowest_cost_path": min_cost_path,
        "lowest_cost": round(min_cost, 2),
    }


def _analyze_capacity(graph: nx.DiGraph) -> dict[str, Any]:
    """Analyze capacity utilization and constraints across the network."""
    # Find capacity-constrained edges (utilization > 80%)
    constrained_edges = []
    for u, v, d in graph.edges(data=True):
        capacity = d.get("capacity", 0)
        if capacity > 0:
            # Estimate utilization based on demand flow
            target_demand = graph.nodes[v].get("demand", 0)
            utilization = (target_demand / capacity * 100) if target_demand > 0 else 0
            if utilization > 80:
                constrained_edges.append({
                    "edge": f"{u}->{v}",
                    "capacity": capacity,
                    "estimated_utilization_pct": round(utilization, 1),
                })

    return {
        "constrained_edges": constrained_edges,
        "total_edges_analyzed": graph.number_of_edges(),
    }


def _compute_resilience(graph: nx.DiGraph) -> dict[str, Any]:
    """Compute a supply chain resilience score (0-100).

    Resilience is based on:
    - Path redundancy: average number of alternative paths
    - Node diversity: number of suppliers/warehouses
    - Connectivity: edge-to-node ratio
    """
    if graph.number_of_nodes() <= 1:
        return {"score": 0, "breakdown": {}}

    # Edge-to-node ratio (higher = more connections = more resilient)
    e_n_ratio = graph.number_of_edges() / graph.number_of_nodes()

    # Supplier diversity
    suppliers = [
        n for n, d in graph.nodes(data=True)
        if d.get("node_type") == "supplier"
    ]
    warehouse_diversity = len([
        n for n, d in graph.nodes(data=True)
        if d.get("node_type") in ("warehouse", "distribution_center")
    ])

    # Average alternative paths between suppliers and customers
    customers = [
        n for n, d in graph.nodes(data=True)
        if d.get("node_type") == "customer"
    ]

    alt_path_scores = []
    for s in suppliers[:3]:
        for c in customers[:3]:
            try:
                paths = list(nx.all_simple_paths(graph, s, c, cutoff=6))
                alt_path_scores.append(len(paths))
            except (nx.NetworkXError, nx.NodeNotFound):
                alt_path_scores.append(0)

    avg_alt_paths = sum(alt_path_scores) / max(len(alt_path_scores), 1)

    # Compute composite score
    connectivity_score = min(e_n_ratio / 2.0, 1.0) * 30
    diversity_score = min((len(suppliers) + warehouse_diversity) / 10.0, 1.0) * 35
    redundancy_score = min(avg_alt_paths / 3.0, 1.0) * 35

    total_score = connectivity_score + diversity_score + redundancy_score

    return {
        "score": round(total_score, 1),
        "breakdown": {
            "connectivity": round(connectivity_score, 1),
            "diversity": round(diversity_score, 1),
            "redundancy": round(redundancy_score, 1),
        },
        "suppliers": len(suppliers),
        "warehouses": warehouse_diversity,
        "avg_alternative_paths": round(avg_alt_paths, 2),
    }


def find_bottleneck_nodes(graph: nx.DiGraph) -> list[dict[str, Any]]:
    """Find nodes that restrict flow through the network.

    Bottlenecks are identified by high betweenness centrality combined
    with capacity constraints. These nodes represent the best candidates
    for capacity expansion investment.
    """
    betweenness = nx.betweenness_centrality(graph, weight="cost_per_unit")

    bottlenecks = []
    for node, bet_score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True):
        if bet_score <= 0:
            continue

        capacity = graph.nodes[node].get("capacity", 0)
        fixed_cost = graph.nodes[node].get("fixed_cost", 0)

        bottlenecks.append({
            "node": node,
            "name": graph.nodes[node].get("name", ""),
            "betweenness_centrality": round(bet_score, 4),
            "capacity": capacity,
            "fixed_cost": fixed_cost,
            "investment_priority": "high" if bet_score > 0.2 else "medium",
        })

    return bottlenecks

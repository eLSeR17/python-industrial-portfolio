"""Tests for graph analytics: centrality, bottleneck, critical path, resilience."""

import networkx as nx
import pytest

from src.utils.graph_utils import (
    _compute_resilience,
    _find_critical_path,
    analyze_network,
    find_bottleneck_nodes,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def linear_chain():
    """A -> B -> C -> D (no redundancy)."""
    g = nx.DiGraph()
    g.add_node("A", node_type="supplier", capacity=1000, fixed_cost=1000)
    g.add_node("B", node_type="warehouse", capacity=800, fixed_cost=500)
    g.add_node("C", node_type="distribution_center", capacity=600, fixed_cost=300)
    g.add_node("D", node_type="customer", demand=200)
    g.add_edge("A", "B", cost_per_unit=5, transit_time_hours=2, capacity=1000)
    g.add_edge("B", "C", cost_per_unit=3, transit_time_hours=1, capacity=800)
    g.add_edge("C", "D", cost_per_unit=2, transit_time_hours=1, capacity=600)
    return g


@pytest.fixture
def diamond_network():
    """Supplier with two parallel paths to customer (redundant)."""
    g = nx.DiGraph()
    g.add_node("S", node_type="supplier", capacity=5000, fixed_cost=1000)
    g.add_node("W1", node_type="warehouse", capacity=3000, fixed_cost=500)
    g.add_node("W2", node_type="warehouse", capacity=3000, fixed_cost=500)
    g.add_node("C", node_type="customer", demand=1000)
    g.add_edge("S", "W1", cost_per_unit=4, transit_time_hours=3, capacity=3000)
    g.add_edge("S", "W2", cost_per_unit=6, transit_time_hours=5, capacity=3000)
    g.add_edge("W1", "C", cost_per_unit=3, transit_time_hours=2, capacity=3000)
    g.add_edge("W2", "C", cost_per_unit=5, transit_time_hours=4, capacity=3000)
    return g


@pytest.fixture
def bottleneck_hub():
    """Star topology: B is the only hub (single point of failure)."""
    g = nx.DiGraph()
    g.add_node("A", node_type="supplier", capacity=1000, fixed_cost=100)
    g.add_node("B", node_type="warehouse", capacity=5000, fixed_cost=100)
    g.add_node("C1", node_type="customer", demand=500)
    g.add_node("C2", node_type="customer", demand=500)
    g.add_edge("A", "B", cost_per_unit=5, transit_time_hours=3, capacity=5000)
    g.add_edge("B", "C1", cost_per_unit=3, transit_time_hours=2, capacity=3000)
    g.add_edge("B", "C2", cost_per_unit=4, transit_time_hours=3, capacity=3000)
    return g


# -----------------------------------------------------------------------
# analyze_network
# -----------------------------------------------------------------------

class TestAnalyzeNetwork:

    def test_returns_all_keys(self, linear_chain):
        result = analyze_network(linear_chain)
        assert "centrality" in result
        assert "bottlenecks" in result
        assert "single_points_of_failure" in result
        assert "critical_path" in result
        assert "capacity_analysis" in result
        assert "resilience_score" in result

    def test_empty_graph(self):
        g = nx.DiGraph()
        result = analyze_network(g)
        assert "error" in result

    def test_centrality_keys(self, linear_chain):
        result = analyze_network(linear_chain)
        centrality = result["centrality"]
        assert "in_degree" in centrality
        assert "out_degree" in centrality
        assert "betweenness" in centrality
        assert "pagerank" in centrality

    def test_bottleneck_nodes_have_betweenness(self, bottleneck_hub):
        result = analyze_network(bottleneck_hub)
        # Hub B should be a bottleneck
        bnode = [b for b in result["bottlenecks"] if b["node"] == "B"]
        assert len(bnode) == 1
        assert bnode[0]["betweenness"] > 0

    def test_linear_chain_has_spof(self, linear_chain):
        result = analyze_network(linear_chain)
        # In a linear chain, removing any intermediate node disconnects
        spof_nodes = {s["node"] for s in result["single_points_of_failure"]}
        assert len(spof_nodes) > 0

    def test_diamond_no_spof(self, diamond_network):
        result = analyze_network(diamond_network)
        # Diamond has redundant paths — removing one warehouse shouldn't disconnect
        spof_nodes = {s["node"] for s in result["single_points_of_failure"]}
        assert "W1" not in spof_nodes
        assert "W2" not in spof_nodes


# -----------------------------------------------------------------------
# find_bottleneck_nodes
# -----------------------------------------------------------------------

class TestFindBottleneckNodes:

    def test_returns_list(self, bottleneck_hub):
        result = find_bottleneck_nodes(bottleneck_hub)
        assert isinstance(result, list)

    def test_hub_is_top_bottleneck(self, bottleneck_hub):
        result = find_bottleneck_nodes(bottleneck_hub)
        if result:
            assert result[0]["node"] == "B"

    def test_empty_graph(self):
        g = nx.DiGraph()
        g.add_node("A")
        result = find_bottleneck_nodes(g)
        assert isinstance(result, list)


# -----------------------------------------------------------------------
# _find_critical_path
# -----------------------------------------------------------------------

class TestCriticalPath:

    def test_returns_dict_keys(self, linear_chain):
        result = _find_critical_path(linear_chain)
        assert "longest_time_path" in result
        assert "longest_time_hours" in result
        assert "lowest_cost_path" in result
        assert "lowest_cost" in result

    def test_linear_chain_critical_path(self, linear_chain):
        result = _find_critical_path(linear_chain)
        assert result["longest_time_hours"] == pytest.approx(4.0)

    def test_cost_and_time_both_positive(self, linear_chain):
        result = _find_critical_path(linear_chain)
        assert result["lowest_cost"] > 0
        assert result["longest_time_hours"] > 0


# -----------------------------------------------------------------------
# _compute_resilience
# -----------------------------------------------------------------------

class TestResilience:

    def test_resilience_score_range(self, diamond_network):
        result = _compute_resilience(diamond_network)
        assert 0 <= result["score"] <= 100

    def test_diamond_more_resilient_than_linear(self, linear_chain, diamond_network):
        r1 = _compute_resilience(linear_chain)
        r2 = _compute_resilience(diamond_network)
        # Diamond has more redundancy
        assert r2["score"] >= r1["score"]

    def test_empty_graph(self):
        g = nx.DiGraph()
        g.add_node("A")
        result = _compute_resilience(g)
        assert result["score"] == 0

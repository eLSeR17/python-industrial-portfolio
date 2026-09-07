"""Supply chain network builder using NetworkX graph algorithms."""

import uuid
from typing import Any

import networkx as nx

from src.models.network import SupplyChainNetwork
from src.models.schemas import (
    EdgeCreate,
    NetworkCreateRequest,
    NetworkCreateResponse,
    NodeType,
)
from src.utils.graph_utils import analyze_network


class NetworkBuilder:
    """Construct and validate a supply chain graph from API input.

    Builds a directed graph where nodes represent facilities (suppliers,
    warehouses, distribution centers, customers) and edges represent
    transport routes. Validates connectivity and capacity feasibility
    before returning a ready-to-optimize network.
    """

    def build(self, request: NetworkCreateRequest) -> tuple[SupplyChainNetwork, NetworkCreateResponse]:
        """Build a validated supply chain network from a creation request.

        Args:
            request: Node and edge definitions for the network.

        Returns:
            Tuple of the constructed network and a summary response.

        Raises:
            ValueError: If edge references non-existent nodes or network
                is disconnected.
        """
        network_id = f"SCN-{uuid.uuid4().hex[:8]}"
        network = SupplyChainNetwork(network_id=network_id)

        # Add nodes
        for node in request.nodes:
            network.add_node(node)

        # Add edges
        for edge in request.edges:
            network.add_edge(edge)

        # Validate structure
        self._validate(network)

        # Run analytics
        analytics = analyze_network(network.graph)

        response = NetworkCreateResponse(
            network_id=network_id,
            node_count=network.graph.number_of_nodes(),
            edge_count=network.graph.number_of_edges(),
            connected_components=nx.number_weakly_connected_components(
                network.graph
            ),
            summary={
                **network.summary(),
                **analytics,
            },
        )

        return network, response

    def _validate(self, network: SupplyChainNetwork) -> None:
        """Validate the network structure and feasibility.

        Raises:
            ValueError: On structural or capacity issues.
        """
        graph = network.graph

        if graph.number_of_nodes() == 0:
            raise ValueError("Network must contain at least one node")

        # Check capacity feasibility
        suppliers = network.suppliers()
        customers = network.customers()

        if not suppliers:
            raise ValueError("Network must contain at least one supplier")
        if not customers:
            raise ValueError("Network must contain at least one customer")

        total_supply = sum(
            graph.nodes[s].get("capacity", 0.0) for s in suppliers
        )
        total_demand = sum(
            graph.nodes[c].get("demand", 0.0) for c in customers
        )

        if total_supply < total_demand:
            raise ValueError(
                f"Insufficient capacity: supply={total_supply:.0f} "
                f"< demand={total_demand:.0f}"
            )

        # Warn about disconnected components but don't block
        num_components = nx.number_weakly_connected_components(graph)
        if num_components > 1:
            # Find which components have demand
            components = list(nx.weakly_connected_components(graph))
            for comp in components:
                comp_demand = sum(
                    graph.nodes[n].get("demand", 0.0)
                    for n in comp
                    if graph.nodes[n].get("node_type") == NodeType.CUSTOMER
                )
                if comp_demand > 0:
                    # Check if this component has supply
                    comp_supply = sum(
                        graph.nodes[n].get("capacity", 0.0)
                        for n in comp
                        if graph.nodes[n].get("node_type") in {
                            NodeType.SUPPLIER, NodeType.WAREHOUSE,
                            NodeType.DISTRIBUTION_CENTER,
                        }
                    )
                    if comp_supply < comp_demand:
                        raise ValueError(
                            f"Disconnected component with demand={comp_demand:.0f} "
                            f"but supply={comp_supply:.0f}"
                        )


def build_demo_network() -> SupplyChainNetwork:
    """Build a realistic demo network with 5 suppliers, 3 warehouses, and 10 customers.

    This creates a multi-tier supply chain spanning:
    - 3 raw material suppliers (domestic)
    - 2 component suppliers (overseas)
    - 3 regional warehouses
    - 10 retail customers

    Returns:
        A fully connected SupplyChainNetwork ready for optimization.
    """
    request = NetworkCreateRequest(
        nodes=[
            # --- Raw material suppliers ---
            NodeCreate(
                id="SUP-01", name="Steel Corp Midwest",
                type=NodeType.SUPPLIER, latitude=41.88, longitude=-87.63,
                capacity=50000, fixed_cost=15000, variable_cost=2.50,
            ),
            NodeCreate(
                id="SUP-02", name="Polymer Industries South",
                type=NodeType.SUPPLIER, latitude=29.76, longitude=-95.37,
                capacity=35000, fixed_cost=12000, variable_cost=3.20,
            ),
            NodeCreate(
                id="SUP-03", name="Electronics West",
                type=NodeType.SUPPLIER, latitude=34.05, longitude=-118.24,
                capacity=20000, fixed_cost=18000, variable_cost=8.00,
            ),
            # --- Overseas component suppliers ---
            NodeCreate(
                id="SUP-04", name="Components Asia",
                type=NodeType.SUPPLIER, latitude=31.23, longitude=121.47,
                capacity=100000, fixed_cost=8000, variable_cost=1.80,
            ),
            NodeCreate(
                id="SUP-05", name="Precision Parts EU",
                type=NodeType.SUPPLIER, latitude=50.11, longitude=8.68,
                capacity=40000, fixed_cost=10000, variable_cost=4.50,
            ),
            # --- Warehouses ---
            NodeCreate(
                id="WH-01", name="Chicago Distribution Hub",
                type=NodeType.WAREHOUSE, latitude=41.88, longitude=-87.63,
                capacity=30000, fixed_cost=25000, variable_cost=0.80,
            ),
            NodeCreate(
                id="WH-02", name="Dallas Regional Warehouse",
                type=NodeType.WAREHOUSE, latitude=32.78, longitude=-96.80,
                capacity=25000, fixed_cost=20000, variable_cost=0.75,
            ),
            NodeCreate(
                id="WH-03", name="Los Angeles Logistics Center",
                type=NodeType.WAREHOUSE, latitude=33.94, longitude=-118.41,
                capacity=35000, fixed_cost=28000, variable_cost=0.90,
            ),
            # --- Customers ---
            NodeCreate(
                id="CUS-01", name="Retail North", type=NodeType.CUSTOMER,
                latitude=45.50, longitude=-73.57, demand=2000,
            ),
            NodeCreate(
                id="CUS-02", name="Retail South", type=NodeType.CUSTOMER,
                latitude=25.76, longitude=-80.19, demand=3500,
            ),
            NodeCreate(
                id="CUS-03", name="Retail Midwest", type=NodeType.CUSTOMER,
                latitude=39.76, longitude=-86.16, demand=2800,
            ),
            NodeCreate(
                id="CUS-04", name="Retail West", type=NodeType.CUSTOMER,
                latitude=37.77, longitude=-122.42, demand=4200,
            ),
            NodeCreate(
                id="CUS-05", name="Industrial East", type=NodeType.CUSTOMER,
                latitude=42.36, longitude=-71.06, demand=1500,
            ),
            NodeCreate(
                id="CUS-06", name="Industrial Central", type=NodeType.CUSTOMER,
                latitude=36.17, longitude=-86.78, demand=1800,
            ),
            NodeCreate(
                id="CUS-07", name="Retail Mountain", type=NodeType.CUSTOMER,
                latitude=39.74, longitude=-104.99, demand=1200,
            ),
            NodeCreate(
                id="CUS-08", name="Retail Pacific NW", type=NodeType.CUSTOMER,
                latitude=47.61, longitude=-122.33, demand=2200,
            ),
            NodeCreate(
                id="CUS-09", name="Industrial Gulf", type=NodeType.CUSTOMER,
                latitude=30.33, longitude=-81.66, demand=3000,
            ),
            NodeCreate(
                id="CUS-10", name="Retail Southwest", type=NodeType.CUSTOMER,
                latitude=33.45, longitude=-112.07, demand=1600,
            ),
        ],
        edges=[
            # --- Suppliers to warehouses (domestic) ---
            EdgeCreate(source="SUP-01", target="WH-01", distance_km=0.5, cost_per_unit=0.10, transit_time_hours=0.5, capacity=50000),
            EdgeCreate(source="SUP-01", target="WH-02", distance_km=1300, cost_per_unit=3.50, transit_time_hours=14, capacity=20000),
            EdgeCreate(source="SUP-02", target="WH-02", distance_km=1.0, cost_per_unit=0.12, transit_time_hours=1, capacity=35000),
            EdgeCreate(source="SUP-02", target="WH-01", distance_km=1500, cost_per_unit=4.20, transit_time_hours=16, capacity=15000),
            EdgeCreate(source="SUP-03", target="WH-03", distance_km=2.0, cost_per_unit=0.15, transit_time_hours=1, capacity=20000),
            EdgeCreate(source="SUP-03", target="WH-01", distance_km=2800, cost_per_unit=7.80, transit_time_hours=28, capacity=10000),
            # --- Overseas suppliers to warehouses (via sea/air) ---
            EdgeCreate(
                source="SUP-04", target="WH-03", distance_km=11000,
                cost_per_unit=5.20, transit_time_hours=720, capacity=80000,
                reliability=0.85,
            ),
            EdgeCreate(
                source="SUP-05", target="WH-01", distance_km=8000,
                cost_per_unit=6.80, transit_time_hours=500, capacity=30000,
                reliability=0.90,
            ),
            # --- Inter-warehouse transfers ---
            EdgeCreate(source="WH-01", target="WH-02", distance_km=1300, cost_per_unit=2.00, transit_time_hours=14, capacity=10000),
            EdgeCreate(source="WH-02", target="WH-03", distance_km=1500, cost_per_unit=2.30, transit_time_hours=16, capacity=10000),
            EdgeCreate(source="WH-03", target="WH-01", distance_km=2800, cost_per_unit=3.50, transit_time_hours=28, capacity=8000),
            # --- Warehouses to customers ---
            EdgeCreate(source="WH-01", target="CUS-01", distance_km=1200, cost_per_unit=3.00, transit_time_hours=12, capacity=5000),
            EdgeCreate(source="WH-01", target="CUS-03", distance_km=300, cost_per_unit=0.80, transit_time_hours=3, capacity=5000),
            EdgeCreate(source="WH-01", target="CUS-05", distance_km=1600, cost_per_unit=4.00, transit_time_hours=16, capacity=3000),
            EdgeCreate(source="WH-01", target="CUS-06", distance_km=500, cost_per_unit=1.30, transit_time_hours=5, capacity=3000),
            EdgeCreate(source="WH-02", target="CUS-02", distance_km=1100, cost_per_unit=2.80, transit_time_hours=11, capacity=6000),
            EdgeCreate(source="WH-02", target="CUS-06", distance_km=800, cost_per_unit=2.00, transit_time_hours=8, capacity=4000),
            EdgeCreate(source="WH-02", target="CUS-07", distance_km=1000, cost_per_unit=2.50, transit_time_hours=10, capacity=3000),
            EdgeCreate(source="WH-02", target="CUS-09", distance_km=1200, cost_per_unit=3.00, transit_time_hours=12, capacity=4000),
            EdgeCreate(source="WH-02", target="CUS-10", distance_km=1100, cost_per_unit=2.70, transit_time_hours=11, capacity=3000),
            EdgeCreate(source="WH-03", target="CUS-04", distance_km=10, cost_per_unit=0.10, transit_time_hours=0.2, capacity=6000),
            EdgeCreate(source="WH-03", target="CUS-07", distance_km=1600, cost_per_unit=4.20, transit_time_hours=16, capacity=3000),
            EdgeCreate(source="WH-03", target="CUS-08", distance_km=1800, cost_per_unit=4.80, transit_time_hours=18, capacity=4000),
            EdgeCreate(source="WH-03", target="CUS-10", distance_km=600, cost_per_unit=1.50, transit_time_hours=6, capacity=3000),
        ],
    )

    builder = NetworkBuilder()
    network, _ = builder.build(request)
    return network

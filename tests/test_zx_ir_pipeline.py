from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import pytest
import stim

from spiderstate.stabilizer_graph import graph_state_stim_circuit
from spiderstate.zx_ir import (
    DiagramValidationError,
    EdgeKind,
    EdgeRole,
    FaultStatus,
    LemmaBStarError,
    NodeKind,
    NodeRole,
    ZXDiagram,
    apply_lemma_b_star,
    build_ideal_graph_state_diagram,
)


def test_ideal_graph_diagram_has_explicit_edge_and_boundary_semantics():
    graph = nx.path_graph(3)
    diagram = build_ideal_graph_state_diagram(
        graph,
        local_corrections={0: ("H",), 2: ("S_DAG", "H")},
    )

    assert len(diagram.nodes_of_kind(NodeKind.Z_SPIDER)) == 3
    assert len(diagram.nodes_of_kind(NodeKind.BOUNDARY)) == 3
    assert len(diagram.nodes_of_kind(NodeKind.LOCAL_CLIFFORD)) == 2

    graph_edges = diagram.edges_of_role(EdgeRole.GRAPH_EDGE)
    assert len(graph_edges) == 2
    assert {
        (data["kind"], data["fault_status"])
        for _, _, data in graph_edges
    } == {(EdgeKind.HADAMARD, FaultStatus.IDEAL)}

    boundary_edges = diagram.edges_of_role(EdgeRole.BOUNDARY_EDGE)
    assert len(boundary_edges) == 5
    assert {
        (data["kind"], data["fault_status"])
        for _, _, data in boundary_edges
    } == {(EdgeKind.SIMPLE, FaultStatus.NOISY)}

    diagram.validate()


def test_lemma_b_star_builds_four_x_spiders_and_a_noisy_k22():
    ideal = build_ideal_graph_state_diagram(nx.path_graph(2))
    rewritten = apply_lemma_b_star(ideal)

    assert ideal.graph.number_of_nodes() == 4
    assert ideal.graph.number_of_edges() == 3
    assert len(rewritten.nodes_of_kind(NodeKind.X_SPIDER)) == 4
    assert not rewritten.edges_of_role(EdgeRole.GRAPH_EDGE)

    x_nodes = rewritten.nodes_of_kind(NodeKind.X_SPIDER)
    assert all(rewritten.graph.degree(node) == 3 for node in x_nodes)
    assert all(
        rewritten.graph.nodes[node]["phase"] == 0
        and rewritten.graph.nodes[node]["role"] is NodeRole.LEMMA_B_STAR
        for node in x_nodes
    )

    lemma_edges = rewritten.edges_of_role(EdgeRole.LEMMA_B_STAR)
    simple_edges = [
        (source, target, data)
        for source, target, data in lemma_edges
        if data["kind"] is EdgeKind.SIMPLE
    ]
    hadamard_edges = [
        (source, target, data)
        for source, target, data in lemma_edges
        if data["kind"] is EdgeKind.HADAMARD
    ]
    assert len(simple_edges) == 4
    assert len(hadamard_edges) == 4
    assert all(
        data["fault_status"] is FaultStatus.NOISY
        for _, _, data in lemma_edges
    )

    left_pair = [
        node for node in x_nodes if ":left:" in node
    ]
    right_pair = [
        node for node in x_nodes if ":right:" in node
    ]
    assert {
        frozenset((source, target))
        for source, target, _ in hadamard_edges
    } == {
        frozenset((left, right))
        for left in left_pair
        for right in right_pair
    }


def test_c5_rewrite_has_twenty_trivalent_x_spiders():
    ideal = build_ideal_graph_state_diagram(nx.cycle_graph(5))
    rewritten = apply_lemma_b_star(ideal)

    assert len(rewritten.nodes_of_kind(NodeKind.X_SPIDER)) == 20
    assert all(
        rewritten.spider_arity(node) == 3
        for node in rewritten.nodes_of_kind(NodeKind.X_SPIDER)
    )
    assert all(
        rewritten.spider_arity(node) == 5
        for node in rewritten.nodes_of_kind(NodeKind.Z_SPIDER)
    )
    assert rewritten.graph.number_of_edges() == 45
    assert all(
        data["fault_status"] is FaultStatus.NOISY
        for _, _, data in rewritten.internal_edges()
    )


def test_lemma_b_star_is_non_mutating_and_idempotent():
    ideal = build_ideal_graph_state_diagram(nx.cycle_graph(4))
    before = ideal.to_json()
    rewritten = apply_lemma_b_star(ideal)
    rewritten_again = apply_lemma_b_star(rewritten)

    assert ideal.to_json() == before
    assert rewritten_again.to_json() == rewritten.to_json()
    assert rewritten_again is not rewritten
    assert rewritten.metadata["lemma_b_star_sources"] == [
        f"graph-edge:{index:06d}" for index in range(4)
    ]


def test_boundary_and_local_clifford_boxes_survive_rewrite_unchanged():
    ideal = build_ideal_graph_state_diagram(
        nx.path_graph(2),
        local_corrections={0: ("SQRT_X_DAG", "S"), 1: ("H",)},
    )
    expected_nodes = {
        node: dict(ideal.graph.nodes[node])
        for node in (
            ideal.nodes_of_kind(NodeKind.BOUNDARY)
            + ideal.nodes_of_kind(NodeKind.LOCAL_CLIFFORD)
        )
    }
    expected_edges = {
        data["edge_id"]: (source, target, dict(data))
        for source, target, data in ideal.edges_of_role(
            EdgeRole.BOUNDARY_EDGE
        )
    }

    rewritten = apply_lemma_b_star(ideal)

    assert {
        node: dict(rewritten.graph.nodes[node])
        for node in expected_nodes
    } == expected_nodes
    actual_edges = {
        data["edge_id"]: (source, target, dict(data))
        for source, target, data in rewritten.edges_of_role(
            EdgeRole.BOUNDARY_EDGE
        )
    }
    assert actual_edges == expected_edges


def test_lemma_requires_an_additional_leg_at_both_endpoints():
    diagram = ZXDiagram()
    for node in ("u", "v"):
        diagram.add_node(
            node,
            kind=NodeKind.Z_SPIDER,
            role=NodeRole.GRAPH_VERTEX,
            phase=0,
            provenance={"source": node},
        )
    diagram.add_edge(
        "u",
        "v",
        kind=EdgeKind.HADAMARD,
        fault_status=FaultStatus.IDEAL,
        role=EdgeRole.GRAPH_EDGE,
        provenance={"source": "uv"},
        edge_id="graph-edge:000000",
    )

    with pytest.raises(LemmaBStarError, match="at least one leg"):
        apply_lemma_b_star(diagram)


def test_validation_rejects_nonzero_spider_phases():
    diagram = ZXDiagram()
    diagram.add_node(
        "z",
        kind=NodeKind.Z_SPIDER,
        role=NodeRole.GRAPH_VERTEX,
        phase=1,
        provenance={"source": "test"},
    )

    with pytest.raises(DiagramValidationError, match="nonzero phase"):
        diagram.validate()


def test_serialization_and_svg_are_deterministic_across_insertion_orders():
    first_graph = nx.Graph()
    first_graph.add_nodes_from(["b", "a", "c"])
    first_graph.add_edges_from([("b", "c"), ("a", "b")])
    second_graph = nx.Graph()
    second_graph.add_nodes_from(["c", "b", "a"])
    second_graph.add_edges_from([("b", "a"), ("c", "b")])

    first = apply_lemma_b_star(build_ideal_graph_state_diagram(first_graph))
    second = apply_lemma_b_star(build_ideal_graph_state_diagram(second_graph))

    assert first.to_json() == second.to_json()
    assert first.render_svg() == second.render_svg()
    assert 'data-kind="hadamard"' in first.render_svg()
    assert 'data-fault-status="noisy"' in first.render_svg()

    restored = ZXDiagram.from_dict(json.loads(first.to_json()))
    assert restored.to_json() == first.to_json()


def test_pyzx_adapter_preserves_lemma_rewrite_semantics_up_to_scalar():
    pyzx = pytest.importorskip("pyzx")
    ideal = build_ideal_graph_state_diagram(nx.path_graph(3))
    rewritten = apply_lemma_b_star(ideal)

    ideal_pyzx = ideal.to_pyzx()
    rewritten_pyzx = rewritten.to_pyzx()

    assert len(ideal_pyzx.outputs()) == 3
    assert len(rewritten_pyzx.outputs()) == 3
    assert pyzx.compare_tensors(
        ideal_pyzx,
        rewritten_pyzx,
        preserve_scalar=False,
    )


def test_pyzx_adapter_expands_supported_local_clifford_words():
    pyzx = pytest.importorskip("pyzx")
    diagram = build_ideal_graph_state_diagram(
        nx.empty_graph(1),
        local_corrections={0: ("H", "S_DAG", "SQRT_X", "Y")},
    )

    graph = diagram.to_pyzx()

    assert len(graph.outputs()) == 1
    assert graph.num_vertices() > diagram.graph.number_of_nodes()
    assert graph.types()[graph.outputs()[0]] is pyzx.VertexType.BOUNDARY


def test_numeric_qubit_order_does_not_become_lexicographic():
    graph = nx.path_graph(11)
    diagram = build_ideal_graph_state_diagram(graph)

    assert diagram.metadata["qubit_order"] == list(range(11))
    expected = stim.Tableau.from_circuit(
        graph_state_stim_circuit(graph)
    ).to_state_vector(endian="big")
    actual = np.asarray(diagram.to_pyzx().to_tensor()).reshape(-1)
    support = np.flatnonzero(np.abs(expected) > 1e-8)
    ratio = actual[support[0]] / expected[support[0]]
    np.testing.assert_allclose(actual, ratio * expected, atol=1e-8)


def test_validation_normalizes_serialized_enum_values():
    graph = nx.Graph()
    graph.add_node(
        "z",
        kind="z_spider",
        role="graph_vertex",
        phase=0,
        provenance={"source": "test"},
    )
    graph.add_node(
        "out",
        kind="boundary",
        role="boundary",
        phase=0,
        provenance={"source": "test"},
    )
    graph.add_edge(
        "z",
        "out",
        edge_id="boundary:test",
        kind="simple",
        fault_status="noisy",
        role="boundary_edge",
        provenance={"source": "test"},
    )
    diagram = ZXDiagram(graph)

    diagram.validate()

    assert diagram.nodes_of_kind(NodeKind.Z_SPIDER) == ["z"]
    assert diagram.graph.nodes["z"]["kind"] is NodeKind.Z_SPIDER
    assert diagram.graph.edges["z", "out"]["kind"] is EdgeKind.SIMPLE
    assert 'data-kind="z_spider"' in diagram.render_svg()


def test_svg_escapes_quotes_in_arbitrary_stable_ids():
    diagram = ZXDiagram()
    diagram.add_node(
        'z"quoted',
        kind=NodeKind.Z_SPIDER,
        role=NodeRole.GRAPH_VERTEX,
        phase=0,
        provenance={"source": "test"},
    )
    diagram.add_node(
        'out"quoted',
        kind=NodeKind.BOUNDARY,
        role=NodeRole.BOUNDARY,
        phase=0,
        provenance={"source": "test"},
    )
    diagram.add_edge(
        'z"quoted',
        'out"quoted',
        edge_id='edge"quoted',
        kind=EdgeKind.SIMPLE,
        fault_status=FaultStatus.NOISY,
        role=EdgeRole.BOUNDARY_EDGE,
        provenance={"source": "test"},
    )

    root = ET.fromstring(diagram.render_svg())

    assert root.tag.endswith("svg")

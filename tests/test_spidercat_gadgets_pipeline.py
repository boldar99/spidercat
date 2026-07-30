from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from spiderstate.spidercat_gadgets import (
    SpiderCatGadgetUnavailable,
    UnsupportedFaultToleranceError,
    clear_spidercat_gadget_cache,
    decompose_spidercats,
    find_t_robustness_violation,
    predicted_spidercat_spider_count,
    verified_spidercat_gadget,
    verify_t_robustness,
)
from spiderstate.zx_ir import (
    EdgeKind,
    EdgeRole,
    FaultStatus,
    NodeKind,
    NodeRole,
    ZXDiagram,
    apply_lemma_b_star,
    build_ideal_graph_state_diagram,
)


@pytest.mark.parametrize("bad_t", [0, 8, -1, True, 1.5])
def test_rejects_fault_levels_outside_verified_range(bad_t):
    with pytest.raises(UnsupportedFaultToleranceError, match="1..7"):
        verified_spidercat_gadget(5, bad_t)


@pytest.mark.parametrize("arity", [4, 5])
@pytest.mark.parametrize("t", [1, 7])
def test_small_cycle_gadgets_are_exactly_verified(arity, t):
    gadget = verified_spidercat_gadget(arity, t)

    assert gadget.construction == f"verified-cycle-{arity}"
    assert gadget.arity == arity
    assert gadget.requested_t == t
    assert len(gadget.attachment_nodes) == arity
    assert set(gadget.attachment_nodes) == set(gadget.graph)
    assert nx.is_isomorphic(gadget.graph, nx.cycle_graph(arity))
    assert max(dict(gadget.graph.degree()).values()) == 2
    assert verify_t_robustness(gadget.graph, t)
    assert predicted_spidercat_spider_count(arity, t) == arity


def test_exact_verifier_reports_actual_cut_size_not_global_t():
    # The middle edge is a size-one cut with two marked ports on each side.
    bad = nx.path_graph(6)
    nx.set_node_attributes(
        bad,
        {0: True, 1: True, 4: True, 5: True},
        "is_mark",
    )

    violation = find_t_robustness_violation(bad, 7)

    assert violation is not None
    assert violation.cut_size == 1
    assert violation.marks_on_side == 2
    assert violation.marks_on_complement == 2
    assert len(violation.cut_edges) == 1
    assert not verify_t_robustness(bad, 7)


def test_repository_expanded_gadget_uses_only_marked_ports_and_is_robust():
    gadget = verified_spidercat_gadget(7, 7)

    assert gadget.arity == 7
    assert gadget.effective_t == 2
    assert gadget.construction == "repository-expanded-marked-graph"
    assert len(gadget.attachment_nodes) == 7
    assert max(dict(gadget.graph.degree()).values()) <= 3
    assert all(gadget.graph.degree(node) == 2 for node in gadget.attachment_nodes)
    assert verify_t_robustness(gadget.graph, 7)


def test_unverified_repository_candidate_fails_typed(monkeypatch):
    bad = nx.path_graph(6)
    nx.set_node_attributes(bad, True, "is_mark")

    monkeypatch.setattr(
        "spiderstate.spidercat_gadgets._load_repository_gadget",
        lambda arity, effective_t: bad.copy(),
    )
    clear_spidercat_gadget_cache()

    with pytest.raises(SpiderCatGadgetUnavailable, match="exact marked-cut"):
        verified_spidercat_gadget(6, 1)

    clear_spidercat_gadget_cache()


def _boundary_decorations(diagram):
    return {
        node: deepcopy(data)
        for node, data in diagram.graph.nodes(data=True)
        if data["kind"] in {NodeKind.BOUNDARY, NodeKind.LOCAL_CLIFFORD}
    }


def _edge_data_by_id(diagram):
    return {
        data["edge_id"]: deepcopy(data)
        for _, _, data in diagram.graph.edges(data=True)
    }


def test_c5_post_lemma_decomposes_to_45_trivalent_spiders_deterministically():
    graph = nx.cycle_graph(5)
    ideal = build_ideal_graph_state_diagram(
        graph,
        local_corrections={0: ("H",)},
    )
    post_lemma = apply_lemma_b_star(ideal)
    boundary_before = _boundary_decorations(post_lemma)
    original_edges = _edge_data_by_id(post_lemma)

    final, metadata = decompose_spidercats(post_lemma, t=1)
    repeated, repeated_metadata = decompose_spidercats(post_lemma, t=1)

    assert final.to_json() == repeated.to_json()
    assert metadata == repeated_metadata
    assert len(metadata.replacements) == 5
    assert all(replacement.arity == 5 for replacement in metadata.replacements)
    assert all(
        replacement.construction == "verified-cycle-5"
        for replacement in metadata.replacements
    )
    assert all(len(replacement.ports) == 5 for replacement in metadata.replacements)
    assert _boundary_decorations(final) == boundary_before

    z_spiders = [
        node
        for node, data in final.graph.nodes(data=True)
        if data["kind"] is NodeKind.Z_SPIDER
    ]
    x_spiders = [
        node
        for node, data in final.graph.nodes(data=True)
        if data["kind"] is NodeKind.X_SPIDER
    ]
    assert len(z_spiders) == 25
    assert len(x_spiders) == 20
    assert len(z_spiders) + len(x_spiders) == 45
    assert max(final.graph.degree(node) for node in z_spiders + x_spiders) == 3
    assert all(
        data["fault_status"] is FaultStatus.NOISY
        for _, _, data in final.internal_edges()
    )

    final_edges = _edge_data_by_id(final)
    for edge_id, source_data in original_edges.items():
        assert edge_id in final_edges
        assert final_edges[edge_id] == source_data
    assert all(
        data.get("provenance") is not None
        for _, data in final.graph.nodes(data=True)
    )
    assert all(
        data.get("provenance") is not None
        for _, _, data in final.graph.edges(data=True)
    )
    assert final.metadata["stage"] == "unidealized_trivalent"
    assert final.metadata["spidercat_requested_t"] == 1
    for entry in final.metadata["graph_vertex_nodes"]:
        assert "node_id" not in entry
        assert entry["source_node_id"].startswith("z:q")
        assert set(entry["gadget_node_ids"]) <= set(final.graph)
        assert set(entry["gadget_attachment_nodes"]) <= set(final.graph)
    final.validate()


def test_high_arity_x_spider_keeps_color_phase_and_port_semantics():
    diagram = ZXDiagram()
    diagram.add_node(
        "x",
        kind=NodeKind.X_SPIDER,
        role=NodeRole.LEMMA_B_STAR,
        phase=0,
        provenance={"source": "test-x"},
    )
    for index in range(4):
        boundary = f"out:{index}"
        diagram.add_node(
            boundary,
            kind=NodeKind.BOUNDARY,
            role=NodeRole.BOUNDARY,
            phase=0,
            provenance={"source": "test-port", "index": index},
        )
        diagram.add_edge(
            "x",
            boundary,
            kind=EdgeKind.SIMPLE,
            fault_status=FaultStatus.NOISY,
            role=EdgeRole.BOUNDARY_EDGE,
            provenance={"source": "test-edge", "index": index},
            edge_id=f"test-edge:{index}",
        )
    diagram.validate()

    final, metadata = decompose_spidercats(diagram, t=3)

    spiders = [
        (node, data)
        for node, data in final.graph.nodes(data=True)
        if data["kind"] in {NodeKind.X_SPIDER, NodeKind.Z_SPIDER}
    ]
    assert len(spiders) == 4
    assert all(data["kind"] is NodeKind.X_SPIDER for _, data in spiders)
    assert all(data["phase"] == 0 for _, data in spiders)
    assert all(final.graph.degree(node) == 3 for node, _ in spiders)
    assert [port.port_index for port in metadata.replacements[0].ports] == [
        0,
        1,
        2,
        3,
    ]
    assert {
        data["edge_id"]
        for _, _, data in final.graph.edges(data=True)
        if data["role"] is EdgeRole.BOUNDARY_EDGE
    } == {f"test-edge:{index}" for index in range(4)}
    final.validate()


def test_no_op_decomposition_still_marks_the_final_stage():
    ideal = build_ideal_graph_state_diagram(nx.empty_graph(1))
    post_lemma = apply_lemma_b_star(ideal)

    final, metadata = decompose_spidercats(post_lemma, t=7)

    assert not metadata.replacements
    assert final.metadata["stage"] == "unidealized_trivalent"
    assert final.metadata["spidercat_requested_t"] == 7
    assert final.metadata["graph_vertex_nodes"][0]["node_id"] == "z:q0"

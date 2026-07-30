"""End-to-end tests for the staged stabilizer-state synthesis API."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pytest
import stim

from spiderstate.spidercat_gadgets import UnsupportedFaultToleranceError
from spiderstate.stabilizer_synthesis import (
    SynthesisStage,
    synthesize_stabilizer_state,
)
from spiderstate.zx_ir import EdgeRole, FaultStatus, NodeKind


FIVE_QUBIT_LOGICAL_ZERO = (
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
    "ZZZZZ",
)


def _assert_vectors_equal_up_to_scalar(actual, expected) -> None:
    actual = np.asarray(actual).reshape(-1)
    expected = np.asarray(expected).reshape(-1)
    support = np.flatnonzero(np.abs(expected) > 1e-8)
    assert len(support)
    ratio = actual[support[0]] / expected[support[0]]
    assert abs(ratio) > 1e-10
    np.testing.assert_allclose(actual, ratio * expected, atol=1e-8)


@pytest.fixture(scope="module")
def five_qubit_result():
    return synthesize_stabilizer_state(FIVE_QUBIT_LOGICAL_ZERO, t=1)


def test_five_qubit_pipeline_returns_c5_and_45_trivalent_spiders(
    five_qubit_result,
):
    result = five_qubit_result

    assert nx.is_isomorphic(result.graph, nx.cycle_graph(5))
    assert result.search_metadata.guaranteed_optimal
    assert result.search_metadata.score[:3] == (45, 2, 5)
    assert result.graph_conversion.validate_certificate()

    counts = Counter(
        data["kind"] for _, data in result.final_diagram.graph.nodes(data=True)
    )
    assert counts[NodeKind.Z_SPIDER] == 25
    assert counts[NodeKind.X_SPIDER] == 20
    assert counts[NodeKind.Z_SPIDER] + counts[NodeKind.X_SPIDER] == 45
    assert max(
        result.final_diagram.graph.degree(node)
        for node, data in result.final_diagram.graph.nodes(data=True)
        if data["kind"] in (NodeKind.Z_SPIDER, NodeKind.X_SPIDER)
    ) == 3

    assert len(result.gadget_metadata.replacements) == 5
    assert set(result.source_to_gadget_ports) == {
        f"z:q{qubit}" for qubit in range(5)
    }
    assert all(
        len(ports) == 5 for ports in result.source_to_gadget_ports.values()
    )


def test_every_final_internal_edge_is_noisy_and_has_provenance(
    five_qubit_result,
):
    diagram = five_qubit_result.final_diagram

    assert not diagram.edges_of_role(EdgeRole.GRAPH_EDGE)
    assert all(
        data["fault_status"] is FaultStatus.NOISY
        for _, _, data in diagram.internal_edges()
    )
    assert all(
        data.get("provenance") is not None
        for _, data in diagram.graph.nodes(data=True)
    )
    assert all(
        data.get("provenance") is not None
        for _, _, data in diagram.graph.edges(data=True)
    )


def test_all_stages_denote_the_exact_five_qubit_input_state(
    five_qubit_result,
):
    expected = stim.Tableau.from_stabilizers(
        [stim.PauliString(pauli) for pauli in FIVE_QUBIT_LOGICAL_ZERO]
    ).to_state_vector(endian="big")

    for stage in SynthesisStage:
        _assert_vectors_equal_up_to_scalar(
            five_qubit_result.to_pyzx(stage).to_tensor(),
            expected,
        )


@pytest.mark.parametrize("stabilizer", ["+Z", "+Y", "-Y"])
def test_lc_boundary_boxes_prepare_signed_single_qubit_states(stabilizer):
    result = synthesize_stabilizer_state([stabilizer], t=7)
    expected = stim.Tableau.from_stabilizers(
        [stim.PauliString(stabilizer)]
    ).to_state_vector(endian="big")

    _assert_vectors_equal_up_to_scalar(
        result.to_pyzx(stage="unidealized").to_tensor(),
        expected,
    )
    assert result.guarantee_metadata.requested_t == 7
    assert result.guarantee_metadata.status == "verified"
    assert not result.guarantee_metadata.theoretical_optimality_claimed


def test_result_serialization_and_svg_are_deterministic(
    five_qubit_result,
    tmp_path,
):
    repeated = synthesize_stabilizer_state(FIVE_QUBIT_LOGICAL_ZERO, t=1)

    assert (
        five_qubit_result.final_diagram.to_json()
        == repeated.final_diagram.to_json()
    )
    first_svg = five_qubit_result.render_svg(stage="final")
    output = tmp_path / "five-qubit-final.svg"
    second_svg = repeated.render_svg(stage="trivalent", path=output)
    assert first_svg == second_svg
    assert output.read_text(encoding="utf-8") == first_svg
    assert 'data-fault-status="noisy"' in first_svg


@pytest.mark.parametrize("bad_t", [0, 8, -1, True, 1.5])
def test_public_api_rejects_unsupported_fault_levels(bad_t):
    with pytest.raises(UnsupportedFaultToleranceError, match="1 through 7"):
        synthesize_stabilizer_state(["Z"], t=bad_t)


def test_stage_selection_has_clear_aliases_and_errors(five_qubit_result):
    assert (
        five_qubit_result.diagram("post-lemma-b-star")
        is five_qubit_result.lemma_b_star_diagram
    )
    assert (
        five_qubit_result.post_lemma_b_star_diagram
        is five_qubit_result.lemma_b_star_diagram
    )
    assert (
        five_qubit_result.unidealized_diagram
        is five_qubit_result.final_diagram
    )
    with pytest.raises(ValueError, match="Unknown synthesis stage"):
        five_qubit_result.diagram("not-a-stage")

"""Focused tests for exact stabilizer-to-LC-graph conversion."""

from __future__ import annotations

import networkx as nx
import pytest
import stim

from spiderstate.stabilizer_graph import (
    DependentGeneratorsError,
    InconsistentStabilizerError,
    InvalidPauliStringError,
    LCSearchConfig,
    LocalClifford,
    NonCommutingGeneratorsError,
    UnderconstrainedStabilizerError,
    css_logical_state_stabilizers,
    stabilizer_state_to_graph,
)


def _certificate_signature(result) -> tuple[tuple[str, ...], ...]:
    return tuple(clifford.gate_word for clifford in result.local_cliffords_to_graph)


@pytest.mark.parametrize(
    ("stabilizers", "expected_edges"),
    [
        (["Z"], 0),
        (["XX", "ZZ"], 1),
        (["XXX", "ZZI", "IZZ"], 2),
    ],
)
def test_product_bell_and_ghz_states(stabilizers, expected_edges):
    result = stabilizer_state_to_graph(stabilizers)

    assert result.graph.number_of_edges() == expected_edges
    assert result.validate_certificate()
    assert len(result.local_cliffords_to_graph) == len(stabilizers[0])


def test_signed_y_generator_tracks_phase_exactly():
    result = stabilizer_state_to_graph(["-Y"])

    assert result.graph.number_of_nodes() == 1
    assert result.graph.number_of_edges() == 0
    assert result.validate_certificate()

    transformed = stim.PauliString("-Y").after(result.certificate_stim_circuit())
    assert transformed == stim.PauliString("+X")


def test_tableau_adapter_uses_z_output_stabilizers():
    tableau = stim.Tableau.from_circuit(stim.Circuit("H 0\nS 0"))

    result = stabilizer_state_to_graph(tableau)

    assert result.input_stabilizers == ("+Y",)
    assert result.validate_certificate()


def test_local_clifford_words_compose_in_application_order_and_invert():
    first = LocalClifford.from_gate_word(["S", "H"])
    second = LocalClifford.from_gate_word(["H", "S_DAG"])
    composed = first.followed_by(second)

    circuit = stim.Circuit("S 0\nH 0\nH 0\nS_DAG 0")
    assert stim.PauliString("X").after(composed.to_stim_circuit()) == (
        stim.PauliString("X").after(circuit)
    )
    assert composed.followed_by(composed.inverse()) == LocalClifford.identity()


@pytest.mark.parametrize(
    ("stabilizers", "error_type"),
    [
        (["X", "Z"], NonCommutingGeneratorsError),
        (["ZZ"], UnderconstrainedStabilizerError),
        (["ZZ", "ZZ"], DependentGeneratorsError),
        (["XX", "-XX"], InconsistentStabilizerError),
        (["XI", "Z"], InvalidPauliStringError),
        (["AX"], InvalidPauliStringError),
    ],
)
def test_rejects_invalid_generator_sets(stabilizers, error_type):
    with pytest.raises(error_type):
        stabilizer_state_to_graph(stabilizers)


def test_rejects_nonhermitian_stim_pauli():
    generator = stim.PauliString("X")
    generator.sign = 1j

    with pytest.raises(InvalidPauliStringError, match="non-Hermitian"):
        stabilizer_state_to_graph([generator])


def test_five_qubit_logical_zero_optimizes_to_c5():
    logical_zero = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ", "ZZZZZ"]

    result = stabilizer_state_to_graph(logical_zero)

    assert nx.is_isomorphic(result.graph, nx.cycle_graph(5))
    assert result.search.guaranteed_optimal
    assert result.search.score[:3] == (45, 2, 5)
    assert result.validate_certificate()


def test_search_is_deterministic():
    stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ", "ZZZZZ"]

    first = stabilizer_state_to_graph(stabilizers)
    second = stabilizer_state_to_graph(stabilizers)

    assert sorted(first.graph.edges()) == sorted(second.graph.edges())
    assert _certificate_signature(first) == _certificate_signature(second)
    assert first.search == second.search


def test_search_uses_supplied_spidercat_arity_cost():
    seen_arities: list[int] = []

    def cost(arity: int) -> int:
        seen_arities.append(arity)
        return 10 * arity

    result = stabilizer_state_to_graph(
        ["XX", "ZZ"],
        lc_search=LCSearchConfig(vertex_arity_cost=cost),
    )

    assert seen_arities
    assert result.search.score[0] == 4 + 2 * 30


def test_css_helper_completes_logical_zero_and_plus_states():
    zero_generators = css_logical_state_stabilizers(
        h_x=[],
        h_z=["110", "011"],
        logical_z=["100"],
        state="0",
    )
    plus_generators = css_logical_state_stabilizers(
        h_x=[],
        h_z=["110", "011"],
        logical_x=["111"],
        state="+",
    )

    assert zero_generators == ("+ZZI", "+IZZ", "+ZII")
    assert plus_generators == ("+ZZI", "+IZZ", "+XXX")
    assert stabilizer_state_to_graph(zero_generators).validate_certificate()
    assert stabilizer_state_to_graph(plus_generators).validate_certificate()


def test_css_helper_supports_signed_logical_eigenstates():
    generators = css_logical_state_stabilizers(
        h_x=[],
        h_z=["110", "011"],
        logical_z=["100"],
        state="0",
        eigenvalues=-1,
    )

    assert generators[-1] == "-ZII"
    assert stabilizer_state_to_graph(generators).validate_certificate()


def test_random_tableau_direct_conversion_regression():
    # This is intentionally a small fixed batch; certificate validation uses a
    # second implementation inside Stim for every generated state.
    for num_qubits in range(1, 7):
        for _ in range(5):
            tableau = stim.Tableau.random(num_qubits)
            result = stabilizer_state_to_graph(
                tableau,
                lc_search=LCSearchConfig(optimize=False),
            )
            assert result.validate_certificate()

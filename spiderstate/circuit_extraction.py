"""Circuit extraction from a stabiliser-state decomposition.

The extraction backend knows how to lower the graph IR to a circuit, but it
does not know how the parity matrix was decomposed.  This is the deliberate
module boundary with ``spiderstate.stabiliser_decomposition``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import stim

from spidercat.circuit_extraction import CatStateExtractor, CircuitBuilder, StimBuilder
from spiderstate.stabiliser_decomposition import StabiliserStateDecomposition


ResourceMetric = Literal[
    "two_qubit_gates",
    "ancilla_qubits",
    "tick_depth",
    "measurements",
    "space_time_volume",
]
_RESOURCE_METRICS = {
    "two_qubit_gates",
    "ancilla_qubits",
    "tick_depth",
    "measurements",
    "space_time_volume",
}
_TWO_QUBIT_GATES = {"CX", "CNOT", "CZ", "SWAP", "CY", "XCZ", "YCX"}
_PHYSICAL_OPERATIONS = {
    "CX",
    "CNOT",
    "CZ",
    "SWAP",
    "CY",
    "XCZ",
    "YCX",
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "S_DAG",
    "R",
    "RX",
    "RY",
    "M",
    "MX",
    "MY",
    "MZ",
    "MR",
}


@dataclass(frozen=True)
class ExtractionPolicy:
    """Lexicographic resource objective for comparing decomposition candidates."""

    objective_order: tuple[ResourceMetric, ...] = (
        "two_qubit_gates",
        "ancilla_qubits",
        "tick_depth",
        "measurements",
    )
    verbose: bool = False
    strict_target_alignment: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.objective_order) - _RESOURCE_METRICS
        if unknown:
            raise ValueError(f"unknown extraction resource metrics: {sorted(unknown)}")
        if len(set(self.objective_order)) != len(self.objective_order):
            raise ValueError("objective_order must not contain duplicate metrics")


@dataclass(frozen=True)
class CircuitResources:
    """Measured resources for one extracted circuit."""

    data_qubits: int
    total_qubits: int
    ancilla_qubits: int
    two_qubit_gates: int
    measurements: int
    detectors: int
    tick_depth: int
    dependency_depth: int
    space_time_volume: int


@dataclass(frozen=True)
class ExtractionResult:
    """Circuit plus the manifest needed to compare and verify it."""

    circuit: stim.Circuit
    resources: CircuitResources
    candidate_id: str
    decomposition_schema_version: int
    target_parity_matrix: tuple[tuple[int, ...], ...]
    distance: int
    fault_budget: int


def _physical_two_qubit_gate_count(circuit: stim.Circuit) -> int:
    count = 0
    for instruction in circuit.flattened():
        if instruction.name not in _TWO_QUBIT_GATES:
            continue
        targets = instruction.targets_copy()
        for index in range(0, len(targets), 2):
            pair = targets[index:index + 2]
            if len(pair) == 2 and all(target.is_qubit_target for target in pair):
                count += 1
    return count


def _tick_depth(circuit: stim.Circuit) -> int:
    """Count non-empty scheduled blocks separated by TICK instructions."""

    depth = 0
    block_has_physical_operation = False
    for instruction in circuit.flattened():
        if instruction.name == "TICK":
            if block_has_physical_operation:
                depth += 1
                block_has_physical_operation = False
        elif instruction.name in _PHYSICAL_OPERATIONS:
            block_has_physical_operation = True
    return depth + int(block_has_physical_operation)


def measure_circuit_resources(
    circuit: stim.Circuit,
    decomposition: StabiliserStateDecomposition,
) -> CircuitResources:
    """Measure extraction resources without running fault verification."""

    total_qubits = int(circuit.num_qubits)
    data_qubits = decomposition.num_data_qubits
    ancilla_qubits = max(0, total_qubits - data_qubits)
    tick_depth = _tick_depth(circuit)
    return CircuitResources(
        data_qubits=data_qubits,
        total_qubits=total_qubits,
        ancilla_qubits=ancilla_qubits,
        two_qubit_gates=_physical_two_qubit_gate_count(circuit),
        measurements=int(circuit.num_measurements),
        detectors=int(circuit.num_detectors),
        tick_depth=tick_depth,
        dependency_depth=decomposition.dependency_depth,
        space_time_volume=total_qubits * tick_depth,
    )


def extraction_score(
    result: ExtractionResult,
    policy: ExtractionPolicy,
) -> tuple[int | str, ...]:
    """Return a deterministic lexicographic score; lower is better."""

    resources = result.resources
    values = tuple(
        int(getattr(resources, metric))
        for metric in policy.objective_order
    )
    return (*values, result.candidate_id)


def extract_stabiliser_state(
    decomposition: StabiliserStateDecomposition,
    *,
    policy: ExtractionPolicy | None = None,
    builder: CircuitBuilder | None = None,
    extractor_factory: Callable[..., CatStateExtractor] = CatStateExtractor,
) -> ExtractionResult:
    """Lower one validated decomposition to a circuit and resource manifest."""

    policy = policy or ExtractionPolicy()
    decomposition.validate(
        strict_target_alignment=policy.strict_target_alignment,
    )
    builder = builder or StimBuilder()
    extractor = extractor_factory(builder, verbose=policy.verbose)
    graph, forest, roots, dependency_graph, primary_paths = (
        decomposition.extraction_inputs()
    )
    circuit = extractor.extract(
        graph,
        forest,
        roots,
        dependency_graph,
        primary_paths,
    )
    if not isinstance(circuit, stim.Circuit):
        # Custom builders are allowed, but resource reporting is a Stim contract.
        # This error is clearer than failing later on ``num_qubits``.
        raise TypeError("resource-aware extraction requires a Stim circuit")

    return ExtractionResult(
        circuit=circuit,
        resources=measure_circuit_resources(circuit, decomposition),
        candidate_id=decomposition.candidate_id,
        decomposition_schema_version=decomposition.schema_version,
        target_parity_matrix=tuple(
            tuple(int(value) for value in row)
            for row in decomposition.parity_matrix
        ),
        distance=decomposition.distance,
        fault_budget=decomposition.fault_budget,
    )


def extract_best_stabiliser_state(
    decompositions: Sequence[StabiliserStateDecomposition],
    *,
    policy: ExtractionPolicy | None = None,
    builder_factory: Callable[[], CircuitBuilder] = StimBuilder,
    extractor_factory: Callable[..., CatStateExtractor] = CatStateExtractor,
) -> ExtractionResult:
    """Compile a decomposition portfolio and keep the best measured circuit."""

    if not decompositions:
        raise ValueError("at least one decomposition candidate is required")
    policy = policy or ExtractionPolicy()
    best_result: ExtractionResult | None = None
    best_score: tuple[int | str, ...] | None = None
    for decomposition in decompositions:
        result = extract_stabiliser_state(
            decomposition,
            policy=policy,
            builder=builder_factory(),
            extractor_factory=extractor_factory,
        )
        score = extraction_score(result, policy)
        if best_score is None or score < best_score:
            best_result = result
            best_score = score
    assert best_result is not None
    return best_result

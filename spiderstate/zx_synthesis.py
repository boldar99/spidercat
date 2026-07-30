"""Fault-equivalent extraction of phase-free, cubic ZX diagrams to Stim.

The production pipeline in this module is deliberately separate from
``spidercat.circuit_extraction``.  SpiderCat's extractor is specialised to cat
states, while this module accepts a general phase-free PyZX diagram whose
internal spiders have degree three.

The implementation uses the following fixed rewrite catalogue.

* A degree-three Z spider is a three-qubit GHZ resource.
* A degree-three X spider is the colour-changed GHZ resource.
* A simple internal wire is a Bell contraction.
* A Hadamard internal wire is a Bell contraction with a Hadamard on one port.
* Random Bell outcomes are reconciled with Pauli feedback obtained from a
  stabilizer-tableau kickback.  Deterministic outcomes become detectors.

Only the choice of half-edge roles, colour-change frames, traversal, physical
qubit reuse, and schedule are optimized.  The rewrite catalogue itself is
shared by both optimization strategies.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from fractions import Fraction
import heapq
import random
from typing import Any, Iterable, Mapping, Sequence

import pyzx as zx
from pyzx.utils import EdgeType, VertexType
import stim

try:
    import z3
except ImportError:  # pragma: no cover - exercised by fallback tests.
    z3 = None


class SynthesisError(ValueError):
    """Raised when a ZX diagram cannot be synthesized by this pipeline."""


class SynthesisStrategy(str, Enum):
    """Supported extraction objectives."""

    GATE_COUNT = "gate_count"
    DEPTH = "depth"


class HalfEdgeRole(IntEnum):
    """Circuit role assigned to one incidence of a degree-three spider."""

    PAST = 0
    FUTURE = 1
    INTERACTION = 2


@dataclass(frozen=True)
class NormalizedEdge:
    """An edge in the immutable normalized representation."""

    index: int
    u: int
    v: int
    hadamard: bool

    def other(self, vertex: int) -> int:
        if vertex == self.u:
            return self.v
        if vertex == self.v:
            return self.u
        raise KeyError(f"Vertex {vertex} is not incident to edge {self.index}.")


@dataclass(frozen=True)
class NormalizedDiagram:
    """Validated, immutable subset of a PyZX graph used by the compiler."""

    original_vertices: tuple[Any, ...]
    vertex_types: tuple[VertexType, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    spiders: tuple[int, ...]
    edges: tuple[NormalizedEdge, ...]
    incident_edges: tuple[tuple[int, ...], ...]

    @property
    def original_to_normalized(self) -> dict[Any, int]:
        return {v: i for i, v in enumerate(self.original_vertices)}

    def edge(self, edge_index: int) -> NormalizedEdge:
        return self.edges[edge_index]

    def neighbors(self, vertex: int) -> tuple[int, ...]:
        return tuple(
            self.edges[e].other(vertex) for e in self.incident_edges[vertex]
        )

    def edge_between(self, u: int, v: int) -> NormalizedEdge:
        for edge_index in self.incident_edges[u]:
            edge = self.edges[edge_index]
            if edge.other(u) == v:
                return edge
        raise KeyError((u, v))


@dataclass(frozen=True)
class MeasurementInfo:
    """Meaning and feedback associated with one measurement record bit."""

    index: int
    qubit: int
    basis: str
    kind: str
    inverted: bool = False
    detector_index: int | None = None
    correction: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class DetectorInfo:
    """A deterministic parity emitted as a Stim ``DETECTOR``."""

    index: int
    measurement_indices: tuple[int, ...]


@dataclass(frozen=True)
class SynthesisMetrics:
    """Resource counts for a synthesized Stim circuit."""

    cx_count: int
    h_count: int
    preparation_count: int
    measurement_count: int
    feedback_count: int
    detector_count: int
    depth: int
    cx_depth: int
    num_qubits: int
    total_operations: int

    @property
    def gate_count_key(self) -> tuple[int, ...]:
        """Lexicographic score used by the gate-count strategy."""

        return (
            self.cx_count,
            self.h_count,
            self.preparation_count + self.measurement_count,
            self.num_qubits,
            self.depth,
        )

    @property
    def depth_key(self) -> tuple[int, ...]:
        """Lexicographic score used by the depth strategy."""

        return (
            self.depth,
            self.cx_depth,
            self.num_qubits,
            self.cx_count,
            self.h_count,
        )

    @property
    def peak_qubits(self) -> int:
        """Alias describing ``num_qubits`` as the peak allocated width."""

        return self.num_qubits


@dataclass(frozen=True)
class SynthesisResult:
    """Circuit and metadata returned by :func:`synthesize_zx`."""

    circuit: stim.Circuit
    input_qubits: Mapping[Any, int]
    output_qubits: Mapping[Any, int]
    metrics: SynthesisMetrics
    measurements: tuple[MeasurementInfo, ...]
    detectors: tuple[DetectorInfo, ...]
    half_edge_roles: Mapping[Any, Mapping[Any, HalfEdgeRole]]
    effective_spider_types: Mapping[Any, str]
    strategy: SynthesisStrategy
    optimizer_status: str
    proven_optimal: bool

    def to_file(self, path: str) -> None:
        """Write the synthesized circuit in Stim's text format."""

        self.circuit.to_file(path)


@dataclass(frozen=True)
class _LayoutPlan:
    order: tuple[int, ...]
    roles: Mapping[tuple[int, int], HalfEdgeRole]
    frames: Mapping[int, bool]
    optimizer_status: str
    proven_optimal: bool


@dataclass(frozen=True)
class _Operation:
    name: str
    qubits: tuple[int, ...] = ()
    condition: int | None = None
    measurement_id: int | None = None
    measurement_deps: tuple[int, ...] = ()
    inverted: bool = False


@dataclass
class _ScheduledProgram:
    circuit: stim.Circuit
    layers: tuple[tuple[_Operation, ...], ...]


def _coerce_strategy(strategy: str | SynthesisStrategy) -> SynthesisStrategy:
    if isinstance(strategy, SynthesisStrategy):
        return strategy
    try:
        return SynthesisStrategy(strategy)
    except ValueError as exc:
        choices = ", ".join(s.value for s in SynthesisStrategy)
        raise SynthesisError(
            f"Unknown synthesis strategy {strategy!r}; expected one of {choices}."
        ) from exc


def _is_zero_phase(phase: Any) -> bool:
    """Return whether a PyZX phase value is exactly zero."""

    try:
        return phase == 0 or Fraction(phase) == 0
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def normalize_zx_diagram(diagram: Any) -> NormalizedDiagram:
    """Validate and copy the supported PyZX graph data into an immutable IR.

    Supported diagrams contain only:

    * zero-phase X and Z spiders of degree three;
    * boundary vertices of degree one;
    * simple and Hadamard edges; and
    * at least one declared input or output in every connected component.

    The input graph is never mutated.
    """

    required_methods = (
        "vertices",
        "type",
        "phase",
        "neighbors",
        "inputs",
        "outputs",
        "edge",
        "edge_type",
    )
    missing = [name for name in required_methods if not hasattr(diagram, name)]
    if missing:
        raise TypeError(
            "diagram must be a PyZX BaseGraph-like object; missing methods: "
            + ", ".join(missing)
        )

    original_vertices = tuple(diagram.vertices())
    if not original_vertices:
        raise SynthesisError("The ZX diagram is empty.")

    original_to_normalized = {
        vertex: index for index, vertex in enumerate(original_vertices)
    }
    input_vertices = tuple(diagram.inputs())
    output_vertices = tuple(diagram.outputs())
    input_set = set(input_vertices)
    output_set = set(output_vertices)

    overlap = input_set & output_set
    if overlap:
        raise SynthesisError(
            "Boundary vertices cannot be both inputs and outputs: "
            f"{sorted(map(str, overlap))}."
        )

    undeclared_boundaries: list[Any] = []
    vertex_types: list[VertexType] = []
    spiders: list[int] = []
    for normalized, original in enumerate(original_vertices):
        vertex_type = diagram.type(original)
        vertex_types.append(vertex_type)
        neighbors = tuple(diagram.neighbors(original))

        if vertex_type == VertexType.BOUNDARY:
            if len(neighbors) != 1:
                raise SynthesisError(
                    f"Boundary vertex {original!r} has degree {len(neighbors)}; "
                    "boundary vertices must have degree one."
                )
            if original not in input_set and original not in output_set:
                undeclared_boundaries.append(original)
            continue

        if vertex_type not in (VertexType.X, VertexType.Z):
            raise SynthesisError(
                f"Vertex {original!r} has unsupported type {vertex_type!r}; "
                "only phase-free X and Z spiders are supported."
            )
        if len(neighbors) != 3:
            raise SynthesisError(
                f"Spider {original!r} has degree {len(neighbors)}; "
                "all internal spiders must have degree three."
            )
        if not _is_zero_phase(diagram.phase(original)):
            raise SynthesisError(
                f"Spider {original!r} has nonzero phase "
                f"{diagram.phase(original)!r}; version one is phase-free."
            )
        spiders.append(normalized)

    if undeclared_boundaries:
        raise SynthesisError(
            "Every boundary must be declared with diagram.set_inputs() or "
            "diagram.set_outputs(); undeclared boundaries: "
            f"{sorted(map(str, undeclared_boundaries))}."
        )

    if set(input_vertices + output_vertices) - set(original_vertices):
        raise SynthesisError("The input/output lists reference missing vertices.")

    normalized_edges: list[NormalizedEdge] = []
    incident: list[list[int]] = [[] for _ in original_vertices]
    seen_edges: set[frozenset[int]] = set()

    for original_u in original_vertices:
        u = original_to_normalized[original_u]
        for original_v in diagram.neighbors(original_u):
            v = original_to_normalized.get(original_v)
            if v is None:
                raise SynthesisError(
                    f"Vertex {original_u!r} references missing neighbor "
                    f"{original_v!r}."
                )
            if u == v:
                raise SynthesisError(
                    f"Self-loop at vertex {original_u!r} is unsupported."
                )
            key = frozenset((u, v))
            if key in seen_edges:
                continue
            seen_edges.add(key)

            edge_type = diagram.edge_type(diagram.edge(original_u, original_v))
            if edge_type not in (EdgeType.SIMPLE, EdgeType.HADAMARD):
                raise SynthesisError(
                    f"Edge ({original_u!r}, {original_v!r}) has unsupported "
                    f"type {edge_type!r}."
                )
            index = len(normalized_edges)
            edge = NormalizedEdge(
                index=index,
                u=u,
                v=v,
                hadamard=edge_type == EdgeType.HADAMARD,
            )
            normalized_edges.append(edge)
            incident[u].append(index)
            incident[v].append(index)

    boundary_indices = {
        original_to_normalized[v] for v in input_vertices + output_vertices
    }
    unvisited = set(range(len(original_vertices)))
    while unvisited:
        start = next(iter(unvisited))
        component: set[int] = set()
        queue = [start]
        while queue:
            vertex = queue.pop()
            if vertex in component:
                continue
            component.add(vertex)
            for edge_index in incident[vertex]:
                queue.append(normalized_edges[edge_index].other(vertex))
        unvisited -= component
        if not component & boundary_indices:
            original_component = [
                original_vertices[v] for v in sorted(component)
            ]
            raise SynthesisError(
                "Closed scalar-only components are unsupported because Stim "
                f"cannot represent their normalization: {original_component!r}."
            )

    return NormalizedDiagram(
        original_vertices=original_vertices,
        vertex_types=tuple(vertex_types),
        inputs=tuple(original_to_normalized[v] for v in input_vertices),
        outputs=tuple(original_to_normalized[v] for v in output_vertices),
        spiders=tuple(spiders),
        edges=tuple(normalized_edges),
        incident_edges=tuple(tuple(sorted(items)) for items in incident),
    )


def _spider_adjacency(
    diagram: NormalizedDiagram,
) -> dict[int, list[int]]:
    spider_set = set(diagram.spiders)
    adjacency = {v: [] for v in diagram.spiders}
    for edge in diagram.edges:
        if edge.u in spider_set and edge.v in spider_set:
            adjacency[edge.u].append(edge.v)
            adjacency[edge.v].append(edge.u)
    for neighbors in adjacency.values():
        neighbors.sort()
    return adjacency


def _boundary_rank_hint(
    diagram: NormalizedDiagram, spider: int
) -> tuple[int, int]:
    input_set = set(diagram.inputs)
    output_set = set(diagram.outputs)
    input_count = 0
    output_count = 0
    for neighbor in diagram.neighbors(spider):
        input_count += int(neighbor in input_set)
        output_count += int(neighbor in output_set)
    return input_count, output_count


def _dfs_order(
    diagram: NormalizedDiagram, rng: random.Random
) -> tuple[int, ...]:
    """Find a low-frontier traversal used by qubit-reusing extraction."""

    adjacency = _spider_adjacency(diagram)
    starts = sorted(
        diagram.spiders,
        key=lambda v: (
            -_boundary_rank_hint(diagram, v)[0],
            -_boundary_rank_hint(diagram, v)[1],
            v,
        ),
    )
    tie_break = {v: rng.random() for v in diagram.spiders}
    order: list[int] = []
    visited: set[int] = set()

    def visit(vertex: int) -> None:
        visited.add(vertex)
        order.append(vertex)
        candidates = [n for n in adjacency[vertex] if n not in visited]
        candidates.sort(
            key=lambda n: (
                len([x for x in adjacency[n] if x not in visited]),
                tie_break[n],
                n,
            )
        )
        for neighbor in candidates:
            if neighbor not in visited:
                visit(neighbor)

    for start in starts:
        if start not in visited:
            visit(start)
    return tuple(order)


def _bfs_order(
    diagram: NormalizedDiagram, rng: random.Random
) -> tuple[int, ...]:
    """Find a balanced, multi-source traversal used by depth extraction."""

    adjacency = _spider_adjacency(diagram)
    boundary_adjacent = [
        v
        for v in diagram.spiders
        if any(
            diagram.vertex_types[n] == VertexType.BOUNDARY
            for n in diagram.neighbors(v)
        )
    ]
    if not boundary_adjacent:
        boundary_adjacent = list(diagram.spiders)
    tie_break = {v: rng.random() for v in diagram.spiders}
    queue = deque(
        sorted(
            set(boundary_adjacent),
            key=lambda v: (tie_break[v], v),
        )
    )
    visited: set[int] = set()
    order: list[int] = []
    while queue:
        vertex = queue.popleft()
        if vertex in visited:
            continue
        visited.add(vertex)
        order.append(vertex)
        neighbors = sorted(
            adjacency[vertex], key=lambda n: (tie_break[n], n)
        )
        queue.extend(n for n in neighbors if n not in visited)
    for vertex in sorted(diagram.spiders):
        if vertex not in visited:
            order.append(vertex)
    return tuple(order)


def _initial_roles(
    diagram: NormalizedDiagram, order: Sequence[int]
) -> dict[tuple[int, int], HalfEdgeRole]:
    position = {vertex: index for index, vertex in enumerate(order)}
    input_set = set(diagram.inputs)
    output_set = set(diagram.outputs)
    result: dict[tuple[int, int], HalfEdgeRole] = {}

    for spider in diagram.spiders:
        neighbors = list(diagram.neighbors(spider))

        def neighbor_key(neighbor: int) -> tuple[int, int]:
            if neighbor in input_set:
                return (-2, neighbor)
            if neighbor in output_set:
                return (len(order) + 2, neighbor)
            return (position.get(neighbor, len(order)), neighbor)

        neighbors.sort(key=neighbor_key)
        for neighbor, role in zip(neighbors, HalfEdgeRole):
            result[(spider, neighbor)] = role
    return result


def _frame_cost(
    diagram: NormalizedDiagram, frames: Mapping[int, bool]
) -> int:
    spider_set = set(diagram.spiders)
    cost = 0
    for edge in diagram.edges:
        if edge.u in spider_set and edge.v in spider_set:
            residual = (
                edge.hadamard ^ frames[edge.u] ^ frames[edge.v]
            )
        else:
            spider = edge.u if edge.u in spider_set else edge.v
            residual = edge.hadamard ^ frames[spider]
        cost += int(residual)
    return cost


def _heuristic_frames(
    diagram: NormalizedDiagram, rng: random.Random
) -> dict[int, bool]:
    frames = {vertex: False for vertex in diagram.spiders}
    vertices = list(diagram.spiders)
    rng.shuffle(vertices)
    improved = True
    while improved:
        improved = False
        before = _frame_cost(diagram, frames)
        for vertex in vertices:
            frames[vertex] = not frames[vertex]
            after = _frame_cost(diagram, frames)
            if after < before:
                before = after
                improved = True
            else:
                frames[vertex] = not frames[vertex]
    return frames


def _layout_proxy_key(
    diagram: NormalizedDiagram,
    order: Sequence[int],
    roles: Mapping[tuple[int, int], HalfEdgeRole],
    frames: Mapping[int, bool],
    strategy: SynthesisStrategy,
) -> tuple[Any, ...]:
    """Evaluate the same deterministic resource proxies used by Z3."""

    spider_set = set(diagram.spiders)
    position = {vertex: index for index, vertex in enumerate(order)}
    template_cost = 0
    spans: list[int] = []
    for edge in diagram.edges:
        if edge.u in spider_set and edge.v in spider_set:
            left_role = roles[(edge.u, edge.v)]
            right_role = roles[(edge.v, edge.u)]
            direct = (
                left_role == HalfEdgeRole.INTERACTION
                and right_role == HalfEdgeRole.INTERACTION
                and _edge_is_directable(diagram, edge)
            )
            temporal = {
                left_role,
                right_role,
            } == {HalfEdgeRole.PAST, HalfEdgeRole.FUTURE}
            template_cost += 0 if direct else 1 if temporal else 3
            spans.append(abs(position[edge.u] - position[edge.v]))
        else:
            spider = edge.u if edge.u in spider_set else edge.v
            boundary = edge.v if edge.u in spider_set else edge.u
            expected = (
                HalfEdgeRole.PAST
                if boundary in set(diagram.inputs)
                else HalfEdgeRole.FUTURE
            )
            template_cost += int(roles[(spider, boundary)] != expected) * 2

    peak_frontier = 0
    for cut in range(max(0, len(order) - 1)):
        peak_frontier = max(
            peak_frontier,
            sum(
                1
                for edge in diagram.edges
                if edge.u in spider_set
                and edge.v in spider_set
                and (
                    (position[edge.u] <= cut < position[edge.v])
                    or (position[edge.v] <= cut < position[edge.u])
                )
            ),
        )
    total_span = sum(spans)
    max_span = max(spans, default=0)
    common_tie_break = (
        tuple(order),
        tuple(
            int(roles[key])
            for key in sorted(roles)
        ),
        tuple(int(frames[vertex]) for vertex in sorted(frames)),
    )
    if strategy == SynthesisStrategy.GATE_COUNT:
        return (
            template_cost,
            _frame_cost(diagram, frames),
            peak_frontier,
            total_span,
            max_span,
            common_tie_break,
        )
    return (
        max_span,
        peak_frontier,
        template_cost,
        _frame_cost(diagram, frames),
        total_span,
        common_tie_break,
    )


def _edge_is_directable(
    diagram: NormalizedDiagram, edge: NormalizedEdge
) -> bool:
    """Check the local colour/edge parity for a CNOT-shaped interaction."""

    type_parity = (
        diagram.vertex_types[edge.u] != diagram.vertex_types[edge.v]
    )
    return bool(type_parity) ^ edge.hadamard


def _template_cost_expression(
    diagram: NormalizedDiagram,
    role_vars: Mapping[tuple[int, int], Any],
) -> Any:
    spider_set = set(diagram.spiders)
    terms: list[Any] = []
    for edge in diagram.edges:
        u_is_spider = edge.u in spider_set
        v_is_spider = edge.v in spider_set
        if u_is_spider and v_is_spider:
            ru = role_vars[(edge.u, edge.v)]
            rv = role_vars[(edge.v, edge.u)]
            temporal = z3.Or(
                z3.And(
                    ru == HalfEdgeRole.FUTURE,
                    rv == HalfEdgeRole.PAST,
                ),
                z3.And(
                    rv == HalfEdgeRole.FUTURE,
                    ru == HalfEdgeRole.PAST,
                ),
            )
            direct = z3.And(
                ru == HalfEdgeRole.INTERACTION,
                rv == HalfEdgeRole.INTERACTION,
                z3.BoolVal(_edge_is_directable(diagram, edge)),
            )
            terms.append(z3.If(direct, 0, z3.If(temporal, 1, 3)))
        else:
            spider = edge.u if u_is_spider else edge.v
            boundary = edge.v if u_is_spider else edge.u
            expected = (
                HalfEdgeRole.PAST
                if boundary in set(diagram.inputs)
                else HalfEdgeRole.FUTURE
            )
            terms.append(
                z3.If(role_vars[(spider, boundary)] == expected, 0, 2)
            )
    return z3.Sum(terms) if terms else z3.IntVal(0)


def _solve_exact_layout(
    diagram: NormalizedDiagram,
    strategy: SynthesisStrategy,
    timeout_seconds: float,
) -> _LayoutPlan | None:
    if z3 is None or not diagram.spiders:
        return None

    optimizer = z3.Optimize()
    optimizer.set(timeout=max(1, int(timeout_seconds * 1000)))
    n = len(diagram.spiders)
    position = {
        vertex: z3.Int(f"zx_pos_{vertex}") for vertex in diagram.spiders
    }
    for value in position.values():
        optimizer.add(value >= 0, value < n)
    optimizer.add(z3.Distinct(*position.values()))

    role_vars: dict[tuple[int, int], Any] = {}
    for spider in diagram.spiders:
        variables = []
        for neighbor in diagram.neighbors(spider):
            variable = z3.Int(f"zx_role_{spider}_{neighbor}")
            optimizer.add(
                variable >= HalfEdgeRole.PAST,
                variable <= HalfEdgeRole.INTERACTION,
            )
            role_vars[(spider, neighbor)] = variable
            variables.append(variable)
        optimizer.add(z3.Distinct(*variables))

    frame_vars = {
        vertex: z3.Bool(f"zx_frame_{vertex}")
        for vertex in diagram.spiders
    }
    spider_set = set(diagram.spiders)
    internal_edges = [
        edge
        for edge in diagram.edges
        if edge.u in spider_set and edge.v in spider_set
    ]

    for edge in internal_edges:
        ru = role_vars[(edge.u, edge.v)]
        rv = role_vars[(edge.v, edge.u)]
        optimizer.add(
            z3.Implies(
                z3.And(
                    ru == HalfEdgeRole.FUTURE,
                    rv == HalfEdgeRole.PAST,
                ),
                position[edge.u] < position[edge.v],
            )
        )
        optimizer.add(
            z3.Implies(
                z3.And(
                    rv == HalfEdgeRole.FUTURE,
                    ru == HalfEdgeRole.PAST,
                ),
                position[edge.v] < position[edge.u],
            )
        )

    template_cost = _template_cost_expression(diagram, role_vars)
    hadamard_terms: list[Any] = []
    for edge in diagram.edges:
        if edge.u in spider_set and edge.v in spider_set:
            residual = z3.Xor(frame_vars[edge.u], frame_vars[edge.v])
            desired = z3.BoolVal(edge.hadamard)
            hadamard_terms.append(z3.If(residual == desired, 0, 1))
        else:
            spider = edge.u if edge.u in spider_set else edge.v
            hadamard_terms.append(
                z3.If(frame_vars[spider] == edge.hadamard, 0, 1)
            )
    hadamard_cost = z3.Sum(hadamard_terms)

    spans: list[Any] = []
    for edge in internal_edges:
        delta = position[edge.u] - position[edge.v]
        spans.append(z3.If(delta >= 0, delta, -delta))
    total_span = z3.Sum(spans) if spans else z3.IntVal(0)
    max_span = z3.Int("zx_max_span")
    optimizer.add(max_span >= 0)
    for span in spans:
        optimizer.add(max_span >= span)

    peak_frontier = z3.Int("zx_peak_frontier")
    optimizer.add(peak_frontier >= 0)
    for cut in range(max(0, n - 1)):
        crossing = []
        for edge in internal_edges:
            crossing.append(
                z3.If(
                    z3.Or(
                        z3.And(
                            position[edge.u] <= cut,
                            position[edge.v] > cut,
                        ),
                        z3.And(
                            position[edge.v] <= cut,
                            position[edge.u] > cut,
                        ),
                    ),
                    1,
                    0,
                )
            )
        if crossing:
            optimizer.add(peak_frontier >= z3.Sum(crossing))

    if strategy == SynthesisStrategy.GATE_COUNT:
        objective_values = (
            template_cost,
            hadamard_cost,
            peak_frontier,
            total_span,
            max_span,
        )
    else:
        objective_values = (
            max_span,
            peak_frontier,
            template_cost,
            hadamard_cost,
            total_span,
        )
    handles = [optimizer.minimize(value) for value in objective_values]

    # Optimize has freedom to return any model when all resource objectives
    # tie.  Pin that freedom lexicographically so identical inputs and options
    # always produce byte-identical Stim circuits.
    for vertex in sorted(diagram.spiders):
        handles.append(optimizer.minimize(position[vertex]))
    for key in sorted(role_vars):
        handles.append(optimizer.minimize(role_vars[key]))
    for vertex in sorted(diagram.spiders):
        handles.append(
            optimizer.minimize(z3.If(frame_vars[vertex], 1, 0))
        )

    status = optimizer.check()
    if status != z3.sat:
        return None
    model = optimizer.model()
    try:
        proven_optimal = all(
            str(optimizer.lower(handle)) == str(optimizer.upper(handle))
            for handle in handles
        )
    except z3.Z3Exception:
        proven_optimal = False

    order = tuple(
        sorted(
            diagram.spiders,
            key=lambda vertex: model.eval(
                position[vertex], model_completion=True
            ).as_long(),
        )
    )
    roles = {
        key: HalfEdgeRole(
            model.eval(variable, model_completion=True).as_long()
        )
        for key, variable in role_vars.items()
    }
    frames = {
        vertex: z3.is_true(
            model.eval(variable, model_completion=True)
        )
        for vertex, variable in frame_vars.items()
    }
    return _LayoutPlan(
        order=order,
        roles=roles,
        frames=frames,
        optimizer_status="optimal" if proven_optimal else "feasible_timeout",
        proven_optimal=proven_optimal,
    )


def _heuristic_layout(
    diagram: NormalizedDiagram,
    strategy: SynthesisStrategy,
    seed: int,
    status: str,
) -> _LayoutPlan:
    candidates: list[
        tuple[
            tuple[Any, ...],
            tuple[int, ...],
            dict[tuple[int, int], HalfEdgeRole],
            dict[int, bool],
        ]
    ] = []
    attempts = max(1, min(16, 2 * len(diagram.spiders)))
    for attempt in range(attempts):
        # A fixed arithmetic progression avoids Python hash randomization and
        # makes every multi-start portfolio reproducible across processes.
        rng = random.Random(seed + 1_000_003 * attempt)
        if strategy == SynthesisStrategy.GATE_COUNT:
            order = _dfs_order(diagram, rng)
        else:
            order = _bfs_order(diagram, rng)
        roles = _initial_roles(diagram, order)
        frames = _heuristic_frames(diagram, rng)
        candidates.append(
            (
                _layout_proxy_key(
                    diagram, order, roles, frames, strategy
                ),
                order,
                roles,
                frames,
            )
        )
    _, order, roles, frames = min(candidates, key=lambda item: item[0])
    return _LayoutPlan(
        order=order,
        roles=roles,
        frames=frames,
        optimizer_status=status,
        proven_optimal=False,
    )


def _find_layout(
    diagram: NormalizedDiagram,
    strategy: SynthesisStrategy,
    exact_max_spiders: int,
    timeout_seconds: float,
    seed: int,
) -> _LayoutPlan:
    if len(diagram.spiders) <= exact_max_spiders and z3 is not None:
        exact = _solve_exact_layout(diagram, strategy, timeout_seconds)
        if exact is not None and exact.proven_optimal:
            return exact
        status = "heuristic_after_exact_timeout"
    elif z3 is None:
        status = "heuristic_without_z3"
    else:
        status = "heuristic_size_limit"
    return _heuristic_layout(diagram, strategy, seed, status)


class _OperationBuilder:
    """Records a causal operation stream and ideal measurement metadata."""

    def __init__(self) -> None:
        self.operations: list[_Operation] = []
        self.measurements: list[MeasurementInfo] = []
        self._next_measurement = 0
        self._next_detector = 0

    def add(self, name: str, *qubits: int) -> None:
        self.operations.append(_Operation(name=name, qubits=tuple(qubits)))

    def add_feedback(
        self, measurement: int, qubit: int, pauli: int
    ) -> None:
        # Stim supports X feedback directly.  A Z correction is H-X-H;
        # Y is X followed by Z, up to an irrelevant global phase.
        if pauli in (1, 2):
            self.operations.append(
                _Operation(
                    name="CX_FEEDBACK",
                    qubits=(qubit,),
                    condition=measurement,
                    measurement_deps=(measurement,),
                )
            )
        if pauli in (2, 3):
            self.add("H", qubit)
            self.operations[-1] = _Operation(
                name="H",
                qubits=(qubit,),
                measurement_deps=(measurement,),
            )
            self.operations.append(
                _Operation(
                    name="CX_FEEDBACK",
                    qubits=(qubit,),
                    condition=measurement,
                    measurement_deps=(measurement,),
                )
            )
            self.add("H", qubit)

    def measure_with_kickback(
        self,
        simulator: stim.TableauSimulator,
        qubit: int,
        basis: str,
        live_qubits: set[int],
    ) -> MeasurementInfo:
        if basis not in ("X", "Z"):
            raise ValueError(f"Unsupported measurement basis {basis!r}.")

        expectation = (
            simulator.peek_x(qubit)
            if basis == "X"
            else simulator.peek_z(qubit)
        )
        if basis == "X":
            simulator.h(qubit)
        result, kickback = simulator.measure_kickback(qubit)
        measurement_id = self._next_measurement
        self._next_measurement += 1

        inverted = kickback is None and expectation == -1
        self.operations.append(
            _Operation(
                name="MX" if basis == "X" else "M",
                qubits=(qubit,),
                measurement_id=measurement_id,
                inverted=inverted,
            )
        )

        if kickback is None:
            detector_id = self._next_detector
            self._next_detector += 1
            self.operations.append(
                _Operation(
                    name="DETECTOR",
                    measurement_deps=(measurement_id,),
                )
            )
            info = MeasurementInfo(
                index=measurement_id,
                qubit=qubit,
                basis=basis,
                kind="check",
                inverted=inverted,
                detector_index=detector_id,
            )
        else:
            # Keep the companion tableau on the all-zero reference branch.
            if result:
                simulator.do(kickback)
            correction: list[tuple[int, str]] = []
            for target in sorted(live_qubits):
                if target == qubit or target >= len(kickback):
                    continue
                pauli = kickback[target]
                if pauli == 0:
                    continue
                correction.append((target, "_XYZ"[pauli]))
                self.add_feedback(measurement_id, target, pauli)
            info = MeasurementInfo(
                index=measurement_id,
                qubit=qubit,
                basis=basis,
                kind="teleportation",
                correction=tuple(correction),
            )

        self.measurements.append(info)
        live_qubits.discard(qubit)
        return info


class _QubitAllocator:
    def __init__(self, num_outputs: int, reuse: bool):
        self.num_outputs = num_outputs
        self.reuse = reuse
        self.next_qubit = num_outputs
        self.free: list[int] = []
        self.allocated: set[int] = set(range(num_outputs))

    def allocate(self) -> int:
        if self.reuse and self.free:
            qubit = heapq.heappop(self.free)
        else:
            qubit = self.next_qubit
            self.next_qubit += 1
        self.allocated.add(qubit)
        return qubit

    def release(self, qubit: int) -> None:
        if qubit < self.num_outputs:
            raise SynthesisError(
                f"Attempted to recycle output data qubit {qubit}."
            )
        if qubit not in self.allocated:
            raise SynthesisError(f"Qubit {qubit} was released twice.")
        self.allocated.remove(qubit)
        if self.reuse:
            heapq.heappush(self.free, qubit)


def _append_reset(
    builder: _OperationBuilder,
    simulator: stim.TableauSimulator,
    qubit: int,
    basis: str,
) -> None:
    if basis == "X":
        builder.add("RX", qubit)
        simulator.reset_x(qubit)
    elif basis == "Z":
        builder.add("R", qubit)
        simulator.reset(qubit)
    else:
        raise ValueError(basis)


def _append_h(
    builder: _OperationBuilder,
    simulator: stim.TableauSimulator,
    qubit: int,
) -> None:
    builder.add("H", qubit)
    simulator.h(qubit)


def _append_cx(
    builder: _OperationBuilder,
    simulator: stim.TableauSimulator,
    control: int,
    target: int,
) -> None:
    builder.add("CX", control, target)
    simulator.cnot(control, target)


def _prepare_spider(
    diagram: NormalizedDiagram,
    layout: _LayoutPlan,
    spider: int,
    port_qubits: Mapping[int, int],
    builder: _OperationBuilder,
    simulator: stim.TableauSimulator,
    live_qubits: set[int],
) -> None:
    neighbors = list(diagram.neighbors(spider))
    # Spider tensors are permutation-symmetric.  Use physical-port order for
    # the preparation fragment so equal-cost role assignments cannot make two
    # otherwise identical synthesis calls differ textually.
    neighbors.sort(key=lambda n: (port_qubits[n], n))
    ports = [port_qubits[neighbor] for neighbor in neighbors]
    live_qubits.update(ports)

    is_x = diagram.vertex_types[spider] == VertexType.X
    if layout.frames[spider]:
        is_x = not is_x

    if not is_x:
        # |000> + |111>: one X-prepared root copied to two Z-prepared ports.
        root, second, third = ports
        _append_reset(builder, simulator, root, "X")
        _append_reset(builder, simulator, second, "Z")
        _append_reset(builder, simulator, third, "Z")
        _append_cx(builder, simulator, root, second)
        _append_cx(builder, simulator, root, third)
    else:
        # H^⊗3 GHZ = even-parity state.  Two X-prepared controls write
        # their parity into one Z-prepared target.
        first, second, target = ports
        _append_reset(builder, simulator, first, "X")
        _append_reset(builder, simulator, second, "X")
        _append_reset(builder, simulator, target, "Z")
        _append_cx(builder, simulator, first, target)
        _append_cx(builder, simulator, second, target)


def _residual_hadamard(
    edge: NormalizedEdge,
    layout: _LayoutPlan,
    spider_set: set[int],
) -> bool:
    if edge.u in spider_set and edge.v in spider_set:
        return (
            edge.hadamard
            ^ layout.frames[edge.u]
            ^ layout.frames[edge.v]
        )
    spider = edge.u if edge.u in spider_set else edge.v
    return edge.hadamard ^ layout.frames[spider]


def _contract_edge(
    edge: NormalizedEdge,
    first_port: int,
    second_port: int,
    residual_hadamard: bool,
    builder: _OperationBuilder,
    simulator: stim.TableauSimulator,
    live_qubits: set[int],
) -> None:
    # <Φ+| = <00| (H ⊗ I) CX.  A Hadamard ZX edge contributes one
    # additional H on either endpoint; H is symmetric, so use first_port.
    if residual_hadamard:
        _append_h(builder, simulator, first_port)
    _append_cx(builder, simulator, first_port, second_port)
    builder.measure_with_kickback(
        simulator, first_port, "X", live_qubits
    )
    builder.measure_with_kickback(
        simulator, second_port, "Z", live_qubits
    )


def _compile_state_candidate(
    diagram: NormalizedDiagram,
    layout: _LayoutPlan,
    *,
    reuse_qubits: bool,
    simulator_seed: int,
) -> tuple[_OperationBuilder, dict[Any, int]]:
    if diagram.inputs:
        raise SynthesisError(
            "Internal error: state compiler received an open-input diagram."
        )

    output_map_normalized = {
        boundary: index for index, boundary in enumerate(diagram.outputs)
    }
    output_map = {
        diagram.original_vertices[boundary]: qubit
        for boundary, qubit in output_map_normalized.items()
    }
    allocator = _QubitAllocator(len(diagram.outputs), reuse_qubits)
    builder = _OperationBuilder()
    simulator = stim.TableauSimulator(seed=simulator_seed)
    live_qubits: set[int] = set()
    spider_set = set(diagram.spiders)

    # A port is identified by (spider, neighbor).
    port_to_qubit: dict[tuple[int, int], int] = {}
    prepared: set[int] = set()
    contracted_edges: set[int] = set()

    for spider in layout.order:
        ports_for_spider: dict[int, int] = {}
        for edge_index in diagram.incident_edges[spider]:
            edge = diagram.edge(edge_index)
            neighbor = edge.other(spider)
            if neighbor in output_map_normalized:
                qubit = output_map_normalized[neighbor]
            else:
                qubit = allocator.allocate()
            port_to_qubit[(spider, neighbor)] = qubit
            ports_for_spider[neighbor] = qubit

        _prepare_spider(
            diagram,
            layout,
            spider,
            ports_for_spider,
            builder,
            simulator,
            live_qubits,
        )
        prepared.add(spider)

        # Reusing mode contracts completed edges immediately, minimizing the
        # live frontier.  Non-reusing mode reaches the same code after every
        # spider has been prepared, allowing maximal preparation parallelism.
        if reuse_qubits:
            ready_edges = [
                diagram.edge(edge_index)
                for edge_index in diagram.incident_edges[spider]
                if diagram.edge(edge_index).other(spider) in prepared
                and diagram.edge(edge_index).other(spider) in spider_set
                and edge_index not in contracted_edges
            ]
            for edge in sorted(ready_edges, key=lambda e: e.index):
                first = port_to_qubit[(edge.u, edge.v)]
                second = port_to_qubit[(edge.v, edge.u)]
                _contract_edge(
                    edge,
                    first,
                    second,
                    _residual_hadamard(edge, layout, spider_set),
                    builder,
                    simulator,
                    live_qubits,
                )
                contracted_edges.add(edge.index)
                allocator.release(first)
                allocator.release(second)

    if not reuse_qubits:
        internal_edges = [
            edge
            for edge in diagram.edges
            if edge.u in spider_set and edge.v in spider_set
        ]
        position = {
            vertex: index for index, vertex in enumerate(layout.order)
        }
        internal_edges.sort(
            key=lambda edge: (
                max(position[edge.u], position[edge.v]),
                min(position[edge.u], position[edge.v]),
                edge.index,
            )
        )
        for edge in internal_edges:
            first = port_to_qubit[(edge.u, edge.v)]
            second = port_to_qubit[(edge.v, edge.u)]
            _contract_edge(
                edge,
                first,
                second,
                _residual_hadamard(edge, layout, spider_set),
                builder,
                simulator,
                live_qubits,
            )
            contracted_edges.add(edge.index)
            allocator.release(first)
            allocator.release(second)

    expected_internal_edges = {
        edge.index
        for edge in diagram.edges
        if edge.u in spider_set and edge.v in spider_set
    }
    if contracted_edges != expected_internal_edges:
        missing = sorted(expected_internal_edges - contracted_edges)
        raise SynthesisError(
            f"Traversal left internal ZX edges uncontracted: {missing}."
        )

    # Terminal Hadamards are local frame changes on data outputs.
    for boundary, qubit in output_map_normalized.items():
        edge_index = diagram.incident_edges[boundary][0]
        edge = diagram.edge(edge_index)
        if _residual_hadamard(edge, layout, spider_set):
            _append_h(builder, simulator, qubit)

    if live_qubits != set(output_map_normalized.values()):
        raise SynthesisError(
            "Extraction ended with dangling live ports. "
            f"Expected outputs {sorted(output_map_normalized.values())}, "
            f"found {sorted(live_qubits)}."
        )
    return builder, output_map


def _schedule_operations(
    operations: Sequence[_Operation],
) -> _ScheduledProgram:
    """ASAP schedule operations while respecting qubit and record causality."""

    next_free: dict[int, int] = defaultdict(int)
    measurement_layer: dict[int, int] = {}
    layers: list[list[_Operation]] = []

    for operation in operations:
        earliest = 0
        if operation.qubits:
            earliest = max(next_free[q] for q in operation.qubits)
        deps = operation.measurement_deps
        if operation.condition is not None:
            deps = tuple(set(deps) | {operation.condition})
        if deps:
            try:
                earliest = max(
                    earliest,
                    max(measurement_layer[m] + 1 for m in deps),
                )
            except KeyError as exc:
                raise SynthesisError(
                    f"Operation depends on unknown measurement {exc.args[0]}."
                ) from exc

        while len(layers) <= earliest:
            layers.append([])
        layers[earliest].append(operation)

        for qubit in operation.qubits:
            next_free[qubit] = earliest + 1
        if operation.measurement_id is not None:
            measurement_layer[operation.measurement_id] = earliest

    nonempty_layers = [layer for layer in layers if layer]
    circuit = stim.Circuit()
    emitted_measurements = 0
    measurement_positions: dict[int, int] = {}

    for layer_index, layer in enumerate(nonempty_layers):
        for operation in layer:
            if operation.name in ("R", "RX", "H"):
                circuit.append(operation.name, operation.qubits)
            elif operation.name == "CX":
                circuit.append("CX", operation.qubits)
            elif operation.name in ("M", "MX"):
                target: int | stim.GateTarget = operation.qubits[0]
                if operation.inverted:
                    target = stim.target_inv(target)
                circuit.append(operation.name, [target])
                if operation.measurement_id is None:
                    raise SynthesisError(
                        "Measurement operation is missing its record id."
                    )
                measurement_positions[
                    operation.measurement_id
                ] = emitted_measurements
                emitted_measurements += 1
            elif operation.name == "CX_FEEDBACK":
                if operation.condition is None:
                    raise SynthesisError(
                        "Feedback operation is missing a condition."
                    )
                record_index = measurement_positions[operation.condition]
                offset = record_index - emitted_measurements
                circuit.append(
                    "CX",
                    [
                        stim.target_rec(offset),
                        operation.qubits[0],
                    ],
                )
            elif operation.name == "DETECTOR":
                targets = []
                for measurement in operation.measurement_deps:
                    record_index = measurement_positions[measurement]
                    targets.append(
                        stim.target_rec(
                            record_index - emitted_measurements
                        )
                    )
                circuit.append("DETECTOR", targets)
            else:
                raise SynthesisError(
                    f"Unknown scheduled operation {operation.name!r}."
                )
        if layer_index != len(nonempty_layers) - 1:
            circuit.append("TICK")

    return _ScheduledProgram(
        circuit=circuit,
        layers=tuple(tuple(layer) for layer in nonempty_layers),
    )


def _metrics_from_program(program: _ScheduledProgram) -> SynthesisMetrics:
    cx_count = 0
    h_count = 0
    preparation_count = 0
    measurement_count = 0
    feedback_count = 0
    detector_count = 0
    cx_depth = 0
    total_operations = 0

    for layer in program.layers:
        layer_has_cx = False
        for operation in layer:
            width = max(1, len(operation.qubits))
            if operation.name == "CX":
                cx_count += 1
                layer_has_cx = True
                total_operations += 1
            elif operation.name == "CX_FEEDBACK":
                feedback_count += 1
                total_operations += 1
            elif operation.name == "H":
                h_count += width
                total_operations += width
            elif operation.name in ("R", "RX"):
                preparation_count += width
                total_operations += width
            elif operation.name in ("M", "MX"):
                measurement_count += width
                total_operations += width
            elif operation.name == "DETECTOR":
                detector_count += 1
        cx_depth += int(layer_has_cx)

    return SynthesisMetrics(
        cx_count=cx_count,
        h_count=h_count,
        preparation_count=preparation_count,
        measurement_count=measurement_count,
        feedback_count=feedback_count,
        detector_count=detector_count,
        depth=len(program.layers),
        cx_depth=cx_depth,
        num_qubits=program.circuit.num_qubits,
        total_operations=total_operations,
    )


def _translate_pyzx_circuit(
    circuit: Any,
) -> tuple[_OperationBuilder, int]:
    """Translate a phase-free PyZX circuit into the supported Stim gate set."""

    builder = _OperationBuilder()
    max_qubit = -1
    for gate in circuit.to_basic_gates().gates:
        name = gate.name
        if name == "HAD":
            target = int(gate.target)
            builder.add("H", target)
            max_qubit = max(max_qubit, target)
        elif name == "CNOT":
            control = int(gate.control)
            target = int(gate.target)
            builder.add("CX", control, target)
            max_qubit = max(max_qubit, control, target)
        elif name == "CZ":
            control = int(gate.control)
            target = int(gate.target)
            builder.add("H", target)
            builder.add("CX", control, target)
            builder.add("H", target)
            max_qubit = max(max_qubit, control, target)
        elif name == "SWAP":
            control = int(gate.control)
            target = int(gate.target)
            builder.add("CX", control, target)
            builder.add("CX", target, control)
            builder.add("CX", control, target)
            max_qubit = max(max_qubit, control, target)
        elif name == "InitAncilla":
            target = int(gate.target)
            state = getattr(gate, "state", "0")
            builder.add("RX" if state == "+" else "R", target)
            max_qubit = max(max_qubit, target)
        elif name == "PostSelect":
            target = int(gate.target)
            measurement = builder._next_measurement
            builder._next_measurement += 1
            basis = "MX" if getattr(gate, "state", "0") == "+" else "M"
            builder.operations.append(
                _Operation(
                    name=basis,
                    qubits=(target,),
                    measurement_id=measurement,
                )
            )
            detector = builder._next_detector
            builder._next_detector += 1
            builder.operations.append(
                _Operation(
                    name="DETECTOR",
                    measurement_deps=(measurement,),
                )
            )
            builder.measurements.append(
                MeasurementInfo(
                    index=measurement,
                    qubit=target,
                    basis="X" if basis == "MX" else "Z",
                    kind="postselection",
                    detector_index=detector,
                )
            )
            max_qubit = max(max_qubit, target)
        else:
            phase = getattr(gate, "phase", None)
            if phase is not None and _is_zero_phase(phase):
                continue
            raise SynthesisError(
                f"PyZX extraction introduced unsupported gate {gate!r}. "
                "The input may not be phase-free or circuit-like."
            )
    return builder, max_qubit + 1


def _compile_open_diagram(
    source_diagram: Any,
    normalized: NormalizedDiagram,
    strategy: SynthesisStrategy,
) -> tuple[
    _OperationBuilder,
    dict[Any, int],
    dict[Any, int],
    str,
]:
    """Use PyZX's gflow extractor for diagrams with live quantum inputs."""

    graph = source_diagram.copy()
    optimize_cnots = 3 if strategy == SynthesisStrategy.GATE_COUNT else 0
    try:
        pyzx_circuit = zx.extract_circuit(
            graph,
            optimize_czs=strategy == SynthesisStrategy.GATE_COUNT,
            optimize_cnots=optimize_cnots,
            up_to_perm=False,
            quiet=True,
        )
        if strategy == SynthesisStrategy.GATE_COUNT:
            # ``extract_circuit`` can leave explicit SWAP objects, while
            # PyZX's basic optimizer accepts only its basic gate expansion.
            pyzx_circuit = zx.optimize.basic_optimization(
                pyzx_circuit.to_basic_gates(),
                do_swaps=True,
                quiet=True,
            )
    except Exception as exc:
        raise SynthesisError(
            "The open ZX diagram has no extractable circuit-like causal flow. "
            "State-preparation diagrams with no inputs are always handled by "
            "the stabilizer tensor-network extractor."
        ) from exc

    builder, num_qubits = _translate_pyzx_circuit(pyzx_circuit)
    if len(normalized.inputs) > num_qubits or len(normalized.outputs) > num_qubits:
        raise SynthesisError(
            "PyZX returned fewer circuit wires than declared boundaries."
        )
    input_map = {
        normalized.original_vertices[boundary]: index
        for index, boundary in enumerate(normalized.inputs)
    }
    output_map = {
        normalized.original_vertices[boundary]: index
        for index, boundary in enumerate(normalized.outputs)
    }
    return builder, input_map, output_map, "pyzx_gflow"


def _public_role_map(
    diagram: NormalizedDiagram,
    layout: _LayoutPlan,
) -> dict[Any, dict[Any, HalfEdgeRole]]:
    result: dict[Any, dict[Any, HalfEdgeRole]] = {}
    for spider in diagram.spiders:
        result[diagram.original_vertices[spider]] = {
            diagram.original_vertices[neighbor]: layout.roles[
                (spider, neighbor)
            ]
            for neighbor in diagram.neighbors(spider)
        }
    return result


def _effective_type_map(
    diagram: NormalizedDiagram,
    layout: _LayoutPlan,
) -> dict[Any, str]:
    result = {}
    for spider in diagram.spiders:
        is_x = diagram.vertex_types[spider] == VertexType.X
        is_x ^= bool(layout.frames[spider])
        result[diagram.original_vertices[spider]] = "X" if is_x else "Z"
    return result


def synthesize_zx(
    diagram: Any,
    strategy: str | SynthesisStrategy = SynthesisStrategy.GATE_COUNT,
    *,
    optimizer: str = "auto",
    exact_max_spiders: int = 24,
    timeout_seconds: float = 10.0,
    seed: int = 0,
) -> SynthesisResult:
    """Synthesize a phase-free, cubic ZX diagram into a Stim circuit.

    Args:
        diagram: A PyZX ``BaseGraph``-like object.
        strategy: ``"gate_count"`` or ``"depth"``.
        optimizer: ``"auto"``, ``"exact"``, or ``"heuristic"``.
        exact_max_spiders: Maximum internal-spider count for automatic Z3 use.
        timeout_seconds: Z3 timeout for the half-edge layout problem.
        seed: Deterministic heuristic and tableau-simulation seed.

    Returns:
        A :class:`SynthesisResult` containing the Stim circuit and metadata.
    """

    chosen_strategy = _coerce_strategy(strategy)
    if optimizer not in {"auto", "exact", "heuristic"}:
        raise SynthesisError(
            "optimizer must be 'auto', 'exact', or 'heuristic'."
        )
    if exact_max_spiders < 0:
        raise SynthesisError("exact_max_spiders must be non-negative.")
    if timeout_seconds <= 0:
        raise SynthesisError("timeout_seconds must be positive.")

    normalized = normalize_zx_diagram(diagram)

    if optimizer == "exact" and z3 is None:
        raise SynthesisError(
            "optimizer='exact' requires the z3-solver package."
        )

    def layout_for(layout_strategy: SynthesisStrategy) -> _LayoutPlan:
        if optimizer == "heuristic":
            return _heuristic_layout(
                normalized,
                layout_strategy,
                seed,
                "heuristic_requested",
            )
        max_spiders = (
            max(exact_max_spiders, len(normalized.spiders))
            if optimizer == "exact"
            else exact_max_spiders
        )
        return _find_layout(
            normalized,
            layout_strategy,
            max_spiders,
            timeout_seconds,
            seed,
        )

    layout = layout_for(chosen_strategy)

    input_map: dict[Any, int] = {}
    if normalized.inputs:
        builder, input_map, output_map, backend_status = (
            _compile_open_diagram(
                diagram, normalized, chosen_strategy
            )
        )
        program = _schedule_operations(builder.operations)
        metrics = _metrics_from_program(program)
        optimizer_status = (
            f"{layout.optimizer_status}+{backend_status}"
        )
        proven_optimal = False
        measurements = tuple(builder.measurements)
    else:
        candidates: list[
            tuple[
                _ScheduledProgram,
                SynthesisMetrics,
                _OperationBuilder,
                dict[Any, int],
                str,
                _LayoutPlan,
            ]
        ] = []

        alternate_strategy = (
            SynthesisStrategy.DEPTH
            if chosen_strategy == SynthesisStrategy.GATE_COUNT
            else SynthesisStrategy.GATE_COUNT
        )
        layouts = (layout, layout_for(alternate_strategy))

        # Both layout objectives and both physical allocations are valid
        # realizations of the same rewrite catalogue. Evaluating their small
        # portfolio makes the public strategies compare actual emitted
        # resources instead of relying solely on a proxy layout score.
        for candidate_layout in layouts:
            for reuse_qubits, label in (
                (True, "reused"),
                (False, "parallel"),
            ):
                builder, output_map = _compile_state_candidate(
                    normalized,
                    candidate_layout,
                    reuse_qubits=reuse_qubits,
                    simulator_seed=seed,
                )
                program = _schedule_operations(builder.operations)
                metrics = _metrics_from_program(program)
                candidates.append(
                    (
                        program,
                        metrics,
                        builder,
                        output_map,
                        label,
                        candidate_layout,
                    )
                )

        if chosen_strategy == SynthesisStrategy.GATE_COUNT:
            selected = min(
                candidates, key=lambda item: item[1].gate_count_key
            )
        else:
            selected = min(
                candidates, key=lambda item: item[1].depth_key
            )
        (
            program,
            metrics,
            builder,
            output_map,
            allocation_label,
            layout,
        ) = selected
        optimizer_status = (
            f"{layout.optimizer_status}+{allocation_label}"
        )
        proven_optimal = layout.proven_optimal
        measurements = tuple(builder.measurements)

    detectors = tuple(
        DetectorInfo(
            index=measurement.detector_index,
            measurement_indices=(measurement.index,),
        )
        for measurement in measurements
        if measurement.detector_index is not None
    )

    return SynthesisResult(
        circuit=program.circuit,
        input_qubits=input_map,
        output_qubits=output_map,
        metrics=metrics,
        measurements=measurements,
        detectors=detectors,
        half_edge_roles=_public_role_map(normalized, layout),
        effective_spider_types=_effective_type_map(normalized, layout),
        strategy=chosen_strategy,
        optimizer_status=optimizer_status,
        proven_optimal=proven_optimal,
    )


def synthesize_stim(
    diagram: Any,
    strategy: str | SynthesisStrategy = SynthesisStrategy.GATE_COUNT,
    **options: Any,
) -> stim.Circuit:
    """Return only the Stim circuit from :func:`synthesize_zx`."""

    return synthesize_zx(diagram, strategy=strategy, **options).circuit


__all__ = [
    "DetectorInfo",
    "HalfEdgeRole",
    "MeasurementInfo",
    "NormalizedDiagram",
    "NormalizedEdge",
    "SynthesisError",
    "SynthesisMetrics",
    "SynthesisResult",
    "SynthesisStrategy",
    "normalize_zx_diagram",
    "synthesize_stim",
    "synthesize_zx",
]

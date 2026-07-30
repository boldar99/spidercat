"""Convert a pure stabilizer state into an LC-equivalent graph state.

The main entry point is :func:`stabilizer_state_to_graph`.  Its local
Clifford certificate has the convention

    ``tensor(local_cliffords_to_graph) |input> = |graph>``.

Only local Clifford operations are used.  Row operations on stabilizer
generators are tracked exactly, including their signs.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import networkx as nx
import stim


class StabilizerGraphError(ValueError):
    """Base class for errors raised by stabilizer-to-graph conversion."""


class StabilizerValidationError(StabilizerGraphError):
    """The supplied operators do not specify a valid pure stabilizer state."""


class InvalidPauliStringError(StabilizerValidationError):
    """A generator is malformed, has the wrong length, or is non-Hermitian."""


class NonCommutingGeneratorsError(StabilizerValidationError):
    """At least two supplied generators anticommute."""


class DependentGeneratorsError(StabilizerValidationError):
    """The supplied generator list contains a redundant generator."""


class UnderconstrainedStabilizerError(StabilizerValidationError):
    """The supplied generators stabilize a space of dimension greater than one."""


class InconsistentStabilizerError(StabilizerValidationError):
    """The generated stabilizer group contains negative identity."""


class LocalCliffordCertificateError(StabilizerGraphError):
    """A local-Clifford certificate does not map the input to the graph state."""


SignedAxis: TypeAlias = int
"""A signed Pauli axis: ``1=X``, ``2=Y``, ``3=Z`` and negatives thereof."""

CliffordKey: TypeAlias = tuple[SignedAxis, SignedAxis]
"""Images of X and Z, respectively, under conjugation by a Clifford."""


_IDENTITY_KEY: CliffordKey = (1, 3)
_GATE_KEYS: dict[str, CliffordKey] = {
    "I": _IDENTITY_KEY,
    "H": (3, 1),
    "S": (2, 3),
    "S_DAG": (-2, 3),
    "X": (1, -3),
    "Y": (-1, -3),
    "Z": (-1, 3),
    "SQRT_X": (1, -2),
    "SQRT_X_DAG": (1, 2),
}


def _axis_to_pauli(axis: SignedAxis) -> tuple[int, int, int]:
    """Return ``(phase, x, z)`` for a signed, Hermitian one-qubit Pauli."""

    sign_phase = 2 if axis < 0 else 0
    unsigned = abs(axis)
    if unsigned == 1:
        return sign_phase, 1, 0
    if unsigned == 2:
        return (sign_phase + 1) % 4, 1, 1
    if unsigned == 3:
        return sign_phase, 0, 1
    raise ValueError(f"Not a signed Pauli axis: {axis!r}")


def _pauli_to_axis(phase: int, x: int, z: int) -> SignedAxis:
    """Convert a non-identity, Hermitian one-qubit Pauli into a signed axis."""

    phase %= 4
    if (x, z) == (1, 0):
        base_phase, axis = 0, 1
    elif (x, z) == (1, 1):
        base_phase, axis = 1, 2
    elif (x, z) == (0, 1):
        base_phase, axis = 0, 3
    else:
        raise ValueError("A Clifford cannot map a Pauli axis to identity.")
    delta = (phase - base_phase) % 4
    if delta == 0:
        return axis
    if delta == 2:
        return -axis
    raise ValueError("The Pauli image is not Hermitian.")


def _multiply_axes(left: SignedAxis, right: SignedAxis) -> tuple[int, int]:
    """Return ``(phase, unsigned_axis)`` for the product of two axes.

    The returned phase is the exponent of ``i`` multiplying the unsigned
    output axis.
    """

    lp, lx, lz = _axis_to_pauli(left)
    rp, rx, rz = _axis_to_pauli(right)
    x = lx ^ rx
    z = lz ^ rz
    phase = (lp + rp + 2 * (lz & rx)) % 4
    if (x, z) == (0, 0):
        return phase, 0
    if (x, z) == (1, 0):
        return phase, 1
    if (x, z) == (1, 1):
        return (phase - 1) % 4, 2
    return phase, 3


def _apply_key_to_axis(key: CliffordKey, axis: SignedAxis) -> SignedAxis:
    sign = -1 if axis < 0 else 1
    unsigned = abs(axis)
    if unsigned == 1:
        return sign * key[0]
    if unsigned == 3:
        return sign * key[1]
    if unsigned != 2:
        raise ValueError(f"Not a signed Pauli axis: {axis!r}")

    # Y = iXZ.  Images of X and Z always anticommute.
    product_phase, product_axis = _multiply_axes(key[0], key[1])
    phase = (product_phase + 1) % 4
    mapped = product_axis if phase == 0 else -product_axis if phase == 2 else None
    if mapped is None:
        raise ValueError(f"Invalid Clifford key: {key!r}")
    return sign * mapped


def _compose_keys(first: CliffordKey, second: CliffordKey) -> CliffordKey:
    """Compose Clifford actions in time order: apply ``first``, then ``second``."""

    return (
        _apply_key_to_axis(second, first[0]),
        _apply_key_to_axis(second, first[1]),
    )


def _canonical_clifford_words() -> dict[CliffordKey, tuple[str, ...]]:
    """Enumerate the 24 single-qubit Cliffords using deterministic H/S words."""

    words: dict[CliffordKey, tuple[str, ...]] = {_IDENTITY_KEY: ()}
    queue: deque[CliffordKey] = deque([_IDENTITY_KEY])
    while queue:
        key = queue.popleft()
        for gate in ("H", "S"):
            nxt = _compose_keys(key, _GATE_KEYS[gate])
            if nxt not in words:
                words[nxt] = (*words[key], gate)
                queue.append(nxt)
    if len(words) != 24:  # pragma: no cover - protects the group implementation.
        raise AssertionError(f"Expected 24 single-qubit Cliffords, got {len(words)}.")
    return words


_CLIFFORD_WORDS = _canonical_clifford_words()


@dataclass(frozen=True, slots=True)
class LocalClifford:
    """An immutable single-qubit Clifford, modulo global phase.

    ``x_image`` and ``z_image`` are signed axes describing conjugation of X
    and Z.  :attr:`gate_word` is a deterministic word over ``H`` and ``S``.
    """

    x_image: SignedAxis = 1
    z_image: SignedAxis = 3

    def __post_init__(self) -> None:
        if (self.x_image, self.z_image) not in _CLIFFORD_WORDS:
            raise ValueError(
                "Images must define a valid orientation-preserving Clifford: "
                f"X->{self.x_image}, Z->{self.z_image}."
            )

    @classmethod
    def identity(cls) -> LocalClifford:
        """Return the identity Clifford."""

        return cls()

    @classmethod
    def from_gate_word(cls, gates: Iterable[str]) -> LocalClifford:
        """Construct a Clifford by applying the named gates from left to right."""

        key = _IDENTITY_KEY
        for raw_gate in gates:
            gate = raw_gate.upper()
            try:
                gate_key = _GATE_KEYS[gate]
            except KeyError as exc:
                raise ValueError(f"Unsupported single-qubit Clifford gate {raw_gate!r}.") from exc
            key = _compose_keys(key, gate_key)
        return cls(*key)

    @property
    def key(self) -> CliffordKey:
        """A compact, hashable representation of this Clifford."""

        return self.x_image, self.z_image

    @property
    def gate_word(self) -> tuple[str, ...]:
        """A deterministic shortest H/S word implementing this Clifford."""

        return _CLIFFORD_WORDS[self.key]

    def conjugate_axis(self, axis: SignedAxis) -> SignedAxis:
        """Return the signed Pauli obtained by conjugating ``axis``."""

        return _apply_key_to_axis(self.key, axis)

    def followed_by(self, later: LocalClifford) -> LocalClifford:
        """Compose in circuit order: apply ``self`` and then ``later``."""

        return LocalClifford(*_compose_keys(self.key, later.key))

    def inverse(self) -> LocalClifford:
        """Return the inverse Clifford."""

        for candidate_key in _CLIFFORD_WORDS:
            if _compose_keys(self.key, candidate_key) == _IDENTITY_KEY:
                return LocalClifford(*candidate_key)
        raise AssertionError("The finite Clifford group is not closed.")  # pragma: no cover

    def to_stim_circuit(self, target: int = 0) -> stim.Circuit:
        """Return a Stim circuit implementing this Clifford on ``target``."""

        if target < 0:
            raise ValueError("A Stim target must be non-negative.")
        circuit = stim.Circuit()
        circuit.append("I", [target])
        for gate in self.gate_word:
            circuit.append(gate, [target])
        return circuit


def _default_vertex_arity_cost(arity: int) -> int:
    # A conservative proxy until the diagram layer supplies exact cached gadget
    # sizes.  It exactly scores the direct/cycle constructions at arities 1..5.
    return 1 if arity <= 3 else arity


@dataclass(frozen=True, slots=True)
class LCSearchConfig:
    """Deterministic local-complementation orbit search settings."""

    optimize: bool = True
    exhaustive_qubits: int = 8
    orbit_state_cap: int = 100_000
    beam_width: int = 256
    beam_rounds: int = 64
    vertex_arity_cost: Callable[[int], int] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.exhaustive_qubits < 0:
            raise ValueError("exhaustive_qubits must be non-negative.")
        if self.orbit_state_cap < 1:
            raise ValueError("orbit_state_cap must be positive.")
        if self.beam_width < 1:
            raise ValueError("beam_width must be positive.")
        if self.beam_rounds < 0:
            raise ValueError("beam_rounds must be non-negative.")


AdjacencyScore: TypeAlias = tuple[int, int, int, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class LCSearchMetadata:
    """Metadata describing how the returned LC representative was selected."""

    mode: Literal["disabled", "exhaustive", "exhaustive_capped", "beam"]
    states_examined: int
    orbit_exhausted: bool
    guaranteed_optimal: bool
    local_complementations: tuple[int, ...]
    score: AdjacencyScore


@dataclass(frozen=True, slots=True)
class _Pauli:
    """An n-qubit Pauli represented as ``i**phase X**x Z**z``."""

    n: int
    x: int
    z: int
    phase: int

    @property
    def vector(self) -> int:
        return self.x | (self.z << self.n)

    @property
    def physical_sign(self) -> int:
        """0 for positive and 1 for negative in the I/X/Y/Z convention."""

        delta = (self.phase - (self.x & self.z).bit_count()) % 4
        if delta == 0:
            return 0
        if delta == 2:
            return 1
        raise InvalidPauliStringError("Encountered a non-Hermitian Pauli operator.")

    def commutes(self, other: _Pauli) -> bool:
        return (
            ((self.x & other.z).bit_count() + (self.z & other.x).bit_count()) & 1
        ) == 0

    def multiplied_by(self, other: _Pauli) -> _Pauli:
        if self.n != other.n:
            raise ValueError("Cannot multiply Paulis with different lengths.")
        return _Pauli(
            n=self.n,
            x=self.x ^ other.x,
            z=self.z ^ other.z,
            phase=(
                self.phase
                + other.phase
                + 2 * (self.z & other.x).bit_count()
            )
            % 4,
        )

    def conjugated_at(self, qubit: int, clifford: LocalClifford) -> _Pauli:
        x_bit = (self.x >> qubit) & 1
        z_bit = (self.z >> qubit) & 1
        if not x_bit and not z_bit:
            return self
        old_axis = 2 if x_bit and z_bit else 1 if x_bit else 3
        new_axis = clifford.conjugate_axis(old_axis)
        sign = self.physical_sign ^ int(new_axis < 0)
        unsigned = abs(new_axis)
        new_x_bit = int(unsigned in (1, 2))
        new_z_bit = int(unsigned in (2, 3))
        mask = 1 << qubit
        new_x = (self.x & ~mask) | (new_x_bit << qubit)
        new_z = (self.z & ~mask) | (new_z_bit << qubit)
        new_phase = ((new_x & new_z).bit_count() + 2 * sign) % 4
        return _Pauli(self.n, new_x, new_z, new_phase)

    def to_signed_string(self) -> str:
        chars: list[str] = []
        for qubit in range(self.n):
            x_bit = (self.x >> qubit) & 1
            z_bit = (self.z >> qubit) & 1
            chars.append(
                "Y"
                if x_bit and z_bit
                else "X"
                if x_bit
                else "Z"
                if z_bit
                else "I"
            )
        return ("-" if self.physical_sign else "+") + "".join(chars)


def _parse_pauli(value: str | stim.PauliString, index: int) -> _Pauli:
    if isinstance(value, stim.PauliString):
        sign = complex(value.sign)
        if sign not in (1 + 0j, -1 + 0j):
            raise InvalidPauliStringError(
                f"Generator {index} is non-Hermitian (phase {sign!r})."
            )
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise InvalidPauliStringError(
            f"Generator {index} must be a string or stim.PauliString, "
            f"not {type(value).__name__}."
        )

    if text.startswith(("+i", "-i")):
        raise InvalidPauliStringError(f"Generator {index} is non-Hermitian: {text!r}.")
    negative = text.startswith("-")
    if text.startswith(("+", "-")):
        text = text[1:]
    if not text:
        raise InvalidPauliStringError(f"Generator {index} is empty.")
    text = text.replace("_", "I")
    invalid = sorted(set(text) - set("IXYZ"))
    if invalid:
        raise InvalidPauliStringError(
            f"Generator {index} contains invalid Pauli symbols {invalid!r}."
        )

    x = 0
    z = 0
    y_count = 0
    for qubit, char in enumerate(text):
        if char in "XY":
            x |= 1 << qubit
        if char in "YZ":
            z |= 1 << qubit
        if char == "Y":
            y_count += 1
    return _Pauli(
        n=len(text),
        x=x,
        z=z,
        phase=(y_count + 2 * int(negative)) % 4,
    )


def _insert_into_pauli_basis(
    pauli: _Pauli,
    basis: dict[int, _Pauli],
) -> _Pauli | None:
    """Insert an independent row, or return its identity dependency."""

    reduced = pauli
    while reduced.vector:
        pivot = reduced.vector.bit_length() - 1
        existing = basis.get(pivot)
        if existing is None:
            basis[pivot] = reduced
            return None
        reduced = reduced.multiplied_by(existing)
    return reduced


def _validate_generators(rows: Sequence[_Pauli]) -> None:
    if not rows:
        raise UnderconstrainedStabilizerError(
            "At least one generator is required to infer a non-empty pure state."
        )
    n = rows[0].n
    for index, row in enumerate(rows):
        if row.n != n:
            raise InvalidPauliStringError(
                f"Generator {index} has length {row.n}; expected {n}."
            )
        # Also validates that the canonical phase is Hermitian.
        row.physical_sign

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if not rows[left].commutes(rows[right]):
                raise NonCommutingGeneratorsError(
                    f"Generators {left} and {right} anticommute."
                )

    basis: dict[int, _Pauli] = {}
    dependent = False
    for row in rows:
        reduced = _insert_into_pauli_basis(row, basis)
        if reduced is not None:
            if reduced.physical_sign:
                raise InconsistentStabilizerError(
                    "The supplied generators multiply to -I."
                )
            dependent = True

    rank = len(basis)
    if dependent:
        raise DependentGeneratorsError(
            f"The generator list is dependent (binary symplectic rank {rank})."
        )
    if rank < n:
        raise UnderconstrainedStabilizerError(
            f"{rank} independent generators constrain {n} qubits; expected {n}."
        )
    if rank > n:
        # An isotropic subspace cannot have rank > n.  Keeping this check makes
        # failures from future representation changes explicit.
        raise StabilizerValidationError(
            f"A commuting stabilizer group on {n} qubits cannot have rank {rank}."
        )

    # Ask Stim to independently certify purity and phase consistency.
    try:
        stim.Tableau.from_stabilizers(
            [stim.PauliString(row.to_signed_string()) for row in rows],
            allow_redundant=False,
            allow_underconstrained=False,
        )
    except ValueError as exc:  # pragma: no cover - our checks should classify it first.
        raise StabilizerValidationError(f"Stim rejected the stabilizer state: {exc}") from exc


def _coerce_generators(
    stabilizers: Sequence[str | stim.PauliString] | stim.Tableau,
) -> tuple[_Pauli, ...]:
    if isinstance(stabilizers, stim.Tableau):
        values: Sequence[str | stim.PauliString] = [
            stabilizers.z_output(qubit) for qubit in range(len(stabilizers))
        ]
    else:
        if isinstance(stabilizers, (str, stim.PauliString)):
            raise InvalidPauliStringError(
                "Pass a sequence of generators, not a single Pauli object."
            )
        try:
            values = list(stabilizers)
        except TypeError as exc:
            raise InvalidPauliStringError(
                "stabilizers must be a generator sequence or stim.Tableau."
            ) from exc
    rows = tuple(_parse_pauli(value, index) for index, value in enumerate(values))
    _validate_generators(rows)
    return rows


def _row_reduce_x(rows: list[_Pauli]) -> tuple[int, ...]:
    """RREF the X block in place and return its pivot columns."""

    n = rows[0].n
    next_row = 0
    pivot_columns: list[int] = []
    for column in range(n):
        pivot = next(
            (row for row in range(next_row, n) if (rows[row].x >> column) & 1),
            None,
        )
        if pivot is None:
            continue
        rows[next_row], rows[pivot] = rows[pivot], rows[next_row]
        for row in range(n):
            if row != next_row and ((rows[row].x >> column) & 1):
                rows[row] = rows[row].multiplied_by(rows[next_row])
        pivot_columns.append(column)
        next_row += 1
    return tuple(pivot_columns)


def _normalize_x_to_identity(rows: list[_Pauli]) -> None:
    n = rows[0].n
    for column in range(n):
        pivot = next(
            (row for row in range(column, n) if (rows[row].x >> column) & 1),
            None,
        )
        if pivot is None:
            raise StabilizerGraphError(
                "Internal error: pivot-H conversion did not make X invertible."
            )
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(n):
            if row != column and ((rows[row].x >> column) & 1):
                rows[row] = rows[row].multiplied_by(rows[column])
    if any(row.x != 1 << index for index, row in enumerate(rows)):
        raise StabilizerGraphError("Internal error: failed to normalize X to identity.")


def _append_local_gate(
    rows: list[_Pauli],
    certificate: list[LocalClifford],
    qubit: int,
    gate: str,
) -> None:
    local_gate = LocalClifford.from_gate_word([gate])
    for index, row in enumerate(rows):
        rows[index] = row.conjugated_at(qubit, local_gate)
    certificate[qubit] = certificate[qubit].followed_by(local_gate)


def _direct_graph_conversion(
    original_rows: Sequence[_Pauli],
) -> tuple[nx.Graph, tuple[LocalClifford, ...]]:
    rows = list(original_rows)
    n = rows[0].n
    certificate = [LocalClifford.identity() for _ in range(n)]

    pivot_columns = set(_row_reduce_x(rows))
    for qubit in range(n):
        if qubit not in pivot_columns:
            _append_local_gate(rows, certificate, qubit, "H")

    _normalize_x_to_identity(rows)

    for row in range(n):
        for column in range(n):
            if ((rows[row].z >> column) & 1) != ((rows[column].z >> row) & 1):
                raise StabilizerGraphError(
                    "Internal error: commutation did not produce a symmetric adjacency."
                )

    for qubit in range(n):
        if (rows[qubit].z >> qubit) & 1:
            _append_local_gate(rows, certificate, qubit, "S")

    if any((rows[index].z >> index) & 1 for index in range(n)):
        raise StabilizerGraphError("Internal error: failed to clear adjacency diagonal.")

    for qubit in range(n):
        if rows[qubit].physical_sign:
            _append_local_gate(rows, certificate, qubit, "Z")

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for left in range(n):
        for right in range(left + 1, n):
            if (rows[left].z >> right) & 1:
                graph.add_edge(left, right)
    return graph, tuple(certificate)


def _graph_to_adjacency(graph: nx.Graph) -> tuple[int, ...]:
    n = len(graph)
    expected_nodes = set(range(n))
    if set(graph.nodes) != expected_nodes:
        raise ValueError("Graph nodes must be the consecutive integers 0..n-1.")
    adjacency = [0] * n
    for left, right in graph.edges:
        if left == right:
            raise ValueError("Graph states do not permit self-loops.")
        adjacency[left] |= 1 << right
        adjacency[right] |= 1 << left
    return tuple(adjacency)


def _adjacency_to_graph(adjacency: tuple[int, ...]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(adjacency)))
    for left, neighbors in enumerate(adjacency):
        for right in range(left + 1, len(adjacency)):
            if (neighbors >> right) & 1:
                graph.add_edge(left, right)
    return graph


def _local_complement(
    adjacency: tuple[int, ...],
    vertex: int,
) -> tuple[int, ...]:
    result = list(adjacency)
    neighbors = [
        qubit
        for qubit in range(len(adjacency))
        if (adjacency[vertex] >> qubit) & 1
    ]
    for index, left in enumerate(neighbors):
        for right in neighbors[index + 1 :]:
            result[left] ^= 1 << right
            result[right] ^= 1 << left
    return tuple(result)


def _adjacency_bits(adjacency: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        (adjacency[left] >> right) & 1
        for left in range(len(adjacency))
        for right in range(left + 1, len(adjacency))
    )


def _score_adjacency(
    adjacency: tuple[int, ...],
    config: LCSearchConfig,
    arity_cost_cache: dict[int, int] | None = None,
) -> AdjacencyScore:
    cost_function = config.vertex_arity_cost or _default_vertex_arity_cost
    degrees = tuple(neighbors.bit_count() for neighbors in adjacency)
    edge_count = sum(degrees) // 2
    vertex_costs: list[int] = []
    for degree in degrees:
        arity = 2 * degree + 1
        if arity_cost_cache is not None and arity in arity_cost_cache:
            cost = arity_cost_cache[arity]
        else:
            cost = cost_function(arity)
            if arity_cost_cache is not None:
                arity_cost_cache[arity] = cost
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
            raise ValueError(
                "vertex_arity_cost must deterministically return a non-negative int; "
                f"got {cost!r} for arity {arity}."
            )
        vertex_costs.append(cost)
    predicted_spiders = 4 * edge_count + sum(vertex_costs)
    return (
        predicted_spiders,
        max(degrees, default=0),
        edge_count,
        _adjacency_bits(adjacency),
    )


def _reconstruct_path(
    parents: dict[tuple[int, ...], tuple[tuple[int, ...], int] | None],
    state: tuple[int, ...],
) -> tuple[int, ...]:
    reversed_path: list[int] = []
    while parents[state] is not None:
        previous, vertex = parents[state]  # type: ignore[misc]
        reversed_path.append(vertex)
        state = previous
    return tuple(reversed(reversed_path))


def _search_lc_orbit(
    initial: tuple[int, ...],
    config: LCSearchConfig,
) -> tuple[tuple[int, ...], LCSearchMetadata]:
    arity_cost_cache: dict[int, int] = {}

    def score(state: tuple[int, ...]) -> AdjacencyScore:
        return _score_adjacency(state, config, arity_cost_cache)

    initial_score = score(initial)
    if not config.optimize:
        return initial, LCSearchMetadata(
            mode="disabled",
            states_examined=1,
            orbit_exhausted=False,
            guaranteed_optimal=False,
            local_complementations=(),
            score=initial_score,
        )

    n = len(initial)
    if n <= config.exhaustive_qubits:
        parents: dict[
            tuple[int, ...],
            tuple[tuple[int, ...], int] | None,
        ] = {initial: None}
        queue: deque[tuple[int, ...]] = deque([initial])
        best = initial
        best_score = initial_score
        capped = False
        while queue and not capped:
            state = queue.popleft()
            for vertex in range(n):
                nxt = _local_complement(state, vertex)
                if nxt in parents:
                    continue
                if len(parents) >= config.orbit_state_cap:
                    capped = True
                    break
                parents[nxt] = (state, vertex)
                queue.append(nxt)
                candidate_score = score(nxt)
                if candidate_score < best_score:
                    best, best_score = nxt, candidate_score
        exhausted = not capped and not queue
        return best, LCSearchMetadata(
            mode="exhaustive" if exhausted else "exhaustive_capped",
            states_examined=len(parents),
            orbit_exhausted=exhausted,
            guaranteed_optimal=exhausted,
            local_complementations=_reconstruct_path(parents, best),
            score=best_score,
        )

    beam: dict[tuple[int, ...], tuple[int, ...]] = {initial: ()}
    seen = {initial}
    best = initial
    best_path: tuple[int, ...] = ()
    best_score = initial_score
    for _round in range(config.beam_rounds):
        candidates: dict[tuple[int, ...], tuple[int, ...]] = {}
        for state in sorted(beam, key=_adjacency_bits):
            path = beam[state]
            for vertex in range(n):
                nxt = _local_complement(state, vertex)
                if nxt in seen:
                    continue
                candidate_path = (*path, vertex)
                old_path = candidates.get(nxt)
                if old_path is None or candidate_path < old_path:
                    candidates[nxt] = candidate_path
        if not candidates:
            break
        seen.update(candidates)
        ranked = sorted(
            candidates,
            key=lambda state: (score(state), candidates[state]),
        )
        beam = {state: candidates[state] for state in ranked[: config.beam_width]}
        for state in ranked:
            candidate_score = score(state)
            path = candidates[state]
            if (candidate_score, path) < (best_score, best_path):
                best, best_score, best_path = state, candidate_score, path

    return best, LCSearchMetadata(
        mode="beam",
        states_examined=len(seen),
        orbit_exhausted=False,
        guaranteed_optimal=False,
        local_complementations=best_path,
        score=best_score,
    )


def _compose_local_complementation_certificate(
    initial: tuple[int, ...],
    initial_certificate: Sequence[LocalClifford],
    path: Sequence[int],
) -> tuple[tuple[int, ...], tuple[LocalClifford, ...]]:
    adjacency = initial
    certificate = list(initial_certificate)
    sqrt_x = LocalClifford.from_gate_word(["SQRT_X"])
    s_dag = LocalClifford.from_gate_word(["S_DAG"])
    for vertex in path:
        neighbors = [
            qubit
            for qubit in range(len(adjacency))
            if (adjacency[vertex] >> qubit) & 1
        ]
        certificate[vertex] = certificate[vertex].followed_by(sqrt_x)
        for qubit in neighbors:
            certificate[qubit] = certificate[qubit].followed_by(s_dag)
        adjacency = _local_complement(adjacency, vertex)
    return adjacency, tuple(certificate)


def local_clifford_layer_to_stim_circuit(
    local_cliffords: Sequence[LocalClifford],
) -> stim.Circuit:
    """Serialize a tensor product of local Cliffords as a Stim circuit."""

    circuit = stim.Circuit()
    if local_cliffords:
        circuit.append("I", range(len(local_cliffords)))
    for qubit, clifford in enumerate(local_cliffords):
        for gate in clifford.gate_word:
            circuit.append(gate, [qubit])
    return circuit


def graph_state_stim_circuit(graph: nx.Graph) -> stim.Circuit:
    """Return the canonical ``H`` then ``CZ`` preparation of a graph state."""

    adjacency = _graph_to_adjacency(graph)
    circuit = stim.Circuit()
    if adjacency:
        circuit.append("H", range(len(adjacency)))
    for left in range(len(adjacency)):
        for right in range(left + 1, len(adjacency)):
            if (adjacency[left] >> right) & 1:
                circuit.append("CZ", [left, right])
    return circuit


def _validate_certificate(
    original_rows: Sequence[_Pauli],
    graph: nx.Graph,
    certificate: Sequence[LocalClifford],
) -> None:
    n = original_rows[0].n
    if len(certificate) != n or len(graph) != n:
        raise LocalCliffordCertificateError(
            "Certificate, graph, and stabilizer state have different qubit counts."
        )

    # First check our exact phase-aware conjugation implementation.
    graph_basis: dict[int, _Pauli] = {}
    adjacency = _graph_to_adjacency(graph)
    for qubit, neighbors in enumerate(adjacency):
        _insert_into_pauli_basis(
            _Pauli(n=n, x=1 << qubit, z=neighbors, phase=0),
            graph_basis,
        )
    transformed_rows: list[_Pauli] = []
    for original in original_rows:
        transformed = original
        for qubit, clifford in enumerate(certificate):
            transformed = transformed.conjugated_at(qubit, clifford)
        transformed_rows.append(transformed)
        remainder = transformed
        while remainder.vector:
            pivot = remainder.vector.bit_length() - 1
            basis_row = graph_basis.get(pivot)
            if basis_row is None:
                break
            remainder = remainder.multiplied_by(basis_row)
        if remainder.vector or remainder.physical_sign:
            raise LocalCliffordCertificateError(
                "The local Clifford certificate does not map the input "
                "stabilizer group to the graph stabilizer group."
            )

    # Independently conjugate and evaluate the generators using Stim.
    local_circuit = local_clifford_layer_to_stim_circuit(certificate)
    graph_simulator = stim.TableauSimulator()
    graph_simulator.do_circuit(graph_state_stim_circuit(graph))
    for original, expected in zip(original_rows, transformed_rows, strict=True):
        stim_transformed = stim.PauliString(original.to_signed_string()).after(local_circuit)
        if str(stim_transformed).replace("_", "I") != expected.to_signed_string():
            raise LocalCliffordCertificateError(
                "Internal Clifford conjugation disagrees with Stim."
            )
        if graph_simulator.peek_observable_expectation(stim_transformed) != 1:
            raise LocalCliffordCertificateError(
                "Stim rejected a transformed generator as a +1 graph stabilizer."
            )


@dataclass(frozen=True, slots=True)
class StabilizerGraphResult:
    """An LC-equivalent graph and an exact preparation certificate."""

    graph: nx.Graph
    local_cliffords_to_graph: tuple[LocalClifford, ...]
    input_stabilizers: tuple[str, ...]
    search: LCSearchMetadata
    direct_graph: nx.Graph

    @property
    def local_cliffords_from_graph(self) -> tuple[LocalClifford, ...]:
        """Boundary corrections mapping the graph state back to the input."""

        return tuple(clifford.inverse() for clifford in self.local_cliffords_to_graph)

    def certificate_stim_circuit(self, *, inverse: bool = False) -> stim.Circuit:
        """Serialize the certificate (or boundary inverse) as a Stim circuit."""

        layer = (
            self.local_cliffords_from_graph
            if inverse
            else self.local_cliffords_to_graph
        )
        return local_clifford_layer_to_stim_circuit(layer)

    def validate_certificate(self) -> bool:
        """Revalidate this result exactly and with Stim; raise on failure."""

        original_rows = tuple(
            _parse_pauli(text, index) for index, text in enumerate(self.input_stabilizers)
        )
        _validate_generators(original_rows)
        _validate_certificate(
            original_rows,
            self.graph,
            self.local_cliffords_to_graph,
        )
        return True


def stabilizer_state_to_graph(
    stabilizers: Sequence[str | stim.PauliString] | stim.Tableau,
    *,
    lc_search: LCSearchConfig = LCSearchConfig(),
) -> StabilizerGraphResult:
    """Return an optimized LC-equivalent graph state and exact certificate.

    Args:
        stabilizers: Exactly ``n`` independent, commuting signed generators on
            ``n`` qubits, or a Stim tableau whose Z outputs define the state.
        lc_search: Deterministic orbit-search and synthesis-cost configuration.

    Returns:
        A graph on nodes ``0..n-1`` and local Cliffords ``U_q`` satisfying
        ``(tensor_q U_q) |input> = |graph>``.
    """

    original_rows = _coerce_generators(stabilizers)
    direct_graph, direct_certificate = _direct_graph_conversion(original_rows)
    _validate_certificate(original_rows, direct_graph, direct_certificate)

    initial_adjacency = _graph_to_adjacency(direct_graph)
    optimized_adjacency, metadata = _search_lc_orbit(initial_adjacency, lc_search)
    reached_adjacency, optimized_certificate = (
        _compose_local_complementation_certificate(
            initial_adjacency,
            direct_certificate,
            metadata.local_complementations,
        )
    )
    if reached_adjacency != optimized_adjacency:
        raise StabilizerGraphError("Internal error reconstructing LC search path.")
    optimized_graph = _adjacency_to_graph(optimized_adjacency)
    _validate_certificate(original_rows, optimized_graph, optimized_certificate)
    return StabilizerGraphResult(
        graph=optimized_graph,
        local_cliffords_to_graph=optimized_certificate,
        input_stabilizers=tuple(row.to_signed_string() for row in original_rows),
        search=metadata,
        direct_graph=direct_graph.copy(),
    )


# Descriptive aliases for callers that prefer a shorter or theorem-style name.
stabilizer_to_graph = stabilizer_state_to_graph
find_lc_graph = stabilizer_state_to_graph


def _normalize_binary_rows(
    rows: Sequence[Sequence[int] | str] | Sequence[int] | str | None,
    *,
    name: str,
) -> list[tuple[int, ...]]:
    if rows is None:
        return []
    if isinstance(rows, str):
        materialized: list[object] = [rows]
    else:
        try:
            materialized = list(rows)
        except TypeError as exc:
            raise InvalidPauliStringError(f"{name} must be a binary row or matrix.") from exc
        if materialized and not isinstance(materialized[0], str):
            try:
                iter(materialized[0])  # type: ignore[arg-type]
            except TypeError:
                materialized = [materialized]

    normalized: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(materialized):
        if isinstance(raw_row, str):
            values: list[object] = list(raw_row.strip())
        else:
            try:
                values = list(raw_row)  # type: ignore[arg-type]
            except TypeError as exc:
                raise InvalidPauliStringError(
                    f"{name}[{row_index}] is not a binary row."
                ) from exc
        converted: list[int] = []
        for value in values:
            try:
                bit = int(value)  # Supports NumPy integer scalar types.
            except (TypeError, ValueError) as exc:
                raise InvalidPauliStringError(
                    f"{name}[{row_index}] contains non-binary value {value!r}."
                ) from exc
            if bit not in (0, 1) or str(value) not in {"0", "1", "False", "True"}:
                # The explicit equality below permits numeric 0/1 without
                # accidentally accepting values such as 0.5 via int().
                if value != bit:
                    raise InvalidPauliStringError(
                        f"{name}[{row_index}] contains non-binary value {value!r}."
                    )
            if bit not in (0, 1):
                raise InvalidPauliStringError(
                    f"{name}[{row_index}] contains non-binary value {value!r}."
                )
            converted.append(bit)
        normalized.append(tuple(converted))
    return normalized


def css_logical_state_stabilizers(
    h_x: Sequence[Sequence[int] | str] | Sequence[int] | str,
    h_z: Sequence[Sequence[int] | str] | Sequence[int] | str,
    *,
    logical_x: Sequence[Sequence[int] | str] | Sequence[int] | str | None = None,
    logical_z: Sequence[Sequence[int] | str] | Sequence[int] | str | None = None,
    state: Literal["0", "+", "zero", "plus", "Z", "X"] = "0",
    eigenvalues: Sequence[int] | int | None = None,
    num_qubits: int | None = None,
) -> tuple[str, ...]:
    """Complete CSS code checks into a signed pure-state generator list.

    ``state="0"`` (or ``"Z"``) appends signed logical-Z generators.
    ``state="+"`` (or ``"X"``) appends signed logical-X generators.  A
    negative eigenvalue selects logical one/minus for the corresponding
    logical qubit.
    """

    x_checks = _normalize_binary_rows(h_x, name="h_x")
    z_checks = _normalize_binary_rows(h_z, name="h_z")
    x_logicals = _normalize_binary_rows(logical_x, name="logical_x")
    z_logicals = _normalize_binary_rows(logical_z, name="logical_z")
    normalized_state = state.lower()
    if normalized_state in {"0", "zero", "z"}:
        chosen_logicals = z_logicals
        logical_pauli = "Z"
    elif normalized_state in {"+", "plus", "x"}:
        chosen_logicals = x_logicals
        logical_pauli = "X"
    else:
        raise ValueError("state must be one of '0', '+', 'zero', 'plus', 'Z', or 'X'.")

    all_rows = [*x_checks, *z_checks, *x_logicals, *z_logicals]
    inferred_widths = {len(row) for row in all_rows}
    if num_qubits is None:
        if not inferred_widths:
            raise InvalidPauliStringError(
                "Cannot infer num_qubits from empty CSS matrices."
            )
        if len(inferred_widths) != 1:
            raise InvalidPauliStringError(
                f"CSS rows have inconsistent lengths {sorted(inferred_widths)}."
            )
        num_qubits = next(iter(inferred_widths))
    if num_qubits < 1:
        raise InvalidPauliStringError("num_qubits must be positive.")
    if any(len(row) != num_qubits for row in all_rows):
        raise InvalidPauliStringError(
            f"Every CSS row must have length num_qubits={num_qubits}."
        )

    if eigenvalues is None:
        signs = [1] * len(chosen_logicals)
    elif isinstance(eigenvalues, int):
        signs = [eigenvalues] * len(chosen_logicals)
    else:
        signs = list(eigenvalues)
    if len(signs) != len(chosen_logicals) or any(sign not in (-1, 1) for sign in signs):
        raise ValueError(
            "eigenvalues must contain exactly one +1/-1 value per chosen logical."
        )

    generators: list[str] = []
    for row in x_checks:
        generators.append("+" + "".join("X" if bit else "I" for bit in row))
    for row in z_checks:
        generators.append("+" + "".join("Z" if bit else "I" for bit in row))
    for row, sign in zip(chosen_logicals, signs, strict=True):
        body = "".join(logical_pauli if bit else "I" for bit in row)
        generators.append(("+" if sign == 1 else "-") + body)

    # The helper promises a completed pure state, so validate it immediately.
    _coerce_generators(generators)
    return tuple(generators)


complete_css_stabilizers = css_logical_state_stabilizers


__all__ = [
    "DependentGeneratorsError",
    "InconsistentStabilizerError",
    "InvalidPauliStringError",
    "LCSearchConfig",
    "LCSearchMetadata",
    "LocalClifford",
    "LocalCliffordCertificateError",
    "NonCommutingGeneratorsError",
    "StabilizerGraphError",
    "StabilizerGraphResult",
    "StabilizerValidationError",
    "UnderconstrainedStabilizerError",
    "complete_css_stabilizers",
    "css_logical_state_stabilizers",
    "find_lc_graph",
    "graph_state_stim_circuit",
    "local_clifford_layer_to_stim_circuit",
    "stabilizer_state_to_graph",
    "stabilizer_to_graph",
]

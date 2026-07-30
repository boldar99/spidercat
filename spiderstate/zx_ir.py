"""Project-owned intermediate representation for staged ZX synthesis.

The repository historically used bare NetworkX graphs for several distinct
objects.  This module deliberately keeps the ZX semantics on every node and
edge so that ideal graph-state wires, noisy Lemma-B* wires, and SpiderCat
gadget wires cannot be confused.

Only phase-zero spiders are represented directly.  Single-qubit Clifford
corrections are kept as opaque boxes on output legs and are expanded only by
the optional PyZX adapter.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from enum import Enum
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import networkx as nx


class NodeKind(str, Enum):
    """Semantic kind of a node in :class:`ZXDiagram`."""

    BOUNDARY = "boundary"
    Z_SPIDER = "z_spider"
    X_SPIDER = "x_spider"
    LOCAL_CLIFFORD = "local_clifford"


class NodeRole(str, Enum):
    """Construction stage that introduced a node."""

    BOUNDARY = "boundary"
    GRAPH_VERTEX = "graph_vertex"
    LEMMA_B_STAR = "lemma_b_star"
    SPIDERCAT = "spidercat"
    LOCAL_CLIFFORD = "local_clifford"


class EdgeKind(str, Enum):
    """ZX interpretation of an edge."""

    SIMPLE = "simple"
    HADAMARD = "hadamard"


class FaultStatus(str, Enum):
    """Whether an edge is an idealized or physical/noisy location."""

    IDEAL = "ideal"
    NOISY = "noisy"


class EdgeRole(str, Enum):
    """Construction stage that introduced an edge."""

    GRAPH_EDGE = "graph_edge"
    LEMMA_B_STAR = "lemma_b_star"
    SPIDERCAT = "spidercat"
    BOUNDARY_EDGE = "boundary_edge"


class ZXDiagramError(ValueError):
    """Base class for errors involving the annotated ZX representation."""


class DiagramValidationError(ZXDiagramError):
    """Raised when a diagram violates an IR invariant."""


class LemmaBStarError(ZXDiagramError):
    """Raised when Lemma B* cannot be applied to a graph-state edge."""


class PyZXAdapterError(ZXDiagramError):
    """Raised when a diagram cannot be faithfully converted to PyZX."""


_NODE_ENUM_FIELDS = {
    "kind": NodeKind,
    "role": NodeRole,
}
_EDGE_ENUM_FIELDS = {
    "kind": EdgeKind,
    "fault_status": FaultStatus,
    "role": EdgeRole,
}


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation of ``value``."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return value.numerator
        return {"numerator": value.numerator, "denominator": value.denominator}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    return {
        "__python_type__": (
            f"{value.__class__.__module__}.{value.__class__.__qualname__}"
        ),
        "repr": repr(value),
    }


def _stable_key(value: Any) -> tuple[Any, ...]:
    """Return a deterministic key without lexicographically sorting numbers."""

    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, float):
        return (2, value)
    if isinstance(value, str):
        return (3, value)
    if isinstance(value, tuple):
        return (4, tuple(_stable_key(item) for item in value))
    return (
        5,
        type(value).__module__,
        type(value).__qualname__,
        json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")),
    )


def _coerce_enum(value: Any, enum_type: type[Enum], field: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise DiagramValidationError(
            f"Invalid {field} {value!r}; expected {enum_type.__name__}."
        ) from exc


def _canonical_gate_name(gate: Any) -> str:
    if isinstance(gate, Enum):
        gate = gate.value
    name = str(gate).strip().upper().replace("-", "_")
    aliases = {
        "IDENTITY": "I",
        "ID": "I",
        "S†": "S_DAG",
        "S_DAGGER": "S_DAG",
        "SDAG": "S_DAG",
        "S^-1": "S_DAG",
        "S**_1": "S_DAG",
        "SX": "SQRT_X",
        "SX_DAG": "SQRT_X_DAG",
        "SX_DAGGER": "SQRT_X_DAG",
        "SQRT_X_DAGGER": "SQRT_X_DAG",
        "SQRT_Z": "S",
        "SQRT_Z_DAG": "S_DAG",
        "SQRT_Z_DAGGER": "S_DAG",
    }
    return aliases.get(name, name)


def normalize_gate_word(correction: Any) -> tuple[str, ...]:
    """Normalize a local-Clifford description into a deterministic gate word.

    Strings (whitespace/comma separated), sequences of gate names, and simple
    certificate objects exposing ``gate_word``, ``gates``, ``operations``, or
    ``word`` are accepted.  Identity operations are removed.
    """

    if correction is None:
        return ()
    for attribute in ("gate_word", "gates", "operations", "word"):
        if hasattr(correction, attribute):
            return normalize_gate_word(getattr(correction, attribute))
    if isinstance(correction, str):
        cleaned = correction.replace(",", " ").replace(";", " ")
        pieces = tuple(piece for piece in cleaned.split() if piece)
    elif isinstance(correction, Sequence):
        pieces = tuple(correction)
    else:
        raise ZXDiagramError(
            "A local Clifford must be a gate-name string, a sequence of gate "
            "names, or expose a gate_word/gates/operations/word attribute."
        )

    normalized = tuple(_canonical_gate_name(piece) for piece in pieces)
    return tuple(gate for gate in normalized if gate != "I")


class ZXDiagram:
    """An annotated, simple, undirected ZX graph.

    The underlying :class:`networkx.Graph` remains available through
    :attr:`graph` for graph algorithms.  Mutation helpers assign and preserve
    stable edge identifiers and deep-copy provenance records.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if graph is None:
            self.graph = nx.Graph()
        else:
            if graph.is_directed() or graph.is_multigraph():
                raise ZXDiagramError(
                    "ZXDiagram requires a simple undirected NetworkX graph."
                )
            self.graph = deepcopy(graph)
        self.metadata: dict[str, Any] = deepcopy(dict(metadata or {}))
        self._next_edge_number = self._infer_next_edge_number()

    def _infer_next_edge_number(self) -> int:
        largest = -1
        for _, _, data in self.graph.edges(data=True):
            edge_id = data.get("edge_id")
            if not isinstance(edge_id, str) or not edge_id.startswith("edge:"):
                continue
            suffix = edge_id.removeprefix("edge:")
            if suffix.isdigit():
                largest = max(largest, int(suffix))
        return largest + 1

    def _new_edge_id(self) -> str:
        existing = {
            data.get("edge_id") for _, _, data in self.graph.edges(data=True)
        }
        while True:
            candidate = f"edge:{self._next_edge_number:06d}"
            self._next_edge_number += 1
            if candidate not in existing:
                return candidate

    def add_node(
        self,
        node_id: str,
        *,
        kind: NodeKind,
        role: NodeRole,
        phase: int | Fraction = 0,
        provenance: Any,
        **attributes: Any,
    ) -> str:
        """Add an annotated node and return its stable identifier."""

        if not isinstance(node_id, str) or not node_id:
            raise ZXDiagramError("Node identifiers must be nonempty strings.")
        if node_id in self.graph:
            raise ZXDiagramError(f"Node identifier {node_id!r} already exists.")
        kind = _coerce_enum(kind, NodeKind, "node kind")
        role = _coerce_enum(role, NodeRole, "node role")
        data = {
            "kind": kind,
            "role": role,
            "phase": phase,
            "provenance": deepcopy(provenance),
            **deepcopy(attributes),
        }
        if kind is NodeKind.LOCAL_CLIFFORD and "gate_word" in data:
            data["gate_word"] = normalize_gate_word(data["gate_word"])
        self.graph.add_node(node_id, **data)
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        kind: EdgeKind,
        fault_status: FaultStatus,
        role: EdgeRole,
        provenance: Any,
        edge_id: str | None = None,
        **attributes: Any,
    ) -> str:
        """Add an annotated edge and return its stable identifier."""

        if source not in self.graph or target not in self.graph:
            missing = [node for node in (source, target) if node not in self.graph]
            raise ZXDiagramError(f"Cannot connect missing node(s): {missing!r}.")
        if source == target:
            raise ZXDiagramError("ZXDiagram does not permit self-loops.")
        if self.graph.has_edge(source, target):
            raise ZXDiagramError(
                f"Parallel edge between {source!r} and {target!r} is not permitted."
            )
        kind = _coerce_enum(kind, EdgeKind, "edge kind")
        fault_status = _coerce_enum(
            fault_status, FaultStatus, "edge fault status"
        )
        role = _coerce_enum(role, EdgeRole, "edge role")
        if edge_id is None:
            edge_id = self._new_edge_id()
        elif not isinstance(edge_id, str) or not edge_id:
            raise ZXDiagramError("Edge identifiers must be nonempty strings.")
        elif any(
            data.get("edge_id") == edge_id
            for _, _, data in self.graph.edges(data=True)
        ):
            raise ZXDiagramError(f"Edge identifier {edge_id!r} already exists.")
        self.graph.add_edge(
            source,
            target,
            edge_id=edge_id,
            kind=kind,
            fault_status=fault_status,
            role=role,
            provenance=deepcopy(provenance),
            **deepcopy(attributes),
        )
        return edge_id

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.graph:
            raise ZXDiagramError(f"Unknown node {node_id!r}.")
        self.graph.remove_node(node_id)

    def remove_edge(self, source: str, target: str) -> None:
        if not self.graph.has_edge(source, target):
            raise ZXDiagramError(
                f"Unknown edge between {source!r} and {target!r}."
            )
        self.graph.remove_edge(source, target)

    def incident_edges(self, node_id: str) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(other_endpoint, attributes)`` records in stable order."""

        if node_id not in self.graph:
            raise ZXDiagramError(f"Unknown node {node_id!r}.")
        records = [
            (other, deepcopy(data))
            for _, other, data in self.graph.edges(node_id, data=True)
        ]
        return sorted(
            records,
            key=lambda record: (
                str(record[1].get("edge_id", "")),
                record[0],
            ),
        )

    def copy(self) -> "ZXDiagram":
        return ZXDiagram(self.graph, metadata=self.metadata)

    def nodes_of_kind(self, kind: NodeKind) -> list[str]:
        kind = _coerce_enum(kind, NodeKind, "node kind")
        return sorted(
            node
            for node, data in self.graph.nodes(data=True)
            if data.get("kind") is kind
        )

    def edges_of_role(
        self, role: EdgeRole
    ) -> list[tuple[str, str, dict[str, Any]]]:
        role = _coerce_enum(role, EdgeRole, "edge role")
        records = [
            (source, target, deepcopy(data))
            for source, target, data in self.graph.edges(data=True)
            if data.get("role") is role
        ]
        return sorted(
            records,
            key=lambda record: (
                str(record[2].get("edge_id", "")),
                min(record[0], record[1]),
                max(record[0], record[1]),
            ),
        )

    def internal_edges(
        self,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Return edges whose endpoints are both spiders."""

        spider_kinds = {NodeKind.Z_SPIDER, NodeKind.X_SPIDER}
        records = []
        for source, target, data in self.graph.edges(data=True):
            if (
                self.graph.nodes[source].get("kind") in spider_kinds
                and self.graph.nodes[target].get("kind") in spider_kinds
            ):
                records.append((source, target, deepcopy(data)))
        return sorted(
            records,
            key=lambda record: str(record[2].get("edge_id", "")),
        )

    def spider_arity(self, node_id: str) -> int:
        if node_id not in self.graph:
            raise ZXDiagramError(f"Unknown node {node_id!r}.")
        if self.graph.nodes[node_id].get("kind") not in {
            NodeKind.Z_SPIDER,
            NodeKind.X_SPIDER,
        }:
            raise ZXDiagramError(f"Node {node_id!r} is not a spider.")
        return self.graph.degree(node_id)

    def validate(self) -> None:
        """Check structural and semantic IR invariants."""

        if self.graph.is_directed() or self.graph.is_multigraph():
            raise DiagramValidationError(
                "ZXDiagram must remain a simple undirected graph."
            )
        if nx.number_of_selfloops(self.graph):
            raise DiagramValidationError("ZXDiagram contains a self-loop.")

        for node_id, data in self.graph.nodes(data=True):
            if not isinstance(node_id, str) or not node_id:
                raise DiagramValidationError(
                    "Every node identifier must be a nonempty string."
                )
            missing = {"kind", "role", "phase", "provenance"} - data.keys()
            if missing:
                raise DiagramValidationError(
                    f"Node {node_id!r} is missing attributes {sorted(missing)!r}."
                )
            kind = _coerce_enum(data["kind"], NodeKind, "node kind")
            role = _coerce_enum(data["role"], NodeRole, "node role")
            data["kind"] = kind
            data["role"] = role
            if kind in {NodeKind.Z_SPIDER, NodeKind.X_SPIDER}:
                if data["phase"] != 0:
                    raise DiagramValidationError(
                        f"Spider {node_id!r} has nonzero phase {data['phase']!r}; "
                        "this IR keeps phase corrections in boundary boxes."
                    )
            if kind is NodeKind.BOUNDARY and self.graph.degree(node_id) != 1:
                raise DiagramValidationError(
                    f"Boundary {node_id!r} must have degree one."
                )
            if kind is NodeKind.LOCAL_CLIFFORD:
                if self.graph.degree(node_id) != 2:
                    raise DiagramValidationError(
                        f"Local-Clifford box {node_id!r} must have degree two."
                    )
                gate_word = normalize_gate_word(data.get("gate_word", ()))
                if not gate_word:
                    raise DiagramValidationError(
                        f"Local-Clifford box {node_id!r} is the identity and "
                        "should be omitted."
                    )

        edge_ids: set[str] = set()
        for source, target, data in self.graph.edges(data=True):
            missing = {
                "edge_id",
                "kind",
                "fault_status",
                "role",
                "provenance",
            } - data.keys()
            if missing:
                raise DiagramValidationError(
                    f"Edge {source!r}-{target!r} is missing attributes "
                    f"{sorted(missing)!r}."
                )
            edge_id = data["edge_id"]
            if not isinstance(edge_id, str) or not edge_id:
                raise DiagramValidationError(
                    f"Edge {source!r}-{target!r} has an invalid edge_id."
                )
            if edge_id in edge_ids:
                raise DiagramValidationError(
                    f"Duplicate edge identifier {edge_id!r}."
                )
            edge_ids.add(edge_id)
            kind = _coerce_enum(data["kind"], EdgeKind, "edge kind")
            status = _coerce_enum(
                data["fault_status"], FaultStatus, "edge fault status"
            )
            role = _coerce_enum(data["role"], EdgeRole, "edge role")
            data["kind"] = kind
            data["fault_status"] = status
            data["role"] = role
            if role is EdgeRole.GRAPH_EDGE and (
                kind is not EdgeKind.HADAMARD
                or status is not FaultStatus.IDEAL
            ):
                raise DiagramValidationError(
                    f"Original graph edge {edge_id!r} must be ideal Hadamard."
                )
            if role is EdgeRole.BOUNDARY_EDGE and kind is not EdgeKind.SIMPLE:
                raise DiagramValidationError(
                    f"Boundary edge {edge_id!r} must be simple."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the diagram to a stable, JSON-compatible dictionary."""

        nodes = []
        for node_id in sorted(self.graph.nodes):
            data = self.graph.nodes[node_id]
            nodes.append(
                {
                    "id": node_id,
                    **{
                        key: _jsonable(value)
                        for key, value in sorted(data.items())
                    },
                }
            )

        edges = []
        records = sorted(
            self.graph.edges(data=True),
            key=lambda record: (
                str(record[2].get("edge_id", "")),
                min(record[0], record[1]),
                max(record[0], record[1]),
            ),
        )
        for source, target, data in records:
            source, target = sorted((source, target))
            edges.append(
                {
                    "source": source,
                    "target": target,
                    **{
                        key: _jsonable(value)
                        for key, value in sorted(data.items())
                    },
                }
            )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "metadata": _jsonable(self.metadata),
            "nodes": nodes,
            "edges": edges,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        separators = None if indent is not None else (",", ":")
        return json.dumps(
            self.to_dict(),
            indent=indent,
            separators=separators,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZXDiagram":
        """Restore a diagram serialized by :meth:`to_dict`."""

        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ZXDiagramError(
                f"Unsupported ZXDiagram schema version "
                f"{payload.get('schema_version')!r}."
            )
        diagram = cls(metadata=payload.get("metadata", {}))
        for serialized in payload.get("nodes", []):
            data = dict(serialized)
            node_id = data.pop("id")
            for field, enum_type in _NODE_ENUM_FIELDS.items():
                data[field] = enum_type(data[field])
            diagram.add_node(node_id, **data)
        for serialized in payload.get("edges", []):
            data = dict(serialized)
            source = data.pop("source")
            target = data.pop("target")
            for field, enum_type in _EDGE_ENUM_FIELDS.items():
                data[field] = enum_type(data[field])
            diagram.add_edge(source, target, **data)
        diagram.validate()
        return diagram

    def render_svg(
        self,
        path: str | Path | None = None,
        *,
        title: str | None = None,
    ) -> str:
        """Return a deterministic, dependency-free SVG rendering.

        If ``path`` is supplied, the exact returned string is also written as
        UTF-8 text.
        """

        svg = _render_svg(self, title=title)
        if path is not None:
            Path(path).write_text(svg, encoding="utf-8")
        return svg

    def to_pyzx(self):
        """Convert this diagram to a PyZX graph, expanding LC boxes."""

        return _to_pyzx(self)


def _ordered_graph_nodes(graph: nx.Graph) -> list[Hashable]:
    return sorted(graph.nodes, key=_stable_key)


def _correction_for_vertex(
    local_corrections: Mapping[Hashable, Any] | Sequence[Any] | None,
    vertex: Hashable,
    index: int,
    count: int,
) -> tuple[str, ...]:
    if local_corrections is None:
        return ()
    if isinstance(local_corrections, Mapping):
        return normalize_gate_word(local_corrections.get(vertex))
    if isinstance(local_corrections, (str, bytes)):
        raise ZXDiagramError(
            "local_corrections must be a mapping or one entry per graph vertex."
        )
    if len(local_corrections) != count:
        raise ZXDiagramError(
            f"Expected {count} local corrections, got {len(local_corrections)}."
        )
    return normalize_gate_word(local_corrections[index])


def build_ideal_graph_state_diagram(
    graph: nx.Graph,
    local_corrections: Mapping[Hashable, Any] | Sequence[Any] | None = None,
) -> ZXDiagram:
    """Build the exact-state boundary diagram for an LC graph representative.

    ``local_corrections`` are understood to be the corrections applied from
    the graph state towards the requested state (typically ``U†`` when the
    graph-state certificate uses the convention ``U|ψ⟩ = |G⟩``).
    """

    if graph.is_directed() or graph.is_multigraph():
        raise ZXDiagramError("The graph-state representative must be simple.")
    if nx.number_of_selfloops(graph):
        raise ZXDiagramError("A graph state cannot contain self-loops.")

    vertices = _ordered_graph_nodes(graph)
    index_of = {vertex: index for index, vertex in enumerate(vertices)}
    diagram = ZXDiagram(
        metadata={
            "stage": "ideal_graph_state",
            "lemma_b_star_applied": False,
            "qubit_order": [_jsonable(vertex) for vertex in vertices],
            "graph_vertex_nodes": [],
        }
    )

    count = len(vertices)
    spacing = 76
    for index, vertex in enumerate(vertices):
        y = 76 + spacing * index
        source_vertex = _jsonable(vertex)
        z_node = f"z:q{index}"
        output = f"out:q{index}"
        diagram.add_node(
            z_node,
            kind=NodeKind.Z_SPIDER,
            role=NodeRole.GRAPH_VERTEX,
            phase=0,
            provenance={
                "source": "graph_vertex",
                "graph_vertex": source_vertex,
                "qubit": index,
            },
            qubit=index,
            source_vertex=source_vertex,
            label=f"q{index}",
            layout_hint=(710, y),
        )
        diagram.add_node(
            output,
            kind=NodeKind.BOUNDARY,
            role=NodeRole.BOUNDARY,
            phase=0,
            provenance={
                "source": "input_qubit",
                "graph_vertex": source_vertex,
                "qubit": index,
            },
            qubit=index,
            source_vertex=source_vertex,
            label=f"q{index}",
            layout_hint=(1010, y),
        )

        gate_word = _correction_for_vertex(
            local_corrections, vertex, index, count
        )
        if gate_word:
            box = f"lc:q{index}"
            diagram.add_node(
                box,
                kind=NodeKind.LOCAL_CLIFFORD,
                role=NodeRole.LOCAL_CLIFFORD,
                phase=0,
                provenance={
                    "source": "local_clifford_correction",
                    "graph_vertex": source_vertex,
                    "qubit": index,
                },
                gate_word=gate_word,
                label=" ".join(gate_word),
                qubit=index,
                source_vertex=source_vertex,
                layout_hint=(875, y),
            )
            diagram.add_edge(
                z_node,
                box,
                kind=EdgeKind.SIMPLE,
                fault_status=FaultStatus.NOISY,
                role=EdgeRole.BOUNDARY_EDGE,
                provenance={
                    "source": "output_leg",
                    "qubit": index,
                    "segment": 0,
                },
                edge_id=f"boundary:q{index}:0",
            )
            diagram.add_edge(
                box,
                output,
                kind=EdgeKind.SIMPLE,
                fault_status=FaultStatus.NOISY,
                role=EdgeRole.BOUNDARY_EDGE,
                provenance={
                    "source": "output_leg",
                    "qubit": index,
                    "segment": 1,
                },
                edge_id=f"boundary:q{index}:1",
            )
        else:
            diagram.add_edge(
                z_node,
                output,
                kind=EdgeKind.SIMPLE,
                fault_status=FaultStatus.NOISY,
                role=EdgeRole.BOUNDARY_EDGE,
                provenance={
                    "source": "output_leg",
                    "qubit": index,
                    "segment": 0,
                },
                edge_id=f"boundary:q{index}:0",
            )
        diagram.metadata["graph_vertex_nodes"].append(
            {
                "graph_vertex": source_vertex,
                "qubit": index,
                "node_id": z_node,
                "output_id": output,
            }
        )

    ordered_edges = []
    for left, right in graph.edges:
        left_index, right_index = index_of[left], index_of[right]
        if right_index < left_index:
            left, right = right, left
            left_index, right_index = right_index, left_index
        ordered_edges.append((left_index, right_index, left, right))
    ordered_edges.sort(key=lambda item: (item[0], item[1]))

    for edge_index, (left_index, right_index, left, right) in enumerate(
        ordered_edges
    ):
        source_id = f"graph-edge:{edge_index:06d}"
        diagram.add_edge(
            f"z:q{left_index}",
            f"z:q{right_index}",
            kind=EdgeKind.HADAMARD,
            fault_status=FaultStatus.IDEAL,
            role=EdgeRole.GRAPH_EDGE,
            provenance={
                "source": "graph_edge",
                "source_edge_id": source_id,
                "endpoints": [_jsonable(left), _jsonable(right)],
                "qubits": [left_index, right_index],
            },
            edge_id=source_id,
        )

    diagram.validate()
    return diagram


def apply_lemma_b_star(diagram: ZXDiagram) -> ZXDiagram:
    """Unidealize every original graph edge exactly once using Lemma B*.

    Each ideal Hadamard graph edge is replaced by four phase-zero X spiders.
    Each endpoint is joined by noisy simple edges to its two local X spiders;
    the two pairs are joined by all four noisy Hadamard edges (a ``K₂,₂``).
    The input diagram is never mutated.  Calling this function on an already
    rewritten result is idempotent.
    """

    diagram.validate()
    result = diagram.copy()
    if result.metadata.get("lemma_b_star_applied", False):
        return result

    graph_edges = result.edges_of_role(EdgeRole.GRAPH_EDGE)
    rewritten_source_ids: list[str] = []
    for rewrite_index, (left, right, edge_data) in enumerate(graph_edges):
        if (
            edge_data["kind"] is not EdgeKind.HADAMARD
            or edge_data["fault_status"] is not FaultStatus.IDEAL
        ):
            raise LemmaBStarError(
                f"Graph edge {edge_data['edge_id']!r} is not an ideal "
                "Hadamard edge."
            )
        for endpoint in (left, right):
            node = result.graph.nodes[endpoint]
            if (
                node.get("kind") is not NodeKind.Z_SPIDER
                or node.get("phase") != 0
            ):
                raise LemmaBStarError(
                    f"Endpoint {endpoint!r} of {edge_data['edge_id']!r} must "
                    "be a phase-zero Z spider."
                )
            if result.graph.degree(endpoint) < 2:
                raise LemmaBStarError(
                    f"Endpoint {endpoint!r} of {edge_data['edge_id']!r} must "
                    "have at least one leg besides the rewritten edge."
                )

        original_edge_id = edge_data["edge_id"]
        original_provenance = deepcopy(edge_data["provenance"])
        left, right = sorted(
            (left, right),
            key=lambda node_id: (
                result.graph.nodes[node_id].get("qubit", math.inf),
                node_id,
            ),
        )
        left_qubit = result.graph.nodes[left].get("qubit")
        right_qubit = result.graph.nodes[right].get("qubit")
        left_y = result.graph.nodes[left].get("layout_hint", (710, 100))[1]
        right_y = result.graph.nodes[right].get("layout_hint", (710, 300))[1]
        center_y = (left_y + right_y) / 2
        lane = rewrite_index % 4
        center_x = 115 + 135 * lane

        result.remove_edge(left, right)
        side_nodes: dict[str, list[str]] = {"left": [], "right": []}
        for side, endpoint, qubit, x_sign in (
            ("left", left, left_qubit, -1),
            ("right", right, right_qubit, 1),
        ):
            for copy_index, y_sign in ((0, -1), (1, 1)):
                node_id = (
                    f"lemma:{original_edge_id}:{side}:x{copy_index}"
                )
                result.add_node(
                    node_id,
                    kind=NodeKind.X_SPIDER,
                    role=NodeRole.LEMMA_B_STAR,
                    phase=0,
                    provenance={
                        "source": "lemma_b_star_vertex",
                        "source_edge_id": original_edge_id,
                        "source_edge": original_provenance,
                        "endpoint": endpoint,
                        "endpoint_qubit": qubit,
                        "side": side,
                        "copy": copy_index,
                    },
                    source_edge_id=original_edge_id,
                    endpoint=endpoint,
                    endpoint_qubit=qubit,
                    copy=copy_index,
                    layout_hint=(
                        center_x + 42 * x_sign,
                        center_y + 24 * y_sign,
                    ),
                )
                side_nodes[side].append(node_id)
                result.add_edge(
                    endpoint,
                    node_id,
                    kind=EdgeKind.SIMPLE,
                    fault_status=FaultStatus.NOISY,
                    role=EdgeRole.LEMMA_B_STAR,
                    provenance={
                        "source": "lemma_b_star_endpoint_leg",
                        "source_edge_id": original_edge_id,
                        "source_edge": original_provenance,
                        "side": side,
                        "copy": copy_index,
                    },
                )

        for left_copy, left_node in enumerate(side_nodes["left"]):
            for right_copy, right_node in enumerate(side_nodes["right"]):
                result.add_edge(
                    left_node,
                    right_node,
                    kind=EdgeKind.HADAMARD,
                    fault_status=FaultStatus.NOISY,
                    role=EdgeRole.LEMMA_B_STAR,
                    provenance={
                        "source": "lemma_b_star_cross_edge",
                        "source_edge_id": original_edge_id,
                        "source_edge": original_provenance,
                        "left_copy": left_copy,
                        "right_copy": right_copy,
                    },
                )
        rewritten_source_ids.append(original_edge_id)

    result.metadata["stage"] = "lemma_b_star"
    result.metadata["lemma_b_star_applied"] = True
    result.metadata["lemma_b_star_sources"] = rewritten_source_ids
    result.validate()
    return result


def _node_layout(diagram: ZXDiagram) -> tuple[dict[str, tuple[float, float]], int]:
    nodes = sorted(diagram.graph.nodes)
    boundary_nodes = sorted(
        (
            node
            for node in nodes
            if diagram.graph.nodes[node]["kind"] is NodeKind.BOUNDARY
        ),
        key=lambda node: (
            diagram.graph.nodes[node].get("qubit", math.inf),
            node,
        ),
    )
    height = max(400, 150 + 76 * max(1, len(boundary_nodes)))
    positions: dict[str, tuple[float, float]] = {}
    for node in nodes:
        hint = diagram.graph.nodes[node].get("layout_hint")
        if (
            isinstance(hint, (tuple, list))
            and len(hint) == 2
            and all(isinstance(value, (int, float)) for value in hint)
        ):
            positions[node] = (float(hint[0]), float(hint[1]))

    for index, node in enumerate(boundary_nodes):
        positions.setdefault(node, (1010.0, 76.0 + 76.0 * index))

    local_boxes = [
        node
        for node in nodes
        if diagram.graph.nodes[node]["kind"] is NodeKind.LOCAL_CLIFFORD
    ]
    for node in local_boxes:
        boundary_neighbors = [
            neighbor
            for neighbor in diagram.graph.neighbors(node)
            if diagram.graph.nodes[neighbor]["kind"] is NodeKind.BOUNDARY
        ]
        if boundary_neighbors:
            y = positions[boundary_neighbors[0]][1]
            positions.setdefault(node, (875.0, y))

    # Keep each SpiderCat replacement visually local to the spider it
    # replaced.  A global circle interleaves independent gadgets and turns
    # their preserved external wires into an unreadable knot.
    source_qubits = {
        entry.get("node_id"): entry.get("qubit")
        for entry in diagram.metadata.get("graph_vertex_nodes", ())
        if isinstance(entry, Mapping)
    }
    spidercat_groups: dict[Any, list[str]] = {}
    for node in nodes:
        data = diagram.graph.nodes[node]
        if data["role"] is not NodeRole.SPIDERCAT:
            continue
        provenance = data.get("provenance")
        source = (
            provenance.get("source_node")
            if isinstance(provenance, Mapping)
            else None
        )
        spidercat_groups.setdefault(source, []).append(node)

    fallback_sources = {
        source: index
        for index, source in enumerate(
            sorted(spidercat_groups, key=_stable_key)
        )
    }
    for source, group in sorted(
        spidercat_groups.items(),
        key=lambda item: _stable_key(item[0]),
    ):
        qubit = source_qubits.get(source)
        if isinstance(qubit, int) and 0 <= qubit < len(boundary_nodes):
            center_y = positions[boundary_nodes[qubit]][1]
        else:
            center_y = 76.0 + 76.0 * fallback_sources[source]
        ordered_group = sorted(group, key=_stable_key)
        radius = min(42.0, max(20.0, 4.5 * len(ordered_group)))
        for index, node in enumerate(ordered_group):
            angle = -math.pi / 2 + 2 * math.pi * index / len(ordered_group)
            positions.setdefault(
                node,
                (
                    710.0 + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                ),
            )

    unplaced = [node for node in nodes if node not in positions]
    count = max(1, len(unplaced))
    radius_x = min(280.0, 26.0 * count)
    radius_y = min((height - 120) / 2, max(80.0, 18.0 * count))
    for index, node in enumerate(unplaced):
        angle = -math.pi / 2 + 2 * math.pi * index / count
        positions[node] = (
            390.0 + radius_x * math.cos(angle),
            height / 2 + radius_y * math.sin(angle),
        )
    return positions, height


def _render_svg(diagram: ZXDiagram, *, title: str | None) -> str:
    diagram.validate()
    positions, height = _node_layout(diagram)
    width = 1100
    title = title or str(diagram.metadata.get("stage", "ZX diagram"))
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img">'
        ),
        f"  <title>{escape(title)}</title>",
        "  <rect width=\"100%\" height=\"100%\" fill=\"#fffdf8\"/>",
        (
            "  <style>text{font-family:ui-monospace,SFMono-Regular,"
            "Menlo,monospace}.edge{fill:none;stroke-width:2.2}"
            ".ideal{stroke-dasharray:7 5}.label{font-size:12px;"
            "fill:#202124}.legend{font-size:11px;fill:#4b5563}</style>"
        ),
    ]

    edge_records = sorted(
        diagram.graph.edges(data=True),
        key=lambda record: (
            str(record[2]["edge_id"]),
            min(record[0], record[1]),
            max(record[0], record[1]),
        ),
    )
    for source, target, data in edge_records:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        color = "#1976d2" if data["kind"] is EdgeKind.HADAMARD else "#202124"
        ideal_class = (
            " ideal" if data["fault_status"] is FaultStatus.IDEAL else ""
        )
        lines.append(
            (
                f"  <line id={quoteattr(str(data['edge_id']))} "
                f'class="edge{ideal_class}" x1="{x1:.2f}" y1="{y1:.2f}" '
                f'x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" '
                f'data-kind="{data["kind"].value}" '
                f'data-fault-status="{data["fault_status"].value}" '
                f'data-role="{data["role"].value}"/>'
            )
        )

    for node in sorted(diagram.graph.nodes):
        data = diagram.graph.nodes[node]
        x, y = positions[node]
        kind = data["kind"]
        common = (
            f"id={quoteattr(str(node))} "
            f"data-kind={quoteattr(kind.value)} "
            f"data-role={quoteattr(data['role'].value)}"
        )
        if kind is NodeKind.Z_SPIDER:
            lines.append(
                f'  <circle {common} cx="{x:.2f}" cy="{y:.2f}" r="8" '
                'fill="#ffffff" stroke="#15803d" stroke-width="3"/>'
            )
        elif kind is NodeKind.X_SPIDER:
            lines.append(
                f'  <circle {common} cx="{x:.2f}" cy="{y:.2f}" r="7" '
                'fill="#c62828" stroke="#7f1d1d" stroke-width="1.5"/>'
            )
        elif kind is NodeKind.BOUNDARY:
            lines.append(
                f'  <circle {common} cx="{x:.2f}" cy="{y:.2f}" r="4" '
                'fill="#111827"/>'
            )
        else:
            label = escape(str(data.get("label", "LC")))
            box_width = max(42, min(118, 12 + 7 * len(label)))
            lines.append(
                f'  <rect {common} x="{x - box_width / 2:.2f}" '
                f'y="{y - 13:.2f}" width="{box_width}" height="26" rx="4" '
                'fill="#ede9fe" stroke="#6d28d9" stroke-width="2"/>'
            )
            lines.append(
                f'  <text class="label" x="{x:.2f}" y="{y + 4:.2f}" '
                f'text-anchor="middle">{label}</text>'
            )

        if kind is NodeKind.BOUNDARY:
            label = escape(str(data.get("label", node)))
            lines.append(
                f'  <text class="label" x="{x + 10:.2f}" y="{y + 4:.2f}">'
                f"{label}</text>"
            )

    lines.extend(
        [
            (
                f'  <text class="legend" x="18" y="{height - 38}">'
                "blue = Hadamard edge; black = simple edge</text>"
            ),
            (
                f'  <text class="legend" x="18" y="{height - 20}">'
                "dashed = ideal; solid = noisy</text>"
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _to_pyzx(diagram: ZXDiagram):
    diagram.validate()
    try:
        import pyzx as zx
    except ImportError as exc:
        raise PyZXAdapterError(
            "PyZX is not installed; install the project's synthesis extras "
            "to use ZXDiagram.to_pyzx()."
        ) from exc

    positions, _ = _node_layout(diagram)
    graph = zx.Graph()
    vertex_map: dict[str, int] = {}
    for node in sorted(diagram.graph.nodes):
        data = diagram.graph.nodes[node]
        if data["kind"] is NodeKind.LOCAL_CLIFFORD:
            continue
        vertex_type = {
            NodeKind.BOUNDARY: zx.VertexType.BOUNDARY,
            NodeKind.Z_SPIDER: zx.VertexType.Z,
            NodeKind.X_SPIDER: zx.VertexType.X,
        }[data["kind"]]
        x, y = positions[node]
        vertex_map[node] = graph.add_vertex(
            ty=vertex_type,
            qubit=y / 76,
            row=x / 76,
            phase=Fraction(0),
        )

    edge_type = {
        EdgeKind.SIMPLE: zx.EdgeType.SIMPLE,
        EdgeKind.HADAMARD: zx.EdgeType.HADAMARD,
    }
    for source, target, data in sorted(
        diagram.graph.edges(data=True),
        key=lambda record: str(record[2]["edge_id"]),
    ):
        if (
            diagram.graph.nodes[source]["kind"] is NodeKind.LOCAL_CLIFFORD
            or diagram.graph.nodes[target]["kind"] is NodeKind.LOCAL_CLIFFORD
        ):
            continue
        graph.add_edge(
            (vertex_map[source], vertex_map[target]),
            edgetype=edge_type[data["kind"]],
        )

    phase_gates: dict[str, tuple[Any, Fraction]] = {
        "S": (zx.VertexType.Z, Fraction(1, 2)),
        "S_DAG": (zx.VertexType.Z, Fraction(-1, 2)),
        "Z": (zx.VertexType.Z, Fraction(1)),
        "SQRT_X": (zx.VertexType.X, Fraction(1, 2)),
        "SQRT_X_DAG": (zx.VertexType.X, Fraction(-1, 2)),
        "X": (zx.VertexType.X, Fraction(1)),
    }
    for box in sorted(diagram.nodes_of_kind(NodeKind.LOCAL_CLIFFORD)):
        neighbors = sorted(diagram.graph.neighbors(box))
        boundary_neighbors = [
            node
            for node in neighbors
            if diagram.graph.nodes[node]["kind"] is NodeKind.BOUNDARY
        ]
        if len(neighbors) != 2 or len(boundary_neighbors) != 1:
            raise PyZXAdapterError(
                f"Local-Clifford box {box!r} must have exactly one boundary "
                "neighbor and one internal neighbor."
            )
        boundary = boundary_neighbors[0]
        internal = next(node for node in neighbors if node != boundary)
        if diagram.graph.nodes[internal]["kind"] is NodeKind.LOCAL_CLIFFORD:
            raise PyZXAdapterError("Chained opaque LC boxes are not supported.")

        current = vertex_map[internal]
        x, y = positions[box]
        gates = list(normalize_gate_word(diagram.graph.nodes[box]["gate_word"]))
        expanded: list[str] = []
        for gate in gates:
            if gate == "Y":
                expanded.extend(("X", "Z"))
            else:
                expanded.append(gate)
        for gate_index, gate in enumerate(expanded):
            if gate == "H":
                vertex = graph.add_vertex(
                    ty=zx.VertexType.Z,
                    qubit=y / 76,
                    row=x / 76 + gate_index / 100,
                    phase=Fraction(0),
                )
                graph.add_edge(
                    (current, vertex), edgetype=zx.EdgeType.HADAMARD
                )
            elif gate in phase_gates:
                vertex_type, phase = phase_gates[gate]
                vertex = graph.add_vertex(
                    ty=vertex_type,
                    qubit=y / 76,
                    row=x / 76 + gate_index / 100,
                    phase=phase,
                )
                graph.add_edge(
                    (current, vertex), edgetype=zx.EdgeType.SIMPLE
                )
            else:
                raise PyZXAdapterError(
                    f"Unsupported local-Clifford gate {gate!r} on box {box!r}."
                )
            current = vertex
        graph.add_edge(
            (current, vertex_map[boundary]), edgetype=zx.EdgeType.SIMPLE
        )

    outputs = sorted(
        diagram.nodes_of_kind(NodeKind.BOUNDARY),
        key=lambda node: (
            diagram.graph.nodes[node].get("qubit", math.inf),
            node,
        ),
    )
    graph.set_inputs(())
    graph.set_outputs(tuple(vertex_map[node] for node in outputs))
    return graph


__all__ = [
    "DiagramValidationError",
    "EdgeKind",
    "EdgeRole",
    "FaultStatus",
    "LemmaBStarError",
    "NodeKind",
    "NodeRole",
    "PyZXAdapterError",
    "ZXDiagram",
    "ZXDiagramError",
    "apply_lemma_b_star",
    "build_ideal_graph_state_diagram",
    "normalize_gate_word",
]

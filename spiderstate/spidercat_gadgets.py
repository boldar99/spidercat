"""Verified SpiderCat replacements for high-arity phase-zero ZX spiders.

This module deliberately stops at the diagram level.  The cached SpiderCat
data contains a spanning forest and circuit-extraction artefacts as well as the
expanded marked graph; only the expanded graph and its marked attachment
vertices are consumed here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx

from spiderstate.zx_ir import (
    EdgeKind,
    EdgeRole,
    FaultStatus,
    NodeKind,
    NodeRole,
    ZXDiagram,
)


MIN_SUPPORTED_FAULT_TOLERANCE = 1
MAX_SUPPORTED_FAULT_TOLERANCE = 7


class UnsupportedFaultToleranceError(ValueError):
    """Raised when ``t`` is outside the verified SpiderCat range."""


class SpiderCatGadgetUnavailable(RuntimeError):
    """Raised when no gadget satisfying the requested guarantee is available."""


class SpiderCatInvariantError(RuntimeError):
    """Raised when a composition would violate the diagram contract."""


@dataclass(frozen=True)
class RobustnessViolation:
    """A marked cut witnessing failure of SpiderCat robustness."""

    cut_size: int
    side: frozenset[Hashable]
    complement: frozenset[Hashable]
    cut_edges: tuple[tuple[Hashable, Hashable], ...]
    marks_on_side: int
    marks_on_complement: int


@dataclass(frozen=True)
class VerifiedSpiderCatGadget:
    """An expanded, exactly checked SpiderCat graph."""

    graph: nx.Graph
    attachment_nodes: tuple[Hashable, ...]
    arity: int
    requested_t: int
    effective_t: int
    construction: str
    guarantee: str = "verified"
    optimality: str = "verified"


@dataclass(frozen=True)
class SpiderCatPort:
    """Stable provenance for one external leg of a replacement."""

    port_index: int
    original_neighbor: Hashable
    original_edge_id: str | None
    original_edge_provenance: Any
    original_edge_kind: EdgeKind
    original_fault_status: FaultStatus
    original_edge_role: EdgeRole
    attachment_node: Hashable


@dataclass(frozen=True)
class SpiderCatReplacement:
    """Metadata for one source-spider replacement."""

    source_node: Hashable
    source_kind: NodeKind
    source_provenance: Any
    arity: int
    construction: str
    requested_t: int
    effective_t: int
    gadget_nodes: tuple[Hashable, ...]
    ports: tuple[SpiderCatPort, ...]


@dataclass(frozen=True)
class SpiderCatDecompositionMetadata:
    """Guarantee and source-to-gadget mapping for a full decomposition."""

    requested_t: int
    replacements: tuple[SpiderCatReplacement, ...]
    guarantee: str = "verified"

    @property
    def source_to_gadget_ports(
        self,
    ) -> Mapping[Hashable, tuple[SpiderCatPort, ...]]:
        return {replacement.source_node: replacement.ports for replacement in self.replacements}


def _validate_requested_t(t: int) -> int:
    if isinstance(t, bool) or not isinstance(t, int):
        raise UnsupportedFaultToleranceError(
            "fault tolerance t must be an integer in the inclusive range 1..7"
        )
    if not MIN_SUPPORTED_FAULT_TOLERANCE <= t <= MAX_SUPPORTED_FAULT_TOLERANCE:
        raise UnsupportedFaultToleranceError(
            "fault tolerance t must be in the inclusive verified range 1..7; "
            f"got {t}"
        )
    return t


def _stable_key(value: Hashable) -> tuple[Any, ...]:
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
    return (5, type(value).__module__, type(value).__qualname__, repr(value))


def _canonical_edge(
    u: Hashable, v: Hashable
) -> tuple[Hashable, Hashable]:
    if _stable_key(u) <= _stable_key(v):
        return u, v
    return v, u


def _sorted_edges(
    edges: Iterable[tuple[Hashable, Hashable]],
) -> tuple[tuple[Hashable, Hashable], ...]:
    return tuple(
        sorted(
            (_canonical_edge(u, v) for u, v in edges),
            key=lambda edge: (_stable_key(edge[0]), _stable_key(edge[1])),
        )
    )


def _validate_expanded_marked_graph(
    graph: nx.Graph,
    *,
    expected_arity: int | None = None,
    mark_attribute: str = "is_mark",
) -> tuple[Hashable, ...]:
    if graph.is_directed() or graph.is_multigraph():
        raise SpiderCatGadgetUnavailable(
            "SpiderCat gadgets must be simple undirected graphs"
        )
    if graph.number_of_nodes() == 0:
        raise SpiderCatGadgetUnavailable("SpiderCat gadget graph is empty")
    if nx.number_of_selfloops(graph):
        raise SpiderCatGadgetUnavailable(
            "SpiderCat gadget graph contains a self-loop"
        )
    if not nx.is_connected(graph):
        raise SpiderCatGadgetUnavailable(
            "SpiderCat gadget graph is disconnected"
        )
    if max(dict(graph.degree()).values(), default=0) > 3:
        raise SpiderCatGadgetUnavailable(
            "expanded SpiderCat graph has a vertex of degree greater than three"
        )

    attachments = tuple(
        sorted(
            (
                node
                for node, data in graph.nodes(data=True)
                if bool(data.get(mark_attribute, False))
            ),
            key=_stable_key,
        )
    )
    if expected_arity is not None and len(attachments) != expected_arity:
        raise SpiderCatGadgetUnavailable(
            "expanded SpiderCat graph has "
            f"{len(attachments)} marked attachment vertices, expected "
            f"{expected_arity}"
        )
    for node in attachments:
        if graph.degree(node) > 2:
            raise SpiderCatGadgetUnavailable(
                "a marked attachment vertex already has degree greater than two"
            )
    return attachments


def find_t_robustness_violation(
    graph: nx.Graph,
    t: int,
    *,
    mark_attribute: str = "is_mark",
) -> RobustnessViolation | None:
    """Find an exact marked-cut violation in an expanded SpiderCat graph.

    For a cut with its *actual* size ``f <= t``, robustness requires that one
    side contains at most ``f`` marked attachment vertices.  The SAT encoding
    searches for a partition with at least ``f + 1`` marks on both sides.  It
    iterates ``f`` in ascending order, so an at-most-``f`` edge constraint is
    exact for detecting the first violating cut.
    """

    _validate_requested_t(t)
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("robustness verification requires a simple undirected graph")
    if nx.number_of_selfloops(graph):
        raise ValueError("robustness verification does not accept self-loops")

    nodes = tuple(sorted(graph.nodes(), key=_stable_key))
    marks = tuple(
        node
        for node in nodes
        if bool(graph.nodes[node].get(mark_attribute, False))
    )
    if len(nodes) < 2 or len(marks) < 2:
        return None

    try:
        from pysat.card import CardEnc, EncType
        from pysat.formula import CNF, IDPool
        from pysat.solvers import Solver
    except ImportError as exc:  # pragma: no cover - pinned runtime dependency
        raise SpiderCatGadgetUnavailable(
            "exact SpiderCat verification requires the python-sat dependency"
        ) from exc

    pool = IDPool()
    side_vars = {
        node: pool.id(("spidercat-side", index))
        for index, node in enumerate(nodes)
    }
    edges = _sorted_edges(graph.edges())
    cut_vars = {
        edge: pool.id(("spidercat-cut", index))
        for index, edge in enumerate(edges)
    }

    base = CNF()
    for u, v in edges:
        side_u = side_vars[u]
        side_v = side_vars[v]
        cut = cut_vars[(u, v)]
        # cut <-> (side_u XOR side_v)
        base.append([side_u, side_v, -cut])
        base.append([side_u, -side_v, cut])
        base.append([-side_u, side_v, cut])
        base.append([-side_u, -side_v, -cut])

    mark_lits = [side_vars[node] for node in marks]
    max_relevant_cut = min(t, len(marks) // 2 - 1)
    for f in range(max_relevant_cut + 1):
        formula = CNF(from_clauses=base.clauses)
        if f < len(cut_vars):
            formula.extend(
                CardEnc.atmost(
                    lits=list(cut_vars.values()),
                    bound=f,
                    vpool=pool,
                    encoding=EncType.seqcounter,
                ).clauses
            )
        formula.extend(
            CardEnc.atleast(
                lits=mark_lits,
                bound=f + 1,
                vpool=pool,
                encoding=EncType.seqcounter,
            ).clauses
        )
        formula.extend(
            CardEnc.atleast(
                lits=[-literal for literal in mark_lits],
                bound=f + 1,
                vpool=pool,
                encoding=EncType.seqcounter,
            ).clauses
        )
        # The two marked-side constraints are symmetric.
        formula.append([mark_lits[0]])

        with Solver(name="g42", bootstrap_with=formula.clauses) as solver:
            if not solver.solve():
                continue
            positive = {literal for literal in solver.get_model() if literal > 0}

        side = frozenset(
            node for node in nodes if side_vars[node] in positive
        )
        complement = frozenset(set(nodes) - set(side))
        crossing = _sorted_edges(
            (u, v)
            for u, v in edges
            if (u in side) != (v in side)
        )
        actual_f = len(crossing)
        marks_on_side = sum(node in side for node in marks)
        marks_on_complement = len(marks) - marks_on_side
        if min(marks_on_side, marks_on_complement) <= actual_f:
            raise AssertionError(
                "internal SAT verifier returned a non-violating marked cut"
            )
        return RobustnessViolation(
            cut_size=actual_f,
            side=side,
            complement=complement,
            cut_edges=crossing,
            marks_on_side=marks_on_side,
            marks_on_complement=marks_on_complement,
        )
    return None


def verify_t_robustness(
    graph: nx.Graph,
    t: int,
    *,
    mark_attribute: str = "is_mark",
) -> bool:
    """Return whether the expanded graph satisfies exact ``t`` robustness."""

    return (
        find_t_robustness_violation(
            graph,
            t,
            mark_attribute=mark_attribute,
        )
        is None
    )


def _effective_t(arity: int, requested_t: int) -> int:
    # Cuts with f >= floor(arity / 2) cannot have more than f ports on
    # both sides, so asking the repository for a larger construction gives no
    # stronger marked-cut guarantee.  The returned graph is still checked
    # against requested_t below.
    return min(requested_t, max(1, arity // 2 - 1))


def _small_cycle_gadget(arity: int) -> nx.Graph:
    graph = nx.cycle_graph(arity)
    nx.set_node_attributes(graph, True, "is_mark")
    return graph


def _load_repository_gadget(arity: int, effective_t: int) -> nx.Graph:
    from spiderstate.well_ordered_cat_state import (
        well_ordered_ft_cat_state_data,
    )

    # Intentionally discard the forest, roots, extraction DAG, and removed
    # edge.  They are circuit-extraction data, not part of the ZX gadget.
    expanded_graph, *_ = well_ordered_ft_cat_state_data(arity, effective_t)
    return nx.Graph(expanded_graph)


@lru_cache(maxsize=None)
def _verified_spidercat_gadget_cached(
    arity: int, requested_t: int
) -> VerifiedSpiderCatGadget:
    _validate_requested_t(requested_t)
    if isinstance(arity, bool) or not isinstance(arity, int) or arity < 4:
        raise SpiderCatGadgetUnavailable(
            "a SpiderCat replacement requires integer arity at least four"
        )

    effective_t = _effective_t(arity, requested_t)
    try:
        if arity in (4, 5):
            graph = _small_cycle_gadget(arity)
            construction = f"verified-cycle-{arity}"
        else:
            graph = _load_repository_gadget(arity, effective_t)
            construction = "repository-expanded-marked-graph"
    except SpiderCatGadgetUnavailable:
        raise
    except Exception as exc:
        raise SpiderCatGadgetUnavailable(
            "unable to load or search a SpiderCat gadget for "
            f"arity={arity}, t={requested_t} (effective t={effective_t})"
        ) from exc

    attachments = _validate_expanded_marked_graph(
        graph,
        expected_arity=arity,
    )
    violation = find_t_robustness_violation(graph, requested_t)
    if violation is not None:
        raise SpiderCatGadgetUnavailable(
            "candidate SpiderCat gadget failed exact marked-cut verification "
            f"for arity={arity}, t={requested_t}: cut size "
            f"{violation.cut_size} separates {violation.marks_on_side} and "
            f"{violation.marks_on_complement} ports"
        )

    return VerifiedSpiderCatGadget(
        graph=graph.copy(),
        attachment_nodes=attachments,
        arity=arity,
        requested_t=requested_t,
        effective_t=effective_t,
        construction=construction,
        optimality=(
            "paper-optimal" if requested_t <= 5 else "verified-not-claimed-optimal"
        ),
    )


def verified_spidercat_gadget(
    arity: int, t: int
) -> VerifiedSpiderCatGadget:
    """Return a defensive copy of an expanded, exactly verified gadget."""

    cached = _verified_spidercat_gadget_cached(arity, t)
    return replace(cached, graph=cached.graph.copy())


def clear_spidercat_gadget_cache() -> None:
    """Clear provider/scoring caches, primarily for deterministic tests."""

    _verified_spidercat_gadget_cached.cache_clear()
    predicted_spidercat_spider_count.cache_clear()


@lru_cache(maxsize=None)
def predicted_spidercat_spider_count(arity: int, t: int) -> int:
    """Predict replacement size for LC-orbit scoring.

    Scoring must remain total even when an optional cached construction is
    absent.  Actual synthesis never uses this fallback: it calls
    :func:`verified_spidercat_gadget` and fails with a typed error.
    """

    _validate_requested_t(t)
    if isinstance(arity, bool) or not isinstance(arity, int) or arity < 1:
        raise ValueError("spider arity must be a positive integer")
    if arity <= 3:
        return 1
    if arity in (4, 5):
        return arity
    effective_t = _effective_t(arity, t)
    try:
        return _load_repository_gadget(arity, effective_t).number_of_nodes()
    except Exception:
        return 3 * arity


def _is_spider(kind: Any) -> bool:
    return kind in (NodeKind.Z_SPIDER, NodeKind.X_SPIDER)


def _fresh_gadget_node_id(
    graph: nx.Graph,
    source: Hashable,
    local_index: int,
) -> str:
    base = (
        "spidercat["
        f"{type(source).__module__}.{type(source).__qualname__}:{source!r}"
        f"].{local_index}"
    )
    candidate = base
    suffix = 0
    while candidate in graph:
        suffix += 1
        candidate = f"{base}.{suffix}"
    return candidate


def _incident_sort_key(
    source: Hashable,
    neighbor: Hashable,
    data: Mapping[str, Any],
) -> tuple[Any, ...]:
    edge_id = data.get("edge_id")
    return (
        0 if edge_id is not None else 1,
        "" if edge_id is None else str(edge_id),
        _stable_key(neighbor),
        _stable_key(source),
    )


def _copy_boundary_decorations(
    graph: nx.Graph,
) -> dict[Hashable, dict[str, Any]]:
    boundary_kinds = (NodeKind.BOUNDARY, NodeKind.LOCAL_CLIFFORD)
    return {
        node: deepcopy(data)
        for node, data in graph.nodes(data=True)
        if data.get("kind") in boundary_kinds
    }


def _set_final_stage_metadata(
    diagram: ZXDiagram,
    *,
    requested_t: int,
    replacements: tuple[SpiderCatReplacement, ...],
) -> None:
    """Record the final stage without leaving references to removed spiders."""

    by_source = {
        replacement.source_node: replacement for replacement in replacements
    }
    graph_vertex_nodes: list[Any] = []
    for raw_entry in diagram.metadata.get("graph_vertex_nodes", ()):
        if not isinstance(raw_entry, Mapping):
            graph_vertex_nodes.append(deepcopy(raw_entry))
            continue
        entry = deepcopy(dict(raw_entry))
        source_node = entry.get("node_id")
        replacement = by_source.get(source_node)
        if replacement is not None:
            entry.pop("node_id", None)
            entry["source_node_id"] = source_node
            entry["gadget_node_ids"] = list(replacement.gadget_nodes)
            entry["gadget_attachment_nodes"] = [
                port.attachment_node for port in replacement.ports
            ]
        graph_vertex_nodes.append(entry)

    diagram.metadata["stage"] = "unidealized_trivalent"
    diagram.metadata["spidercat_requested_t"] = requested_t
    diagram.metadata["spidercat_guarantee"] = "verified"
    diagram.metadata["spidercat_replacements"] = [
        {
            "source_node": replacement.source_node,
            "arity": replacement.arity,
            "construction": replacement.construction,
            "effective_t": replacement.effective_t,
            "gadget_node_ids": list(replacement.gadget_nodes),
            "attachment_node_ids": [
                port.attachment_node for port in replacement.ports
            ],
        }
        for replacement in replacements
    ]
    if "graph_vertex_nodes" in diagram.metadata:
        diagram.metadata["graph_vertex_nodes"] = graph_vertex_nodes


def decompose_spidercats(
    diagram: ZXDiagram,
    *,
    t: int,
) -> tuple[ZXDiagram, SpiderCatDecompositionMetadata]:
    """Replace every phase-zero spider of arity greater than three.

    Source spiders and all incident ports are snapshotted before mutation, so
    an edge joining two high-arity spiders is also handled without order
    dependence.  External edge attributes are copied verbatim; only new gadget
    edges are introduced as noisy simple SpiderCat edges.
    """

    requested_t = _validate_requested_t(t)
    diagram.validate()
    source_graph = diagram.graph
    boundary_before = _copy_boundary_decorations(source_graph)
    high_arity = tuple(
        sorted(
            (
                node
                for node, data in source_graph.nodes(data=True)
                if _is_spider(data.get("kind")) and source_graph.degree(node) > 3
            ),
            key=_stable_key,
        )
    )

    if not high_arity:
        result = diagram.copy()
        _set_final_stage_metadata(
            result,
            requested_t=requested_t,
            replacements=(),
        )
        _validate_final_diagram(result, boundary_before)
        return result, SpiderCatDecompositionMetadata(
            requested_t=requested_t,
            replacements=(),
        )

    high_set = set(high_arity)
    source_node_data = {
        node: deepcopy(dict(source_graph.nodes[node]))
        for node in high_arity
    }
    source_edges = [
        (u, v, deepcopy(dict(data)))
        for u, v, data in source_graph.edges(data=True)
        if u in high_set or v in high_set
    ]

    gadgets: dict[Hashable, VerifiedSpiderCatGadget] = {}
    incident_ports: dict[
        Hashable, list[tuple[Hashable, dict[str, Any]]]
    ] = {}
    for source in high_arity:
        data = source_node_data[source]
        if data.get("phase", 0) != 0:
            raise SpiderCatInvariantError(
                f"high-arity spider {source!r} has nonzero phase; "
                "local Clifford phases must remain in boundary boxes"
            )
        incident = [
            (
                v if u == source else u,
                deepcopy(edge_data),
            )
            for u, v, edge_data in source_edges
            if u == source or v == source
        ]
        incident.sort(
            key=lambda item: _incident_sort_key(source, item[0], item[1])
        )
        incident_ports[source] = incident
        gadgets[source] = verified_spidercat_gadget(len(incident), requested_t)

    result = diagram.copy()
    for source in high_arity:
        result.remove_node(source)

    port_endpoint: dict[tuple[Hashable, Hashable], Hashable] = {}
    replacements: list[SpiderCatReplacement] = []

    for source in high_arity:
        source_data = source_node_data[source]
        gadget = gadgets[source]
        ordered_gadget_nodes = tuple(
            sorted(gadget.graph.nodes(), key=_stable_key)
        )
        node_map: dict[Hashable, Hashable] = {}
        for local_index, gadget_node in enumerate(ordered_gadget_nodes):
            new_node = _fresh_gadget_node_id(
                result.graph,
                source,
                local_index,
            )
            node_map[gadget_node] = new_node
            result.add_node(
                new_node,
                kind=source_data["kind"],
                role=NodeRole.SPIDERCAT,
                phase=0,
                provenance={
                    "transform": "spidercat",
                    "source_node": source,
                    "source_provenance": deepcopy(
                        source_data.get("provenance")
                    ),
                    "gadget_vertex": gadget_node,
                    "arity": gadget.arity,
                    "requested_t": gadget.requested_t,
                    "effective_t": gadget.effective_t,
                },
            )
        for edge_index, (u, v) in enumerate(
            _sorted_edges(gadget.graph.edges())
        ):
            result.add_edge(
                node_map[u],
                node_map[v],
                kind=EdgeKind.SIMPLE,
                fault_status=FaultStatus.NOISY,
                role=EdgeRole.SPIDERCAT,
                provenance={
                    "transform": "spidercat",
                    "source_node": source,
                    "gadget_edge": (u, v),
                    "edge_index": edge_index,
                },
            )

        ports: list[SpiderCatPort] = []
        for port_index, (
            (neighbor, edge_data),
            attachment,
        ) in enumerate(
            zip(incident_ports[source], gadget.attachment_nodes, strict=True)
        ):
            attachment_node = node_map[attachment]
            port_endpoint[(source, neighbor)] = attachment_node
            ports.append(
                SpiderCatPort(
                    port_index=port_index,
                    original_neighbor=neighbor,
                    original_edge_id=(
                        None
                        if edge_data.get("edge_id") is None
                        else str(edge_data["edge_id"])
                    ),
                    original_edge_provenance=deepcopy(
                        edge_data.get("provenance")
                    ),
                    original_edge_kind=edge_data["kind"],
                    original_fault_status=edge_data["fault_status"],
                    original_edge_role=edge_data["role"],
                    attachment_node=attachment_node,
                )
            )
        replacements.append(
            SpiderCatReplacement(
                source_node=source,
                source_kind=source_data["kind"],
                source_provenance=deepcopy(
                    source_data.get("provenance")
                ),
                arity=gadget.arity,
                construction=gadget.construction,
                requested_t=gadget.requested_t,
                effective_t=gadget.effective_t,
                gadget_nodes=tuple(node_map[node] for node in ordered_gadget_nodes),
                ports=tuple(ports),
            )
        )

    for u, v, edge_data in source_edges:
        new_u = port_endpoint[(u, v)] if u in high_set else u
        new_v = port_endpoint[(v, u)] if v in high_set else v
        if result.graph.has_edge(new_u, new_v):
            raise SpiderCatInvariantError(
                "SpiderCat replacement would create a parallel edge between "
                f"{new_u!r} and {new_v!r}"
            )
        # Keep the source port's edge identity, semantics, fault status, role,
        # provenance, and extension attributes exactly.  New internal edges
        # alone receive new IDs.
        restored = deepcopy(edge_data)
        result.add_edge(
            new_u,
            new_v,
            edge_id=restored.pop("edge_id"),
            kind=restored.pop("kind"),
            fault_status=restored.pop("fault_status"),
            role=restored.pop("role"),
            provenance=restored.pop("provenance"),
            **restored,
        )

    replacement_tuple = tuple(replacements)
    _set_final_stage_metadata(
        result,
        requested_t=requested_t,
        replacements=replacement_tuple,
    )
    _validate_final_diagram(result, boundary_before)
    return result, SpiderCatDecompositionMetadata(
        requested_t=requested_t,
        replacements=replacement_tuple,
    )


def _validate_final_diagram(
    diagram: ZXDiagram,
    boundary_before: Mapping[Hashable, Mapping[str, Any]],
) -> None:
    diagram.validate()
    graph = diagram.graph
    for node, data in graph.nodes(data=True):
        if _is_spider(data.get("kind")) and graph.degree(node) > 3:
            raise SpiderCatInvariantError(
                f"spider {node!r} still has arity {graph.degree(node)}"
            )

    for u, v, data in graph.edges(data=True):
        if _is_spider(graph.nodes[u].get("kind")) and _is_spider(
            graph.nodes[v].get("kind")
        ):
            if data.get("fault_status") is not FaultStatus.NOISY:
                raise SpiderCatInvariantError(
                    f"internal edge {(u, v)!r} remains ideal after decomposition"
                )
        if data.get("provenance") is None:
            raise SpiderCatInvariantError(
                f"edge {(u, v)!r} has no provenance"
            )

    boundary_after = _copy_boundary_decorations(graph)
    if boundary_after != dict(boundary_before):
        raise SpiderCatInvariantError(
            "SpiderCat decomposition changed a boundary or local-Clifford box"
        )

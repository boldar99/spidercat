"""Staged synthesis of a stabilizer state into a noisy trivalent ZX diagram.

The public certificate convention is

``local_cliffords_to_graph |psi> = |G>``.

Consequently, the ideal ZX diagram places the inverse local Cliffords on its
output legs and denotes the exact requested state, not merely an
LC-equivalent graph-state representative.  As usual for ZX diagrams, global
nonzero scalar factors are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import stim

from spiderstate.spidercat_gadgets import (
    SpiderCatDecompositionMetadata,
    UnsupportedFaultToleranceError,
    decompose_spidercats,
    predicted_spidercat_spider_count,
)
from spiderstate.stabilizer_graph import (
    LCSearchConfig,
    LCSearchMetadata,
    LocalClifford,
    StabilizerGraphResult,
    css_logical_state_stabilizers,
    stabilizer_state_to_graph,
)
from spiderstate.zx_ir import (
    EdgeRole,
    FaultStatus,
    NodeKind,
    ZXDiagram,
    apply_lemma_b_star,
    build_ideal_graph_state_diagram,
)


class SynthesisStage(str, Enum):
    """Named snapshots exposed by :class:`StabilizerSynthesisResult`."""

    IDEAL = "ideal"
    LEMMA_B_STAR = "lemma_b_star"
    FINAL = "final"


class StabilizerSynthesisError(ValueError):
    """Base error for a failed end-to-end stabilizer synthesis."""


class SynthesisInvariantError(StabilizerSynthesisError):
    """Raised if a construction stage violates its structural contract."""


@dataclass(frozen=True)
class FaultToleranceGuarantee:
    """Machine-readable scope of the returned fault-tolerance guarantee."""

    requested_t: int
    status: str
    scalar_convention: str
    theoretical_optimality_claimed: bool
    construction_count: int


@dataclass(frozen=True)
class StabilizerSynthesisResult:
    """All stable products and certificates from the staged pipeline."""

    graph: nx.Graph
    local_cliffords_to_graph: tuple[LocalClifford, ...]
    ideal_diagram: ZXDiagram
    lemma_b_star_diagram: ZXDiagram
    final_diagram: ZXDiagram
    search_metadata: LCSearchMetadata
    gadget_metadata: SpiderCatDecompositionMetadata
    guarantee_metadata: FaultToleranceGuarantee
    input_stabilizers: tuple[str, ...]
    graph_conversion: StabilizerGraphResult

    @property
    def post_lemma_b_star_diagram(self) -> ZXDiagram:
        """Alias matching the stage name used in the papers and plan."""

        return self.lemma_b_star_diagram

    @property
    def unidealized_diagram(self) -> ZXDiagram:
        """The final noisy trivalent diagram."""

        return self.final_diagram

    @property
    def local_cliffords_from_graph(self) -> tuple[LocalClifford, ...]:
        """Boundary corrections that map ``|G>`` back to the input state."""

        return self.graph_conversion.local_cliffords_from_graph

    @property
    def source_to_gadget_ports(self) -> Mapping[str, Any]:
        """Stable source-spider to final-gadget attachment provenance."""

        return self.gadget_metadata.source_to_gadget_ports

    def diagram(self, stage: str | SynthesisStage = SynthesisStage.FINAL) -> ZXDiagram:
        """Select one of the three immutable-by-convention stage snapshots."""

        normalized = _coerce_stage(stage)
        if normalized is SynthesisStage.IDEAL:
            return self.ideal_diagram
        if normalized is SynthesisStage.LEMMA_B_STAR:
            return self.lemma_b_star_diagram
        return self.final_diagram

    def to_pyzx(self, stage: str | SynthesisStage = SynthesisStage.FINAL):
        """Convert a selected stage to a PyZX graph."""

        return self.diagram(stage).to_pyzx()

    def render_svg(
        self,
        stage: str | SynthesisStage = SynthesisStage.FINAL,
        path: str | Path | None = None,
    ) -> str:
        """Render a selected stage deterministically and optionally save it."""

        selected = _coerce_stage(stage)
        return self.diagram(selected).render_svg(
            path,
            title=f"stabilizer synthesis: {selected.value}",
        )


def _coerce_stage(stage: str | SynthesisStage) -> SynthesisStage:
    if isinstance(stage, SynthesisStage):
        return stage
    normalized = str(stage).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "graph": SynthesisStage.IDEAL,
        "ideal_graph": SynthesisStage.IDEAL,
        "ideal_graph_state": SynthesisStage.IDEAL,
        "lemma": SynthesisStage.LEMMA_B_STAR,
        "lemma_b*": SynthesisStage.LEMMA_B_STAR,
        "post_lemma_b_star": SynthesisStage.LEMMA_B_STAR,
        "post_lemma": SynthesisStage.LEMMA_B_STAR,
        "unidealized": SynthesisStage.FINAL,
        "trivalent": SynthesisStage.FINAL,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return SynthesisStage(normalized)
    except ValueError as exc:
        choices = ", ".join(stage.value for stage in SynthesisStage)
        raise ValueError(
            f"Unknown synthesis stage {stage!r}; expected one of {choices}."
        ) from exc


def _search_config_for_t(config: LCSearchConfig, t: int) -> LCSearchConfig:
    if config.vertex_arity_cost is not None:
        return config

    def vertex_arity_cost(arity: int) -> int:
        return predicted_spidercat_spider_count(arity, t)

    return replace(config, vertex_arity_cost=vertex_arity_cost)


def _boundary_snapshot(diagram: ZXDiagram) -> dict[str, dict[str, Any]]:
    return {
        node: dict(diagram.graph.nodes[node])
        for node in sorted(diagram.graph.nodes)
        if diagram.graph.nodes[node]["kind"]
        in (NodeKind.BOUNDARY, NodeKind.LOCAL_CLIFFORD)
    }


def _validate_pipeline_invariants(
    *,
    ideal: ZXDiagram,
    lemma: ZXDiagram,
    final: ZXDiagram,
    qubit_count: int,
) -> None:
    ideal.validate()
    lemma.validate()
    final.validate()

    if len(ideal.nodes_of_kind(NodeKind.BOUNDARY)) != qubit_count:
        raise SynthesisInvariantError(
            "The ideal graph diagram does not have one output per input qubit."
        )
    if _boundary_snapshot(ideal) != _boundary_snapshot(lemma):
        raise SynthesisInvariantError(
            "Lemma B* changed an output boundary or local-Clifford box."
        )
    if _boundary_snapshot(ideal) != _boundary_snapshot(final):
        raise SynthesisInvariantError(
            "SpiderCat decomposition changed an output boundary or "
            "local-Clifford box."
        )

    remaining_graph_edges = final.edges_of_role(EdgeRole.GRAPH_EDGE)
    if remaining_graph_edges:
        edge_ids = [data["edge_id"] for _, _, data in remaining_graph_edges]
        raise SynthesisInvariantError(
            f"Original ideal graph edges remain after Lemma B*: {edge_ids}."
        )

    for node, data in final.graph.nodes(data=True):
        if data["kind"] in (NodeKind.Z_SPIDER, NodeKind.X_SPIDER):
            degree = final.graph.degree(node)
            if degree > 3:
                raise SynthesisInvariantError(
                    f"Final spider {node!r} has arity {degree}, exceeding three."
                )
        if data.get("provenance") is None:
            raise SynthesisInvariantError(
                f"Final node {node!r} has no source provenance."
            )

    for source, target, data in final.graph.edges(data=True):
        if (
            data["role"] is not EdgeRole.BOUNDARY_EDGE
            and data["fault_status"] is not FaultStatus.NOISY
        ):
            raise SynthesisInvariantError(
                f"Internal edge {data['edge_id']!r} remains ideal."
            )
        if data.get("provenance") is None:
            raise SynthesisInvariantError(
                f"Final edge {data['edge_id']!r} between {source!r} and "
                f"{target!r} has no source provenance."
            )


def synthesize_stabilizer_state(
    stabilizers: Sequence[str | stim.PauliString] | stim.Tableau,
    *,
    t: int,
    lc_search: LCSearchConfig = LCSearchConfig(),
) -> StabilizerSynthesisResult:
    """Return an optimized LC graph and its verified noisy trivalent diagram.

    Args:
        stabilizers: A complete, independent signed generating set for a pure
            stabilizer state, or a Stim tableau whose Z outputs define it.
        t: Inclusive number of internal edge faults to tolerate.  Version one
            supports ``1 <= t <= 7``.
        lc_search: Deterministic local-complementation search controls.
    """

    if isinstance(t, bool) or not isinstance(t, int) or not 1 <= t <= 7:
        raise UnsupportedFaultToleranceError(
            f"t must be an integer from 1 through 7 inclusive; got {t!r}."
        )

    graph_result = stabilizer_state_to_graph(
        stabilizers,
        lc_search=_search_config_for_t(lc_search, t),
    )
    graph_result.validate_certificate()

    ideal = build_ideal_graph_state_diagram(
        graph_result.graph,
        local_corrections=graph_result.local_cliffords_from_graph,
    )
    lemma = apply_lemma_b_star(ideal)
    final, gadget_metadata = decompose_spidercats(lemma, t=t)
    _validate_pipeline_invariants(
        ideal=ideal,
        lemma=lemma,
        final=final,
        qubit_count=len(graph_result.input_stabilizers),
    )

    guarantee = FaultToleranceGuarantee(
        requested_t=t,
        status=gadget_metadata.guarantee,
        scalar_convention="ignore_nonzero_global_scalar",
        theoretical_optimality_claimed=False,
        construction_count=len(gadget_metadata.replacements),
    )
    return StabilizerSynthesisResult(
        graph=graph_result.graph.copy(),
        local_cliffords_to_graph=graph_result.local_cliffords_to_graph,
        ideal_diagram=ideal,
        lemma_b_star_diagram=lemma,
        final_diagram=final,
        search_metadata=graph_result.search,
        gadget_metadata=gadget_metadata,
        guarantee_metadata=guarantee,
        input_stabilizers=graph_result.input_stabilizers,
        graph_conversion=graph_result,
    )


__all__ = [
    "FaultToleranceGuarantee",
    "LCSearchConfig",
    "StabilizerSynthesisError",
    "StabilizerSynthesisResult",
    "SynthesisInvariantError",
    "SynthesisStage",
    "css_logical_state_stabilizers",
    "synthesize_stabilizer_state",
]

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Literal, Sequence

import networkx as nx
import numpy as np

from spidercat.fault_analysis import (
    GraphDEM,
    GraphEdge,
    InternalEdge,
    PrecomputedFaultTable,
    make_boundary_edge,
)
from spidercat.utils import ed

EdgeWeightValue = int | tuple[int, int]


@dataclass(frozen=True)
class NodeDecision:
    """
    Describes the outgoing edge weight choices at a single node in the oriented graph.

    - Root node (`is_root=True`): has 0 incoming edges and 2 outgoing tree edges.
      Options have total weight 3: `((2, 1), (1, 2), (3, 0), (0, 3))` (or `((2, 1), (1, 2))`
      when `allow_root_3_0=False`).
    - Green degree-3 node (`is_root=False`): has 1 incoming tree edge from its parent
      and 2 outgoing edges (which may be tree edges, non-tree flag edges in `G \\ F`,
      or a boundary edge `(node, "boundary")` if `is_mark=True`).
      Options are `((2, 1), (1, 2))` (or locked to a single option if `is_locked=True`).
    """

    node: int
    is_root: bool
    is_mark: bool
    in_edge: InternalEdge | None
    out_edges: tuple[GraphEdge, GraphEdge]
    out_edge_indices: tuple[int, int]
    options: tuple[tuple[int, int], ...]
    default_option_idx: int = 0

    @property
    def num_options(self) -> int:
        return len(self.options)

    @property
    def is_locked(self) -> bool:
        return len(self.options) == 1


@dataclass(frozen=True)
class FlagEdgeConstraint:
    """
    Represents a non-tree flag edge `edge = (a, b) in E(G_exp) \\ E(F_exp)` (with `a < b`).
    It is an outgoing edge of both green nodes `a` and `b`.
    - Node `a` assigns weight `w_a in {1, 2}` via its decision `decision_idx_a` (leg `leg_a in {0, 1}`).
    - Node `b` assigns weight `w_b in {1, 2}` via its decision `decision_idx_b` (leg `leg_b in {0, 1}`).
    - At least one of `w_a` or `w_b` must be 2, so `(w_a, w_b) in {(2, 1), (1, 2), (2, 2)}`
      and the combined weight is `w_a + w_b in {3, 4}`.
    """

    edge: InternalEdge
    edge_idx: int
    node_a: int
    node_b: int
    decision_idx_a: int
    leg_a: int
    decision_idx_b: int
    leg_b: int
    forbidden_option_idx_a: int
    forbidden_option_idx_b: int


@dataclass(frozen=True)
class WeightCandidate:
    """
    A complete assignment of weights to all edges of `GraphDEM` and its evaluated fault polynomial.
    """

    coeffs: tuple[int, ...]
    polynomial_str: str
    edge_weights: dict[GraphEdge, EdgeWeightValue]
    effective_weights: np.ndarray
    node_choices: dict[int, dict[GraphEdge, int]]
    option_indices: tuple[int, ...]
    score: float

    def __repr__(self) -> str:
        return (
            f"WeightCandidate(polynomial={self.polynomial_str!r}, "
            f"coeffs={self.coeffs}, score={self.score:.6g})"
        )


def compute_polynomial_score(
    coeffs: Sequence[int | float] | np.ndarray,
    p: float = 1e-3,
    mode: Literal["expected_errors", "expected_fault_events"] = "expected_errors",
) -> float:
    """
    Evaluates the scalar objective for a fault polynomial `sum_{k=1}^L c_k p^k`.

    Args:
        coeffs: Sequence of polynomial coefficients `[c_1, c_2, ..., c_L]`.
        p: Physical error probability `p` (default `1e-3`).
        mode:
            - `"expected_errors"` (default): `sum_{k=1}^L k * c_k * p^k`
              (expected number of undetectable boundary errors).
            - `"expected_fault_events"`: `sum_{k=1}^L c_k * p^k`
              (total probability of any undetectable boundary fault up to weight L).
    """
    c_arr = np.asarray(coeffs, dtype=np.float64)
    L = c_arr.shape[-1]
    ks = np.arange(1, L + 1, dtype=np.float64)
    p_powers = np.power(float(p), ks)
    if mode == "expected_errors":
        weights = ks * p_powers
    elif mode == "expected_fault_events":
        weights = p_powers
    else:
        raise ValueError(f"Unknown score mode: {mode!r}")
    return float(np.sum(c_arr * weights))


def _compute_batch_scores(
    batch_coeffs: np.ndarray,
    p: float = 1e-3,
    mode: Literal["expected_errors", "expected_fault_events"] = "expected_errors",
) -> np.ndarray:
    c_arr = np.asarray(batch_coeffs, dtype=np.float64)
    L = c_arr.shape[1]
    ks = np.arange(1, L + 1, dtype=np.float64)
    p_powers = np.power(float(p), ks)
    if mode == "expected_errors":
        weights = ks * p_powers
    elif mode == "expected_fault_events":
        weights = p_powers
    else:
        raise ValueError(f"Unknown score mode: {mode!r}")
    return np.sum(c_arr * weights[np.newaxis, :], axis=1)


class WeightSpace:
    """
    Encapsulates all node-level out-edge weight decisions and flag-edge constraints
    for a given `GraphDEM` or `PrecomputedFaultTable`.
    """

    def __init__(
        self,
        dem: GraphDEM,
        decisions: Sequence[NodeDecision],
        flag_constraints: Sequence[FlagEdgeConstraint],
        passthrough_edges: dict[int, tuple[int, int]] | None = None,
    ):
        self.dem = dem
        self.decisions: tuple[NodeDecision, ...] = tuple(decisions)
        self.node_to_decision_idx: dict[int, int] = {
            d.node: idx for idx, d in enumerate(self.decisions)
        }
        self.flag_constraints: tuple[FlagEdgeConstraint, ...] = tuple(flag_constraints)
        self.flag_edge_to_constraint: dict[InternalEdge, FlagEdgeConstraint] = {
            fc.edge: fc for fc in self.flag_constraints
        }
        # Optional mapping for legacy degree-2 `is_flag` nodes if ever present:
        # maps dem_edge_idx -> (decision_idx, leg_idx)
        self.passthrough_edges: dict[int, tuple[int, int]] = passthrough_edges or {}

        self.free_decision_indices: tuple[int, ...] = tuple(
            idx for idx, d in enumerate(self.decisions) if not d.is_locked
        )

        # Map each decision to any flag constraints it participates in
        self.decision_to_flag_constraints: dict[int, list[FlagEdgeConstraint]] = {
            idx: [] for idx in range(len(self.decisions))
        }
        for fc in self.flag_constraints:
            self.decision_to_flag_constraints[fc.decision_idx_a].append(fc)
            self.decision_to_flag_constraints[fc.decision_idx_b].append(fc)

        # Precompute a 3D lookup table of shape (num_decisions, max_options, num_edges)
        # so converting a batch of option_indices (batch_size, num_decisions) into
        # effective_weights (batch_size, num_edges) is a single vectorized gather + sum!
        max_opts = max((d.num_options for d in self.decisions), default=1)
        self._contrib_table = np.zeros(
            (len(self.decisions), max_opts, self.dem.num_edges), dtype=np.int64
        )
        for d_idx, d in enumerate(self.decisions):
            idx0, idx1 = d.out_edge_indices
            for opt_idx, (w0, w1) in enumerate(d.options):
                self._contrib_table[d_idx, opt_idx, idx0] += w0
                self._contrib_table[d_idx, opt_idx, idx1] += w1

        for edge_idx, (d_idx, leg_idx) in self.passthrough_edges.items():
            d = self.decisions[d_idx]
            for opt_idx, pair in enumerate(d.options):
                self._contrib_table[d_idx, opt_idx, edge_idx] += pair[leg_idx]

    @property
    def num_decisions(self) -> int:
        return len(self.decisions)

    @property
    def num_free_decisions(self) -> int:
        return len(self.free_decision_indices)

    def count_valid_assignments(self) -> int:
        """
        Returns the total number of valid weight assignments in this space
        (respecting all flag-edge constraints `(w_a, w_b) != (1, 1)`).
        """
        count = 0
        for _ in self.iter_valid_option_indices():
            count += 1
        return count

    def default_option_indices(self) -> tuple[int, ...]:
        """
        Returns the default option index tuple (0 for every decision node),
        repaired if necessary so all flag-edge constraints are satisfied.
        """
        opts = [d.default_option_idx for d in self.decisions]
        opts = self.repair_option_indices(opts)
        return tuple(opts)

    def is_valid_option_indices(self, option_indices: Sequence[int]) -> bool:
        """
        Checks that no flag edge in `G \\ F` is assigned weight `(1, 1)`.
        """
        for fc in self.flag_constraints:
            if (
                option_indices[fc.decision_idx_a] == fc.forbidden_option_idx_a
                and option_indices[fc.decision_idx_b] == fc.forbidden_option_idx_b
            ):
                return False
        return True

    def repair_option_indices(self, option_indices: Sequence[int]) -> list[int]:
        """
        Repairs a list of option indices in-place/copy so that every flag edge
        has at least one endpoint with weight 2.
        """
        opts = list(option_indices)
        for fc in self.flag_constraints:
            if (
                opts[fc.decision_idx_a] == fc.forbidden_option_idx_a
                and opts[fc.decision_idx_b] == fc.forbidden_option_idx_b
            ):
                if not self.decisions[fc.decision_idx_b].is_locked:
                    opts[fc.decision_idx_b] = 1 - fc.forbidden_option_idx_b
                elif not self.decisions[fc.decision_idx_a].is_locked:
                    opts[fc.decision_idx_a] = 1 - fc.forbidden_option_idx_a
                else:
                    raise ValueError(
                        f"Flag edge {fc.edge} has both endpoints locked to weight 1."
                    )
        return opts

    def iter_valid_option_indices(self) -> itertools.Iterator[tuple[int, ...]]:
        """
        Backtracking generator yielding all option index tuples `(opt_0, ..., opt_{D-1})`
        that satisfy all flag-edge constraints `(w_a, w_b) != (1, 1)`.
        Prunes invalid branches immediately as soon as both endpoints of a flag edge are set.
        """
        num_d = len(self.decisions)
        # For each decision step i, store constraints whose other endpoint is < i
        check_at_step: list[list[tuple[int, int, int]]] = [[] for _ in range(num_d)]
        for fc in self.flag_constraints:
            da, db = fc.decision_idx_a, fc.decision_idx_b
            if da < db:
                check_at_step[db].append((da, fc.forbidden_option_idx_a, fc.forbidden_option_idx_b))
            else:
                check_at_step[da].append((db, fc.forbidden_option_idx_b, fc.forbidden_option_idx_a))

        option_ranges = [range(d.num_options) for d in self.decisions]
        current = [0] * num_d

        def _backtrack(step: int):
            if step == num_d:
                yield tuple(current)
                return
            checks = check_at_step[step]
            for opt in option_ranges[step]:
                valid = True
                for prev_idx, prev_forbid, curr_forbid in checks:
                    if opt == curr_forbid and current[prev_idx] == prev_forbid:
                        valid = False
                        break
                if valid:
                    current[step] = opt
                    yield from _backtrack(step + 1)

        yield from _backtrack(0)

    def option_indices_to_effective_weights(
        self, option_indices: Sequence[int] | np.ndarray
    ) -> np.ndarray:
        """
        Converts a single option-index tuple `(D,)` or a 2D batch `(batch_size, D)`
        into effective integer edge weights `(num_edges,)` or `(batch_size, num_edges)`.
        For flag edges `(a, b)`, the effective weight is `w_a + w_b in {3, 4}`.
        """
        arr = np.asarray(option_indices, dtype=np.int64)
        d_arange = np.arange(len(self.decisions))
        if arr.ndim == 1:
            return self._contrib_table[d_arange, arr, :].sum(axis=0)
        if arr.ndim == 2:
            # arr has shape (batch_size, num_decisions)
            # self._contrib_table[d_arange, arr] has shape (batch_size, num_decisions, num_edges)
            return self._contrib_table[d_arange[np.newaxis, :], arr, :].sum(axis=1)
        raise ValueError(f"Expected 1D or 2D option_indices, got {arr.ndim}D")

    def build_candidate(
        self,
        table: PrecomputedFaultTable,
        option_indices: Sequence[int],
        coeffs: Sequence[int] | np.ndarray | None = None,
        effective_weights: np.ndarray | None = None,
        p: float = 1e-3,
        score_mode: Literal["expected_errors", "expected_fault_events"] = "expected_errors",
        poly_mode: Literal["boundary_weight", "fault_weight"] = "boundary_weight",
    ) -> WeightCandidate:
        """
        Constructs a full `WeightCandidate` (including tuple weights `(w_a, w_b)` for flag edges)
        from an `option_indices` assignment.
        """
        opt_tuple = tuple(int(x) for x in option_indices)
        if effective_weights is None:
            effective_weights = self.option_indices_to_effective_weights(opt_tuple)
        else:
            effective_weights = np.asarray(effective_weights, dtype=np.int64)

        if coeffs is None:
            coeffs_arr = table.evaluate_polynomial(effective_weights, mode=poly_mode)
        else:
            coeffs_arr = np.asarray(coeffs, dtype=np.int64)

        coeffs_tuple = tuple(int(c) for c in coeffs_arr)

        # Format polynomial string
        terms = []
        for idx, c in enumerate(coeffs_tuple, start=1):
            if c == 0:
                continue
            exp_str = "p" if idx == 1 else f"p^{idx}"
            terms.append(f"{c}{exp_str}")
        poly_str = " + ".join(terms) if terms else "0"

        # Build node_choices and edge_weights (with tuples for flag edges)
        node_choices: dict[int, dict[GraphEdge, int]] = {}
        for d_idx, d in enumerate(self.decisions):
            w0, w1 = d.options[opt_tuple[d_idx]]
            e0, e1 = d.out_edges
            node_choices[d.node] = {e0: int(w0), e1: int(w1)}

        edge_weights: dict[GraphEdge, EdgeWeightValue] = {}
        for idx, e in enumerate(self.dem.edges):
            if e in self.flag_edge_to_constraint:
                fc = self.flag_edge_to_constraint[e]
                wa = self.decisions[fc.decision_idx_a].options[opt_tuple[fc.decision_idx_a]][fc.leg_a]
                wb = self.decisions[fc.decision_idx_b].options[opt_tuple[fc.decision_idx_b]][fc.leg_b]
                edge_weights[e] = (int(wa), int(wb))
            else:
                edge_weights[e] = int(effective_weights[idx])

        score = compute_polynomial_score(coeffs_tuple, p=p, mode=score_mode)
        return WeightCandidate(
            coeffs=coeffs_tuple,
            polynomial_str=poly_str,
            edge_weights=edge_weights,
            effective_weights=effective_weights,
            node_choices=node_choices,
            option_indices=opt_tuple,
            score=score,
        )


def build_weight_space(
    table_or_dem: PrecomputedFaultTable | GraphDEM,
    fix_boundary_weight_one: bool = True,
    allow_root_3_0: bool = True,
) -> WeightSpace:
    """
    Constructs the `WeightSpace` of node decisions and flag-edge constraints from a
    `PrecomputedFaultTable` or `GraphDEM`.

    Rules:
    1. Root node (degree 2): 2 outgoing tree edges, total weight 3 -> `(2, 1)`, `(1, 2)`,
       and (if `allow_root_3_0=True`) `(3, 0)`, `(0, 3)`.
    2. Each green node (degree 3 in the open graph): 1 incoming tree edge from its parent
       in `spanning_tree`, and 2 outgoing edges with weights `(2, 1)` or `(1, 2)`.
       - If `is_mark=True` and `fix_boundary_weight_one=True`, the node's `(node, "boundary")`
         out-edge is locked to weight `1` and its other out-edge is locked to weight `2`.
       - If `fix_boundary_weight_one=False`, `is_mark` nodes can also choose `(2, 1)` or `(1, 2)`
         (with `(boundary=1, other=2)` as default option 0).
    3. Each flag edge `(a, b) in G_exp \\ F_exp` is an outgoing edge of both green nodes `a` and `b`.
       Its weight is represented as `(w_a, w_b)` summing to `w_a + w_b`, subject to the constraint
       that at least one of `w_a, w_b` is 2 (so effective weight is 3 or 4).
    """
    dem = table_or_dem.dem if isinstance(table_or_dem, PrecomputedFaultTable) else table_or_dem
    G_exp = dem.G_exp
    T = dem.spanning_tree
    root = dem.root

    bfs_tree = nx.bfs_tree(T, root)
    bfs_order = list(bfs_tree.nodes())
    parent_of: dict[int, int | None] = {root: None}
    for u, v in bfs_tree.edges():
        parent_of[v] = u

    decisions: list[NodeDecision] = []
    node_to_d_idx: dict[int, int] = {}
    passthrough_edges: dict[int, tuple[int, int]] = {}

    for node in bfs_order:
        is_root = node == root
        node_data = G_exp.nodes[node]
        is_mark = bool(node_data.get("is_mark", False))
        is_flag = bool(node_data.get("is_flag", False))

        if is_Root := is_root:
            children = sorted(bfs_tree.successors(node))
            if len(children) != 2:
                raise ValueError(
                    f"Expected root node {node} to have degree 2 in spanning tree, got {len(children)}."
                )
            e0 = ed(node, children[0])
            e1 = ed(node, children[1])
            idx0 = dem.edge_to_idx[e0]
            idx1 = dem.edge_to_idx[e1]
            root_opts: tuple[tuple[int, int], ...] = (
                ((2, 1), (1, 2), (3, 0), (0, 3)) if allow_root_3_0 else ((2, 1), (1, 2))
            )
            node_to_d_idx[node] = len(decisions)
            decisions.append(
                NodeDecision(
                    node=node,
                    is_root=True,
                    is_mark=False,
                    in_edge=None,
                    out_edges=(e0, e1),
                    out_edge_indices=(idx0, idx1),
                    options=root_opts,
                    default_option_idx=0,
                )
            )
            continue

        parent = parent_of[node]
        assert parent is not None
        in_edge = ed(parent, node)

        # In case an older graph has a degree-2 `is_flag` node, pass its parent's weight through
        if is_flag and not is_mark and G_exp.degree(node) == 2:
            other_nbrs = [n for n in G_exp.neighbors(node) if n != parent]
            if len(other_nbrs) == 1 and parent in node_to_d_idx:
                p_d_idx = node_to_d_idx[parent]
                p_dec = decisions[p_d_idx]
                leg = 0 if p_dec.out_edges[0] == in_edge else 1
                out_e = ed(node, other_nbrs[0])
                passthrough_edges[dem.edge_to_idx[out_e]] = (p_d_idx, leg)
            continue

        # Collect outgoing edges for this green node
        inc_internal = [
            ed(node, nbr) for nbr in sorted(G_exp.neighbors(node)) if nbr != parent
        ]
        if is_mark:
            if len(inc_internal) != 1:
                raise ValueError(
                    f"Expected is_mark node {node} to have 1 non-parent internal edge, got {inc_internal}."
                )
            e_int = inc_internal[0]
            e_bnd = make_boundary_edge(node)
            # Put (e_int, e_bnd) in order so option (2, 1) assigns 2 to e_int and 1 to e_bnd!
            out_edges: tuple[GraphEdge, GraphEdge] = (e_int, e_bnd)
            out_indices = (dem.edge_to_idx[e_int], dem.edge_to_idx[e_bnd])
            if fix_boundary_weight_one:
                opts: tuple[tuple[int, int], ...] = ((2, 1),)
            else:
                opts = ((2, 1), (1, 2))
        else:
            if len(inc_internal) != 2:
                raise ValueError(
                    f"Expected internal green node {node} to have 2 non-parent edges, got {inc_internal}."
                )
            out_edges = (inc_internal[0], inc_internal[1])
            out_indices = (dem.edge_to_idx[out_edges[0]], dem.edge_to_idx[out_edges[1]])
            opts = ((2, 1), (1, 2))

        node_to_d_idx[node] = len(decisions)
        decisions.append(
            NodeDecision(
                node=node,
                is_root=False,
                is_mark=is_mark,
                in_edge=in_edge,
                out_edges=out_edges,
                out_edge_indices=out_indices,
                options=opts,
                default_option_idx=0,
            )
        )

    # Build FlagEdgeConstraint for each non-tree edge in G_exp \ T
    flag_constraints: list[FlagEdgeConstraint] = []
    for edge_idx in dem.non_tree_edge_indices:
        f_edge = dem.edges[edge_idx]
        u, v = f_edge  # already normalized with u < v
        if u in node_to_d_idx and v in node_to_d_idx:
            da = node_to_d_idx[u]
            db = node_to_d_idx[v]
            dec_a = decisions[da]
            dec_b = decisions[db]
            leg_a = 0 if dec_a.out_edges[0] == f_edge else 1
            leg_b = 0 if dec_b.out_edges[0] == f_edge else 1

            # Find which option index (if any) assigns weight 1 to `f_edge`
            forbid_a = next(
                (opt_i for opt_i, pair in enumerate(dec_a.options) if pair[leg_a] == 1),
                -1,
            )
            forbid_b = next(
                (opt_i for opt_i, pair in enumerate(dec_b.options) if pair[leg_b] == 1),
                -1,
            )
            flag_constraints.append(
                FlagEdgeConstraint(
                    edge=f_edge,
                    edge_idx=edge_idx,
                    node_a=u,
                    node_b=v,
                    decision_idx_a=da,
                    leg_a=leg_a,
                    decision_idx_b=db,
                    leg_b=leg_b,
                    forbidden_option_idx_a=forbid_a,
                    forbidden_option_idx_b=forbid_b,
                )
            )

    return WeightSpace(
        dem=dem,
        decisions=decisions,
        flag_constraints=flag_constraints,
        passthrough_edges=passthrough_edges,
    )


def _compare_SortKey(
    coeffs: tuple[int, ...],
    score: float,
    objective: Literal["lex", "eval"] = "lex",
) -> tuple:
    if objective == "lex":
        return (coeffs, score)
    elif objective == "eval":
        return (score, coeffs)
    raise ValueError(f"Unknown objective: {objective!r}")


# =============================================================================
# 1. Exhaustive Search
# =============================================================================


def exhaustive_weight_search(
    table: PrecomputedFaultTable,
    *,
    fix_boundary_weight_one: bool = True,
    allow_root_3_0: bool = True,
    objective: Literal["lex", "eval"] = "lex",
    p: float = 1e-3,
    score_mode: Literal["expected_errors", "expected_fault_events"] = "expected_errors",
    poly_mode: Literal["boundary_weight", "fault_weight"] = "boundary_weight",
    batch_size: int = 4096,
    max_results: int | None = None,
) -> list[WeightCandidate]:
    """
    Enumerates all valid weight choices across the root and degree-3 green nodes
    (enforcing that every flag edge has at least one 2, i.e., weight 3 or 4),
    evaluates all fault polynomials in fast vectorized batches, and returns the
    candidates sorted from best to worst.

    Args:
        table: `PrecomputedFaultTable` for `(G_exp, F_exp, root)`.
        fix_boundary_weight_one: If True (default), locks `node -> boundary` edges to
            weight 1 and the other out-edge of `is_mark` nodes to weight 2. If False,
            also explores swapping `(1, 2)` on `is_mark` nodes.
        allow_root_3_0: If True (default), allows root out-edge weights `(2, 1), (1, 2), (3, 0), (0, 3)`.
            If False, restricts root to `(2, 1)` and `(1, 2)`.
        objective: Sorting criterion:
            - `"lex"` (default): Sorts lexicographically by `(c_1, c_2, ..., c_L)` first
              (prioritizing lower powers of `p`), breaking ties by `score`.
            - `"eval"`: Sorts by scalar `score` at error probability `p`.
        p: Target physical error rate `p` used to compute `candidate.score` (default `1e-3`).
        score_mode: `"expected_errors"` (`sum k c_k p^k`) or `"expected_fault_events"` (`sum c_k p^k`).
        poly_mode: Passed to `table.evaluate_polynomial` (`"boundary_weight"` or `"fault_weight"`).
        batch_size: Number of weight vectors evaluated per vectorized `evaluate_polynomial` call.
        max_results: Optional cap on the number of top candidates returned (`None` returns all).

    Returns:
        List of `WeightCandidate` objects sorted from best to worst.
    """
    space = build_weight_space(
        table,
        fix_boundary_weight_one=fix_boundary_weight_one,
        allow_root_3_0=allow_root_3_0,
    )

    all_opt_tuples: list[tuple[int, ...]] = []
    all_coeffs_list: list[np.ndarray] = []
    all_eff_weights_list: list[np.ndarray] = []

    iterator = space.iter_valid_option_indices()
    while True:
        chunk = list(itertools.islice(iterator, batch_size))
        if not chunk:
            break
        opt_batch = np.asarray(chunk, dtype=np.int64)
        w_batch = space.option_indices_to_effective_weights(opt_batch)
        coeffs_batch = table.evaluate_polynomial(w_batch, mode=poly_mode)

        all_opt_tuples.extend(chunk)
        all_eff_weights_list.append(w_batch)
        all_coeffs_list.append(coeffs_batch)

    if not all_opt_tuples:
        return []

    full_eff_weights = np.vstack(all_eff_weights_list)
    full_coeffs = np.vstack(all_coeffs_list)
    full_scores = _compute_batch_scores(full_coeffs, p=p, mode=score_mode)

    # Sort indices before materializing full dictionaries if max_results is set
    num_total = len(all_opt_tuples)
    coeffs_tuples = [tuple(int(x) for x in row) for row in full_coeffs]

    sorted_indices = sorted(
        range(num_total),
        key=lambda i: _compare_SortKey(coeffs_tuples[i], float(full_scores[i]), objective),
    )
    if max_results is not None:
        sorted_indices = sorted_indices[:max_results]

    results: list[WeightCandidate] = []
    for i in sorted_indices:
        results.append(
            space.build_candidate(
                table=table,
                option_indices=all_opt_tuples[i],
                coeffs=coeffs_tuples[i],
                effective_weights=full_eff_weights[i],
                p=p,
                score_mode=score_mode,
                poly_mode=poly_mode,
            )
        )
    return results


# =============================================================================
# 2. Greedy Search
# =============================================================================


def _enumerate_valid_neighbor_moves(
    space: WeightSpace,
    current_opts: Sequence[int],
) -> list[tuple[int, ...]]:
    """
    Generates all valid 1-step neighbor configurations from `current_opts`:
    - Flipping any single unlocked decision node `d_idx` to another option `opt`
      (if that preserves `(w_a, w_b) != (1, 1)` on all flag edges).
    - If flipping `d_idx` to `opt` would create `(1, 1)` on a flag edge `fc` whose
      other endpoint `other_idx` is unlocked, couples the move by simultaneously
      flipping `other_idx` to weight 2 (i.e. swapping `(2, 1) <-> (1, 2)` on the flag edge).
    """
    base = list(current_opts)
    seen: set[tuple[int, ...]] = {tuple(base)}
    neighbors: list[tuple[int, ...]] = []

    for d_idx in space.free_decision_indices:
        d = space.decisions[d_idx]
        curr_opt = base[d_idx]
        for new_opt in range(d.num_options):
            if new_opt == curr_opt:
                continue
            cand = base.copy()
            cand[d_idx] = new_opt

            # Check flag constraints incident to d_idx
            possible = True
            for fc in space.decision_to_flag_constraints[d_idx]:
                if (
                    cand[fc.decision_idx_a] == fc.forbidden_option_idx_a
                    and cand[fc.decision_idx_b] == fc.forbidden_option_idx_b
                ):
                    other_idx = fc.decision_idx_b if d_idx == fc.decision_idx_a else fc.decision_idx_a
                    other_forbid = (
                        fc.forbidden_option_idx_b
                        if d_idx == fc.decision_idx_a
                        else fc.forbidden_option_idx_a
                    )
                    if space.decisions[other_idx].is_locked:
                        possible = False
                        break
                    cand[other_idx] = 1 - other_forbid

            if possible and space.is_valid_option_indices(cand):
                cand_tuple = tuple(cand)
                if cand_tuple not in seen:
                    seen.add(cand_tuple)
                    neighbors.append(cand_tuple)

    return neighbors


def greedy_weight_search(
    table: PrecomputedFaultTable,
    *,
    fix_boundary_weight_one: bool = True,
    allow_root_3_0: bool = True,
    objective: Literal["lex", "eval"] = "lex",
    p: float = 1e-3,
    score_mode: Literal["expected_errors", "expected_fault_events"] = "expected_errors",
    poly_mode: Literal["boundary_weight", "fault_weight"] = "boundary_weight",
    polish: bool = True,
) -> WeightCandidate:
    """
    Constructs a weight assignment greedily by always picking the choice at each node
    that reduces the undetectable boundary error polynomial the most, followed by
    steepest-descent local polishing.

    How it works:
    1. Initializes `is_mark` nodes to the heuristic `(boundary=1, internal=2)` and
       undecided internal green nodes to neutral fractional weights `(1.5, 1.5)`
       (so undecided flag edges start at `1.5 + 1.5 = 3.0` or `2.0 + 1.5 = 3.5`).
    2. Sweeps through the decision nodes in BFS order from the root; at each node,
       evaluates all valid options (respecting the flag-edge constraint that at least
       one side of every flag edge has weight 2) and locks in the option that minimizes
       the error polynomial.
    3. If `polish=True` (default), performs coordinate-descent local search over all
       valid 1-node flips (and coupled 2-node flag swaps `(2, 1) <-> (1, 2)`),
       evaluating the entire neighborhood in a single vectorized batch call at each
       iteration until a local optimum is reached.
    """
    space = build_weight_space(
        table,
        fix_boundary_weight_one=fix_boundary_weight_one,
        allow_root_3_0=allow_root_3_0,
    )

    num_d = space.num_decisions
    chosen_opts: list[int | None] = [
        0 if d.is_locked or d.is_mark else None for d in space.decisions
    ]

    def _build_partial_weights(trial_d_idx: int, trial_opt: int) -> np.ndarray:
        w = np.zeros(table.dem.num_edges, dtype=np.float64)
        for idx, d in enumerate(space.decisions):
            e0_idx, e1_idx = d.out_edge_indices
            if idx == trial_d_idx:
                w0, w1 = d.options[trial_opt]
            elif chosen_opts[idx] is not None:
                w0, w1 = d.options[chosen_opts[idx]]  # type: ignore[index]
            else:
                # Neutral expected weight (1.5, 1.5) for undecided nodes
                w0, w1 = 1.5, 1.5
            w[e0_idx] += w0
            w[e1_idx] += w1
        for edge_idx, (d_idx, leg_idx) in space.passthrough_edges.items():
            d = space.decisions[d_idx]
            opt_val = trial_opt if d_idx == trial_d_idx else chosen_opts[d_idx]
            pair = d.options[opt_val] if opt_val is not None else (1.5, 1.5)
            w[edge_idx] += pair[leg_idx]
        return w

    # Phase 1: Constructive BFS greedy pass over undecided nodes (and unlocked mark nodes)
    # First decide undecided internal/root nodes, then optionally revisit unlocked mark nodes
    pass_order = [
        idx for idx, d in enumerate(space.decisions) if not d.is_locked and not d.is_mark
    ] + [
        idx for idx, d in enumerate(space.decisions) if not d.is_locked and d.is_mark
    ]

    for d_idx in pass_order:
        d = space.decisions[d_idx]
        valid_opts: list[int] = []
        for opt_idx in range(d.num_options):
            # Check that opt_idx doesn't conflict with an already-decided flag partner
            ok = True
            for fc in space.decision_to_flag_constraints[d_idx]:
                if d_idx == fc.decision_idx_a:
                    other_idx = fc.decision_idx_b
                    my_forbid = fc.forbidden_option_idx_a
                    other_forbid = fc.forbidden_option_idx_b
                else:
                    other_idx = fc.decision_idx_a
                    my_forbid = fc.forbidden_option_idx_b
                    other_forbid = fc.forbidden_option_idx_a

                if opt_idx == my_forbid and chosen_opts[other_idx] == other_forbid:
                    ok = False
                    break
            if ok:
                valid_opts.append(opt_idx)

        if not valid_opts:
            valid_opts = list(range(d.num_options))

        if len(valid_opts) == 1:
            chosen_opts[d_idx] = valid_opts[0]
            continue

        cand_weights = np.vstack([_build_partial_weights(d_idx, opt) for opt in valid_opts])
        cand_coeffs = table.evaluate_polynomial(cand_weights, mode=poly_mode)
        cand_scores = _compute_batch_scores(cand_coeffs, p=p, mode=score_mode)

        best_local_i = min(
            range(len(valid_opts)),
            key=lambda i: _compare_SortKey(
                tuple(float(x) for x in cand_coeffs[i]),  # type: ignore[arg-type]
                float(cand_scores[i]),
                objective,
            ),
        )
        chosen_opts[d_idx] = valid_opts[best_local_i]

    current_opts = space.repair_option_indices([int(x or 0) for x in chosen_opts])
    current_eff_w = space.option_indices_to_effective_weights(current_opts)
    current_coeffs_arr = table.evaluate_polynomial(current_eff_w, mode=poly_mode)
    current_coeffs = tuple(int(x) for x in current_coeffs_arr)
    current_score = compute_polynomial_score(current_coeffs, p=p, mode=score_mode)
    current_key = _compare_SortKey(current_coeffs, current_score, objective)

    # Phase 2: Iterative steepest-descent coordinate polish
    if polish and space.num_free_decisions > 0:
        while True:
            neighbors = _enumerate_valid_neighbor_moves(space, current_opts)
            if not neighbors:
                break
            nb_batch = np.asarray(neighbors, dtype=np.int64)
            nb_eff_w = space.option_indices_to_effective_weights(nb_batch)
            nb_coeffs = table.evaluate_polynomial(nb_eff_w, mode=poly_mode)
            nb_scores = _compute_batch_scores(nb_coeffs, p=p, mode=score_mode)

            best_nb_i = -1
            best_nb_key = current_key
            for i in range(len(neighbors)):
                c_tup = tuple(int(x) for x in nb_coeffs[i])
                sc = float(nb_scores[i])
                k = _compare_SortKey(c_tup, sc, objective)
                if k < best_nb_key:
                    best_nb_key = k
                    best_nb_i = i

            if best_nb_i < 0:
                break

            current_opts = list(neighbors[best_nb_i])
            current_eff_w = nb_eff_w[best_nb_i]
            current_coeffs = tuple(int(x) for x in nb_coeffs[best_nb_i])
            current_score = float(nb_scores[best_nb_i])
            current_key = best_nb_key

    return space.build_candidate(
        table=table,
        option_indices=current_opts,
        coeffs=current_coeffs,
        effective_weights=current_eff_w,
        p=p,
        score_mode=score_mode,
        poly_mode=poly_mode,
    )


# =============================================================================
# 3. Simulated Annealing
# =============================================================================


@dataclass(frozen=True)
class SimulatedAnnealingResult:
    """
    Result of `simulated_annealing_weight_search`, containing the best candidate found,
    the initial (greedy) candidate, and the annealing score trajectory.
    """

    best_candidate: WeightCandidate
    initial_candidate: WeightCandidate
    score_history: tuple[float, ...]
    best_score_history: tuple[float, ...]
    num_accepted: int
    num_steps: int


def simulated_annealing_weight_search(
    table: PrecomputedFaultTable,
    *,
    init_candidate: WeightCandidate | None = None,
    fix_boundary_weight_one: bool = True,
    allow_root_3_0: bool = True,
    objective: Literal["lex", "eval"] = "lex",
    p: float = 1e-3,
    score_mode: Literal["expected_errors", "expected_fault_events"] = "expected_errors",
    poly_mode: Literal["boundary_weight", "fault_weight"] = "boundary_weight",
    num_steps: int = 1000,
    T_init: float | None = None,
    T_final: float | None = None,
    seed: int | np.random.Generator | None = 0,
    return_history: bool = False,
) -> WeightCandidate | SimulatedAnnealingResult:
    """
    Starts from the greedy weight assignment (with `polish=False` or `init_candidate`)
    and performs Simulated Annealing where each step swaps the two outgoing weights at
    a randomly chosen node (preserving the flag-edge constraint `w(f) in {3, 4}`).

    Details of the Simulated Annealing loop:
    1. **Initialization**:
       - Starts from `init_candidate` if provided, or runs `greedy_weight_search`
         (without local polish so SA can explore freely from the constructive greedy point,
         while also evaluating the polished greedy solution so `best_candidate` is never
         worse than polished greedy).
    2. **Move Proposal (Swap Two Outputs)**:
       - At each step, selects a random unlocked decision node `v` and swaps its two
         outgoing edge weights `(2, 1) <-> (1, 2)` (or picks another root split).
       - If swapping `v` alone would give `(1, 1)` on an incident flag edge `(v, u)`,
         automatically couples the swap with node `u` so the flag edge transitions
         `(2, 1) <-> (1, 2)` (keeping total flag weight 3).
    3. **Energy & Cooling Schedule**:
       - Uses the expected boundary error rate `E(w) = compute_polynomial_score(coeffs, p)`
         as the energy function (normalized by `p` for numerical scale stability).
       - Auto-calibrates `T_init` from the median positive `|Delta E|` of initial neighbor
         moves if `T_init is None`, and cools geometrically to `T_final = 1e-3 * T_init`.
       - Accepts improving moves (`Delta E <= 0`) always, and uphill moves (`Delta E > 0`)
         with Metropolis probability `exp(-Delta E / T)`.

    Returns:
        `WeightCandidate` (best found) by default, or `SimulatedAnnealingResult` if
        `return_history=True`.
    """
    rng = np.random.default_rng(seed)
    space = build_weight_space(
        table,
        fix_boundary_weight_one=fix_boundary_weight_one,
        allow_root_3_0=allow_root_3_0,
    )

    if init_candidate is None:
        # Start the Markov walk from the constructive greedy state, and also keep track
        # of the polished greedy state so best_candidate is guaranteed >= polished greedy.
        initial_candidate = greedy_weight_search(
            table,
            fix_boundary_weight_one=fix_boundary_weight_one,
            allow_root_3_0=allow_root_3_0,
            objective=objective,
            p=p,
            score_mode=score_mode,
            poly_mode=poly_mode,
            polish=False,
        )
        polished_greedy = greedy_weight_search(
            table,
            fix_boundary_weight_one=fix_boundary_weight_one,
            allow_root_3_0=allow_root_3_0,
            objective=objective,
            p=p,
            score_mode=score_mode,
            poly_mode=poly_mode,
            polish=True,
        )
    else:
        initial_candidate = init_candidate
        polished_greedy = init_candidate

    curr_opts = list(initial_candidate.option_indices)
    curr_eff_w = initial_candidate.effective_weights.copy()
    curr_coeffs = initial_candidate.coeffs
    curr_score = initial_candidate.score

    best_opts = list(polished_greedy.option_indices)
    best_eff_w = polished_greedy.effective_weights.copy()
    best_coeffs = polished_greedy.coeffs
    best_score = polished_greedy.score
    best_key = _compare_SortKey(best_coeffs, best_score, objective)

    curr_key = _compare_SortKey(curr_coeffs, curr_score, objective)
    if curr_key < best_key:
        best_opts = list(curr_opts)
        best_eff_w = curr_eff_w.copy()
        best_coeffs = curr_coeffs
        best_score = curr_score
        best_key = curr_key

    if space.num_free_decisions == 0 or num_steps <= 0:
        best_cand = space.build_candidate(
            table=table,
            option_indices=best_opts,
            coeffs=best_coeffs,
            effective_weights=best_eff_w,
            p=p,
            score_mode=score_mode,
            poly_mode=poly_mode,
        )
        if return_history:
            return SimulatedAnnealingResult(
                best_candidate=best_cand,
                initial_candidate=initial_candidate,
                score_history=(curr_score,),
                best_score_history=(best_score,),
                num_accepted=0,
                num_steps=0,
            )
        return best_cand

    # Auto-calibrate temperature from initial neighborhood energy differences if T_init is None
    if T_init is None:
        init_neighbors = _enumerate_valid_neighbor_moves(space, curr_opts)
        if init_neighbors:
            nb_w = space.option_indices_to_effective_weights(np.asarray(init_neighbors, dtype=np.int64))
            nb_c = table.evaluate_polynomial(nb_w, mode=poly_mode)
            nb_scores = _compute_batch_scores(nb_c, p=p, mode=score_mode)
            deltas = np.abs(nb_scores - curr_score)
            pos_deltas = deltas[deltas > 0]
            if len(pos_deltas) > 0:
                # Set T_init so median uphill move has ~50% acceptance probability (-Delta / ln(0.5))
                T_init = float(np.median(pos_deltas) / math.log(2.0))
            else:
                T_init = max(curr_score * 0.05, 1e-12)
        else:
            T_init = max(curr_score * 0.05, 1e-12)

    if T_final is None:
        T_final = max(T_init * 1e-3, 1e-15)

    score_history: list[float] = [curr_score]
    best_score_history: list[float] = [best_score]
    num_accepted = 0
    free_indices = space.free_decision_indices

    for step in range(num_steps):
        # Geometric cooling schedule
        frac = step / max(num_steps - 1, 1)
        temp = T_init * ((T_final / T_init) ** frac) if T_init > 0 else 0.0

        # Propose a single random node swap (with automatic flag constraint repair)
        d_idx = int(free_indices[rng.integers(0, len(free_indices))])
        d = space.decisions[d_idx]
        curr_opt = curr_opts[d_idx]
        if d.num_options == 2:
            new_opt = 1 - curr_opt
        else:
            other_opts = [o for o in range(d.num_options) if o != curr_opt]
            new_opt = int(other_opts[rng.integers(0, len(other_opts))])

        prop_opts = curr_opts.copy()
        prop_opts[d_idx] = new_opt

        # Enforce flag edge constraint: if (1, 1) occurs on an incident flag edge,
        # couple the move by flipping the partner node to 2
        valid_move = True
        for fc in space.decision_to_flag_constraints[d_idx]:
            if (
                prop_opts[fc.decision_idx_a] == fc.forbidden_option_idx_a
                and prop_opts[fc.decision_idx_b] == fc.forbidden_option_idx_b
            ):
                other_idx = fc.decision_idx_b if d_idx == fc.decision_idx_a else fc.decision_idx_a
                other_forbid = (
                    fc.forbidden_option_idx_b
                    if d_idx == fc.decision_idx_a
                    else fc.forbidden_option_idx_a
                )
                if space.decisions[other_idx].is_locked:
                    valid_move = False
                    break
                prop_opts[other_idx] = 1 - other_forbid

        if not valid_move or not space.is_valid_option_indices(prop_opts):
            score_history.append(curr_score)
            best_score_history.append(best_score)
            continue

        prop_eff_w = space.option_indices_to_effective_weights(prop_opts)
        prop_coeffs_arr = table.evaluate_polynomial(prop_eff_w, mode=poly_mode)
        prop_coeffs = tuple(int(x) for x in prop_coeffs_arr)
        prop_score = compute_polynomial_score(prop_coeffs, p=p, mode=score_mode)

        delta_e = prop_score - curr_score
        if delta_e <= 0.0 or (temp > 0.0 and rng.random() < math.exp(-delta_e / temp)):
            curr_opts = prop_opts
            curr_eff_w = prop_eff_w
            curr_coeffs = prop_coeffs
            curr_score = prop_score
            num_accepted += 1

            prop_key = _compare_SortKey(prop_coeffs, prop_score, objective)
            if prop_key < best_key:
                best_opts = list(prop_opts)
                best_eff_w = prop_eff_w.copy()
                best_coeffs = prop_coeffs
                best_score = prop_score
                best_key = prop_key

        score_history.append(curr_score)
        best_score_history.append(best_score)

    best_candidate = space.build_candidate(
        table=table,
        option_indices=best_opts,
        coeffs=best_coeffs,
        effective_weights=best_eff_w,
        p=p,
        score_mode=score_mode,
        poly_mode=poly_mode,
    )

    if return_history:
        return SimulatedAnnealingResult(
            best_candidate=best_candidate,
            initial_candidate=initial_candidate,
            score_history=tuple(score_history),
            best_score_history=tuple(best_score_history),
            num_accepted=num_accepted,
            num_steps=num_steps,
        )
    return best_candidate

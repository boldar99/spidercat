from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable, Iterator, Literal, Sequence

import networkx as nx
import numpy as np
import stim

from spidercat.utils import ed

BoundaryEdge = tuple[int, str]
InternalEdge = tuple[int, int]
GraphEdge = InternalEdge | BoundaryEdge


def make_boundary_edge(node: int) -> BoundaryEdge:
    """Returns the canonical boundary edge representation for an `is_mark` node."""
    return (int(node), "boundary")


def normalize_edge(edge: tuple) -> GraphEdge:
    """Normalizes an internal edge `(u, v)` to `u < v` or a boundary edge to `(node, 'boundary')`."""
    if len(edge) != 2:
        raise ValueError(f"Expected a 2-tuple edge, got: {edge!r}")
    u, v = edge
    if v == "boundary" or u == "boundary":
        node = u if v == "boundary" else v
        return make_boundary_edge(int(node))
    return ed(int(u), int(v))


def _complete_spanning_tree(G_exp: nx.Graph, F_exp: nx.Graph) -> nx.Graph:
    """
    If `F_exp` is already a single spanning tree of `G_exp`, returns a copy of `F_exp`.
    If `F_exp` has multiple connected components (a forest with p > 1 trees), augments it
    with p - 1 inter-tree edges from `G_exp` (matching CatStateExtractor's meta-graph
    feedback links) so that cut propagation is well-defined across the entire graph.
    """
    if not nx.is_connected(G_exp):
        raise ValueError("G_exp must be a connected graph.")

    T = F_exp.copy()
    components = [set(c) for c in nx.connected_components(T)]
    if len(components) == 1:
        return T

    comp_of = {}
    for comp in components:
        cid = min(comp)
        for node in comp:
            comp_of[node] = cid

    meta = nx.Graph()
    for u, v in sorted(ed(a, b) for a, b in G_exp.edges() if not T.has_edge(a, b)):
        cu, cv = comp_of[u], comp_of[v]
        if cu != cv and not meta.has_edge(cu, cv):
            meta.add_edge(cu, cv, bridge=(u, v))

    root_cid = min(meta.nodes())
    preds = dict(nx.bfs_predecessors(meta, root_cid))
    for cid, pred_cid in preds.items():
        u, v = meta[pred_cid][cid]["bridge"]
        T.add_edge(u, v)

    return T


def _elementary_symmetric_poly_1d(weights: np.ndarray, max_degree: int) -> np.ndarray:
    """
    Computes elementary symmetric polynomials [e_0, e_1, ..., e_{max_degree}]
    for a 1D array of weights in O(len(weights) * max_degree) time.
    """
    dp = np.zeros(max_degree + 1, dtype=weights.dtype)
    dp[0] = 1
    for idx, w in enumerate(weights):
        if w == 0:
            continue
        upper = min(idx + 1, max_degree)
        dp[1 : upper + 1] += w * dp[0:upper]
    return dp


def _elementary_symmetric_poly_2d(weights: np.ndarray, max_degree: int) -> np.ndarray:
    """
    Computes elementary symmetric polynomials [e_0, e_1, ..., e_{max_degree}]
    for a 2D batch of weights of shape (batch_size, num_items).
    Returns an array of shape (batch_size, max_degree + 1).
    """
    batch_size, num_items = weights.shape
    dp = np.zeros((batch_size, max_degree + 1), dtype=weights.dtype)
    dp[:, 0] = 1
    for idx in range(num_items):
        w = weights[:, idx : idx + 1]
        upper = min(idx + 1, max_degree)
        dp[:, 1 : upper + 1] += w * dp[:, 0:upper]
    return dp


@dataclass(frozen=True)
class GraphDEM:
    """
    Detector Error Model constructed directly from `(G_exp, F_exp)`.

    - Detectors `D_0 .. D_{C-1}` correspond to a cycle basis of `G_exp`.
    - Logical observables `L_0 .. L_{n-1}` correspond to the `n` boundary qubits
      connected to `is_mark=True` nodes.
    - Each internal edge in `G_exp` and each of the `n` boundary edges maps to
      the detectors (cycles containing the edge) and observables (boundary qubits
      it propagates to, folded modulo the all-output cat-state stabilizer `X^n`).
    """

    G_exp: nx.Graph
    F_exp: nx.Graph
    spanning_tree: nx.Graph
    root: int
    mark_nodes: tuple[int, ...]
    internal_edges: tuple[InternalEdge, ...]
    boundary_edges: tuple[BoundaryEdge, ...]
    edges: tuple[GraphEdge, ...]
    edge_to_idx: dict[GraphEdge, int]
    tree_edge_indices: tuple[int, ...]
    non_tree_edge_indices: tuple[int, ...]
    boundary_edge_indices: tuple[int, ...]
    cycles: tuple[tuple[InternalEdge, ...], ...]
    det_masks: tuple[int, ...]
    obs_masks: tuple[int, ...]
    unfolded_obs_masks: tuple[int, ...]

    @property
    def num_qubits(self) -> int:
        return len(self.mark_nodes)

    @property
    def num_detectors(self) -> int:
        return len(self.cycles)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def edge_targets(self, edge: GraphEdge | int) -> tuple[list[int], list[int]]:
        """
        Returns `(detector_ids, observable_ids)` flipped by a fault on `edge`
        (specified either as an edge tuple or integer edge index).
        """
        idx = edge if isinstance(edge, (int, np.integer)) else self.edge_to_idx[normalize_edge(edge)]
        d_mask = self.det_masks[idx]
        o_mask = self.obs_masks[idx]
        det_ids = [i for i in range(self.num_detectors) if (d_mask >> i) & 1]
        obs_ids = [j for j in range(self.num_qubits) if (o_mask >> j) & 1]
        return det_ids, obs_ids

    def to_stim_dem(
        self,
        probabilities: float | dict[GraphEdge, float] | Sequence[float] = 0.001,
        include_empty_errors: bool = False,
    ) -> stim.DetectorErrorModel:
        """
        Converts this graph-forest DEM into a `stim.DetectorErrorModel`.
        Each edge `e` produces an `error(p_e) D... L...` instruction.
        """
        if isinstance(probabilities, (float, int, np.floating, np.integer)):
            p_list = [float(probabilities)] * self.num_edges
        elif isinstance(probabilities, dict):
            p_list = []
            for e in self.edges:
                if e in probabilities:
                    p_list.append(float(probabilities[e]))
                elif e[1] != "boundary" and (e[1], e[0]) in probabilities:
                    p_list.append(float(probabilities[(e[1], e[0])]))
                elif e[1] == "boundary" and e[0] in probabilities:
                    p_list.append(float(probabilities[e[0]]))
                else:
                    raise KeyError(f"Missing probability for edge {e!r}")
        else:
            p_list = [float(p) for p in probabilities]
            if len(p_list) != self.num_edges:
                raise ValueError(f"Expected {self.num_edges} probabilities, got {len(p_list)}")

        dem = stim.DetectorErrorModel()
        for idx, p in enumerate(p_list):
            if p <= 0:
                continue
            det_ids, obs_ids = self.edge_targets(idx)
            targets = [stim.target_relative_detector_id(d) for d in det_ids] + [
                stim.target_logical_observable_id(o) for o in obs_ids
            ]
            if targets or include_empty_errors:
                dem.append("error", p, targets)
        return dem


def build_graph_dem(
    G_exp: nx.Graph,
    F_exp: nx.Graph,
    root: int | dict[int, int] | None = None,
    cycle_basis_mode: Literal["fundamental", "nx"] = "fundamental",
) -> GraphDEM:
    """
    Constructs a `GraphDEM` from `(G_exp, F_exp)` and an optional `root`.

    Args:
        G_exp: Expanded graph where marked degree-2 nodes have `is_mark=True`.
        F_exp: Expanded spanning tree / forest of `G_exp`.
        root: Optional root node ID (or `{tree_id: root_node}` dict) used to orient
            downward propagation on tree edges before folding modulo `X^n`.
        cycle_basis_mode:
            - `"fundamental"` (default): Uses the fundamental cycle basis induced by
              `F_exp`, where each non-tree edge belongs to exactly one cycle.
            - `"nx"`: Uses `nx.cycle_basis(G_exp)`.
    """
    T = _complete_spanning_tree(G_exp, F_exp)

    if root is None:
        chosen_root = min(T.nodes())
    elif isinstance(root, dict):
        chosen_root = min(root.values())
    else:
        chosen_root = int(root)
    if chosen_root not in T:
        raise ValueError(f"Root node {chosen_root} is not in F_exp.")

    mark_nodes = tuple(sorted(v for v, d in G_exp.nodes(data=True) if d.get("is_mark", False)))
    n = len(mark_nodes)
    mark_to_bit = {v: i for i, v in enumerate(mark_nodes)}
    all_obs_mask = (1 << n) - 1

    tree_edges = sorted(ed(u, v) for u, v in T.edges())
    non_tree_edges = sorted(ed(u, v) for u, v in G_exp.edges() if not T.has_edge(u, v))
    internal_edges = tuple(tree_edges + non_tree_edges)
    boundary_edges = tuple(make_boundary_edge(m) for m in mark_nodes)
    edges: tuple[GraphEdge, ...] = internal_edges + boundary_edges
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}

    tree_edge_indices = tuple(range(len(tree_edges)))
    non_tree_edge_indices = tuple(range(len(tree_edges), len(internal_edges)))
    boundary_edge_indices = tuple(range(len(internal_edges), len(edges)))

    # Build cycle basis of G_exp
    cycles_list: list[tuple[InternalEdge, ...]] = []
    if cycle_basis_mode == "fundamental":
        for u, v in non_tree_edges:
            path = nx.shortest_path(T, u, v)
            cyc_edges = [ed(path[i], path[i + 1]) for i in range(len(path) - 1)] + [ed(u, v)]
            cycles_list.append(tuple(sorted(cyc_edges)))
    elif cycle_basis_mode == "nx":
        for cyc_nodes in nx.cycle_basis(G_exp, chosen_root):
            cyc_edges = [
                ed(cyc_nodes[i], cyc_nodes[(i + 1) % len(cyc_nodes)])
                for i in range(len(cyc_nodes))
            ]
            cycles_list.append(tuple(sorted(cyc_edges)))
    else:
        raise ValueError(f"Unknown cycle_basis_mode: {cycle_basis_mode!r}")

    cycles = tuple(cycles_list)

    # Map each edge to the cycles it belongs to (detector bitmask)
    det_masks_list = [0] * len(edges)
    for cyc_idx, cyc_edges in enumerate(cycles):
        bit = 1 << cyc_idx
        for e in cyc_edges:
            det_masks_list[edge_to_idx[e]] |= bit

    # Map each edge to the boundary qubits it propagates to (observable bitmask)
    # Orient tree T away from chosen_root so downward subtrees are well-defined.
    bfs_tree = nx.bfs_tree(T, chosen_root)
    descendants_cache: dict[int, set[int]] = {}

    def _get_subtree_nodes(node: int) -> set[int]:
        if node not in descendants_cache:
            descendants_cache[node] = {node} | nx.descendants(bfs_tree, node)
        return descendants_cache[node]

    unfolded_obs_list = [0] * len(edges)
    folded_obs_list = [0] * len(edges)

    for e in tree_edges:
        idx = edge_to_idx[e]
        u, v = e
        child = v if bfs_tree.has_edge(u, v) else u
        subtree_nodes = _get_subtree_nodes(child)
        raw_mask = 0
        for m in subtree_nodes:
            if m in mark_to_bit:
                raw_mask |= 1 << mark_to_bit[m]
        unfolded_obs_list[idx] = raw_mask
        # Fold modulo all-output stabilizer X^n if weight > n // 2
        if raw_mask.bit_count() > n // 2:
            folded_obs_list[idx] = all_obs_mask ^ raw_mask
        else:
            folded_obs_list[idx] = raw_mask

    # Non-tree edges are measurement/flag errors: they flip detectors only (obs_mask = 0).
    # Boundary edges flip their single boundary qubit and no detectors.
    for bit_idx, b_edge in enumerate(boundary_edges):
        idx = edge_to_idx[b_edge]
        mask = 1 << bit_idx
        if mask.bit_count() > n // 2 and n > 0:
            mask ^= all_obs_mask
        unfolded_obs_list[idx] = 1 << bit_idx
        folded_obs_list[idx] = mask

    return GraphDEM(
        G_exp=G_exp,
        F_exp=F_exp,
        spanning_tree=T,
        root=chosen_root,
        mark_nodes=mark_nodes,
        internal_edges=internal_edges,
        boundary_edges=boundary_edges,
        edges=edges,
        edge_to_idx=edge_to_idx,
        tree_edge_indices=tree_edge_indices,
        non_tree_edge_indices=non_tree_edge_indices,
        boundary_edge_indices=boundary_edge_indices,
        cycles=cycles,
        det_masks=tuple(det_masks_list),
        obs_masks=tuple(folded_obs_list),
        unfolded_obs_masks=tuple(unfolded_obs_list),
    )


@dataclass(frozen=True)
class InternalCut:
    """
    Represents a minimal undetectable internal fault cut in `G_exp` that propagates
    to `bnd_weight` boundary errors (after folding modulo the all-output stabilizer `X^n`)
    using `num_faults <= bnd_weight` internal edges.
    """

    edge_indices: tuple[int, ...]
    edges: tuple[InternalEdge, ...]
    bnd_mask: int
    bnd_bit_indices: tuple[int, ...]
    bnd_nodes: tuple[int, ...]
    complement_bnd_bit_indices: tuple[int, ...]
    num_faults: int
    bnd_weight: int


def _is_valid_minimal_cut(
    G_exp: nx.Graph,
    cut_edges: Sequence[InternalEdge],
    mark_set: set[int],
) -> bool:
    """
    Checks that an internal cut `cut_edges` in `G_exp`:
    1. Is a matching in `G_exp` (no vertex has >= 2 incident edges in `cut_edges`),
       which eliminates any 2-faults-on-one-spider reducible configurations
       (including 2 internal faults on an `is_mark` node producing a weight-1 fault).
    2. Has no connected component after removing `cut_edges` where the component's
       cut size exceeds its number of `is_mark` nodes (in particular, no 0-mark
       internal islands and no wasteful sub-cuts piggybacking on a k > t cut).
    """
    seen_vertices: set[int] = set()
    for u, v in cut_edges:
        if u in seen_vertices or v in seen_vertices:
            return False
        seen_vertices.add(u)
        seen_vertices.add(v)

    G_cut = G_exp.copy()
    G_cut.remove_edges_from(cut_edges)
    components = list(nx.connected_components(G_cut))
    if len(components) == 2:
        # For a 2-component cut, the caller already verified num_faults <= min(|M_A|, |M_B|).
        return True

    # If removing cut_edges splits G_exp into > 2 components, verify each component
    # has at least as many mark nodes as incident cut edges.
    for comp in components:
        num_marks_in_comp = len(comp & mark_set)
        if num_marks_in_comp == 0:
            return False
        comp_cut_size = sum(1 for u, v in cut_edges if (u in comp) ^ (v in comp))
        if comp_cut_size > num_marks_in_comp:
            return False

    return True


@dataclass
class PrecomputedFaultTable:
    """
    Precomputed structure of all undetectable fault combinations up to boundary error
    weight `L` for a given `(G_exp, F_exp)`.

    Supports both:
    - Fast polynomial substitution (`evaluate_polynomial`) for arbitrary edge-weight
      assignments `w(e)` (single vector or 2D batch of weight vectors).
    - Explicit enumeration (`get_fault_sets`) of fault edge tuples for any `k <= L`.
    """

    dem: GraphDEM
    L: int
    n: int
    internal_cuts: tuple[InternalCut, ...]
    cuts_by_bnd_weight: dict[int, tuple[InternalCut, ...]]

    def get_fault_sets(
        self,
        k: int,
        as_indices: bool = False,
    ) -> Iterator[tuple[GraphEdge, ...] | tuple[int, ...]]:
        """
        Yields all undetectable fault sets of size `<= k` that produce a boundary
        error of weight `k` (modulo the all-output cat-state stabilizer `X^n`).

        Args:
            k: Target boundary error weight (`1 <= k <= L`).
            as_indices: If True, yields tuples of integer edge indices into `dem.edges`.
                If False (default), yields tuples of edge objects `(u, v)` / `(m, 'boundary')`.
        """
        if k < 1 or k > self.L:
            raise ValueError(f"k must be between 1 and L={self.L}, got {k}")

        bnd_source = self.dem.boundary_edge_indices if as_indices else self.dem.boundary_edges

        # 1. Purely boundary faults: all combinations of k boundary edges
        yield from itertools.combinations(bnd_source, k)

        # 2. Internal cuts producing m_S <= k boundary errors + (k - m_S) disjoint boundary faults
        for m_s in range(1, k + 1):
            b_needed = k - m_s
            for cut in self.cuts_by_bnd_weight.get(m_s, ()):
                prefix = cut.edge_indices if as_indices else cut.edges
                if b_needed == 0:
                    yield prefix
                else:
                    avail_bnd = [bnd_source[i] for i in cut.complement_bnd_bit_indices]
                    for bnd_comb in itertools.combinations(avail_bnd, b_needed):
                        yield prefix + bnd_comb

    def count_fault_sets(self, k: int | None = None) -> int | dict[int, int]:
        """
        Returns the total number of valid fault sets producing boundary error weight `k`
        (or a dictionary `{1: count_1, ..., L: count_L}` if `k is None`).
        """
        if k is not None:
            if k < 1 or k > self.L:
                raise ValueError(f"k must be between 1 and L={self.L}, got {k}")
            total = math.comb(self.n, k)
            for m_s in range(1, k + 1):
                b_needed = k - m_s
                num_cuts = len(self.cuts_by_bnd_weight.get(m_s, ()))
                if num_cuts:
                    total += num_cuts * math.comb(self.n - m_s, b_needed)
            return total

        return {k_val: self.count_fault_sets(k_val) for k_val in range(1, self.L + 1)}

    def _normalize_weights_array(
        self,
        edge_weights: dict | Sequence[int | float] | np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Converts `edge_weights` (dict, 1D sequence/array, or 2D batch array) into a
        2D numpy array of shape `(batch_size, num_edges)` and a flag `is_1d`.
        """
        num_edges = self.dem.num_edges
        if isinstance(edge_weights, dict):
            arr = np.zeros(num_edges, dtype=np.int64)
            for idx, e in enumerate(self.dem.edges):
                if e in edge_weights:
                    val = edge_weights[e]
                elif e[1] != "boundary" and (e[1], e[0]) in edge_weights:
                    val = edge_weights[(e[1], e[0])]
                elif e[1] == "boundary" and e[0] in edge_weights:
                    val = edge_weights[e[0]]
                else:
                    raise KeyError(f"Missing weight for edge {e!r}")
                arr[idx] = val
            return arr[np.newaxis, :], True

        arr = np.asarray(edge_weights)
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int64, copy=False)
        else:
            arr = arr.astype(np.float64, copy=False)

        if arr.ndim == 1:
            if arr.shape[0] != num_edges:
                raise ValueError(f"Expected weight vector of length {num_edges}, got {arr.shape[0]}")
            return arr[np.newaxis, :], True
        if arr.ndim == 2:
            if arr.shape[1] != num_edges:
                raise ValueError(f"Expected weight matrix with {num_edges} columns, got {arr.shape[1]}")
            return arr, False
        raise ValueError(f"edge_weights must be a dict, 1D array, or 2D array; got {arr.ndim}D")

    def evaluate_polynomial(
        self,
        edge_weights: dict | Sequence[int | float] | np.ndarray,
        mode: Literal["boundary_weight", "fault_weight"] = "boundary_weight",
    ) -> np.ndarray:
        """
        Substitutes edge weights `w(e)` into the precomputed fault table to compute
        the polynomial coefficients `[c_1, c_2, ..., c_L]` for `c_1 p + c_2 p^2 + ... + c_L p^L`.

        Args:
            edge_weights: Mapping `{edge: weight}`, 1D array of shape `(num_edges,)`,
                or 2D batch array of shape `(batch_size, num_edges)`.
            mode:
                - `"boundary_weight"` (default): `c_k` is the total number of ways to
                  obtain a boundary error of weight `k` (from fault sets of size `<= k`).
                - `"fault_weight"`: `c_s` accumulates contributions by the number of
                  faulty edges `s = |F|` (`1 <= s <= L`). Note that for `L <= t`,
                  `|F| == k` for every valid fault set so both modes are identical.

        Returns:
            1D numpy array of shape `(L,)` if input is a dict or 1D array, or
            2D numpy array of shape `(batch_size, L)` if input is a 2D batch array.
            Entry `i` (or `[:, i]`) is the coefficient of `p^{i+1}`.
        """
        if mode not in ("boundary_weight", "fault_weight"):
            raise ValueError(f"Unknown mode: {mode!r}")

        w_batch, is_1d = self._normalize_weights_array(edge_weights)
        batch_size = w_batch.shape[0]
        coeffs = np.zeros((batch_size, self.L), dtype=w_batch.dtype)

        bnd_indices = list(self.dem.boundary_edge_indices)
        w_bnd_all = w_batch[:, bnd_indices]

        # 1. Purely boundary faults: e_1 .. e_L over all n boundary edge weights
        esp_all = _elementary_symmetric_poly_2d(w_bnd_all, self.L)
        coeffs[:, :] += esp_all[:, 1 : self.L + 1]

        # 2. Internal cuts paired with elementary symmetric polynomials over remaining boundary edges
        for cut in self.internal_cuts:
            m_s = cut.bnd_weight
            max_b = self.L - m_s
            if max_b < 0:
                continue

            internal_prod = np.prod(w_batch[:, cut.edge_indices], axis=1)
            if np.all(internal_prod == 0):
                continue

            if max_b == 0:
                target_deg = m_s if mode == "boundary_weight" else cut.num_faults
                if 1 <= target_deg <= self.L:
                    coeffs[:, target_deg - 1] += internal_prod
            else:
                avail_cols = [bnd_indices[i] for i in cut.complement_bnd_bit_indices]
                w_avail = w_batch[:, avail_cols]
                esp_avail = _elementary_symmetric_poly_2d(w_avail, max_b)
                for b in range(max_b + 1):
                    target_deg = (m_s + b) if mode == "boundary_weight" else (cut.num_faults + b)
                    if 1 <= target_deg <= self.L:
                        coeffs[:, target_deg - 1] += internal_prod * esp_avail[:, b]

        return coeffs[0] if is_1d else coeffs

    def format_polynomial(
        self,
        edge_weights: dict | Sequence[int | float] | np.ndarray,
        var: str = "p",
        mode: Literal["boundary_weight", "fault_weight"] = "boundary_weight",
    ) -> str:
        """
        Evaluates and formats the fault polynomial as a string like `'32p + 43p^2 + 120p^3'`.
        """
        coeffs = self.evaluate_polynomial(edge_weights, mode=mode)
        if coeffs.ndim != 1:
            raise ValueError("format_polynomial expects a single weight vector, not a 2D batch.")
        terms = []
        for idx, c in enumerate(coeffs, start=1):
            if c == 0:
                continue
            exp_str = f"{var}" if idx == 1 else f"{var}^{idx}"
            terms.append(f"{c}{exp_str}")
        return " + ".join(terms) if terms else "0"


def compute_boundary_fault_table(
    G_exp: nx.Graph,
    F_exp: nx.Graph,
    L: int,
    root: int | dict[int, int] | None = None,
) -> PrecomputedFaultTable:
    """
    Builds the `GraphDEM` for `(G_exp, F_exp)` and precomputes all undetectable fault
    structures producing boundary errors of weight `k = 1 .. L` (modulo the cat-state
    all-output stabilizer `X^n`).

    Args:
        G_exp: Expanded graph with `is_mark=True` on boundary-connected nodes.
        F_exp: Expanded spanning forest/tree of `G_exp`.
        L: Maximum boundary error weight `k` to compute (`1 <= L <= n // 2`).
        root: Optional root node or `{tree_id: root_node}` dict.

    Returns:
        `PrecomputedFaultTable` ready for inspecting fault sets and substituting edge weights.
    """
    # Use the fundamental cycle basis of F_exp so each non-tree edge flips a single unique detector.
    dem = build_graph_dem(G_exp, F_exp, root=root, cycle_basis_mode="fundamental")
    n = dem.num_qubits
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    effective_L = min(L, n // 2) if n > 0 else 0
    all_obs_mask = (1 << n) - 1
    mark_set = set(dem.mark_nodes)

    tree_indices = dem.tree_edge_indices
    non_tree_indices = dem.non_tree_edge_indices
    det_masks = dem.det_masks
    unfolded_obs = dem.unfolded_obs_masks

    seen_cut_edge_sets: set[tuple[int, ...]] = set()
    internal_cuts: list[InternalCut] = []

    # Any undetectable internal fault of total size <= effective_L has at most effective_L tree edges.
    max_tree_faults = min(effective_L, len(tree_indices))

    for r in range(1, max_tree_faults + 1):
        for tree_comb in itertools.combinations(tree_indices, r):
            d_mask = 0
            o_mask = 0
            for t_idx in tree_comb:
                d_mask ^= det_masks[t_idx]
                o_mask ^= unfolded_obs[t_idx]

            num_flags = d_mask.bit_count()
            total_faults = r + num_flags
            if total_faults > effective_L:
                continue

            # Fold modulo the all-output stabilizer X^n so boundary weight is <= n // 2
            pop = o_mask.bit_count()
            if pop > n // 2:
                folded_o_mask = all_obs_mask ^ o_mask
                b_weight = n - pop
            else:
                folded_o_mask = o_mask
                b_weight = pop

            # Must produce a non-empty boundary error of weight <= effective_L
            # and satisfy total_faults <= b_weight
            if b_weight == 0 or b_weight > effective_L or total_faults > b_weight:
                continue

            flag_comb = tuple(
                non_tree_indices[bit_i]
                for bit_i in range(len(non_tree_indices))
                if (d_mask >> bit_i) & 1
            )
            full_edge_indices = tuple(sorted(tree_comb + flag_comb))
            if full_edge_indices in seen_cut_edge_sets:
                continue

            cut_edges = tuple(dem.edges[idx] for idx in full_edge_indices)
            if not _is_valid_minimal_cut(G_exp, cut_edges, mark_set):
                continue

            seen_cut_edge_sets.add(full_edge_indices)
            bnd_bits = tuple(i for i in range(n) if (folded_o_mask >> i) & 1)
            comp_bnd_bits = tuple(i for i in range(n) if not ((folded_o_mask >> i) & 1))
            bnd_nodes = tuple(dem.mark_nodes[i] for i in bnd_bits)

            internal_cuts.append(
                InternalCut(
                    edge_indices=full_edge_indices,
                    edges=cut_edges,
                    bnd_mask=folded_o_mask,
                    bnd_bit_indices=bnd_bits,
                    bnd_nodes=bnd_nodes,
                    complement_bnd_bit_indices=comp_bnd_bits,
                    num_faults=total_faults,
                    bnd_weight=b_weight,
                )
            )

    internal_cuts.sort(key=lambda c: (c.bnd_weight, c.num_faults, c.edge_indices))
    cuts_by_weight: dict[int, tuple[InternalCut, ...]] = {}
    for w in range(1, effective_L + 1):
        cuts_by_weight[w] = tuple(c for c in internal_cuts if c.bnd_weight == w)

    return PrecomputedFaultTable(
        dem=dem,
        L=effective_L,
        n=n,
        internal_cuts=tuple(internal_cuts),
        cuts_by_bnd_weight=cuts_by_weight,
    )

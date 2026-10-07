"""
Marked graphs that are fault tolerant at every fault weight (t = infinity), built by expander
composition (SpiderCat, Theorem C.6) and certified.

Notation
--------
* Starting graph ``H``: a (multi)graph on ``N`` vertices.  Vertex ``v`` receives ``k_v`` outputs.
* Gadget: a cubic marked graph whose marks are *terminals*.  It is robust at every weight w.r.t.
  its terminals.
* Final marked graph ``F``: every vertex ``v`` of ``H`` is replaced by a gadget with
  ``deg_H(v) + k_v`` terminals.  ``deg_H(v)`` terminals become *ports* (degree-3 vertices joined
  along the ``H``-edges), ``k_v`` stay marks (the CAT outputs).

Robustness model (SpiderCat Definition 5.3, with the cut size as the bound)
--------------------------------------------------------------------------
Let ``F'`` be ``F`` with every mark turned into a degree-2 vertex on its edge.  ``F`` is robust at
every weight iff for every ``S`` subset of ``V(F')``::

    min(m(S), n - m(S)) <= |delta_{F'}(S)|

where ``m(S)`` counts mark-vertices in ``S`` and ``n`` is the total number of marks.

Composition lemma
-----------------
If ``|delta_H(R)| >= min(k(R), k(V_H - R))`` for every ``R`` and every gadget is robust w.r.t. its
terminals (at least ``deg_H(v) + k_v`` of them; unused terminals are dropped, which keeps a gadget
robust), then ``F`` is robust at every weight.  For uniform ``k`` the hypothesis on ``H`` follows
from ``mu_2(L_H) = d - lambda_2(A_H) >= 2k``, i.e. ``lambda_2 <= 6`` for ``d = 10, k = 2``: only the
one-sided Ramanujan bound is needed.

Routes
------
* ``ramanujan_tinf_marked_graph``: random 10-regular ``H`` certified by ``lambda2_certificate`` (a
  rigorous floating-point Cholesky of ``6 I - A + J``), 2 outputs per vertex, the 12-terminal
  Heawood gadget; 24 internal vertices per starting-graph vertex (ratio 12), any n >= 22.
* ``dense_tinf_marked_graph``: ``K_N`` or a small random regular ``H`` whose expansion is computed
  exactly, with larger exhaustively certified gadgets (``load_gadget_library``); ratio ~4-8 for
  n <= 100.

Certification and circuits
--------------------------
``verify_composition_certificate`` re-checks a marked graph from the graph, its marks and the
cluster partition only.  ``max_margin_milp`` proves robustness end to end for small graphs.
``spider_ordering_tree_milp`` + ``extract_circuit_sparse`` extract a CAT circuit with V + n + 1
CNOTs (the repository extractor assumes every non-tree edge is marked, i.e. vertex ratio <= 2), and
``circuit_fault_check`` enumerates circuit faults of weight <= 2.

Command line: ``python -m spidercat.graphs_ramanujan N --route dense|ramanujan --out file.json``.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from spidercat.utils import ed, load_solution_triplet

UNIT_ROUNDOFF = 2.0 ** -53


# ----------------------------------------------------------------------------------------------
# Marked graph helpers
# ----------------------------------------------------------------------------------------------

def normalise_marks(G: nx.Graph, M: dict) -> dict[tuple, int]:
    """Return marks as ``{ed(u, v): count}`` over all edges of ``G``."""
    out = {ed(u, v): 0 for u, v in G.edges()}
    for (u, v), c in M.items():
        key = ed(u, v)
        if key not in out:
            raise ValueError(f"Marked edge {key} is not an edge of G")
        out[key] += int(c)
    return out


def subdivide_marks(G: nx.Graph, M: dict) -> tuple[nx.Graph, list]:
    """Build ``G'``: each mark on edge (u, v) becomes a degree-2 vertex ``('m', u, v, i)``."""
    marks = normalise_marks(G, M)
    Gp = nx.Graph()
    Gp.add_nodes_from(G.nodes())
    mark_nodes = []
    for (u, v), c in marks.items():
        chain = [u] + [("m", u, v, i) for i in range(c)] + [v]
        mark_nodes.extend(chain[1:-1])
        nx.add_path(Gp, chain)
    return Gp, mark_nodes


def is_cubic_simple(G: nx.Graph) -> bool:
    return (not isinstance(G, nx.MultiGraph)) and nx.number_of_selfloops(G) == 0 \
        and all(d == 3 for _, d in G.degree())


# ----------------------------------------------------------------------------------------------
# Exact robustness checks (exhaustive; small graphs only)
# ----------------------------------------------------------------------------------------------

def max_margin_exhaustive(G: nx.Graph, M: dict, chunk_bits: int = 20) -> tuple[int, set | None]:
    """
    Exact maximum of ``min(m(S), n - m(S)) - |delta(S)|`` over all cuts, enumerating vertex
    subsets of ``G`` (not ``G'``).

    For a vertex set ``S`` with ``a`` marks on edges inside ``S``, ``b`` inside the complement,
    ``s`` on cut edges and ``c`` cut edges, the adversary places each cut point anywhere along a cut
    edge, so it can split the ``s`` marks arbitrarily and reach ``min(a + s, b + s, n // 2)``.
    Pulling marks out of an uncut edge costs two cut points per edge and gains at most two marks,
    so it never helps when every edge carries at most two marks (asserted).

    Robust at every weight iff the returned margin is ``<= 0``.
    Returns (max margin, a maximising S or None if the maximum is attained only by trivial cuts).
    """
    marks = normalise_marks(G, M)
    if marks and max(marks.values()) > 2:
        raise ValueError("Use max_margin_exhaustive_subdivided for edges with more than 2 marks")
    nodes = list(G.nodes())
    V = len(nodes)
    if V > 34:
        raise ValueError("Too many vertices for exhaustive enumeration")
    idx = {v: i for i, v in enumerate(nodes)}
    n = sum(marks.values())
    half = n // 2
    edges = [(idx[u], idx[v], m) for (u, v), m in marks.items()]

    # The margin is invariant under S <-> complement, so fix the last vertex outside S.
    free = V - 1
    total = 1 << free
    chunk = 1 << min(chunk_bits, free)
    best, best_mask = 0, None
    for start in range(0, total, chunk):
        masks = np.arange(start, min(start + chunk, total), dtype=np.int64)
        bits = [((masks >> i) & 1).astype(np.int16) for i in range(free)]
        zero = np.zeros_like(masks, dtype=np.int16)
        bits.append(zero)
        cut = np.zeros_like(zero)
        a = np.zeros_like(zero)
        s = np.zeros_like(zero)
        for u, v, m in edges:
            xu, xv = bits[u], bits[v]
            x = xu ^ xv
            cut += x
            if m:
                s += m * x
                a += m * (xu & xv)
        b = n - a - s
        margin = np.minimum(np.minimum(a + s, b + s), half) - cut
        i = int(np.argmax(margin))
        if margin[i] > best:
            best, best_mask = int(margin[i]), int(masks[i])
    if best_mask is None:
        return best, None
    return best, {nodes[i] for i in range(free) if (best_mask >> i) & 1}


def max_margin_exhaustive_subdivided(G: nx.Graph, M: dict, chunk_bits: int = 20) -> tuple[int, set | None]:
    """
    Exact maximum of ``min(m(S), n - m(S)) - |delta_{G'}(S)|`` enumerating ALL subsets of ``V(G')``
    (marks included as vertices).  No modelling shortcut; use it to cross-check.
    """
    Gp, mark_nodes = subdivide_marks(G, M)
    nodes = list(Gp.nodes())
    V = len(nodes)
    if V > 32:
        raise ValueError("Too many vertices for exhaustive enumeration")
    idx = {v: i for i, v in enumerate(nodes)}
    is_mark = np.zeros(V, dtype=np.int16)
    for m in mark_nodes:
        is_mark[idx[m]] = 1
    n = len(mark_nodes)
    edges = [(idx[u], idx[v]) for u, v in Gp.edges()]
    free = V - 1
    total = 1 << free
    chunk = 1 << min(chunk_bits, free)
    best, best_mask = 0, None
    for start in range(0, total, chunk):
        masks = np.arange(start, min(start + chunk, total), dtype=np.int64)
        bits = [((masks >> i) & 1).astype(np.int16) for i in range(free)]
        bits.append(np.zeros_like(masks, dtype=np.int16))
        cut = np.zeros_like(bits[0])
        ms = np.zeros_like(bits[0])
        for i in range(V):
            if is_mark[i]:
                ms += bits[i]
        for u, v in edges:
            cut += bits[u] ^ bits[v]
        margin = np.minimum(ms, n - ms) - cut
        i = int(np.argmax(margin))
        if margin[i] > best:
            best, best_mask = int(margin[i]), int(masks[i])
    if best_mask is None:
        return best, None
    return best, {nodes[i] for i in range(free) if (best_mask >> i) & 1}


def cut_margin(G: nx.Graph, M: dict, S: set) -> tuple[int, int, int, int]:
    """
    Recompute, from scratch, the best adversarial margin for the vertex set ``S`` of ``G``.
    Returns (margin, cut size, marks forced into S, marks forced out of S, ...) as
    (margin, c, a + s, b + s) where the adversary's achievable side counts are bounded by a + s
    and b + s.
    """
    marks = normalise_marks(G, M)
    n = sum(marks.values())
    a = b = s = c = 0
    for (u, v), m in marks.items():
        iu, iv = u in S, v in S
        if iu and iv:
            a += m
        elif not iu and not iv:
            b += m
        else:
            c += 1
            s += m
    return min(a + s, b + s, n // 2) - c, c, a + s, b + s


# ----------------------------------------------------------------------------------------------
# Exact robustness check by mixed-integer programming (medium graphs)
# ----------------------------------------------------------------------------------------------

def max_margin_milp(G: nx.Graph, M: dict, time_limit: float = 600.0, verbose: bool = False,
                    max_cut: int | None = None) -> dict:
    """
    Maximise ``m(S) - |delta_{G'}(S)|`` subject to ``m(S) <= n // 2`` over ``S`` subset of ``V(G')``
    with HiGHS.  By symmetry (S <-> complement) this equals the maximum robustness margin.
    Robust at every weight iff the optimum is ``<= 0``.  With ``max_cut = t`` only cuts of weight at
    most t are considered, which decides SpiderCat's t-robustness.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    Gp, mark_nodes = subdivide_marks(G, M)
    nodes = list(Gp.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    V, E = len(nodes), Gp.number_of_edges()
    n = len(mark_nodes)
    nvar = V + E
    c = np.zeros(nvar)
    for m in mark_nodes:
        c[idx[m]] = -1.0          # maximise marks in S
    c[V:] = 1.0                   # minimise cut edges
    extra = 1 if max_cut is not None else 0
    A = lil_matrix((2 * E + 1 + extra, nvar))
    lb = np.zeros(2 * E + 1 + extra)
    ub = np.full(2 * E + 1 + extra, np.inf)
    if max_cut is not None:
        for j in range(E):
            A[2 * E + 1, V + j] = 1
        lb[2 * E + 1], ub[2 * E + 1] = 0, max_cut
    for j, (u, v) in enumerate(Gp.edges()):
        iu, iv = idx[u], idx[v]
        A[2 * j, V + j], A[2 * j, iu], A[2 * j, iv] = 1, -1, 1
        A[2 * j + 1, V + j], A[2 * j + 1, iu], A[2 * j + 1, iv] = 1, 1, -1
    for m in mark_nodes:
        A[2 * E, idx[m]] = 1
    lb[2 * E], ub[2 * E] = 0, n // 2
    integrality = np.concatenate([np.ones(V), np.zeros(E)])
    t0 = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=integrality,
               bounds=Bounds(0, 1), options={"time_limit": time_limit, "disp": verbose, "mip_rel_gap": 0.0})
    elapsed = time.perf_counter() - t0
    out = {"status": res.status, "message": res.message, "time": elapsed, "vertices_Gprime": V}
    if res.x is not None:
        out["best_margin_found"] = int(round(-res.fun))
        S = {nodes[i] for i in range(V) if res.x[i] > 0.5}
        out["witness"] = S
    dual = getattr(res, "mip_dual_bound", None)
    if dual is not None:
        out["upper_bound"] = -dual
    out["proved_robust"] = bool(res.status == 0 and int(round(-res.fun)) <= 0)
    return out


# ----------------------------------------------------------------------------------------------
# Heuristic search for violating cuts (sanity check on large graphs; not a certificate)
# ----------------------------------------------------------------------------------------------

def search_violations_heuristic(G: nx.Graph, M: dict, restarts: int = 20, seed: int = 0,
                                n_vectors: int = 6) -> dict:
    """
    Look for cuts with positive margin using spectral sweep cuts on ``G'`` followed by greedy
    single-vertex moves.  Returns the best margin found (``<= 0`` means nothing violating was found).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    rng = np.random.default_rng(seed)
    Gp, mark_nodes = subdivide_marks(G, M)
    nodes = list(Gp.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    V = len(nodes)
    n = len(mark_nodes)
    is_mark = np.zeros(V, dtype=np.int64)
    for m in mark_nodes:
        is_mark[idx[m]] = 1
    nbrs = [[idx[w] for w in Gp.neighbors(v)] for v in nodes]

    def margin_of(x):
        ms = int(is_mark[x].sum())
        cut = sum(1 for u in range(V) for w in nbrs[u] if u < w and x[u] != x[w])
        return min(ms, n - ms) - cut

    def improve(x):
        # greedy single-vertex flips evaluated incrementally
        ms = int(is_mark[x].sum())
        cut = sum(1 for u in range(V) for w in nbrs[u] if u < w and x[u] != x[w])
        cur = min(ms, n - ms) - cut
        improved = True
        while improved:
            improved = False
            for u in rng.permutation(V):
                same = sum(1 for w in nbrs[u] if x[w] == x[u])
                d_cut = same - (len(nbrs[u]) - same)
                d_ms = is_mark[u] * (-1 if x[u] else 1)
                new = min(ms + d_ms, n - ms - d_ms) - (cut + d_cut)
                if new > cur:
                    x[u] = not x[u]
                    ms += d_ms
                    cut += d_cut
                    cur = new
                    improved = True
        return cur, x

    best, best_S = -math.inf, None
    L = nx.laplacian_matrix(Gp, nodelist=nodes).astype(float)
    k = min(n_vectors + 1, V - 2)
    try:
        _, vecs = spla.eigsh(L + sp.identity(V) * 1e-9, k=k, sigma=0, which="LM")
    except Exception:
        _, vecs = np.linalg.eigh(L.toarray())
        vecs = vecs[:, :k]
    candidates = []
    for j in range(1, vecs.shape[1]):
        order = np.argsort(vecs[:, j])
        x = np.zeros(V, dtype=bool)
        # sweep: keep the best prefix
        ms = cut = 0
        inS = np.zeros(V, dtype=bool)
        sweep_best, sweep_t = -math.inf, 0
        for t, u in enumerate(order[:-1]):
            inS[u] = True
            ms += is_mark[u]
            for w in nbrs[u]:
                cut += -1 if inS[w] else 1
            val = min(ms, n - ms) - cut
            if val > sweep_best:
                sweep_best, sweep_t = val, t
        x[order[:sweep_t + 1]] = True
        candidates.append(x)
    for _ in range(restarts):
        # random BFS ball in G'
        start = int(rng.integers(V))
        size = int(rng.integers(1, max(2, V // 2)))
        ball = [start]
        seen = {start}
        q = [start]
        while q and len(ball) < size:
            u = q.pop(0)
            for w in rng.permutation(nbrs[u]):
                if w not in seen:
                    seen.add(w)
                    ball.append(w)
                    q.append(w)
        x = np.zeros(V, dtype=bool)
        x[ball] = True
        candidates.append(x)
    for x in candidates:
        val, x = improve(x.copy())
        if val > best:
            best, best_S = val, x.copy()
    assert best_S is None or margin_of(best_S) == best
    return {"best_margin": int(best), "candidates": len(candidates),
            "witness": {nodes[i] for i in range(V) if best_S[i]} if best > 0 else None}


# ----------------------------------------------------------------------------------------------
# Rigorous spectral certificates
# ----------------------------------------------------------------------------------------------

def certify_positive_definite(M: np.ndarray, eps: float | None = None) -> dict:
    """
    Floating-point certificate that the symmetric matrix ``M`` (entries exactly representable in
    binary64) is positive definite.

    Cholesky of ``M - eps I`` computes ``L`` with ``L L^T = M - eps I + dM`` and
    ``|dM| <= gamma_{N+1} |L||L^T|`` (Higham, Accuracy and Stability, Thm 10.3).  Hence
    ``lambda_min(M) >= eps - gamma_{N+1} ||L||_F^2``.  ``eps`` is a power of two so ``M - eps I`` is
    formed exactly for integer matrices of moderate size.
    """
    N = M.shape[0]
    gamma = (N + 1) * UNIT_ROUNDOFF / (1 - (N + 1) * UNIT_ROUNDOFF)
    if eps is None:
        eps = 2.0 ** -10
    t0 = time.perf_counter()
    try:
        L = np.linalg.cholesky(M - eps * np.eye(N))
    except np.linalg.LinAlgError:
        return {"certified": False, "reason": "cholesky failed", "eps": eps,
                "time": time.perf_counter() - t0}
    frob2 = float(np.einsum("ij,ij->", L, L))
    bound = gamma * frob2 * 1.01
    return {"certified": bool(bound < eps), "eps": eps, "error_bound": bound,
            "time": time.perf_counter() - t0}


def is_positive_definite_exact(M) -> bool:
    """Exact Sylvester test with fraction-free (Bareiss) elimination on an integer matrix."""
    A = np.array(M, dtype=object)
    n = A.shape[0]
    prev = 1
    for k in range(n):
        piv = A[k, k]
        if piv <= 0:
            return False
        if k + 1 < n:
            A[k + 1:, k + 1:] = (A[k + 1:, k + 1:] * piv - np.outer(A[k + 1:, k], A[k, k + 1:])) // prev
        prev = piv
    return True


def adjacency_int(H: nx.Graph, nodelist=None) -> np.ndarray:
    """Integer adjacency matrix (parallel edges counted, loops forbidden)."""
    if nx.number_of_selfloops(H):
        raise ValueError("Loops are not allowed in the starting graph")
    nodelist = list(H.nodes()) if nodelist is None else nodelist
    return nx.to_numpy_array(H, nodelist=nodelist, dtype=np.int64, weight=None,
                             multigraph_weight=sum)


def lambda2_certificate(H: nx.Graph, bound: int, exact_limit: int = 0, numeric: bool = True) -> dict:
    """
    Certify ``lambda_i(A_H) < bound`` for every eigenvalue on the complement of the all-ones vector
    (``H`` regular).  Equivalent to positive definiteness of ``bound I - A + J`` when
    ``N > d - bound``.  Also reports numerical lambda_2 and lambda_min.
    """
    A = adjacency_int(H)
    N = A.shape[0]
    degs = A.sum(axis=1)
    if not np.all(degs == degs[0]):
        raise ValueError("lambda2_certificate expects a regular graph")
    d = int(degs[0])
    out = {"N": N, "d": d, "bound": bound}
    if numeric:
        t0 = time.perf_counter()
        ev = np.linalg.eigvalsh(A.astype(float))
        out["lambda_2"] = float(ev[-2])
        out["lambda_min"] = float(ev[0])
        out["eig_time"] = time.perf_counter() - t0
    if bound - d + N <= 0:
        out["certified"] = False
        out["reason"] = "N too small for the J shift"
        return out
    Mmat = (bound * np.eye(N, dtype=np.int64) - A + 1).astype(float)
    eps = 2.0 ** -10
    if numeric:
        gap = bound - out["lambda_2"]
        while eps > gap / 2 and eps > 2.0 ** -40:
            eps /= 2
    t0 = time.perf_counter()
    cert = certify_positive_definite(Mmat, eps)
    while not numeric and not cert["certified"] and eps > 2.0 ** -30:
        eps /= 16
        cert = certify_positive_definite(Mmat, eps)
    out.update({"certified": cert["certified"], "eps": cert["eps"],
                "error_bound": cert.get("error_bound"), "cert_time": time.perf_counter() - t0})
    if N <= exact_limit:
        t0 = time.perf_counter()
        out["certified_exact"] = is_positive_definite_exact(bound * np.eye(N, dtype=np.int64) - A + 1)
        out["exact_time"] = time.perf_counter() - t0
    return out


def two_sided_ramanujan_certificate(H: nx.Graph) -> dict:
    """
    OpenAI-style final test: every nonconstant eigenvalue lies strictly inside (-2 sqrt(d-1),
    2 sqrt(d-1)) iff ``4(d-1) I - A^2 + c J`` is positive definite for a large enough integer c.
    """
    A = adjacency_int(H)
    N = A.shape[0]
    degs = A.sum(axis=1)
    if not np.all(degs == degs[0]):
        return {"certified": False, "reason": "not regular"}
    d = int(degs[0])
    q = d - 1
    c = (d * d - 4 * q) // N + 1
    Af = A.astype(float)          # A @ A is exact in binary64: entries are small non-negative integers
    M2 = 4 * q * np.eye(N) - Af @ Af + c
    t0 = time.perf_counter()
    cert = certify_positive_definite(M2, 2.0 ** -12)
    cert["time"] = time.perf_counter() - t0
    return cert


def edge_expansion_exact(H: nx.Graph, k: dict) -> tuple[int, set | None]:
    """
    Exact maximum over R of ``min(k(R), k(V - R)) - |delta_H(R)|`` by enumeration (small H only).
    The H-expansion hypothesis of the composition lemma holds iff the result is ``<= 0``.
    """
    nodes = list(H.nodes())
    N = len(nodes)
    if N > 30:
        raise ValueError("Too many vertices for exhaustive enumeration")
    idx = {v: i for i, v in enumerate(nodes)}
    kv = np.array([k[v] for v in nodes], dtype=np.int64)
    K = int(kv.sum())
    edges = [(idx[u], idx[v]) for u, v in H.edges()]
    free = N - 1
    masks = np.arange(1 << free, dtype=np.int64)
    bits = [((masks >> i) & 1).astype(np.int64) for i in range(free)] + [np.zeros_like(masks)]
    kin = sum(bits[i] * kv[i] for i in range(N))
    cut = sum(bits[u] ^ bits[v] for u, v in edges)
    margin = np.minimum(kin, K - kin) - cut
    i = int(np.argmax(margin))
    if margin[i] <= 0:
        return int(margin[i]), None
    return int(margin[i]), {nodes[j] for j in range(free) if (masks[i] >> j) & 1}


# ----------------------------------------------------------------------------------------------
# Starting graphs
# ----------------------------------------------------------------------------------------------

def random_regular_graph(d: int, N: int, seed: int) -> nx.Graph:
    """Uniform-ish random simple d-regular graph (complement trick for dense cases)."""
    if d >= N:
        raise ValueError(f"No simple {d}-regular graph on {N} vertices")
    if (d * N) % 2:
        raise ValueError("d * N must be even")
    if d > (N - 1) // 2:
        Hc = nx.random_regular_graph(N - 1 - d, N, seed=seed) if N - 1 - d > 0 else nx.empty_graph(N)
        return nx.complement(Hc)
    return nx.random_regular_graph(d, N, seed=seed)


def certified_starting_graph(N: int, d: int, k: int, seed: int = 0, max_tries: int = 200,
                             exact_limit: int = 0) -> tuple[nx.Graph, dict]:
    """
    Sample random simple d-regular graphs on N vertices until one is certified to satisfy
    ``lambda_2(A_H) < d - 2k`` (so ``mu_2 > 2k`` and the H-expansion hypothesis holds with ``k``
    outputs per vertex).  This replaces the OpenAI construction, which only applies to
    astronomically large N.
    """
    bound = d - 2 * k
    t0 = time.perf_counter()
    info = {"tries": 0}
    for attempt in range(max_tries):
        H = random_regular_graph(d, N, seed=seed + 7919 * attempt)
        info["tries"] = attempt + 1
        if not nx.is_connected(H):
            continue
        cert = lambda2_certificate(H, bound, exact_limit=exact_limit, numeric=False)
        if cert["certified"]:
            info.update(cert)
            info["gen_time"] = time.perf_counter() - t0
            return H, info
    raise RuntimeError(f"No certified starting graph found in {max_tries} tries")


# ----------------------------------------------------------------------------------------------
# Gadgets
# ----------------------------------------------------------------------------------------------

def load_gadget(T: int) -> tuple[nx.Graph, dict] | None:
    """
    Load a SpiderCat graph with ``T`` marks that is robust at every weight (``t = floor(T/2) - 1``
    suffices because a side never has more than floor(T/2) marks).  Relabelled to 0..V-1.
    """
    t_needed = T // 2 - 1
    for t in range(t_needed, 8):
        sol = load_solution_triplet(T, t, 1)
        if sol is None:
            continue
        G, _, M, _ = sol
        G = nx.convert_node_labels_to_integers(G, ordering="sorted", label_attribute="orig")
        back = {data["orig"]: v for v, data in G.nodes(data=True)}
        Mn = {ed(back[u], back[v]): c for (u, v), c in M.items() if c}
        return nx.Graph(G.edges()), Mn
    return None


def certify_gadget(G: nx.Graph, M: dict, cross_check: bool = False) -> dict:
    """Exact certificate that a gadget is robust at every weight w.r.t. its terminals (marks)."""
    out = {"vertices": G.number_of_nodes(), "terminals": sum(normalise_marks(G, M).values()),
           "cubic": is_cubic_simple(G), "connected": nx.is_connected(G)}
    t0 = time.perf_counter()
    margin, S = max_margin_exhaustive(G, M)
    out["max_margin"], out["witness"], out["time"] = margin, S, time.perf_counter() - t0
    if cross_check:
        t0 = time.perf_counter()
        out["max_margin_subdivided"], _ = max_margin_exhaustive_subdivided(G, M)
        out["time_subdivided"] = time.perf_counter() - t0
    out["certified"] = out["cubic"] and out["connected"] and margin <= 0 and \
        (not cross_check or out["max_margin_subdivided"] <= 0)
    return out


# ----------------------------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------------------------

def compose(H: nx.Graph, k: dict, gadget) -> tuple[nx.Graph, dict, dict]:
    """
    Replace every vertex ``v`` of the starting graph ``H`` (a loopless graph or multigraph) by a
    copy of its gadget whose terminals are used as ``deg_H(v)`` ports and ``k[v]`` outputs; unused
    terminals are dropped (dropping terminals keeps a gadget robust).  ``gadget`` is either one
    ``(G, marks)`` pair for all vertices or a dict ``v -> (G, marks)``.

    Returns (F, marks, role) where ``role[x] = (v, 'internal' | 'port')``.
    """
    gadget_of = gadget if isinstance(gadget, dict) else {v: gadget for v in H.nodes()}
    F = nx.Graph()
    marks: dict[tuple, int] = {}
    role: dict[int, tuple] = {}
    port_of: dict[tuple, list] = defaultdict(list)   # H-vertex -> list of port nodes (in order)
    nxt = 0
    H_edges = list(H.edges(keys=True)) if H.is_multigraph() else [(u, v, 0) for u, v in H.edges()]
    deg = Counter()
    for u, v, _ in H_edges:
        if u == v:
            raise ValueError("Loops are not allowed in the starting graph")
        deg[u] += 1
        deg[v] += 1
    for v in H.nodes():
        Gg, Mg = gadget_of[v]
        Mg = normalise_marks(Gg, Mg)
        slots = [(e, i) for e, c in sorted(Mg.items()) for i in range(c)]
        T = len(slots)
        need = deg[v] + k[v]
        if need > T:
            raise ValueError(f"Gadget has {T} terminals but vertex {v} needs {need}")
        kinds = ["port"] * deg[v] + ["out"] * k[v] + ["drop"] * (T - need)
        kind_of = dict(zip(slots, kinds))
        local = {}
        for x in Gg.nodes():
            local[x] = nxt
            role[nxt] = (v, "internal")
            nxt += 1
        for (a, c), cnt in Mg.items():
            chain = [local[a]]
            pending = 0          # output marks on the current segment
            for i in range(cnt):
                kind = kind_of[((a, c), i)]
                if kind == "port":
                    p = nxt
                    nxt += 1
                    role[p] = (v, "port")
                    port_of[v].append(p)
                    F.add_edge(chain[-1], p)
                    if pending:
                        marks[ed(chain[-1], p)] = pending
                    chain.append(p)
                    pending = 0
                elif kind == "out":
                    pending += 1
            F.add_edge(chain[-1], local[c])
            if pending:
                marks[ed(chain[-1], local[c])] = marks.get(ed(chain[-1], local[c]), 0) + pending
        for a, c in Gg.edges():
            if Mg[ed(a, c)] == 0:
                F.add_edge(local[a], local[c])
    used = Counter()
    for u, v, _ in H_edges:
        pu, pv = port_of[u][used[u]], port_of[v][used[v]]
        used[u] += 1
        used[v] += 1
        F.add_edge(pu, pv)
    marks = {ed(u, v): marks.get(ed(u, v), 0) for u, v in F.edges()}
    return F, marks, role


# ----------------------------------------------------------------------------------------------
# Independent certificate verification
# ----------------------------------------------------------------------------------------------

def _cluster_gadget(F: nx.Graph, marks: dict, cluster: set) -> tuple[nx.Graph, dict]:
    """
    Rebuild the gadget seen by one cluster: internal vertices are cluster vertices with all three
    neighbours inside; boundary vertices (one neighbour outside) are suppressed into terminal marks
    on the edge they subdivide, together with the output marks along that path.
    """
    inside = {x for x in cluster if all(w in cluster for w in F.neighbors(x))}
    boundary = cluster - inside
    if not inside:
        raise ValueError("Cluster has no internal vertex")
    for x in boundary:
        outside = [w for w in F.neighbors(x) if w not in cluster]
        if len(outside) != 1:
            raise ValueError(f"Boundary vertex {x} has {len(outside)} outside neighbours")
    G = nx.MultiGraph()
    G.add_nodes_from(inside)
    seen_paths = set()
    visited_boundary = set()
    for a in inside:
        for w in F.neighbors(a):
            if w in inside:
                if a < w:
                    G.add_edge(a, w, m=marks[ed(a, w)])
                continue
            # walk along boundary vertices until the next internal vertex
            path, prev, cur = [a], a, w
            count = marks[ed(a, w)]
            while cur in boundary:
                count += 1        # the port itself is a terminal
                visited_boundary.add(cur)
                path.append(cur)
                nxt_ = [y for y in F.neighbors(cur) if y in cluster and y != prev]
                if len(nxt_) != 1:
                    raise ValueError("Unexpected boundary structure")
                prev, cur = cur, nxt_[0]
                count += marks[ed(prev, cur)]
            path.append(cur)
            key = tuple(path) if path[0] < path[-1] or (path[0] == path[-1] and path[1] < path[-2]) \
                else tuple(reversed(path))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            G.add_edge(path[0], path[-1], m=count)
    if visited_boundary != boundary:
        raise ValueError("Cluster has boundary vertices not on a path between internal vertices")
    if any(d != 3 for _, d in G.degree()) or nx.number_of_selfloops(G):
        raise ValueError("Cluster gadget is not cubic and loopless")
    if any(len(G[u][v]) > 1 for u, v in G.edges()):
        raise ValueError("Cluster gadget has parallel edges")
    Gs = nx.Graph()
    Gs.add_nodes_from(G.nodes())
    Ms = {}
    for u, v, data in G.edges(data=True):
        Gs.add_edge(u, v)
        Ms[ed(u, v)] = data["m"]
    return Gs, Ms


def verify_composition_certificate(F: nx.Graph, marks: dict, cluster_of: dict,
                                   exact_quotient_limit: int = 20) -> dict:
    """
    Independent check that ``F`` is robust at every weight, given only the partition
    ``cluster_of`` (vertex -> cluster label):

    1. ``F`` is simple, cubic and connected with at most two marks per edge.
    2. Every cluster, seen as a gadget (ports = terminals), is exactly certified by enumeration
       (identical clusters are certified once, matched by isomorphism).
    3. The quotient multigraph satisfies the H-expansion hypothesis, exactly for few clusters,
       otherwise via the rigorous spectral certificate ``mu_2 > 2 k_max``.
    """
    t_start = time.perf_counter()
    report: dict = {}
    marks = normalise_marks(F, marks)
    report["cubic_simple"] = is_cubic_simple(F)
    report["connected"] = nx.is_connected(F)
    report["max_marks_per_edge"] = max(marks.values())
    report["n"] = sum(marks.values())
    report["V"] = F.number_of_nodes()

    clusters = defaultdict(set)
    for x, c in cluster_of.items():
        clusters[c].add(x)
    if set(cluster_of) != set(F.nodes()):
        raise ValueError("Partition does not cover F")
    # every mark must lie inside a cluster: the lemma has no outputs on port-port edges
    report["marks_between_clusters"] = sum(c for (u, v), c in marks.items() if cluster_of[u] != cluster_of[v])
    if report["marks_between_clusters"]:
        report["certified"] = False
        return report

    # 2. gadgets
    t0 = time.perf_counter()
    reps: list[tuple[nx.Graph, dict, dict]] = []
    k_out = {}
    n_iso_checks = 0
    for c, nodes in clusters.items():
        try:
            Gc, Mc = _cluster_gadget(F, marks, nodes)
            if max(Mc.values(), default=0) > 2:
                raise ValueError("More than two terminals on one gadget edge")
        except ValueError as exc:
            report.update({"gadgets_certified": False, "failed_cluster": c, "reason": str(exc), "certified": False})
            return report
        k_out[c] = sum(marks[ed(u, v)] for u, v in F.subgraph(nodes).edges())
        nx.set_edge_attributes(Gc, {e: Mc[ed(*e)] for e in Gc.edges()}, "m")
        for Gr, Mr, cert in reps:
            n_iso_checks += 1
            if nx.is_isomorphic(Gc, Gr, edge_match=lambda x, y: x["m"] == y["m"]):
                break
        else:
            cert = certify_gadget(Gc, Mc)
            reps.append((Gc, Mc, cert))
            if not cert["certified"]:
                report.update({"gadgets_certified": False, "failed_cluster": c, "certified": False})
                return report
    report["gadgets_certified"] = True
    report["distinct_gadgets"] = len(reps)
    report["gadget_reports"] = [{k: v for k, v in cert.items() if k != "witness"} for _, _, cert in reps]
    report["gadget_time"] = time.perf_counter() - t0

    # 3. quotient
    t0 = time.perf_counter()
    Q = nx.MultiGraph()
    Q.add_nodes_from(clusters)
    for u, v in F.edges():
        cu, cv = cluster_of[u], cluster_of[v]
        if cu != cv:
            Q.add_edge(cu, cv)
    report["clusters"] = Q.number_of_nodes()
    report["k_out"] = dict(Counter(k_out.values()))
    if Q.number_of_nodes() <= exact_quotient_limit:
        margin, _ = edge_expansion_exact(Q, k_out)
        report["quotient_method"] = "exhaustive"
        report["quotient_margin"] = margin
        report["quotient_certified"] = margin <= 0
    else:
        k_max = max(k_out.values())
        A = adjacency_int(Q)
        degs = A.sum(axis=1)
        if not np.all(degs == degs[0]):
            report.update({"quotient_certified": False, "certified": False,
                           "reason": "spectral quotient certificate needs a regular quotient"})
            return report
        d = int(degs[0])
        cert = lambda2_certificate(nx.MultiGraph(Q), d - 2 * k_max, numeric=False)
        report["quotient_method"] = f"spectral: lambda_2 < {d - 2 * k_max}"
        report["quotient_eps"] = cert.get("eps")
        report["quotient_certified"] = cert["certified"]
    report["quotient_time"] = time.perf_counter() - t0
    report["certified"] = bool(report["cubic_simple"] and report["connected"]
                               and report["max_marks_per_edge"] <= 2 and report["gadgets_certified"]
                               and report["quotient_certified"])
    report["total_time"] = time.perf_counter() - t_start
    return report


# ----------------------------------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------------------------------

def ramanujan_tinf_marked_graph(n: int, d: int = 10, k: int = 2, seed: int = 0,
                                gadget: tuple[nx.Graph, dict] | None = None) -> dict:
    """
    SpiderCat Theorem C.6 with a certified random starting graph: N = ceil(n / k) vertices,
    d-regular with lambda_2 < d - 2k, k outputs per vertex (fewer on the last vertices if k does
    not divide n), each vertex replaced by a (d + k)-terminal gadget.
    """
    t0 = time.perf_counter()
    if gadget is None:
        gadget = load_gadget(d + k)
    N = max(-(-n // k), d + 1)
    if (N * d) % 2:
        N += 1
    H, hinfo = certified_starting_graph(N, d, k, seed=seed)
    kv = {v: k for v in H.nodes()}
    surplus = N * k - n
    for v in list(H.nodes())[::-1]:
        if surplus == 0:
            break
        drop = min(surplus, kv[v])
        kv[v] -= drop
        surplus -= drop
    F, marks, role = compose(H, kv, gadget)
    return {"F": F, "marks": marks, "role": role, "H": H, "k": kv, "H_info": hinfo,
            "gen_time": time.perf_counter() - t0, "N": N, "n": sum(marks.values()),
            "V": F.number_of_nodes()}


def save_marked_graph(path: Path, F: nx.Graph, marks: dict, meta: dict) -> None:
    """Save in the layout of ``circuits_data`` (``G.edges`` and ``M_inv``) plus metadata."""
    M_inv = defaultdict(list)
    for e in F.edges():
        M_inv[marks[ed(*e)]].append(list(e))
    path.write_text(json.dumps({"G.edges": [list(e) for e in F.edges()], "M_inv": dict(M_inv), **meta}))


# ----------------------------------------------------------------------------------------------
# Gadget search (exact objective, small graphs)
# ----------------------------------------------------------------------------------------------

def search_gadget(T: int, V: int, seed: int = 0, iters: int = 400, moves: int = 24,
                  graph_move_prob: float = 0.25, time_limit: float = 600.0) -> tuple[nx.Graph, dict] | None:
    """
    Local search for a cubic graph on ``V`` vertices with ``T`` single marks that is robust at every
    weight.  The objective (largest margin, number of violating vertex sets) is evaluated exactly
    over all vertex subsets, so a returned gadget is certified by construction (re-check it with
    ``certify_gadget`` anyway).
    """
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()
    while True:
        G = nx.random_regular_graph(3, V, seed=int(rng.integers(1 << 30)))
        if nx.is_connected(G):
            break
    free = V - 1
    masks = np.arange(1 << free, dtype=np.int64)
    X = [((masks >> i) & 1).astype(bool) for i in range(free)] + [np.zeros(1 << free, dtype=bool)]
    del masks
    half = T // 2

    def arrays(e):
        u, v = e
        return X[u] & X[v], X[u] ^ X[v]

    edges = [ed(u, v) for u, v in G.edges()]
    cut = np.zeros(1 << free, dtype=np.int8)
    for e in edges:
        cut += arrays(e)[1]
    # initial marks: random edges, at most two marked edges per vertex
    marked = set()
    for e in rng.permutation(len(edges)):
        e = edges[e]
        if len(marked) == T:
            break
        if sum(1 for f in marked if e[0] in f) < 2 and sum(1 for f in marked if e[1] in f) < 2:
            marked.add(e)
    for e in edges:
        if len(marked) == T:
            break
        marked.add(e)
    a = np.zeros(1 << free, dtype=np.int8)
    s = np.zeros(1 << free, dtype=np.int8)
    for e in marked:
        ins, ct = arrays(e)
        a += ins
        s += ct

    def score(a_, s_, cut_):
        margin = np.minimum(np.minimum(a_ + s_, T - a_), half) - cut_
        mx = int(margin.max())
        return (mx, int(np.count_nonzero(margin > 0)) if mx > 0 else 0)

    cur = score(a, s, cut)
    for it in range(iters):
        if cur[0] <= 0:
            return nx.Graph(G.edges()), {e: 1 for e in marked}
        if time.perf_counter() - t_start > time_limit:
            return None
        best = None
        if rng.random() < graph_move_prob:
            # double edge swap on unmarked edges: (u,v),(x,y) -> (u,x),(v,y)
            unmarked = [e for e in edges if e not in marked]
            for _ in range(moves):
                (u, v), (x, y) = [unmarked[i] for i in rng.choice(len(unmarked), 2, replace=False)]
                if rng.random() < 0.5:
                    x, y = y, x
                if len({u, v, x, y}) < 4 or G.has_edge(u, x) or G.has_edge(v, y):
                    continue
                G2 = G.copy()
                G2.remove_edges_from([(u, v), (x, y)])
                G2.add_edges_from([(u, x), (v, y)])
                if not nx.is_connected(G2):
                    continue
                c2 = cut - arrays((u, v))[1] - arrays((x, y))[1] + arrays((u, x))[1] + arrays((v, y))[1]
                sc = score(a, s, c2)
                if best is None or sc < best[0]:
                    best = (sc, "graph", (G2, c2))
        else:
            mlist, ulist = sorted(marked), [e for e in edges if e not in marked]
            for _ in range(moves):
                e_out = mlist[rng.integers(len(mlist))]
                e_in = ulist[rng.integers(len(ulist))]
                i1, c1 = arrays(e_out)
                i2, c2 = arrays(e_in)
                a2 = a - i1 + i2
                s2 = s - c1 + c2
                sc = score(a2, s2, cut)
                if best is None or sc < best[0]:
                    best = (sc, "mark", (e_out, e_in, a2, s2))
        if best is None:
            continue
        accept = best[0] <= cur or rng.random() < 0.05
        if accept:
            cur = best[0]
            if best[1] == "graph":
                G, cut = best[2]
                edges = [ed(u, v) for u, v in G.edges()]
            else:
                e_out, e_in, a, s = best[2]
                marked.remove(e_out)
                marked.add(e_in)
    return None


GADGET_DIR = Path(__file__).parent / "tinf_gadgets"


def load_gadget_library(folder: Path | None = GADGET_DIR) -> dict[int, tuple[nx.Graph, dict]]:
    """
    Gadgets keyed by number of terminals: SpiderCat data (T <= 13) plus searched ones from
    ``folder`` (``tinf_gadgets``: T = 14..17 on 20, 20, 24, 26 vertices).  All of them are
    re-certified exactly wherever they are used (``verify_composition_certificate``).
    """
    lib = {}
    for T in range(4, 14):
        g = load_gadget(T)
        if g is not None:
            lib[T] = g
    if folder is not None and Path(folder).is_dir():
        for f in sorted(Path(folder).glob("gadget_T*.json")):
            data = json.loads(f.read_text())
            G = nx.Graph([tuple(e) for e in data["edges"]])
            M = {ed(*e): 1 for e in data["marked"]}
            T = len(M)
            if T not in lib or lib[T][0].number_of_nodes() > G.number_of_nodes():
                lib[T] = (G, M)
    return lib


def complete_graph_plan(n: int, lib_sizes: dict[int, int]) -> dict | None:
    """
    Cheapest K_N composition for n outputs: vertex v of K_N gets k_v <= ceil(N/2) outputs and a
    gadget with at least N - 1 + k_v terminals.  For K_N, |delta(R)| = |R|(N - |R|) >=
    ceil(N/2) min(|R|, N - |R|), so the H-expansion hypothesis holds for any such k.
    ``lib_sizes`` maps terminal count T -> internal vertices of the smallest certified gadget.
    Returns the plan minimising the total number of internal vertices.
    """
    best = None
    Tmax = max(lib_sizes)
    cost_T = {}
    for T in range(1, Tmax + 1):
        cands = [V for t, V in lib_sizes.items() if t >= T]
        cost_T[T] = min(cands) if cands else None
    for N in range(2, Tmax + 2):
        kmax = -(-N // 2)
        options = []
        for k in range(0, kmax + 1):
            T = N - 1 + k
            if T > Tmax or cost_T.get(T) is None:
                continue
            options.append((k, cost_T[T] + (N - 1)))
        if not options:
            continue
        # knapsack: exactly N vertices, total outputs >= n, minimise internal vertices
        INF = math.inf
        dp = [[INF] * (n + 1) for _ in range(N + 1)]
        choice = [[None] * (n + 1) for _ in range(N + 1)]
        dp[0][0] = 0
        for i in range(N):
            for tot in range(n + 1):
                if dp[i][tot] == INF:
                    continue
                for k, c in options:
                    t2 = min(n, tot + k)
                    if dp[i][tot] + c < dp[i + 1][t2]:
                        dp[i + 1][t2] = dp[i][tot] + c
                        choice[i + 1][t2] = (k, tot)
        if dp[N][n] == INF:
            continue
        ks, tot = [], n
        for i in range(N, 0, -1):
            k, prev = choice[i][tot]
            ks.append(k)
            tot = prev
        plan = {"N": N, "k": sorted(ks, reverse=True), "V": dp[N][n]}
        if best is None or plan["V"] < best["V"]:
            best = plan
    return best


def complete_graph_tinf_marked_graph(n: int, lib: dict[int, tuple[nx.Graph, dict]]) -> dict:
    """Build the cheapest K_N composition for ``n`` outputs from a certified gadget library."""
    t0 = time.perf_counter()
    sizes = {T: g[0].number_of_nodes() for T, g in lib.items()}
    plan = complete_graph_plan(n, sizes)
    if plan is None:
        raise ValueError("No K_N plan with the available gadgets")
    N = plan["N"]
    H = nx.complete_graph(N)
    kv = dict(enumerate(plan["k"]))
    surplus = sum(kv.values()) - n
    for v in sorted(kv, key=lambda v: kv[v]):
        if surplus == 0:
            break
        drop = min(surplus, kv[v])
        kv[v] -= drop
        surplus -= drop
    gadgets = {}
    for v in H.nodes():
        need = N - 1 + kv[v]
        T = min((t for t in lib if t >= need), key=lambda t: (lib[t][0].number_of_nodes(), t))
        gadgets[v] = lib[T]
    F, marks, role = compose(H, kv, gadgets)
    return {"F": F, "marks": marks, "role": role, "H": H, "k": kv, "plan": plan,
            "gen_time": time.perf_counter() - t0, "N": N, "n": sum(marks.values()),
            "V": F.number_of_nodes()}


# ----------------------------------------------------------------------------------------------
# Circuit extraction support for sparse markings
# ----------------------------------------------------------------------------------------------

def circuit_fault_check(circ, n: int, max_weight: int = 2) -> dict:
    """
    Circuit-level check of an extracted CAT-state circuit (data qubits 0..n-1, every measurement
    post-selected on 0).  Fault locations are the wire segments of the circuit: the initial segment
    of every qubit and the segment after every operation on each qubit it acts on; each carries an
    X, Y or Z fault.  Pauli frames of all single faults are propagated together (bit matrices), and
    every undetected combination of at most ``max_weight`` faults (max_weight <= 2) must leave an
    output error of weight at most its own weight modulo the CAT stabilisers.
    """
    import stim
    out = {}
    sim = stim.TableauSimulator()
    sim.do(circ)
    out["noiseless_measurements_all_zero"] = not any(sim.current_measurement_record())
    zz = all(sim.peek_observable_expectation(_zz(i)) == 1 for i in range(1, n))
    xx = sim.peek_observable_expectation(stim.PauliString("X" * n)) == 1
    out["noiseless_cat"] = bool(zz and xx)

    if max_weight > 2:
        raise ValueError("Only fault combinations of weight <= 2 are enumerated")
    flat = circ.flattened()
    nq = flat.num_qubits
    # one entry per gate: stim merges consecutive gates of one type into a single instruction
    gates = []
    for op in flat:
        if op.name in ("H", "CX", "CNOT", "M"):
            for group in op.target_groups():
                gates.append((op.name, group))
        elif op.name not in ("DETECTOR", "TICK", "SHIFT_COORDS", "QUBIT_COORDS"):
            raise ValueError(f"Unsupported instruction {op.name}")
    seg_q = list(range(nq))                     # initial segments
    after = defaultdict(list)
    for i, (name, group) in enumerate(gates):
        for t in group:
            if t.is_qubit_target:
                after[i].append(len(seg_q))
                seg_q.append(t.value)
    S = len(seg_q)
    B = 3 * S
    X = np.zeros((nq, B), dtype=bool)
    Z = np.zeros((nq, B), dtype=bool)

    def inject(segs):
        for j in segs:
            q = seg_q[j]
            X[q, 3 * j] ^= True
            X[q, 3 * j + 1] ^= True
            Z[q, 3 * j + 1] ^= True
            Z[q, 3 * j + 2] ^= True

    inject(range(nq))
    flips = []
    for i, (name, ts) in enumerate(gates):
        if name == "H":
            for t in ts:
                q = t.value
                X[q], Z[q] = Z[q].copy(), X[q].copy()
        elif name in ("CX", "CNOT"):
            for c, t in zip(ts[::2], ts[1::2]):
                if c.is_measurement_record_target:
                    X[t.value] ^= flips[len(flips) + c.value]
                else:
                    X[t.value] ^= X[c.value]
                    Z[c.value] ^= Z[t.value]
        elif name == "M":
            for t in ts:
                flips.append(X[t.value].copy())
        inject(after.get(i, ()))
    F_ = np.array(flips) if flips else np.zeros((0, B), dtype=bool)
    syn = [F_[:, j].tobytes() for j in range(B)]
    weights_x = X[:n].T
    weights_z = Z[:n].T
    xs = [int.from_bytes(np.packbits(weights_x[j]).tobytes(), "big") for j in range(B)]
    zs = [int.from_bytes(np.packbits(weights_z[j]).tobytes(), "big") for j in range(B)]

    def reduced(x, z):
        wx = bin(x).count("1")
        wx = min(wx, n - wx)
        return wx if wx else bin(z).count("1") % 2

    zero = np.zeros(F_.shape[0], dtype=bool).tobytes()
    worst = -max_weight
    by_syn = defaultdict(list)
    for j in range(B):
        by_syn[syn[j]].append(j)
        if syn[j] == zero:
            worst = max(worst, reduced(xs[j], zs[j]) - 1)
    pairs = 0
    if max_weight >= 2:
        for group in by_syn.values():
            for a_, b_ in itertools.combinations(group, 2):
                if a_ // 3 == b_ // 3:
                    continue
                pairs += 1
                worst = max(worst, reduced(xs[a_] ^ xs[b_], zs[a_] ^ zs[b_]) - 2)
    out.update({"locations": S, "undetected_pairs_checked": pairs, "worst_excess": worst,
                "ft_up_to_weight": max_weight if worst <= 0 else None})
    return out


def _zz(i):
    import stim
    p = stim.PauliString(i + 1)
    p[0] = "Z"
    p[i] = "Z"
    return p


def spider_ordering_tree_milp(F: nx.Graph, marks: dict, time_limit: float = 300.0) -> tuple[nx.Graph, dict] | None:
    """
    Spanning tree of F in which every vertex of tree degree one is matched to a distinct mark on an
    incident NON-tree edge (Definition 4.2: all leaves are boundary spiders), solved as a MILP
    (HiGHS) with a single-commodity flow for connectivity.  Marked edges may be tree edges; their
    marks then sit inside a qubit line (one CNOT each, as in the repository extractor).
    Returns (tree, matches) in the format of ``match_forest_leaves_to_marked_edges`` or None.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    marks = normalise_marks(F, marks)
    edges = [ed(*e) for e in F.edges()]
    nodes = list(F.nodes())
    V = len(nodes)
    root = nodes[0]
    slots = [(e, i) for e in edges if marks[e] for i in range(marks[e])]
    ys = [(x, j) for j, (e, _) in enumerate(slots) for x in e]
    nx_, nf, ny = len(edges), 2 * len(edges), len(ys)
    nvar = nx_ + nf + ny
    xi = {e: i for i, e in enumerate(edges)}
    rows = [({xi[e]: 1 for e in edges}, V - 1, V - 1)]
    for e, i in xi.items():
        rows.append(({nx_ + 2 * i: 1, nx_ + 2 * i + 1: 1, i: -(V - 1)}, -np.inf, 0))
    cons = defaultdict(dict)
    for e, i in xi.items():
        u, v = e
        cons[v][nx_ + 2 * i] = cons[v].get(nx_ + 2 * i, 0) + 1
        cons[u][nx_ + 2 * i] = cons[u].get(nx_ + 2 * i, 0) - 1
        cons[u][nx_ + 2 * i + 1] = cons[u].get(nx_ + 2 * i + 1, 0) + 1
        cons[v][nx_ + 2 * i + 1] = cons[v].get(nx_ + 2 * i + 1, 0) - 1
    for v in nodes:
        if v != root:
            rows.append((cons[v], 1, 1))
    by_slot = defaultdict(dict)
    by_vertex = defaultdict(dict)
    for k, (x, j) in enumerate(ys):
        by_slot[j][nx_ + nf + k] = 1
        by_vertex[x][nx_ + nf + k] = 1
        # a mark can end a leaf only if its edge is not a tree edge: y <= 1 - x_e
        rows.append(({nx_ + nf + k: 1, xi[slots[j][0]]: 1}, -np.inf, 1))
    for j, row in by_slot.items():
        rows.append((row, 0, 1))
    inc = defaultdict(dict)
    for e, i in xi.items():
        inc[e[0]][i] = 1
        inc[e[1]][i] = 1
    for v in nodes:
        row = dict(inc[v])
        row.update(by_vertex[v])
        rows.append((row, 2, np.inf))
    A = lil_matrix((len(rows), nvar))
    lb = np.zeros(len(rows))
    ub = np.zeros(len(rows))
    for r, (row, lo, hi) in enumerate(rows):
        for c, val in row.items():
            A[r, c] = val
        lb[r], ub[r] = lo, hi
    integrality = np.concatenate([np.ones(nx_), np.zeros(nf), np.ones(ny)])
    upper = np.concatenate([np.ones(nx_), np.full(nf, V - 1), np.ones(ny)])
    # prefer leaving marked edges out of the tree (marks at line ends)
    cost = np.zeros(nvar)
    for e, i in xi.items():
        cost[i] = 1.0 if marks[e] else 0.0
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=integrality,
               bounds=Bounds(0, upper), options={"time_limit": time_limit})
    if res.x is None:
        return None
    T = nx.Graph()
    T.add_nodes_from(nodes)
    T.add_edges_from(e for e, i in xi.items() if res.x[i] > 0.5)
    if T.number_of_edges() != V - 1 or not nx.is_connected(T):
        return None
    matches = defaultdict(list)
    used = Counter()
    for k, (x, j) in enumerate(ys):
        e = slots[j][0]
        if res.x[nx_ + nf + k] > 0.5 and T.degree(x) == 1 and not matches.get(x) and not T.has_edge(*e):
            matches[x].append(e)
            used[e] += 1
    for e, c in marks.items():
        if T.has_edge(*e):
            continue                     # marks on tree edges are expanded inside the line
        for _ in range(c - used[e]):
            u, v = e
            matches[u if T.degree(u) >= T.degree(v) else v].append(e)
    leaves = [x for x in T if T.degree(x) == 1]
    assert all(matches.get(x) for x in leaves), "unmatched leaf"
    return T, dict(matches)


def expand_graph_and_forest_with_flags(graph: nx.Graph, forest: nx.Graph, markings: dict, matchings: dict):
    """
    Variant of ``circuit_extraction.expand_graph_and_forest`` for graphs with unmarked non-tree
    edges.  The original subdivides each such edge with a flag node and hangs that node on the
    forest as a leaf, so a qubit line can end on a flag instead of an output (harmless when every
    non-tree edge is marked, as in SpiderCat's own graphs with vertex ratio <= 2).  Here unmarked
    non-tree edges are left unsubdivided: both endpoints then see the same non-forest edge, and the
    extractor implements it as one ancilla with a CNOT from each endpoint's qubit line followed by
    post-selection (Procedure 4.1, step 4).  Marked edges are expanded exactly as in the original.
    """
    G_new = graph.copy()
    F_new = forest.copy()
    edge_to_matches = defaultdict(list)
    for matched_node, edges in matchings.items():
        for edge in edges:
            edge_to_matches[tuple(sorted(edge))].append(matched_node)
    for edge, count in ((tuple(sorted(e)), c) for e, c in markings.items() if c > 0):
        u, v = edge
        G_new.remove_edge(u, v)
        is_forest_edge = F_new.has_edge(u, v)
        if is_forest_edge:
            F_new.remove_edge(u, v)
        next_id = max(G_new.nodes()) + 1
        new_nodes = []
        for _ in range(count):
            G_new.add_node(next_id, is_mark=True, is_flag=False, original_edge=edge)
            F_new.add_node(next_id, is_mark=True, is_flag=False, original_edge=edge)
            new_nodes.append(next_id)
            next_id += 1
        path = [u] + new_nodes + [v]
        steps = list(zip(path, path[1:]))
        G_new.add_edges_from(steps)
        if is_forest_edge:
            F_new.add_edges_from(steps)
        else:
            gap = min(edge_to_matches.get(edge, []).count(u), count)
            F_new.add_edges_from(s for i, s in enumerate(steps) if i != gap)
    return G_new, F_new


def extract_circuit_sparse(F: nx.Graph, marks: dict, tree: nx.Graph, matches: dict):
    """SpiderCat's CatStateExtractor with flag nodes kept outside the forest."""
    from spidercat.circuit_extraction import CatStateExtractor, StimBuilder
    from spidercat.spanning_tree import find_min_height_roots
    marks = normalise_marks(F, marks)
    G_new, F_new = expand_graph_and_forest_with_flags(F, tree, marks, matches)
    roots = find_min_height_roots(tree)
    return CatStateExtractor(StimBuilder()).extract(G_new, F_new, roots)


def count_cnots(circ) -> int:
    return sum(len(op.targets_copy()) // 2 for op in circ if op.name in ("CX", "CNOT"))


def derived_spidercat_gadgets(max_T: int = 21) -> dict[int, tuple[nx.Graph, dict]]:
    """
    Gadgets obtained from stored SpiderCat graphs by dropping marks.  A t-robust graph with n'
    marks is robust at every weight for any T <= min(n', 2t + 3) of its marks (a side holds at most
    floor(T/2) <= t + 1 terminals, and removing marks never creates a violation).
    """
    best: dict[int, tuple[nx.Graph, dict]] = {}
    for t in range(2, 8):
        for n_ in range(4, 51):
            sol = load_solution_triplet(n_, t, 1)
            if sol is None:
                continue
            G, _, M, _ = sol
            G = nx.convert_node_labels_to_integers(G, ordering="sorted", label_attribute="orig")
            back = {data["orig"]: v for v, data in G.nodes(data=True)}
            slots = [ed(back[u], back[v]) for (u, v), c in sorted(M.items()) for _ in range(c)]
            for T in range(4, min(n_, 2 * t + 3, max_T) + 1):
                if T in best and best[T][0].number_of_nodes() <= G.number_of_nodes():
                    continue
                Mt = Counter(slots[:T])
                best[T] = (nx.Graph(G.edges()), dict(Mt))
    return best


def exact_cheeger(H: nx.Graph) -> tuple[float, int]:
    """Exact min over R of |delta(R)| / min(|R|, N - |R|) (enumeration) and the largest integer k with
    |delta(R)| >= k min(|R|, N - |R|) for all R."""
    nodes = list(H.nodes())
    N = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in H.edges()]
    free = N - 1
    masks = np.arange(1, 1 << free, dtype=np.int64)
    bits = [((masks >> i) & 1).astype(np.int16) for i in range(free)] + [np.zeros(len(masks), dtype=np.int16)]
    size = sum(bits)
    cut = np.zeros(len(masks), dtype=np.int32)
    for u, v in edges:
        cut += bits[u] ^ bits[v]
    small = np.minimum(size, N - size)
    h = float(np.min(cut / small))
    k = int(np.min(cut // small))
    return h, k


def dense_graph_plan(n: int, lib: dict[int, tuple[nx.Graph, dict]], max_N: int = 22, samples: int = 4,
                     seed: int = 0) -> dict | None:
    """
    Cheapest small top-level graph for n outputs: K_N (mixed k_v via ``complete_graph_plan``) or a
    random d-regular H on N <= max_N vertices whose exact Cheeger constant h gives k = floor(h)
    outputs per vertex.  Every vertex gets a gadget with >= d + k terminals.  Candidates are priced
    first; the exponential exact expansion check only runs on candidates that would be cheaper.
    """
    sizes = {T: g[0].number_of_nodes() for T, g in lib.items()}
    Tmax = max(sizes)
    cost_T = {T: min((V for t, V in sizes.items() if t >= T), default=None) for T in range(1, Tmax + 1)}
    best = None
    kn = complete_graph_plan(n, sizes)
    if kn is not None:
        best = {"kind": "complete", "N": kn["N"], "V": kn["V"], "k": kn["k"]}
    rng = np.random.default_rng(seed)
    cands = []
    for N in range(6, max_N + 1):
        k_use = -(-n // N)
        for d in range(3, N - 1):
            T = d + k_use
            # an h >= k_use needs d >= 2 k_use (N - 1) / N roughly; skip hopeless pairs
            if (N * d) % 2 or T > Tmax or cost_T.get(T) is None or d * N < 2 * k_use * (N - 1):
                continue
            cands.append((N * (cost_T[T] + d), N, d, k_use))
    for V, N, d, k_use in sorted(cands):
        if best is not None and V >= best["V"]:
            break
        for _ in range(samples):
            H = random_regular_graph(d, N, seed=int(rng.integers(1 << 30)))
            if not nx.is_connected(H):
                continue
            h, k = exact_cheeger(H)
            if k >= k_use:
                best = {"kind": "regular", "N": N, "d": d, "h": h, "k_cert": k, "k": k_use, "V": V, "H": H}
                break
    return best


def dense_tinf_marked_graph(n: int, lib: dict[int, tuple[nx.Graph, dict]], **kw) -> dict:
    """Build the plan of ``dense_graph_plan``."""
    t0 = time.perf_counter()
    plan = dense_graph_plan(n, lib, **kw)
    if plan is None:
        raise ValueError("No dense plan")
    if plan["kind"] == "complete":
        r = complete_graph_tinf_marked_graph(n, lib)
        r["plan"] = {k: v for k, v in plan.items() if k != "H"}
        r["gen_time"] = time.perf_counter() - t0
        return r
    H = plan["H"]
    kv = {v: plan["k"] for v in H.nodes()}
    surplus = sum(kv.values()) - n
    for v in list(H.nodes())[::-1]:
        if surplus == 0:
            break
        drop = min(surplus, kv[v])
        kv[v] -= drop
        surplus -= drop
    gadgets = {}
    for v in H.nodes():
        need = H.degree(v) + kv[v]
        T = min((t for t in lib if t >= need), key=lambda t: (lib[t][0].number_of_nodes(), t))
        gadgets[v] = lib[T]
    F, marks, role = compose(H, kv, gadgets)
    return {"F": F, "marks": marks, "role": role, "H": H, "k": kv,
            "plan": {k: v for k, v in plan.items() if k != "H"}, "gen_time": time.perf_counter() - t0,
            "N": H.number_of_nodes(), "n": sum(marks.values()), "V": F.number_of_nodes()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Certified t = infinity marked graph for an n-qubit CAT state")
    parser.add_argument("n", type=int, help="number of CAT outputs (marks)")
    parser.add_argument("--route", choices=["ramanujan", "dense"], default="dense",
                        help="ramanujan: random certified 10-regular H with 2 outputs per vertex (any n >= 22); "
                             "dense: small dense H (K_N or exactly-expanding random regular) with larger gadgets")
    parser.add_argument("--gadgets", type=Path, default=GADGET_DIR, help="folder with extra gadget_T*.json files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.route == "ramanujan":
        res = ramanujan_tinf_marked_graph(args.n, seed=args.seed)
    else:
        library = load_gadget_library(args.gadgets)
        for T_, g_ in derived_spidercat_gadgets().items():
            if T_ not in library or g_[0].number_of_nodes() < library[T_][0].number_of_nodes():
                library[T_] = g_
        res = dense_tinf_marked_graph(args.n, library, seed=args.seed)
    gen_time = time.perf_counter() - t0
    clusters = {x: res["role"][x][0] for x in res["F"].nodes()}
    cert = verify_composition_certificate(res["F"], res["marks"], clusters)
    print(f"n={res['n']} starting-graph vertices N={res['N']} internal vertices V={res['V']} "
          f"ratio={res['V'] / res['n']:.2f} generated in {gen_time:.2f}s")
    print(f"certified={cert['certified']} ({cert.get('quotient_method')}) in {cert.get('total_time', 0):.2f}s")
    if args.out is not None:
        save_marked_graph(args.out, res["F"], res["marks"],
                          {"n": res["n"], "t": "inf", "route": args.route, "N": res["N"], "V": res["V"],
                           "cluster_of": {str(x): str(c) for x, c in clusters.items()},
                           "certified": cert["certified"]})
        print(f"saved to {args.out}")

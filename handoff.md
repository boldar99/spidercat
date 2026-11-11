# Handoff: Graph-Forest DEM, Boundary Fault Analysis & Edge Weight Optimization

## 1. Context & Goal

When extracting a fault-tolerant cat-state preparation circuit from an expanded graph `G_exp`, an expanded spanning forest/tree `F_exp`, and a `root`, the set of possible **undetectable faults** and the **boundary errors** they produce on the $n$ output qubits are completely determined by `(G_exp, F_exp)`.

This work is organized into two completed parts:
1. **Part 1 (Completed)**: Given `(G_exp, F_exp)` and a maximum boundary error weight $L$, construct a Detector Error Model (DEM) directly from the graph and forest, precompute how weight-$k$ boundary faults ($k = 1, \dots, L$) can be obtained from $\le k$ edge faults, and return a data structure (`PrecomputedFaultTable`) optimized for fast substitution of integer edge weights.
2. **Part 2 (Completed)**: Model the valid edge-weight assignments $w(e)$ induced by local CNOT orderings at the root and degree-3 green nodes (including combined tuple weights on non-tree flag edges in $G_{\text{exp}} \setminus F_{\text{exp}}$), and optimize edge weights to minimize the undetectable boundary fault polynomial $c_1 p + c_2 p^2 + \dots + c_L p^L$ via **Exhaustive Search**, **Greedy Search**, and **Simulated Annealing**, plus edge-weight visualization in `display_digraph_on_graph`.

---

## 2. Mathematical Model

### 2.1 Open Graph Structure
- **Internal edges**: Raw edges $(u, v) \in E(G_{\text{exp}})$ (normalized with $u < v$).
- **Boundary edges**: Each of the $n$ nodes $m \in V(G_{\text{exp}})$ with `is_mark=True` has degree 2 in `G_exp`. Adding a 3rd imaginary edge `(m, "boundary")` from each `is_mark` node to the boundary forms the open graph $\tilde{G}$ with $n$ boundary outputs.

### 2.2 Graph-Forest DEM (`GraphDEM`)
1. **Detectors (`D_0 .. D_{C-1}`)**:
   - A fault is undetectable iff it forms a cut of `G_exp` (i.e., intersects every cycle of `G_exp` an even number of times).
   - Taking a cycle basis $\mathcal{C} = \{C_0, \dots, C_{C-1}\}$ of `G_exp`, each basis cycle $C_i$ defines a detector `D_i`.
   - Using the **fundamental cycle basis** induced by `F_exp` (where each non-forest edge $f_i \in E(G_{\text{exp}}) \setminus E(F_{\text{exp}})$ defines a single cycle $C_i$ together with the unique path in `F_exp` between its endpoints) means each non-forest edge belongs to **exactly one** cycle (`det_mask[f_i] = 1 << i`).
2. **Logical Observables (`L_0 .. L_{n-1}`)**:
   - Indexed by the sorted $n$ `is_mark` nodes $M = (m_0, \dots, m_{n-1})$.
3. **Edge-to-(Detectors, Observables) Mapping**:
   - **Tree edge $e \in E(F_{\text{exp}})$**: Flips all detectors `D_i` whose cycle $C_i$ contains $e$, and propagates to all boundary qubits $m_j$ in the subtree below $e$ (when cutting `F_exp` at $e$).
   - **Non-tree edge $f \in E(G_{\text{exp}}) \setminus E(F_{\text{exp}})$**: Equivalent to a flag measurement error; flips detectors `D_i` for cycles containing $f$ and flips **no** boundary observables (`obs_mask = 0`).
   - **Boundary edge `(m_j, "boundary")`**: Not part of any cycle (`det_mask = 0`); flips only `L_j`.
4. **All-Output Cat-State Stabilizer ($X^{\otimes n}$)**:
   - Because the prepared state is a cat state stabilized by $X^{\otimes n}$ (`L_0 L_1 ... L_{n-1}`), any boundary error subset of size $m > n // 2$ is equivalent (via multiplying by $X^{\otimes n}$, i.e., bitwise complement `all_obs_mask ^ mask`) to an error of weight $n - m \le n // 2$.

### 2.3 Precomputing Weight-$k$ Boundary Faults & Factored Substitution
- Because each non-tree edge $f_i$ has `det_mask[f_i] = 1 << i` in the fundamental cycle basis, **any subset of tree edges $E_T \subseteq E(F_{\text{exp}})$ uniquely determines the exact set of non-tree edges $E_{\text{flag}}(E_T)$ required to make the internal fault undetectable**:
  $$E_{\text{flag}}(E_T) = \left\{f_i : \text{bit } i \text{ of } \left(\bigoplus_{e \in E_T} \text{det\_mask}[e]\right) \text{ is } 1\right\}$$
- **Filtering Criteria**:
  1. Total internal faults $s_{\text{int}} = |E_T| + |E_{\text{flag}}(E_T)| \le m_S$, where $m_S = \min(\text{popcount}(B_{\text{int}}), n - \text{popcount}(B_{\text{int}})) \in \{1, \dots, L\}$ is the folded boundary error weight.
  2. **Irreducible / Matching Cut**: No vertex in `G_exp` has $\ge 2$ incident edges in $F_{\text{int}} = E_T \cup E_{\text{flag}}(E_T)$. Since every spider has degree $\le 3$, two faults on two legs of the same spider are always reducible to a single fault on the 3rd leg (this automatically rules out two internal faults on an `is_mark` node producing a weight-1 boundary fault).
  3. **Component Validity**: If removing $F_{\text{int}}$ splits `G_exp` into $> 2$ connected components, every component $C$ must satisfy $1 \le |\delta(C)| \le |M \cap C|$ (preventing 0-mark islands or wasteful subcuts from piggybacking on a $k > t$ cut).
- **Factored Representation for Weight Substitution**:
  - Every valid fault set producing a weight-$k$ boundary error is a disjoint union $F = F_{\text{int}} \sqcup B$, where $F_{\text{int}}$ flips $m_S \le k$ boundary nodes $M_S$ and $B$ is a combination of $b = k - m_S$ boundary edges chosen from $E_{\text{bnd}} \setminus M_S$.
  - Rather than expanding all $\binom{n - m_S}{k - m_S}$ boundary combinations in memory, each internal cut $F_{\text{int}}$ is stored once as an `InternalCut` and paired during evaluation with the elementary symmetric polynomial $e_{k - m_S}$ over the remaining $n - m_S$ boundary edge weights:
    $$\sum_{B \in \binom{E_{\text{bnd}} \setminus M_S}{k - m_S}} \prod_{e \in F_{\text{int}} \sqcup B} w(e) = \left(\prod_{e \in F_{\text{int}}} w(e)\right) \cdot e_{k - m_S}\!\left(\{w(b_j) : j \notin M_S\}\right)$$

### 2.4 Edge Weight Assignment Rules & Flag Constraints
Orienting the spanning tree $T$ away from `root` gives each node $v \neq \text{root}$ a unique tree parent $p(v)$ (one incoming edge $(p(v), v) \in E(T)$) and two outgoing legs:
1. **Root node (`v == root`, degree 2)**:
   - Has 0 incoming edges and 2 outgoing tree edges $(e_1, e_2)$ with total weight 3: `(2, 1)`, `(1, 2)`, `(3, 0)`, or `(0, 3)`.
2. **Green degree-3 nodes (`v != root`)**:
   - Have 1 incoming tree edge $(p(v), v)$ and 2 outgoing legs $(e_1, e_2)$, assigned weights `(2, 1)` or `(1, 2)`.
   - For **boundary mark nodes** (`is_mark=True`), the two outgoing legs are the internal/flag edge $e_{\text{int}}$ and the boundary edge $e_{\text{bnd}} = (v, \text{"boundary"})$. By default (`fix_boundary_weight_one=True`), $w(e_{\text{bnd}}) = 1$ and $w(e_{\text{int}}) = 2$.
3. **Flag edges in $G_{\text{exp}} \setminus F_{\text{exp}}$**:
   - A non-tree edge $f = (a, b) \in E(G_{\text{exp}}) \setminus E(T)$ (with $a < b$) is an outgoing leg of **both** green nodes $a$ and $b$.
   - Its weight is represented as the tuple $(w_a, w_b)$ where $w_a \in \{1, 2\}$ is chosen at node $a$ and $w_b \in \{1, 2\}$ is chosen at node $b$.
   - **Flag Constraint**: At least one of $w_a$ or $w_b$ must be $2$ ($(w_a, w_b) \neq (1, 1)$), so $(w_a, w_b) \in \{(2, 1), (1, 2), (2, 2)\}$ and the effective weight substituted into the fault polynomial is $w(f) = w_a + w_b \in \{3, 4\}$.

---

## 3. Codebase Changes & API Reference

### Added / Updated Files
- [spidercat/fault_analysis.py](file:///Users/boldizsar.poor/PycharmProjects/spidercat/spidercat/fault_analysis.py): Core implementation of `GraphDEM`, `InternalCut`, `PrecomputedFaultTable`, `build_graph_dem`, and `compute_boundary_fault_table`. Automatically sums tuple weights `(w_a, w_b)` on flag edges when passed in a dictionary.
- [spidercat/weight_optimization.py](file:///Users/boldizsar.poor/PycharmProjects/spidercat/spidercat/weight_optimization.py): Implementation of `NodeDecision`, `FlagEdgeConstraint`, `WeightSpace`, `WeightCandidate`, `SimulatedAnnealingResult`, `build_weight_space`, `exhaustive_weight_search`, `greedy_weight_search`, and `simulated_annealing_weight_search`.
- [spidercat/draw.py](file:///Users/boldizsar.poor/PycharmProjects/spidercat/spidercat/draw.py): Updated `display_digraph_on_graph(di_graph, G, weights=None, ...)` to accept a weight dict or `WeightCandidate` and render edge weights on both directed tree edges and dashed non-tree flag edges.
- [spidercat/__init__.py](file:///Users/boldizsar.poor/PycharmProjects/spidercat/spidercat/__init__.py): Exports the public API from `spidercat.fault_analysis` and `spidercat.weight_optimization`.

### Key Classes & Functions in `spidercat.fault_analysis`
- `build_graph_dem(G_exp, F_exp, root=None, cycle_basis_mode="fundamental") -> GraphDEM`
- `compute_boundary_fault_table(G_exp, F_exp, L, root=None) -> PrecomputedFaultTable`
  - `table.count_fault_sets(k=None)`
  - `table.get_fault_sets(k, as_indices=False)`
  - `table.evaluate_polynomial(edge_weights, mode="boundary_weight")` (accepts dicts with `int` or `(w_a, w_b)` tuples, 1D arrays, or 2D batch arrays)
  - `table.format_polynomial(edge_weights, var="p", mode="boundary_weight") -> str`

### Key Classes & Functions in `spidercat.weight_optimization`
- `build_weight_space(table_or_dem, fix_boundary_weight_one=True, allow_root_3_0=True) -> WeightSpace`
  - Extracts all root and green-node decisions (`space.decisions`) and flag-edge constraints (`space.flag_constraints`), and provides fast vectorized conversion from option indices to effective edge weights.
- `exhaustive_weight_search(table, *, fix_boundary_weight_one=True, allow_root_3_0=True, objective="lex", p=1e-3, ...) -> list[WeightCandidate]`
  - Enumerates all valid weight choices (pruning any $(1, 1)$ flag configuration), evaluates their polynomials in vectorized batches, and returns a sorted list of `WeightCandidate` objects from best to worst.
- `greedy_weight_search(table, *, fix_boundary_weight_one=True, allow_root_3_0=True, objective="lex", p=1e-3, polish=True) -> WeightCandidate`
  - Performs a constructive top-down BFS pass (starting from neutral `1.5` weights on undecided internal edges and `(2, 1)` on boundary `is_mark` nodes), followed by steepest-descent local coordinate polishing over all valid 1-node flips and coupled 2-node flag swaps $(2, 1) \leftrightarrow (1, 2)$.
- `simulated_annealing_weight_search(table, *, init_candidate=None, fix_boundary_weight_one=True, allow_root_3_0=True, objective="lex", p=1e-3, num_steps=1000, T_init=None, T_final=None, seed=0, return_history=False) -> WeightCandidate | SimulatedAnnealingResult`
  - Starts from the greedy solution and performs a Metropolis walk where each step swaps the two outgoing weights at a randomly selected node (automatically coupling the swap across an incident flag edge if needed to preserve $w(f) \in \{3, 4\}$), with auto-calibrated geometric cooling.

---

## 4. Usage Example

```python
from spidercat.utils import load_solution_triplet
from spidercat.spanning_tree import match_forest_leaves_to_marked_edges, inject_root_nodes
from spidercat.circuit_extraction import expand_graph_and_forest, build_traversal_digraph
from spidercat.fault_analysis import compute_boundary_fault_table
from spidercat.weight_optimization import (
    exhaustive_weight_search,
    greedy_weight_search,
    simulated_annealing_weight_search,
)
from spidercat.draw import display_digraph_on_graph

# 1. Prepare expanded graph, spanning forest, and root
grf, forest, M, matchings = load_solution_triplet(8, 3, 1)
matchings = match_forest_leaves_to_marked_edges(grf, forest, M)
G_exp, F_exp = expand_graph_and_forest(grf, forest, M, matchings)
roots = inject_root_nodes(G_exp, F_exp)

# 2. Precompute boundary fault table up to L = 4
table = compute_boundary_fault_table(G_exp, F_exp, L=4, root=roots)

# 3. Run Exhaustive, Greedy, or Simulated Annealing weight search
all_candidates = exhaustive_weight_search(table, fix_boundary_weight_one=True)
best_exhaustive = all_candidates[0]
print("Exhaustive best:", best_exhaustive.polynomial_str)
print("Edge weights (flag edges as tuples):", best_exhaustive.edge_weights)

best_greedy = greedy_weight_search(table, fix_boundary_weight_one=True)
print("Greedy best    :", best_greedy.polynomial_str)

best_sa = simulated_annealing_weight_search(table, fix_boundary_weight_one=True, num_steps=1000)
print("Sim. Annealing :", best_sa.polynomial_str)

# 4. Visualize the directed traversal graph with edge weights
digraph = build_traversal_digraph(G_exp, F_exp, roots[0])
display_digraph_on_graph(digraph, G_exp, weights=best_greedy, figsize=(10, 6))
```

---

## 5. Next Steps

1. **Circuit Scheduling from Optimal Node Choices**:
   - Use `candidate.node_choices` (which specifies which of the two outgoing legs at each node has weight 1 vs. weight 2, and whether a flag edge is `(2, 1)`, `(1, 2)`, or `(2, 2)`) in [spidercat/circuit_extraction.py](file:///Users/boldizsar.poor/PycharmProjects/spidercat/spidercat/circuit_extraction.py) to order the CNOT gates emitted by `CatStateExtractor` so the physical Stim circuit realizes the optimized fault polynomial.
2. **Joint Spanning-Tree / Root / Weight Optimization**:
   - Search across multiple candidate roots or spanning forests `F_exp` using `greedy_weight_search` / `simulated_annealing_weight_search` as the inner evaluator to find the globally best `(F_exp, root, edge_weights)` triple.

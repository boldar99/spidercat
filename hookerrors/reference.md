# Reference: Fault-Tolerant ZX-Calculus Hook Error Analyzer

This document provides comprehensive context, theoretical definitions, and implementation details for the `hookerrors` codebase. It serves as a guide for understanding the algebraic splitting of stabilizer generators and evaluating the safety of the resulting hook errors during FT-circuit synthesis in the ZX-calculus.

## 1. Context and Theoretical Background

In Fault-Tolerant (FT) state preparation via ZX-calculus (or analogous circuit-model approaches), high-weight stabilizer generators (represented as high-degree spiders) must be broken down into smaller structures (e.g., degree-3 spiders) to be compiled into physical circuits.

**The Hook Error:**
When a generator is algebraically split, an internal edge is created. A single physical fault (weight-1) on this internal edge can propagate to multiple data qubits. This propagated multi-qubit Pauli error is a "hook error" ($E$).

*   **X Generators** produce **X-type** hook errors.
*   **Z Generators** produce **Z-type** hook errors.
*(Note: This convention maps naturally to standard circuit-level synthesis, where an X fault on an ancilla preparing an X-stabilizer propagates through CNOT control nodes to form multi-qubit X errors on the data block.)*

A hook error is malignant if it acts as a weight-amplified error that bypasses the code's distance. However, certain hook errors can be safely tolerated due to two "Saving Graces".

## 2. The Two Saving Graces (Safe Hooks)

Let $d = 2t + 1$ be the distance of the code, where $t$ is the maximum number of correctable adversarial faults. We define the preparation stabilizer group $\mathcal{S}_{\text{prep}}$ based on the logical basis being prepared:
*   **Z-Basis Preparation ($|0\rangle_L$):** $\mathcal{S}_{\text{prep}} = \langle H_x, H_z, L_z \rangle$. Logical failures are caused by $L_x$.
*   **X-Basis Preparation ($|+\rangle_L$):** $\mathcal{S}_{\text{prep}} = \langle H_x, H_z, L_x \rangle$. Logical failures are caused by $L_z$.

### Saving Grace 1: Trivially Safe Hooks (State Symmetries)
If the hook error $E$ can be multiplied by elements of $\mathcal{S}_{\text{prep}}$ such that its total weight is reduced to $\le 1$, it acts exactly as a single physical fault on the target state. 
$$ w_{\text{min}}^* = \min_{S \in \mathcal{S}_{\text{prep}}} \text{weight}(E \cdot S) \le 1 $$
*Classification:* **Trivially Safe.** No distance is consumed.

### Saving Grace 2: Decoder-Benign Hooks (Perfect Decoding)
If $w_{\text{min}}^* > 1$, the hook consumes some of the code's distance. However, it is safe if the minimum-weight decoder can still tolerate $t-1$ additional ambient faults without causing a logical failure. 

To cause a logical failure, the combined error (Hook $E$ + Ambient + Decoder Guess) must form a non-trivial logical operator $L \notin \mathcal{S}_{\text{prep}}$. The condition for the hook to be safely decoded is:
$$ \min_{L, S \in \mathcal{S}_{\text{prep}}} \text{weight}(E \cdot L \cdot S) \ge 2t $$
*Classification:* **Decoder-Benign.** The heuristic decoder will never mistakenly form a logical operator when this hook occurs alongside $\le t-1$ ambient faults.

---

## 3. Codebase Architecture

The `hookerrors` codebase uses a modular, plugin-based architecture designed to aggressively scale from small codes (e.g. `7_1_3`) to massive layouts (e.g. `92_2_14`).

### `hookerrors/filters.py` (The Oracle Strategies)
This module evaluates whether a specific hook error split $E$ satisfies the Saving Graces. It decouples the X and Z evaluations since X and Z errors do not destructively interfere in CSS codes.

*   **`LookupStrategy`**: Precomputes a complete Breadth-First Search (BFS) dictionary mapping syndromes to minimum weights up to depth $2t$. 
    *   *Pros:* Instantaneous ($O(1)$) evaluations. Solves Tier 1 and Tier 3 simultaneously.
    *   *Cons:* Memory and compute scale as $\binom{n}{2t}$. Only viable for $t \le 3$.
*   **`TieredStrategy`**: Implements the strict, three-tier filtering oracle:
    *   *Tier 1 (GF(2) RREF):* Instantly checks if $E$ or $E \oplus e_1$ is in the row space of $\mathcal{S}_{\text{prep}}$.
    *   *Tier 2 (Heuristic BP-OSD):* Feeds the syndrome of $(E \cdot L)$ to a `bposd` decoder. If the decoder finds an error of weight $< 2t$, the hook is instantly flagged as Malignant.
    *   *Tier 3 (MILP Exact Solver):* If BP-OSD fails to find a low-weight error, a rigorous `scipy.optimize.milp` Integer Linear Program proves whether the exact minimum weight is $\ge 2t$.
*   **`HeuristicOnlyStrategy`**: Identical to `TieredStrategy`, but skips Tier 3 entirely. By cranking up the BP-OSD order (`osd_10`), we trust the tight heuristic bound and avoid NP-hard ILP calls. Highly scalable for large $d$.
*   **`MILPStrategy`**: Purely exhaustive exact ILP solving without heuristics. Primarily for testing.

### `hookerrors/searchers.py` (The Pruning Strategies)
This module dictates which algebraic splits (subsets of a generator's support) are generated and evaluated.

*   **Complementary Deduplication:** An internal edge bipartitions the support $W$ into $A$ and $B$. Since $A \cdot B = W \in \mathcal{S}_{\text{prep}}$, error $A$ is equivalent to error $B$. Searchers strictly cap evaluations at `size <= w // 2` and filter out complementary subsets to exactly halve the search space.
*   **`ExhaustiveSearcher`**: Evaluates all $\binom{w}{w/2}$ subsets.
*   **`EarlyExitSearcher`**: A greedy searcher that stops evaluating a generator the moment it finds `max_splits` (e.g., 1) valid safe splits. Critical for synthesizing large codes where exhaustive listing is unnecessary. Supports `max_split_size` bounding to prioritize testing small $k$ splits which are inherently more likely to be safe.

### `hookerrors/solver.py` (The CLI)
The main executable script tying the system together. 
*   Dynamically loads the chosen strategy and searcher based on CLI arguments.
*   Constructs the X and Z cosets independently based on the `--basis` argument.
*   Example: `python hookerrors/solver.py --code 92_2_14 --method heuristic --searcher early_exit --max-weight 2`

---

## 4. Notable Code Behaviors
1.  **Asymmetric Safety (Basis Dependence):** If `--basis Z` is chosen, the target state is $|0\rangle_L$, so $L_z \in \mathcal{S}_{\text{prep}}$. As a result, Z-type hook errors are generally benign (or trivially safe), while X-type hook errors consume distance. The solver correctly reflects this extreme asymmetry in the output counts.
2.  **Missing Logical Cosets:** If a hook error acts entirely in a basis that has no non-trivial logical operators left to test (e.g., testing Z-errors during Z-basis prep), the solver identifies that `len(L_cosets) == 0`. It instantly returns `True` since a logical failure in that basis is impossible.
3.  **Formatting:** Subsets and generators are parsed into human-readable tuples, e.g., `"X(4, 5, 6, 7)"`.

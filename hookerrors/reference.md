# Hook Errors Reference Guide

## 1. Terminology
*   **Physical Fault:** A single hardware failure on an internal ZX edge (weight 1).
*   **Propagated Error:** The resulting multi-qubit Pauli error on the data block.
*   **Hook Error:** A propagated error that has a physical weight $> 1$.
*   **X Generators** produce **X-type** hook errors.
*   **Z Generators** produce **Z-type** hook errors.

## 2. The Saving Grace (State Symmetries)

Let $d = 2t + 1$ be the distance of the code, where $t$ is the maximum number of correctable adversarial faults. We define the preparation stabilizer group $\mathcal{S}_{\text{prep}}$ based on the logical basis being prepared:
*   **Z-Basis Preparation ($|0\rangle_L$):** $\mathcal{S}_{\text{prep}} = \langle H_x, H_z, L_z \rangle$.
*   **X-Basis Preparation ($|+\rangle_L$):** $\mathcal{S}_{\text{prep}} = \langle H_x, H_z, L_x \rangle$.

If the hook error $E$ can be multiplied by elements of $\mathcal{S}_{\text{prep}}$ such that its total weight is reduced to $\le 1$, it acts exactly as a single physical fault on the target state. 
$$ w_{\text{min}}^* = \min_{S \in \mathcal{S}_{\text{prep}}} \text{weight}(E \cdot S) \le 1 $$
*Classification:* **Strictly Safe.** No distance is consumed beyond the 1 physical fault that caused it. Because this metric satisfies the triangle inequality, any combination of $k$ strictly safe hooks is guaranteed to consume $\le k$ effective distance, ensuring the protocol remains fault-tolerant.

---

## 3. Codebase Architecture

The `hookerrors` codebase uses a highly scalable $\mathcal{O}(N)$ architecture.

### `hookerrors/filters.py` (The Oracle Strategies)
This module evaluates whether a specific hook error split $E$ satisfies the Strict Safety condition. 
*   **`AlgebraicStrategy`**: Uses a fast GF(2) Breadth-First Search over the syndrome table to instantly check if $w_{\text{min}}^* \le 1$. Solves the problem in strictly polynomial time without heuristics or ILP.

### `hookerrors/searchers.py` (The Pruning Strategies)
This module dictates which algebraic splits (subsets of a generator's support) are generated and evaluated.
*   **Complementary Deduplication:** An internal edge bipartitions the support $W$ into $A$ and $B$. Searchers typically cap evaluations and filter out complementary subsets to exactly halve the search space.
*   **`ExhaustiveSearcher`**: Evaluates all subsets up to size $w-1$.
*   **`EarlyExitSearcher`**: A greedy searcher that stops evaluating a generator the moment it finds `max_splits` valid safe splits. Critical for synthesizing large codes.

### `hookerrors/combinations.py` (Global Assignment)
Computes a **Globally Safe Assignment**.
*   **Maximal Multi-Splittings (Chains):** A single generator can be split into multiple pieces (e.g., partitioning a degree-8 spider into 5 smaller spiders). This corresponds to a chain of nested safe hook errors (e.g., $S_1 \subset S_2 \subset S_3 \dots$). The algorithm finds the longest valid chain of nested safe splits for each generator.
*   **$\mathcal{O}(N)$ Assembly:** Because all allowed individual splits are strictly safe ($w_{\text{min}}^* \le 1$), their combinations are mathematically guaranteed to be safe by the linearity of the GF(2) vector space. The global assignment is constructed instantly by uniting the maximal chains of each generator, bypassing exponential combination checks.

### `hookerrors/solver.py` (The CLI)
The main executable script tying the system together. 
*   Dynamically loads the chosen strategy and searcher based on CLI arguments.
*   Constructs the X and Z cosets independently based on the `--basis` argument.
*   Example: `python hookerrors/solver.py --code 23_1_7 --method algebraic --searcher exhaustive`

---

## 4. Notable Code Behaviors
1.  **Asymmetric Safety (Basis Dependence):** If `--basis Z` is chosen, the target state is $|0\rangle_L$, so $L_z \in \mathcal{S}_{\text{prep}}$. As a result, Z-type hook errors can leverage the logical operator to reduce their weight, while X-type hook errors cannot.
2.  **Formatting:** Subsets and generators are parsed into human-readable tuples, e.g., `"X(4, 5, 6, 7)"`.
3.  **Triviality Filter:** The solver automatically drops assignments if a generator can only be split trivially (e.g. into size 1 chunks), as this corresponds to physically pulling off single qubits rather than meaningfully shattering the spider body.

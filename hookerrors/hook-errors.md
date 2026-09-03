# Algebraic Stabilizer Splitting and Safe Hook Errors

## 1. Context and Theoretical Background
This project aims to synthesize Fault-Tolerant (FT) state preparation circuits using the ZX-calculus. The paradigm is **FT-by-construction** using diagrammatic rewrites, meaning we are strictly optimizing the topology of a tensor network.

In a ZX-diagram representing state preparation:
*   Boundary nodes represent the physical data qubits.
*   High-degree spiders (stabilizer generators) must be split into lower-degree spiders to ensure the graph can be extracted into a compilable circuit.
*   **The Hook Error:** If a spider is split, an internal edge is created. A single physical fault on this internal edge propagates to a multi-qubit Pauli error on the data block. 

## 2. Mathematical Definition of a "Safe" Hook
We operate under a fault-tolerant distance $d = 2t + 1$. 

1.  **The Physical Fault ($f$):** An original physical error occurring on an internal ZX edge. By definition, $\text{weight}(f) = 1$.
2.  **The Propagated Error ($E$):** The resulting multi-qubit Pauli operator on the boundary data block.
3.  **The Preparation Stabilizers ($\mathcal{S}_{\text{prep}}$):** Because the diagram prepares specific logical states, we possess additional stabilizers beyond the code group. For example, when preparing $|0\rangle_L$, the state is stabilized by both the Z-stabilizers $H_Z$ and the logical Z operator $L_Z$. Thus, $\mathcal{S}_{\text{prep}} = \langle H_Z, L_Z \rangle$.

We define the equivalent physical weight $w_{\text{min}}^*$:
$$ w_{\text{min}}^* = \min_{S \in \mathcal{S}_{\text{prep}}} \text{weight}(E \cdot S) $$

### The Strict Safety Condition
A hook error is classified as **Safe** if and only if it acts exactly as a weight-1 fault on the target state:
$$ w_{\text{min}}^* \le 1 $$

If this condition holds, the hook error is harmless because it is algebraically identical (modulo the state symmetries) to a single physical fault, meaning no dangerous weight amplification has occurred.

## 3. Combinations of Safe Hooks (The Linearity Guarantee)
If a circuit contains multiple internal edges, an adversary with a budget of $t$ faults might trigger multiple hook errors simultaneously.

However, because equivalence is defined over a vector space (the row space of $\mathcal{S}_{\text{prep}}$ over GF(2)), the equivalent weight metric satisfies the triangle inequality:
$$ w_{\text{min}}^*(E_1 \oplus E_2) \le w_{\text{min}}^*(E_1) + w_{\text{min}}^*(E_2) $$

If the adversary triggers $k$ faults anywhere in the circuit, they trigger $k$ hook errors. The total combined error is $E_{\text{comb}} = \bigoplus_{i=1}^k E_i$. 
Because we strictly enforced that every individual split has $w_{\text{min}}^*(E_i) \le 1$, we get:
$$ w_{\text{min}}^*(E_{\text{comb}}) \le \sum_{i=1}^k 1 = k $$

**Conclusion:** $k$ physical faults in the circuit will always algebraically map to $\le k$ physical faults on the data block. They can never conspire to jump the weight up to a logical failure. **Combinations of strictly safe hook errors are mathematically guaranteed to be safe.**

## 4. Software Implementation ($\mathcal{O}(N)$ Scalability)
Because combinations are guaranteed safe by linearity, the synthesis software does not need to cross-check combinations of hook errors across different generators.

The solver evaluates the circuit in strictly $\mathcal{O}(N)$ time:
1.  **GF(2) Evaluation:** For each generator, evaluate all possible algebraic splits (subsets of the support). Use a fast GF(2) null-space check to determine if $w_{\text{min}}^* \le 1$.
2.  **Independent Assignment:** For each generator, independently find the longest nested chain of safe splits (the maximal multi-splitting).
3.  **Global Assignment:** Return the exact union of these chains. It is mathematically guaranteed to be a globally safe assignment.

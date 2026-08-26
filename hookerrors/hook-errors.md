# Implementation Plan: Algebraic Stabilizer Splitting and Safe Hook Errors

## 1. Context and Theoretical Background
This project aims to synthesize Fault-Tolerant (FT) state preparation circuits using the ZX-calculus. The paradigm is **FT-by-construction** using diagrammatic rewrites, meaning we are strictly optimizing the topology of a tensor network rather than time-scheduling CNOT gates.

In a ZX-diagram representing state preparation:
*   Boundary nodes represent the physical data qubits.
*   High-degree spiders (stabilizer generators) must be split into lower-degree spiders (e.g., degree-3) to ensure the graph can be extracted into a compilable circuit.
*   **The Hook Error:** If a spider is split, an internal edge is created. A single physical fault on this internal edge propagates to a multi-qubit Pauli error on the data block. This is our "hook error".

## 2. Mathematical Definition of a "Safe" Hook

We operate under a fault-tolerant distance $d = 2t + 1$. The system must track the evolution of a physical fault through the following exact definitions:

1.  **The Physical Fault ($f$):** An original physical error occurring on an internal ZX edge. By definition, $\text{weight}(f) = 1$.
2.  **The Propagated Error ($E$):** The resulting multi-qubit Pauli operator on the boundary data block after $f$ propagates. Its weight is denoted as $w_{\text{prop}} = \text{weight}(E)$.
3.  **The Code-Level Minimum Weight ($w_{\text{min}}$):** The minimum weight representative of $E$ up to the standard code stabilizers ($\mathcal{S}_{\text{code}}$).
    $$ w_{\text{min}} = \min_{S \in \mathcal{S}_{\text{code}}} \text{weight}(E \cdot S) $$

**Definition of a Hook Error:** If $w_{\text{min}} > 1$, the propagated error $E$ has undergone weight amplification. It is classified as a Hook Error. 

By default, an FT state preparation library must flag all hook errors. However, a hook error $E$ can be reclassified as **Safe** if it satisfies one of two sequential "Saving Graces".

## 2. The Two Saving Graces (Classification of Safe Hooks)

### Saving Grace 1: Trivially Safe Hooks (State Symmetries)
Because the diagram prepares specific logical states, we possess additional stabilizers beyond the code group. Let $\mathcal{S}_{\text{state}}$ be the logical operators stabilizing the specific target state (e.g., $Z_L$ for a logical $|0\rangle_L$). The complete preparation stabilizer group is $\mathcal{S}_{\text{prep}} = \langle \mathcal{S}_{\text{code}}, \mathcal{S}_{\text{state}} \rangle$.

We define the state-level minimum weight $w_{\text{min}}^*$:
$$ w_{\text{min}}^* = \min_{S \in \mathcal{S}_{\text{prep}}} \text{weight}(E \cdot S) $$

*   **Condition:** If we can reduce the weight such that $w_{\text{min}}^* \le 1$, the hook error acts exactly as a weight-1 fault on the target state. 
*   **Classification:** It is a **Trivially Safe Hook**. The system does not need to flag it.

### Saving Grace 2: Decoder-Benign Hooks (Perfect Decoding)
If $w_{\text{min}}^* > 1$, the hook error genuinely consumes a portion of the code's distance. It can only be saved by the decoding bounds. 

The original physical fault $f$ consumed 1 fault from the adversarial budget of $t$. The adversary has $t-1$ physical faults remaining to inject ambient weight-1 errors ($E_{\text{amb}}$). For $E$ to be safe, a minimum-weight decoder must correctly identify the combined error $E \cdot E_{\text{amb}}$. The decoder will guess an error $E_{\text{dec}}$ of weight $\le t$.

A logical failure occurs if the combined physical and decoded errors form a non-trivial logical operator $L \in \mathcal{L} \setminus I$:
$$ E \cdot E_{\text{amb}} \cdot E_{\text{dec}} = L \pmod{\mathcal{S}_{\text{prep}}} $$

The maximum combined weight of the remaining faults and the decoder's guess is $(t-1) + t = 2t - 1$.
*   **Condition:** For $E$ to be safe, the minimum weight of $E$ multiplied by any non-trivial logical operator and stabilizer must exceed this bound:
    $$ \min_{L \in \mathcal{L} \setminus I, S \in \mathcal{S}_{\text{prep}}} \text{weight}(E \cdot L \cdot S) \ge 2t $$
*   **Classification:** If this holds, $E$ is a **Decoder-Benign Hook**. Despite having weight $> 1$, the decoder can still tolerate up to $t-1$ additional faults. It is safe.

## 3. Combinations of Safe Hooks
Safe hook errors do not form a basis, and their combinations are highly non-linear. The safety of individual hooks $h_1$ and $h_2$ does not guarantee the safety of $h_1 \cdot h_2$. 

If a specific combination of $k$ independent safe hooks (where $k \le t$) occurs, they consume $k$ physical faults. The adversary has $t-k$ faults remaining. The maximum weight of the remaining faults plus the decoder's guess is $(t-k) + t = 2t - k$.
*   **Combination Condition:** A specific combination of $k$ hooks $\{h_1, \dots, h_k\}$ is safe if and only if:
    $$ \min_{L \in \mathcal{L} \setminus I, S \in \mathcal{S}_{\text{prep}}} \text{weight}\left(\prod_{i=1}^k h_i \cdot L \cdot S\right) \ge 2t - k + 1 $$

## 4. Software Implementation Architecture
The software component evaluating these conditions will face NP-hard scaling if it relies on brute-force enumeration of static lookup tables. The solver must implement a **Tiered Oracle** to evaluate proposed algebraic splits lazily:

1.  **Tier 1 Filter (GF(2) Reduction):** Evaluates *Saving Grace 1*. Uses Gaussian elimination over the binary symplectic representation of $\mathcal{S}_{\text{prep}}$ to determine if $w_{\text{min}}^* \le 1$.
2.  **Tier 2 Filter (Heuristic Bound):** Evaluates *Saving Grace 2*. Feeds $(E \cdot L)$ to a fast heuristic decoder (e.g., BP-OSD). If the returned minimum error chain is $\le 2t - 1$, the hook is definitively malignant. 
3.  **Tier 3 Filter (Exact Boolean Solver):** Evaluates *Saving Grace 2*. Uses a boolean MaxSAT/ILP solver (e.g., PySAT) to rigorously prove the minimum weight is $\ge 2t$ for the hooks that pass Tier 2.

## 5. Component Output
The software should return a mapping of each generator to its set of valid safe splits. *(Note: subsets are strictly drawn from their parent generator).*

```json
{
  "X1X2X3X4": ["X1X2"],
  "X4X5X6X7X8X9": ["X4X5", "X4X5X6"],
  "Z1Z2Z3Z4": ["Z1Z2"],
  "Z10Z11Z12Z13": []
}
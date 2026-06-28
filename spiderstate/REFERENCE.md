# Fault Tolerant Circuit Synthesis (ZX) - Reference Guide

This document is a comprehensive guide for LLMs working on this project. It outlines the project objectives, architecture, environment requirements, and core entry points for Fault Tolerant State Preparation (FTSP) of CSS quantum error-correcting codes.

## 1. Environment Requirements

**CRITICAL INSTRUCTION FOR LLMS:** 
All Python scripts and tests in this project must be executed using the `zxlive` Conda environment. All dependencies are already installed.

To run a script from the terminal, ensure you run using the environment's python executable:
```bash
conda run -n zxlive python <script_name.py>
```

## 2. Project Overview & Objective

The primary aim of this project is to perform **Fault Tolerant State Preparation (FTSP) of Quantum Error Correcting Codes (specifically CSS states)**.

The strategy revolves around preparing a state based on a parity check matrix $H$ (representing X or Z stabilizers). The workflow is roughly as follows:
1. **Matrix Requirements**: The matrix must represent a bipartite graph state. This is computationally verified using the `has_unique_ones_property`.
2. **Matrix Operations**:
   - **Row Operations**: Correspond to changing the basis. These are "free" operations.
   - **Column Operations**: Correspond to appending CNOT gates *after* the prepared state.
3. **Verification**: The CNOT gates from column operations are assumed to be perfect. Therefore, to ensure the state preparation is truly Fault Tolerant, any faults (up to weight $t$) stemming from these column-operation CNOTs **must be detected**. Alternatively, a state preparation is also considered Fault Tolerant if a fault evades internal flags but produces a bounded residual data syndrome that requires an equivalent or lesser weight to correct ($W_{residual} \leq W_{init}$).
4. **Optimization**: The goal is to find a sequence of column operations (CNOTs) and a good set of stabilizers to measure such that the overall cost is minimized.

## 3. Code Architecture & Modules

The project is split into two main packages, `spidercat` and `spiderstate`.
- **`spidercat`**: Originally written for fault-tolerant preparation of CAT states. It contains useful utilities, but active development and CSS state FTSP work should **not** happen here.
    *   `fao_se_circuit`: Returns a fault-tolerant measurement of a stabilizer up to $t$ faults. It modifies the GHZ state preparation scheme from Flag-at-Origin.
- **`spiderstate`**: The active package where the FTSP project resides. It utilizes `spidercat` under the hood.

### Key Components in `spiderstate`

*   **Matrix Optimization (`spiderstate.optimize_parity_matrix`)**
    *   `has_unique_ones_property`: Guarantees the matrix represents a valid bipartite graph.
    *   `cnot_cost`: Calculates the cost of preparing a given state in constant time.
    *   This module handles row and column operations to find a suitable matrix that minimizes the overall FTSP cost.

*   **Heuristic Verification Cost (`spiderstate.fast_verification`)**
    *   `TrueBackwardTracker`: A highly efficient, exact physical fault tracker used during the matrix optimization phase. By "prepending" CNOTs (building the circuit backwards), it maintains the true physical faults by updating the Heisenberg unitary operator $U$ in $O(c)$ time per step. It feeds these *true* faults into a greedy set-cover based on specific matrix heuristics:
        *   **`overlap` (Baseline)**: Penalizes faults with a syndrome weight $\ge 2$ (support intersection modulo 2) against the candidate stabilizer basis.
        *   **`zero_tolerance`**: Strictly penalizes *any* fault with a syndrome weight $\ge 1$. This encourages the simulated annealing to find inherently robust circuits where faults self-cancel or become full stabilizers.
        *   **`weighted_syndrome`**: Penalizes faults if the physical CNOT cost of the stabilizers they trigger is above the mean.

*   **Stabilizer Finding (`spiderstate.verification`)**
    *   `TrackedFaultSet`: Uses a Detector Error Model (DEM)-like approach to track exactly how physical faults are formed from lower-weight combinations. It replaces the naive `PureFaultSet` by computing minimum-weight representatives and preserving the generation history (`ways_to_form`), allowing for algebraic, higher-order analysis of error propagation before verification.
    *   `find_lookahead_verification_stabilizers`: Once column operations are chosen, this module finds a very efficient set of stabilizers to measure. It uses a beam search and greedy set-cover to operate in layers (detecting 1, 2, ... $t$ faults progressively). To respect intra-layer `overlap <= 2` constraints while achieving target coverages $\ge 2$, it tracks unresolved faults and fulfills their total required coverage across multiple temporal layers. **Crucial Rule**: Basis cross-verification is required. To detect Z-faults, the algorithm must measure X-stabilizers, and to detect X-faults, it must measure Z-stabilizers.

*   **Scheduling (`spiderstate.cnot_scheduler` & `spiderstate.circuit_merger`)**
    *   `cnot_scheduler`: Uses the Z3 SMT solver to determine the precise time-ordering of data-ancilla CNOTs for a layer of stabilizer measurements. It orders the CNOTs to prevent internal faults from maliciously cascading and cancelling out their own syndromes. If no safe ordering exists mathematically, it uses soft constraints to find an optimal schedule that requires the minimum number of physical flags, returning these "violations" for later injection.
    *   `circuit_merger`: Acts as the final assembler. It takes the scheduled CNOTs and chosen stabilizers, injects physical flags around unschedulable data CNOTs, and merges everything into the final `stim.Circuit`.
    *   *(Note: `spiderstate.circuit_finder` and `spiderstate.between_shor_and_steane` contain methods for synthesizing particular syndrome measurement circuits, but they are not necessary for the main pipeline. We use `cnot_scheduler` instead.)*

*   **Fast Fault Analysis (`spiderstate.fast_faults`)**
    *   `FastFaultSet`: A wrapper of `PureFaultSet` that uses a lookup table and caching instead of a SAT solver for much faster execution.

*   **State Generation (`spiderstate.cat_at_origin`)**
    *   `cat_at_origin`: A "black box" method that takes the parity check matrix and distance, returning a `stim.Circuit` that always produces a provably FT state for any CSS state (though not necessarily optimally). -> cnot_cost and ancilla_cost calculates the cost for this method.
    *   **`cat_at_origin_with_verification`**: The **primary end-to-end execution method** that combines state preparation and verification into a complete protocol.

## 4. Testing and Execution

To run end-to-end generation and evaluate if the generated circuits are genuinely fault-tolerant:
- Use **`spiderstate.fault_tolerance_verification.py`**. 
- This module is the main testing ground. You can call it with a code name, and it will execute the generation pipeline and verify the fault tolerance of the output circuits.
- **The Core Verifier (`verify_ftsp`):** This script uses the orchestrator, `verify_ftsp`, which coordinates verification of X and Z faults: 
  - **Primary Basis (ILP):** The primary basis verification relies on the `mip` library to solve an exact Integer Linear Program (ILP). It evaluates if there exists a catastrophic cascade where `W(E_init) + W(E_data) <= d - 1`.
  - **Conjugate Basis (Combinatorial BFS):** Uses a fast bitwise breadth-first search against a pre-computed Maximum Likelihood Decoder dictionary to evaluate uncorrectable syndrome mass.


**Example execution (using the correct environment):**
```bash
conda run -n zxlive python spiderstate/fault_tolerance_verification.py --code <code_name>
```

*(Note: Verify the specific arguments needed for the tester, but `fault_tolerance_verification.py` is the entry point for validation.)*

## 5. Current Implementation Status & Benchmarks

*   **Architecture Note on Heuristics**: Because the `DynamicCoverageTracker` uses a loose overlap heuristic, it intentionally passes highly entangled candidate matrices to the verification module to keep CNOT counts extremely low. While this works beautifully for smaller codes (e.g., yielding records like 40 CNOTs for 12_2_4), for higher distance codes (distance > 5), the heuristic can occasionally under-penalize complex faults to a degree where the subsequent lookahead verification module struggles to find a fully valid stabilizer set without scheduling violations.
*   **Baseline Method**: Running the pipeline with 0 column operations corresponds directly to pure `cat_at_origin`.
*   **Benchmarks**: Below are some reliable tester configurations and expected performance metrics:
    *   **7_1_3**: Best FT result uses 11 CNOT gates.
    *   **9_1_3**: Best FT result uses 11 CNOT gates.
    *   **15_7_3**: Best FT result uses 28 CNOT gates.
    *   **12_2_4**: Best FT result uses 40 CNOT gates.
    *   **16_6_4**: Best FT result uses 63 CNOT gates.
    *   **17_1_5**: Best FT result uses 57 CNOT gates.
    *   **19_1_5**: Best FT result uses 86 CNOT gates.
    *   **20_2_6**: Takes about 1 to 3 minutes to run.
    *   **23_1_7**: Takes a similar time to 20_2_6 (1 to 3 minutes).

---
*If you are an LLM reading this file, use the above references to navigate the project's logic and remember to run everything inside the `zxlive` conda environment.*

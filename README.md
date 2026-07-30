
# SpiderCat: Optimal Fault-Tolerant Cat State Preparation

This repository contain the source code to generate $\textsf{CAT}$ states using the method described in https://arxiv.org/abs/2603.05391.

<img src="figures/spidercat.jpeg" align="left" width="300px" />

**Abstract of the Paper**

The ability to fault-tolerantly prepare $\textsf{CAT}$ states, also known as multi-qubit GHZ states, is an important primitive for quantum error correction. It is required for Shor-style syndrome extraction, and can also be used as a subroutine for doing fault-tolerant state preparation of CSS codewords. Existing approaches to fault-tolerant $\textsf{CAT}$ state preparations have been found using computationally expensive heuristics involving SAT solving, reinforcement learning, or exhaustive analysis.

In this paper, we constructively find optimal circuits for $\textsf{CAT}$ states in a more scalable way. In particular, we derive formal lower bounds on the number of CNOT gates required for circuits implementing $n$-qubit $\textsf{CAT}$ states that do not spread errors of weight at most $t$ for $1\leq t \leq 5$. We do this by using fault-equivalent rewrites of ZX-diagrams to reduce it to a problem of characterising certain 3-regular simple graphs. We then provide families of such optimal graphs for infinitely many values of $n$ and $t\leq5$.

By encoding the construction of optimal graphs as a constraint satisfaction problem we find explicit constructions for circuits that match this lower bound on CNOT count for all $n\leq50$ and $t \leq 5$ and for nearly all pairs $(n,t)$ with $n\leq 100$ and $t\leq 5$ or $n\leq 50$ and $t\leq 7$, significantly extending the regimes that were achievable by previous methods and improving the resource counts for existing constructions. We additionally show how to trade CNOT count against depth, allowing us to construct constant-depth fault-tolerant implementations using $O(n)$ ancilla and $O(n)$ CNOT gates.

## Stabilizer-State ZX Pipeline

The `spiderstate` package also exposes a staged, diagram-level synthesis API:

```python
from spiderstate import synthesize_stabilizer_state

logical_zero_5q = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
    "ZZZZZ",
]

result = synthesize_stabilizer_state(logical_zero_5q, t=1)
result.render_svg(stage="final", path="five_qubit_unidealized.svg")
pyzx_graph = result.to_pyzx(stage="final")
```

The result retains the optimized LC-equivalent NetworkX graph, the exact
local-Clifford certificate, the ideal graph-state ZX diagram, the post-Lemma-B*
diagram, the final verified trivalent SpiderCat diagram, and stable
source-to-port provenance.  The certificate convention is
$U\lvert\psi\rangle=\lvert G\rangle$; the inverse corrections are kept on the
output boundary, so every returned diagram denotes the original input state
up to a nonzero global scalar.

Fault tolerance `t` is explicit and inclusive.  The diagram provider supports
verified constructions for `t=1..7`; it raises a typed error if the corrected
marked-cut check rejects a cached candidate or no verified gadget is
available.

Create the pinned Python 3.12 development environment and run the tests with:

```shell
uv sync --python python3.12
.venv/bin/pytest
```

## Repository Structure

The core code for generating $\textsf{CAT}$ states is located in the `spidercat` folder. This folder contains the implementation of the methods, as well as scripts used to obtain new circuits, simulations, and visualizations.

### Core Methods:
 - `benchmark.py`: Benchmarks different spanning forests.
 - `circuit_extraction.py`: Contains the code necessary to extract a circuit from a graph, a spanning forest, roots for the trees in the forest, markings on the graph, and perfect matchings from leaves to markings. Note that the paper primarily describes methods using a spanning tree, in which case the spanning forest has a single component.
 - `draw.py`: Functions to visualize the data structures used for $\textsf{CAT}$ state extraction.
 - `graphs_amsterdam.py`: An experimental method for generating high-$t$ $\textsf{CAT}$ states by construction.
 - `graphs_circular.py`: Methods for generating solutions based on Hamiltonian graphs.
 - `graphs_random.py`: Implements the hill-climbing algorithm described in the paper to generate graphs with no non-local cuts.
 - `markings.py`: Methods to find valid markings for a given $t$ on a graph.
 - `nonlocal_cut.py`: A SAT-solver-based approach to quickly check for the presence of a non-local $t$-cut.
 - `path_cover.py`: An experimental method to extract circuits using path covers or Hamiltonian paths via SAT solvers.
 - `spanning_tree.py`: Functions for finding spanning trees and forests as described in the paper.
 - `utils.py`: General utility functions.

The general stabiliser-state pipeline lives in `spiderstate`:

 - `stabiliser_decomposition.py`: Decomposes a bipartite CSS-state parity matrix into a validated, circuit-independent graph IR.
 - `circuit_extraction.py`: Lowers that IR to a circuit, reports measured resources, and selects the best circuit from a decomposition portfolio.
 - `cat_at_origin.py`: Preserves the existing public API as a thin orchestrator over those two modules.

See `docs/stabiliser_state_pipeline.md` for the module contract, resource-aware
selection loop, and staged verification design.

### Scripts:
 - `generate.py`: (Re)generates $\textsf{CAT}$ states of various sizes $n$, distances $t$, and spanning forest components $p$. Generated circuits are saved to the `circuits` folder; the intermediate data structures are stored in `circuits_data`.
 - `simulate.py`: Automatically generates simulation data using Stim, utilizing the circuits in the `circuits` folder. Results are saved in `simulation_data`.
 - `visualise.py`: Generates visualizations based on the simulation data.

### Notebooks:
Some demos and experimental ideas can be found in the `notebooks` folder:
 - `cat_state_density_lower_bound.ipynb`: Simple implementation of the lower bounds on the number of CNOTs and flags required to implement a $\textsf{CAT}$ state.
 - `circuit_extraction_demo.ipynb`: A demonstration of the circuit extraction process.
 - `decoding.ipynb`: A proof-of-concept implementation and test of decoding using Tesseract.

## Phase-Free Stabiliser-State Synthesis

`spiderstate.zx_synthesis` extracts a simulation-ready Stim circuit from a
phase-free PyZX diagram whose internal X/Z spiders have degree three. Choose
between a CNOT-count-first layout and a depth-first layout:

```python
from spiderstate.zx_synthesis import synthesize_zx

result = synthesize_zx(diagram, strategy="gate_count")
stim_circuit = result.circuit

depth_result = synthesize_zx(diagram, strategy="depth")
depth_result.circuit.to_file("state_preparation.stim")
```

The result also reports input/output qubit maps, resource metrics, detector and
feedback metadata, the selected half-edge roles, and whether the bounded Z3
layout optimization proved optimality. Pure state-preparation diagrams use the
spider/Bell rewrite catalogue directly. Diagrams with live inputs are passed
through PyZX's causal-flow extractor and then translated to the same Stim gate
set. By default, diagrams with at most 24 internal spiders receive a
10-second exact-optimization attempt; larger instances and timeouts use the
deterministic seeded multi-start heuristic. Pass `optimizer="heuristic"` to
select that fallback directly.

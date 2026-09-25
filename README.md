
# SpiderCat: Optimal Fault-Tolerant Cat State Preparation

This repository contain the source code to generate $\textsf{CAT}$ states using the method described in https://arxiv.org/abs/2603.05391.

<img src="figures/spidercat.jpeg" align="left" width="300px" />

**Abstract of the Paper**

The ability to fault-tolerantly prepare $\textsf{CAT}$ states, also known as multi-qubit GHZ states, is an important primitive for quantum error correction. It is required for Shor-style syndrome extraction, and can also be used as a subroutine for doing fault-tolerant state preparation of CSS codewords. Existing approaches to fault-tolerant $\textsf{CAT}$ state preparations have been found using computationally expensive heuristics involving SAT solving, reinforcement learning, or exhaustive analysis.

In this paper, we constructively find optimal circuits for $\textsf{CAT}$ states in a more scalable way. In particular, we derive formal lower bounds on the number of CNOT gates required for circuits implementing $n$-qubit $\textsf{CAT}$ states that do not spread errors of weight at most $t$ for $1\leq t \leq 5$. We do this by using fault-equivalent rewrites of ZX-diagrams to reduce it to a problem of characterising certain 3-regular simple graphs. We then provide families of such optimal graphs for infinitely many values of $n$ and $t\leq5$.

By encoding the construction of optimal graphs as a constraint satisfaction problem we find explicit constructions for circuits that match this lower bound on CNOT count for all $n\leq50$ and $t \leq 5$ and for nearly all pairs $(n,t)$ with $n\leq 100$ and $t\leq 5$ or $n\leq 50$ and $t\leq 7$, significantly extending the regimes that were achievable by previous methods and improving the resource counts for existing constructions. We additionally show how to trade CNOT count against depth, allowing us to construct constant-depth fault-tolerant implementations using $O(n)$ ancilla and $O(n)$ CNOT gates.

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
 - `mdsf.py`: Algorithms for minimum degree/diameter spanning forests and approximate k-centers.
 - `nonlocal_cut.py`: A SAT-solver-based approach to quickly check for the presence of a non-local $t$-cut.
 - `path_cover.py`: An experimental method to extract circuits using path covers or Hamiltonian paths via SAT solvers.
 - `spanning_tree.py`: Functions for finding spanning trees and forests as described in the paper.
 - `syndrome_measurement.py`: Methods to build and extract syndrome measurement circuits for fault-tolerant error correction.
 - `utils.py`: General utility functions.

### Scripts:
 - `generate.py`: (Re)generates $\textsf{CAT}$ states of various sizes $n$, distances $t$, and spanning forest components $p$. Generated circuits are saved to the `circuits` folder; the intermediate data structures are stored in `circuits_data`.
 - `simulate.py`: Automatically generates simulation data using Stim, utilizing the circuits in the `circuits` folder. Results are saved in `simulation_data`.
 - `visualise.py`: Generates visualizations based on the simulation data.

### Notebooks:
Some demos and experimental ideas can be found in the `notebooks` folder:
 - `cat_state_density_lower_bound.ipynb`: Simple implementation of the lower bounds on the number of CNOTs and flags required to implement a $\textsf{CAT}$ state.
 - `circ_to_dag.ipynb`: Converts circuits to DAG representations for analysis.
 - `circuit_extract_demo.ipynb`: A demonstration of the circuit extraction process.
 - `decoding.ipynb`: A proof-of-concept implementation and test of decoding using Tesseract.
 - `demo.ipynb`: General demonstration of the repository's capabilities.
 - `estimate_logical_error_rate.ipynb`: Estimates logical error rates for extracted circuits.
 - `random.ipynb`: Experiments with random graph constructions.
 - `shor_syndrome_measurement.ipynb`: Explores Shor-style syndrome measurement.
 - `syndrome_measurement.ipynb`: Demonstrates syndrome measurement techniques.
 - `test_split_cat.ipynb`: Tests for splitting cat states.
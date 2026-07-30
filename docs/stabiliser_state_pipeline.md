# Stabiliser-state decomposition and extraction pipeline

## Decision

Keep decomposition and circuit extraction as separate modules, joined by a
versioned intermediate representation (IR):

```text
parity matrix H, distance d
          |
          v
stabiliser_decomposition.py
  - select pivots and CAT components
  - select roots, primary paths, and leg matchings
  - assemble global graph, forest, and dependency DAG
  - validate structural invariants
          |
          v
StabiliserStateDecomposition (schema v1)
          |
          v
circuit_extraction.py
  - lower the IR with a circuit backend
  - measure actual resources
  - rank a portfolio by an ExtractionPolicy
          |
          v
ExtractionResult(circuit, resources, provenance)
          |
          v
ideal-state check -> exact fault-tolerance check
```

`spiderstate.cat_at_origin.cat_at_origin` is a compatibility orchestrator. It
does not contain either algorithm: it calls the two modules and returns the
Stim circuit as before.

## Module contracts

### 1. Decomposition

`decompose_stabiliser_state(H, d)` returns a
`StabiliserStateDecomposition`. It contains:

- the target parity matrix, distance, and fault budget;
- the assembled ZX graph and spanning forest;
- one root and one primary path for each CAT-state component;
- a global acyclic dependency graph;
- component-to-matrix provenance;
- every inter-component CNOT coupling; and
- a candidate ID and schema version.

Its public contract contains no Stim objects and it creates no circuit. (Some
legacy graph-generation utilities still import from `spidercat` internally.)
It owns all choices whose validity depends on the state decomposition: pivot
columns, component sizes, CAT graphs, roots, primary paths, and cycle-free leg
matching.

The default validator checks that the graph objects have consistent node sets,
the forest and dependency graph are acyclic, roots and paths are well formed,
and every coupling joins a Z spider to an X spider. The optional strict target
check also enforces the current `CatStateExtractor` convention that the number
of remaining marked nodes equals the number of data qubits.

### 2. Circuit extraction

`extract_stabiliser_state(decomposition, policy=...)` accepts only the IR. It
does not inspect the original parity matrix to reconstruct decomposition
choices. It returns an `ExtractionResult` containing:

- the circuit;
- data, ancilla, and total qubit counts;
- physical two-qubit-gate count (classical feedback is excluded);
- measurement and detector counts;
- scheduled tick depth and dependency-DAG depth;
- spacetime volume; and
- the decomposition candidate ID and schema version.

`extract_best_stabiliser_state(candidates, policy=...)` compiles every
candidate and selects the lowest lexicographic resource score. This is the
feedback point between state structure and resource efficiency, without
coupling the two algorithms.

## Why decomposition structure affects resources

| Structural choice | Extraction effect |
|---|---|
| CAT component graph and marks | Physical CNOT and flag-ancilla count |
| Root of each forest component | Traversal height and circuit depth |
| Primary root-to-main path | Which branch reuses a data wire and which branches allocate new wires |
| Leg matching between Z and X components | Cross-component CNOT placement, flag reuse, and whether the dependency graph stays acyclic |
| Dependency-DAG width and critical path | Available parallelism and tick depth |
| Component order | Deterministic data-qubit assignment and downstream routing opportunities |

The decomposition module may therefore emit a portfolio that varies roots,
primary paths, valid matchings, or CAT gadgets. The extraction module measures
the resulting circuits under a caller-selected objective such as:

```python
ExtractionPolicy(
    objective_order=(
        "two_qubit_gates",
        "ancilla_qubits",
        "tick_depth",
        "measurements",
    )
)
```

Using a lexicographic objective keeps the optimization interpretable. For
example, the policy above never trades an extra physical CNOT for a depth
reduction. A hardware-oriented run can put `space_time_volume` or
`ancilla_qubits` first.

## Verification gates

Verification is split into increasingly expensive gates so failures are local
and reproducible:

1. **Algebraic input validation** in decomposition: binary matrix, positive
   distance, and the unique-pivot property.
2. **IR structural validation** before extraction: forest, roots, paths,
   couplings, and dependency DAG.
3. **Ideal functional verification** after extraction: use a Stim tableau to
   check the target stabilisers and check that noiseless detectors are silent.
4. **Exact fault-tolerance verification** only for surviving candidates: call
   `spiderstate.fault_tolerance_verification.verify_ftsp` with the full code
   checks and logicals.

The candidate ID, schema version, target matrix, and resource manifest should
be stored with verification results. This allows structural and ideal checks
to be cached independently of exact fault enumeration, and makes a failing
circuit traceable to the decomposition that produced it.

## Extension points

- Add decomposition strategies that enumerate alternative roots, primary
  paths, and cycle-free matchings.
- Add extraction backends by implementing the existing `CircuitBuilder`
  interface.
- Add hard resource constraints before scoring, for example maximum ancillas
  or maximum tick depth.
- Cache IRs by `(schema_version, H, d, decomposition_strategy)` and exact
  verification by `(IR digest, extraction policy, verifier configuration)`.

The important constraint is that optimization feedback crosses the boundary
only through candidate IRs and resource manifests. The extraction module must
not mutate the decomposition or silently choose new pivots/matchings, and the
decomposition module must not emit Stim instructions.

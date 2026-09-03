# Website Plan for "CSSCat"

This is a plan for a website that aim to support the paper titled "CSSCat: Scalable Fault-Tolerant CSS State Preparation."

The aims of the website are to
- promote the works developed in the paper
- support the explanation of the results of the paper through interactive visualisations
- be usable as a GUI to generate fault-tolerant CSS states


Thinking about the interactive nature of the website, one should be able to specify a CSS code through its:
- If it's the logical |0> or |+> state.
- stabilizers, inputted as:
  - binary H_x and H_z matrices, or
  - text-like stabilizers (e.g. IIIXXXX, IIZZIZZ, ...)
- distance.
One should also be able to pick from a list of pre-specified codes.
This is Phase 1.

==================================================

Then, for Phase 2 we want to find a good basis.
Here, there are two aims: (1) explain and (2) use as GUI.

When we want to explain the method, we should be able to see the steps as we:
1. (re-)generate a permutation
2. permute the rows of the matrix
3. perform Gaussian eliminations
4. un-permute the rows
From the final unpermuted matrix, we should see:
- the cost of the state preparation
- the bipartite ZX-diagram with idealized internal edges that corresponds to the matrix

When we want to generate a state, we should be able to set max_basis_tries, and the best matrix will be picked out of X many random trials from the above procedure.

==================================================

At this point, we have an appropriate matrix and a corresponding ZX-diagram. I want to have two options, potentially as buttons:
(a) [Show ZX-rewrites]: unidealize internal edges, then FE-unfuse spiders using the precomputed library.
This is the bit about the theory, but this is not how things are implemented in practice.
(b) [Show Pipeline]: shows how a circuit is extracted from the ZX-diagram. This is Phase 3.

==================================================

For Phase 3 we start with the initialized global data structures and find and perform the part that matches individual edges of spiders according to the connectivity.
Here, we should be able to step back and forth or press play as we add more and more edges to the global data structures.
I would want the diagraph and the graph displayed as they are being mutated, the "target" ZX-diagram and which of its edges have been added, and also a graph showing the backtracing.

==================================================

Phase 4 is circuit extraction. This should have the graph, the digraph, and the current state of the circuit displayed, and optionally the log printed.
Again, we should be able to step back and forth or press play as we traverse the graphs.
The currently processed node + edge should be highlighted.
Should have the option to export as stim.

==================================================

Phase 5, optionally, if there are simulation data about the selected code, that should be displayed after circuit extraction. 
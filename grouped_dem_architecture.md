# Grouped DEM Architecture & Implementation Plan

This document summarizes the transition from the current layer-independent verification approach to the new Detector Error Model (DEM)-based approach for Fault-Tolerant Circuit Synthesis.

## 1. Previous Implementation

In the current verification algorithm:
- **Independent Layers**:  
- **Simple Filtering**: Faults were strictly weight-1 and weight-2 elementary errors. They were filtered purely by their effective weight ($w_{eff} \ge 2$).
- **Naive Target Coverage**: The algorithm evaluated coverage independently. For example, it required a fault to be detected exactly 1 time in Layer 1, and 1 time in Layer 2. 
- **The Core Flaw**: This independent evaluation failed to accurately model reality. It could not track how multiple independent faults occurring in different spacetime locations combine. For instance, if fault $f_1$ occurs before Layer 1 and fault $f_2$ occurs before Layer 2, their combined effect is $f_1 \oplus f_2$. The old algorithm assumed all faults happen before the first layer, leading to blind spots where multi-fault combinations bypassed detection entirely.

## 2. User Requests & Ideas

The user correctly identified the blind spot in the legacy approach:
- **The Problem**: A multi-layer fault combination (e.g., 2 faults in the initial phase, plus 1 fault on the data qubit later) was bypassing detection because the last layer expected the previous layers to have caught the faults, oblivious to the fact that the faults were injected dynamically between layers.
- **The Proposed Solution (DEM)**: The user proposed building a data structure akin to a Detector Error Model (DEM) that inherently knows the fault-location (layer ID) and the fault vector.
- **Composition & Grouping**: To optimize, the user suggested calculating the cumulative effect of composed faults across layers. Since many faults have the identical physical effect when propagated, they can be merged into equivalence classes (groups). By tracking the "origins" of faults and their grouped effects, the set cover algorithm can precisely filter and detect realistic spacetime faults.

## 3. Our Ideas, Remarks, and Discoveries

Based on the user's concepts, we implemented the **Grouped DEM**:
- **Spacetime Faults**: We explicitly assigned timestamps to faults: `(layer_idx, fault_vector)`. We then generated all possible combinations up to size $t$ and computed their cumulative syndrome effects across all subsequent layers.
- **Coset Reduction**: To prevent combinatorial explosion, we mapped elementary faults to their coset leaders with respect to the conjugate basis stabilizers. This drastically reduced the number of unique fault vectors before combining them.
- **Global History Tracking**: The beam search was rewritten to track `det_counts[h]`—the total number of times a specific multi-fault combination `h` has been detected across *all* layers processed so far.

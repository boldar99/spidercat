import z3
import functools

@functools.lru_cache(maxsize=1024)
def _schedule_layer_cnots_cached(stabs_qubits_tuple: tuple[tuple[int, ...], ...]) -> list[list[tuple[int, int]]]:
    stabs_qubits = [list(q) for q in stabs_qubits_tuple]
    
    # Quick fast-path for empty or single stabilizer
    if not stabs_qubits:
        return []
    if len(stabs_qubits) == 1:
        return [[(0, q)] for q in stabs_qubits[0]]

    opt = z3.Optimize()
    
    # T[i, q] will store the tick (time step) for the CNOT between stabilizer i and qubit q
    T = {}
    for i, qubits in enumerate(stabs_qubits):
        for q in qubits:
            v = z3.Int(f"T_{i}_{q}")
            opt.add(v >= 0)
            T[(i, q)] = v
            
    # Constraint 1: A stabilizer can do at most 1 CNOT per tick
    for i, qubits in enumerate(stabs_qubits):
        for j1 in range(len(qubits)):
            for j2 in range(j1 + 1, len(qubits)):
                q1, q2 = qubits[j1], qubits[j2]
                opt.add(T[(i, q1)] != T[(i, q2)])
                
    # Constraint 2: A data qubit can be involved in at most 1 CNOT per tick
    qubit_to_stabs = {}
    for i, qubits in enumerate(stabs_qubits):
        for q in qubits:
            if q not in qubit_to_stabs:
                qubit_to_stabs[q] = []
            qubit_to_stabs[q].append(i)
            
    for q, stabs in qubit_to_stabs.items():
        for j1 in range(len(stabs)):
            for j2 in range(j1 + 1, len(stabs)):
                i1, i2 = stabs[j1], stabs[j2]
                opt.add(T[(i1, q)] != T[(i2, q)])
                
    # Constraint 3: If two stabilizers share EXACTLY 2 qubits, enforce the specific interlace pattern
    for i1 in range(len(stabs_qubits)):
        for i2 in range(i1 + 1, len(stabs_qubits)):
            shared = list(set(stabs_qubits[i1]) & set(stabs_qubits[i2]))
            if len(shared) == 2:
                q1, q2 = shared[0], shared[1]
                t11 = T[(i1, q1)]
                t12 = T[(i1, q2)]
                t21 = T[(i2, q1)]
                t22 = T[(i2, q2)]
                
                # The user's specific ordering: q1 in s1, q2 in s2, q1 in s2, q2 in s1
                # This corresponds to: t11 < t22 < t21 < t12
                # But since s1/s2 and q1/q2 are arbitrary relative to i1/i2/q1/q2,
                # any of the 4 symmetric orderings is valid:
                opt.add(z3.Or(
                    z3.And(t11 < t22, t22 < t21, t21 < t12),
                    z3.And(t12 < t21, t21 < t22, t22 < t11),
                    z3.And(t21 < t12, t12 < t11, t11 < t22),
                    z3.And(t22 < t11, t11 < t12, t12 < t21)
                ))

    # Minimize the maximum tick (circuit depth)
    max_tick = z3.Int("max_tick")
    for (i, q), v in T.items():
        opt.add(max_tick >= v)
        
    opt.minimize(max_tick)
    
    if opt.check() != z3.sat:
        raise RuntimeError("Failed to find a valid CNOT schedule!")
        
    m = opt.model()
    schedule = {}
    for (i, q), v in T.items():
        tick = m.evaluate(v).as_long()
        if tick not in schedule:
            schedule[tick] = []
        schedule[tick].append((i, q))
        
    ticks = []
    for t in sorted(schedule.keys()):
        ticks.append(schedule[t])
        
    return ticks

def schedule_layer_cnots(stabs_qubits: list[list[int]]) -> list[list[tuple[int, int]]]:
    """
    Schedules the CNOTs for a layer of stabilizers to prevent correlated errors.
    
    stabs_qubits[i] is the list of data qubits that stabilizer i interacts with.
    
    Returns a list of 'ticks'. Each tick is a list of (stab_idx, qubit) pairs
    that should be executed simultaneously.
    """
    stabs_qubits_tuple = tuple(tuple(q) for q in stabs_qubits)
    return _schedule_layer_cnots_cached(stabs_qubits_tuple)

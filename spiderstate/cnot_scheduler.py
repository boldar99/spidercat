import z3
import functools
from dataclasses import dataclass

@dataclass
class DangerousFault:
    D_E: frozenset[tuple[int, int]]
    T_E_Q: dict[tuple[int, ...], int]

def schedule_all_verification_layers(
    layers: list[list[list[int]]], 
    dangerous_faults: list[DangerousFault]
) -> list[list[list[tuple[int, int]]]]:
    all_scheduled_layers = []
    all_violations = []
    
    for L, stabs_qubits in enumerate(layers):
        if not stabs_qubits:
            all_scheduled_layers.append([])
            continue
        if len(stabs_qubits) == 1:
            all_scheduled_layers.append([[(0, q)] for q in stabs_qubits[0]])
            continue

        opt = z3.Optimize()
        
        T = {}
        qubit_to_stabs = {}
        for i, qubits in enumerate(stabs_qubits):
            for q in qubits:
                v = z3.Int(f"T_{i}_{q}")
                opt.add(v >= 0)
                T[(i, q)] = v
                if q not in qubit_to_stabs:
                    qubit_to_stabs[q] = []
                qubit_to_stabs[q].append(i)
                
        for i, qubits in enumerate(stabs_qubits):
            for j1 in range(len(qubits)):
                for j2 in range(j1 + 1, len(qubits)):
                    q1, q2 = qubits[j1], qubits[j2]
                    opt.add(T[(i, q1)] != T[(i, q2)])
                    
        for q, stabs in qubit_to_stabs.items():
            for j1 in range(len(stabs)):
                for j2 in range(j1 + 1, len(stabs)):
                    i1, i2 = stabs[j1], stabs[j2]
                    opt.add(T[(i1, q)] != T[(i2, q)])
                    
        all_flags = []
        flag_details = []
        flag_idx = 0
                    
        # Precompute subsets of stabilizers for quick XOR sum condition
        # For a tuple of qubits Q, stabs_Q is all stabs containing at least one q in Q
        for df in dangerous_faults:
            D_E = df.D_E
            abs_D_E = len(D_E)
            D_gt_L = { (l_idx, s_idx) for (l_idx, s_idx) in D_E if l_idx > L }
            D_eq_L_E = { s_idx for (l_idx, s_idx) in D_E if l_idx == L }
            
            for Q, req_T_E_Q in df.T_E_Q.items():
                syn_gt_L_Q = set()
                stabs_Q = set()
                
                for q in Q:
                    # Collect all stabs for Q in layer L
                    if q in qubit_to_stabs:
                        stabs_Q.update(qubit_to_stabs[q])
                    # Compute future syndrome for Q
                    for l_idx in range(L + 1, len(layers)):
                        for s_idx, st in enumerate(layers[l_idx]):
                            if q in st:
                                if (l_idx, s_idx) in syn_gt_L_Q:
                                    syn_gt_L_Q.remove((l_idx, s_idx))
                                else:
                                    syn_gt_L_Q.add((l_idx, s_idx))
                                    
                xor_set = D_gt_L.symmetric_difference(syn_gt_L_Q)
                M_E_Q = abs_D_E - len(D_gt_L) + len(xor_set) - req_T_E_Q
                
                # Check if it's possible to violate the constraint
                # Only stabilizers in D_eq_L_E that interact with Q can be removed from the detection pool
                max_possible_sum = 0
                w_E = {}
                for s_i in stabs_Q:
                    w = 1 if s_i in D_eq_L_E else -1
                    w_E[s_i] = w
                    if w > 0:
                        max_possible_sum += 1
                        
                if max_possible_sum > M_E_Q:
                    import itertools
                    Q_list = list(Q)
                    if len(Q_list) == 0:
                        # Empty Q shouldn't reach here if max_possible_sum > M_E_Q since stabs_Q is empty and sum is 0
                        continue
                        
                    # All possible injection points for each q in Q
                    # A fault can be injected BEFORE any stabilizer s_j that interacts with it
                    injection_choices = [qubit_to_stabs.get(q, [-1]) for q in Q_list]
                    
                    for J in itertools.product(*injection_choices):
                        sum_expr = []
                        for s_i in stabs_Q:
                            # Does an odd number of faults in Q trigger s_i?
                            triggers = []
                            for q_idx, q in enumerate(Q_list):
                                if q in layers[L][s_i]:
                                    triggers.append(T[(s_i, q)] >= T[(J[q_idx], q)])
                                    
                            if len(triggers) == 1:
                                sum_expr.append(z3.If(triggers[0], w_E[s_i], 0))
                            elif len(triggers) > 1:
                                sum_expr.append(z3.If(z3.Xor(*triggers), w_E[s_i], 0))
                                
                        if sum_expr:
                            flag = z3.Int(f"flag_{flag_idx}")
                            opt.add(flag >= 0)
                            opt.add(z3.Sum(sum_expr) <= M_E_Q + flag)
                            all_flags.append(flag)
                            flag_details.append({"layer": L, "Q": Q, "J": J, "D_E": D_E, "M_E_Q": M_E_Q})
                            flag_idx += 1

        max_tick = z3.Int("max_tick")
        for (i, q), v in T.items():
            opt.add(max_tick >= v)
            
        if all_flags:
            opt.minimize(1000 * z3.Sum(all_flags) + max_tick)
        else:
            opt.minimize(max_tick)
            
        opt.set("timeout", 2000) # 2 seconds timeout per layer
        
        status = opt.check()
        if status == z3.unsat:
            raise RuntimeError(f"Failed to find a valid CNOT schedule for layer {L} even with soft constraints!")
            
        try:
            m = opt.model()
        except z3.Z3Exception:
            raise RuntimeError(f"Z3 timed out and could not find any valid model for layer {L}!")
        
        for idx, flag in enumerate(all_flags):
            # Evaluate might return a Real or an uninterpreted expression if Z3 didn't fully resolve it,
            # but for Integer variables it should be safe.
            try:
                flag_val = m.evaluate(flag).as_long()
                if flag_val > 0:
                    detail = flag_details[idx]
                    detail["violation_amount"] = flag_val
                    all_violations.append(detail)
            except z3.Z3Exception:
                pass # If it couldn't evaluate, assume 0
        
        schedule = {}
        for (i, q), v in T.items():
            try:
                tick = m.evaluate(v).as_long()
                if tick not in schedule:
                    schedule[tick] = []
                schedule[tick].append((i, q))
            except z3.Z3Exception:
                raise RuntimeError(f"Failed to evaluate tick for stab {i} on qubit {q} after timeout.")
            
        ticks = []
        for t_tick in sorted(schedule.keys()):
            ticks.append(schedule[t_tick])
            
        all_scheduled_layers.append(ticks)
        
    return all_scheduled_layers, all_violations

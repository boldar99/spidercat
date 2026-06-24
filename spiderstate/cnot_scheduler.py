import z3
import functools
from dataclasses import dataclass

@dataclass
class DangerousFault:
    D_E: frozenset[tuple[int, int]]
    T_E_q: dict[int, int]

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
                    
        for q, stabs in qubit_to_stabs.items():
            for df in dangerous_faults:
                D_E = df.D_E
                req_T_E_q = df.T_E_q[q]
                
                abs_D_E = len(D_E)
                D_gt_L = { (l_idx, s_idx) for (l_idx, s_idx) in D_E if l_idx > L }
                
                syn_gt_L = set()
                for l_idx in range(L + 1, len(layers)):
                    for s_idx, st in enumerate(layers[l_idx]):
                        if q in st:
                            syn_gt_L.add((l_idx, s_idx))
                            
                xor_set = D_gt_L.symmetric_difference(syn_gt_L)
                M_E_q = abs_D_E - len(D_gt_L) + len(xor_set) - req_T_E_q
                
                if M_E_q < 0:
                    raise RuntimeError(f"Impossible to schedule: M_E_q = {M_E_q} for layer {L}, q {q}")
                    
                w_E = {s_i: 1 if (L, s_i) in D_E else -1 for s_i in stabs}
                max_possible_sum = sum(v for v in w_E.values() if v > 0)
                
                if max_possible_sum > M_E_q:
                    for s_j in stabs:
                        sum_expr = []
                        for s_i in stabs:
                            sum_expr.append(z3.If(T[(s_i, q)] >= T[(s_j, q)], w_E[s_i], 0))
                            
                        flag = z3.Int(f"flag_{flag_idx}")
                        opt.add(flag >= 0)
                        opt.add(z3.Sum(sum_expr) <= M_E_q + flag)
                        all_flags.append(flag)
                        flag_details.append({"layer": L, "q": q, "s_j": s_j, "D_E": D_E, "M_E_q": M_E_q})
                        flag_idx += 1

        max_tick = z3.Int("max_tick")
        for (i, q), v in T.items():
            opt.add(max_tick >= v)
            
        if all_flags:
            opt.minimize(1000 * z3.Sum(all_flags) + max_tick)
        else:
            opt.minimize(max_tick)
        
        if opt.check() != z3.sat:
            raise RuntimeError(f"Failed to find a valid CNOT schedule for layer {L} even with soft constraints!")
            
        m = opt.model()
        
        for idx, flag in enumerate(all_flags):
            flag_val = m.evaluate(flag).as_long()
            if flag_val > 0:
                detail = flag_details[idx]
                detail["violation_amount"] = flag_val
                all_violations.append(detail)
        
        schedule = {}
        for (i, q), v in T.items():
            tick = m.evaluate(v).as_long()
            if tick not in schedule:
                schedule[tick] = []
            schedule[tick].append((i, q))
            
        ticks = []
        for t_tick in sorted(schedule.keys()):
            ticks.append(schedule[t_tick])
            
        all_scheduled_layers.append(ticks)
        
    return all_scheduled_layers, all_violations

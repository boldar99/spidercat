import stim
from spiderstate.cnot_scheduler import schedule_layer_cnots

def synthesize_and_merge_layer(stabs_qubits: list[list[int]], t: int, ancilla_start: int, basis: str) -> tuple[stim.Circuit, int]:
    from spidercat.syndrome_measurement import bare_se_circuit, fao_se_circuit
    
    if not stabs_qubits:
        return stim.Circuit(), ancilla_start

    # 1. Schedule data CNOTs
    ticks = schedule_layer_cnots(stabs_qubits)
    
    # 2. Determine ordered qubits for each stabilizer
    ordered_qubits = [[] for _ in range(len(stabs_qubits))]
    for tick_ops in ticks:
        for stab_idx, q in tick_ops:
            ordered_qubits[stab_idx].append(q)
            
    # 3. Generate circuits
    meas_circs = []
    current_ancilla = ancilla_start
    for qubits in ordered_qubits:
        if t == 0:
            circ = bare_se_circuit(qubits=qubits, ancilla=current_ancilla, basis=basis)
        else:
            circ = fao_se_circuit(qubits=qubits[::-1], ancilla_start=current_ancilla, t=t, basis=basis)
            circ.append("DETECTOR", [stim.target_rec(-1)])
            
        # Unfold the circuit so every CX pair is a separate instruction
        unfolded = stim.Circuit()
        for inst in circ:
            if inst.name == "CX":
                targets = inst.targets_copy()
                for j in range(0, len(targets), 2):
                    unfolded.append("CX", [targets[j].value, targets[j+1].value])
            else:
                unfolded.append(inst)
                
        meas_circs.append(unfolded)
        current_ancilla = unfolded.num_qubits

    # 4. Merge via iterators
    merged = stim.Circuit()
    
    # Flatten instructions to avoid Stim's automatic folding
    flat_circs = []
    for circ in meas_circs:
        flat = []
        for inst in circ:
            if inst.name == "CX":
                targets = inst.targets_copy()
                for j in range(0, len(targets), 2):
                    flat.append(("CX", [targets[j].value, targets[j+1].value]))
            elif inst.name == "DETECTOR":
                # DETECTOR targets are RecordTargets, keep them as is
                flat.append(("DETECTOR", inst.targets_copy()))
            else:
                targets = [t.value if t.is_qubit_target else t for t in inst.targets_copy()]
                flat.append((inst.name, targets))
        flat_circs.append(flat)
        
    circ_iters = [iter(flat) for flat in flat_circs]
    current_insts = [next(it, None) for it in circ_iters]
    
    # Extract data H gates (applied by fao_se_circuit)
    data_qubits_needing_h = set()
    
    def is_data_cx(inst_tuple, stab_idx):
        if inst_tuple is None:
            return False
        name, targets = inst_tuple
        if name != "CX":
            return False
        data_set = set(stabs_qubits[stab_idx])
        for t in targets:
            if t in data_set:
                return True
        return False

    def advance_until_data_cx(stab_idx):
        while current_insts[stab_idx] is not None:
            inst_tuple = current_insts[stab_idx]
            name, targets = inst_tuple
            if name == "TICK":
                current_insts[stab_idx] = next(circ_iters[stab_idx], None)
                continue
                
            if is_data_cx(inst_tuple, stab_idx):
                break
                
            # Handle H on data qubits
            is_data_h = False
            if name == "H":
                data_set = set(stabs_qubits[stab_idx])
                for t in targets:
                    if t in data_set:
                        data_qubits_needing_h.add(t)
                        is_data_h = True
                        
            if not is_data_h:
                if name == "DETECTOR":
                    merged.append(name, targets)
                else:
                    merged.append(name, targets)
                
            current_insts[stab_idx] = next(circ_iters[stab_idx], None)

    # Initial advance for all circuits
    for i in range(len(meas_circs)):
        advance_until_data_cx(i)
        
    if data_qubits_needing_h:
        merged.append("H", sorted(list(data_qubits_needing_h)))
        merged.append("TICK")
        
    for tick_ops in ticks:
        tick_targets = []
        for stab_idx, q in tick_ops:
            inst_tuple = current_insts[stab_idx]
            if inst_tuple is None or not is_data_cx(inst_tuple, stab_idx):
                raise ValueError(f"Expected data CX instruction for stab {stab_idx} on qubit {q}, got {inst_tuple}")
            
            name, targets = inst_tuple
            # The tuple only has one CX pair [t1, t2]
            cx_pair = targets
                
            tick_targets.extend(cx_pair)
            
            # Advance past this specific data CX
            current_insts[stab_idx] = next(circ_iters[stab_idx], None)
            
        merged.append("CX", tick_targets)
        merged.append("TICK")
        
        # Advance iterators to queue up next ancilla operations
        for stab_idx, _ in tick_ops:
            advance_until_data_cx(stab_idx)
            
    if data_qubits_needing_h:
        merged.append("H", sorted(list(data_qubits_needing_h)))
        merged.append("TICK")
        
    return merged, current_ancilla

if __name__ == '__main__':
    stabs = ['01001100000010000', '10001000111010101']

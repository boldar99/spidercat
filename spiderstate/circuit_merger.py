from typing import Literal

import stim

from spiderstate.stim_utils import explode_circuit


def split_at_ticks(circ: stim.Circuit) -> list[stim.Circuit]:
    blocks = []
    curr = stim.Circuit()
    for inst in circ:
        if inst.name == "TICK":
            blocks.append(curr)
            curr = stim.Circuit()
        else:
            curr.append(inst)
    if len(curr) > 0:
        blocks.append(curr)
    return blocks


def open_flag_qubit(circ: stim.Circuit, basis: str, f: int, q: int):
    reset_op = "R" if basis == "Z" else "RX"
    c, n = (q, f) if basis == "Z" else (f, q)

    circ.append(reset_op, [f])
    circ.append("TICK")
    circ.append("CX", [c, n])
    circ.append("TICK")


def close_flag_qubit(circ: stim.Circuit, basis: str, f: int, q: int):
    c, n = (q, f) if basis == "Z" else (f, q)
    measure_op = "M" if basis == "Z" else "MX"

    circ.append("CX", [c, n])
    circ.append("TICK")
    circ.append(measure_op, [f])
    circ.append("DETECTOR", [stim.target_rec(-1)])
    circ.append("TICK")

def splice_flag_injection(circ: stim.Circuit, q: int, f: int, basis: str) -> stim.Circuit:
    blocks = split_at_ticks(circ)

    target_idx = -1
    for i in range(len(blocks) - 1, -1, -1):
        block = blocks[i]
        involves_q = False
        for inst in block:
            for t in inst.targets_copy():
                if t.is_qubit_target and t.value == q:
                    involves_q = True
                    break
            if involves_q: break
        if involves_q:
            target_idx = i
            break

    if target_idx == -1:
        target_idx = 0

    new_circ = stim.Circuit()
    for i, b in enumerate(blocks):
        if i == target_idx:
            open_flag_qubit(new_circ, basis, f, q)
        for inst in b:
            new_circ.append(inst)
        if i < len(blocks) - 1:
            new_circ.append("TICK")

    if target_idx == len(blocks):
        open_flag_qubit(new_circ, basis, f, q)

    return new_circ


# TODO: ideally make the implementation very clean
def synthesize_and_merge_layer(previous_circ: stim.Circuit, stabs_qubits: list[list[int]],
                               ticks: list[list[tuple[int, int]]], t: int, ancilla_start: int, basis: Literal["X", "Z"],
                               layer_violations: list[dict] = None) -> stim.Circuit:
    from spidercat.syndrome_measurement import bare_se_circuit, fao_se_circuit

    if layer_violations is None:
        layer_violations = []

    if not stabs_qubits:
        return previous_circ

    # 1. Schedule data CNOTs
    # ticks is passed as argument

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

        circ.flattened_operations()

        meas_circs.append(circ)
        current_ancilla = circ.num_qubits

    # 4. Merge via iterators
    merged = stim.Circuit()

    # Flatten instructions to avoid Stim's automatic folding
    flat_circs = [explode_circuit(circ) for circ in meas_circs]
    circ_iters = [iter(flat) for flat in flat_circs]
    current_insts = [next(it, None) for it in circ_iters]

    def is_data_cx(inst: stim.CircuitInstruction, stab_idx):
        if inst.name != "CX":
            return False
        data_set = set(stabs_qubits[stab_idx])
        for t in inst.targets_copy():
            if t.value in data_set:
                return True
        return False

    def advance_until_data_cx(stab_idx):
        while current_insts[stab_idx] is not None:
            inst = current_insts[stab_idx]
            if inst.name == "TICK":
                current_insts[stab_idx] = next(circ_iters[stab_idx], None)
                continue

            if is_data_cx(inst, stab_idx):
                break

            merged.append(inst)

            current_insts[stab_idx] = next(circ_iters[stab_idx], None)

    # Initial advance for all circuits
    for i in range(len(meas_circs)):
        advance_until_data_cx(i)

    # --- FLAG INJECTION START ---
    violating_qubits = set()
    for v in layer_violations:
        if "Q" in v:
            violating_qubits.update(v["Q"])
        elif "q" in v:  # Fallback just in case
            violating_qubits.add(v["q"])

    flags_to_insert = []
    for q in sorted(list(violating_qubits)):
        f_qubit = current_ancilla
        current_ancilla += 1
        flags_to_insert.append((q, f_qubit))

    for q, f in flags_to_insert:
        previous_circ = splice_flag_injection(previous_circ, q, f, basis)
    # --- FLAG INJECTION END ---

    for tick_ops in ticks:
        tick_targets = []

        for stab_idx, q in tick_ops:
            inst = current_insts[stab_idx]
            if inst is None or not is_data_cx(inst, stab_idx):
                raise ValueError(f"Expected data CX instruction for stab {stab_idx} on qubit {q}, got {inst}")

            tick_targets.extend(inst.targets_copy())

            # Advance past this specific data CX
            current_insts[stab_idx] = next(circ_iters[stab_idx], None)

        merged.append("CX", tick_targets)
        merged.append("TICK")

        # Advance iterators to queue up next ancilla operations
        for stab_idx, _ in tick_ops:
            advance_until_data_cx(stab_idx)

    # --- FLAG EXTRACTION START ---
    for q, f in flags_to_insert:
        close_flag_qubit(merged, basis, f, q)
    # --- FLAG EXTRACTION END ---

    return previous_circ + merged


if __name__ == '__main__':
    stabs = ['01001100000010000', '10001000111010101']

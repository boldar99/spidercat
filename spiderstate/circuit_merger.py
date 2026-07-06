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


class MeasurementCircuitMerger:
    def __init__(self, meas_circs: list[stim.Circuit], stabs_qubits: list[list[int]]):
        self.stab_data_sets = [set(qubits) for qubits in stabs_qubits]
        self.circ_iters = [iter(explode_circuit(circ)) for circ in meas_circs]
        self.current_insts = [next(it, None) for it in self.circ_iters]
        self.merged = stim.Circuit()

    def is_data_cx(self, inst: stim.CircuitInstruction, stab_idx: int) -> bool:
        if inst.name != "CX":
            return False
        data_set = self.stab_data_sets[stab_idx]
        for t in inst.targets_copy():
            if t.value in data_set:
                return True
        return False

    def advance_until_data_cx(self, stab_idx: int):
        while self.current_insts[stab_idx] is not None:
            inst = self.current_insts[stab_idx]
            if inst.name == "TICK":
                self.current_insts[stab_idx] = next(self.circ_iters[stab_idx], None)
                continue

            if self.is_data_cx(inst, stab_idx):
                break

            self.merged.append(inst)
            self.current_insts[stab_idx] = next(self.circ_iters[stab_idx], None)

    def merge(self, ticks: list[list[tuple[int, int]]]) -> stim.Circuit:
        for i in range(len(self.circ_iters)):
            self.advance_until_data_cx(i)

        for tick_ops in ticks:
            tick_targets = []
            for stab_idx, q in tick_ops:
                inst = self.current_insts[stab_idx]
                if inst is None or not self.is_data_cx(inst, stab_idx):
                    raise ValueError(f"Expected data CX instruction for stab {stab_idx} on qubit {q}, got {inst}")

                tick_targets.extend(inst.targets_copy())
                self.current_insts[stab_idx] = next(self.circ_iters[stab_idx], None)

            self.merged.append("CX", tick_targets)
            self.merged.append("TICK")

            for stab_idx, _ in tick_ops:
                self.advance_until_data_cx(stab_idx)

        return self.merged


def _extract_ordered_qubits(stabs_qubits: list[list[int]], ticks: list[list[tuple[int, int]]]) -> list[list[int]]:
    ordered_qubits = [[] for _ in range(len(stabs_qubits))]
    for tick_ops in ticks:
        for stab_idx, q in tick_ops:
            ordered_qubits[stab_idx].append(q)
    return ordered_qubits


def _generate_measurement_circuits(ordered_qubits: list[list[int]], t: int, ancilla_start: int,
                                   basis: Literal["X", "Z"]) -> tuple[list[stim.Circuit], int]:
    from spidercat.syndrome_measurement import bare_se_circuit, fao_se_circuit

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

    return meas_circs, current_ancilla


def _determine_flags_to_insert(layer_violations: list[dict], current_ancilla: int) -> tuple[list[tuple[int, int]], int]:
    violating_qubits = set()
    for v in layer_violations:
        if "Q" in v:
            violating_qubits.update(v["Q"])
        elif "q" in v:
            violating_qubits.add(v["q"])

    flags_to_insert = []
    for q in sorted(list(violating_qubits)):
        flags_to_insert.append((q, current_ancilla))
        current_ancilla += 1

    return flags_to_insert, current_ancilla


def _inject_flags(circ: stim.Circuit, flags_to_insert: list[tuple[int, int]], basis: Literal["X", "Z"]) -> stim.Circuit:
    for q, f in flags_to_insert:
        circ = splice_flag_injection(circ, q, f, basis)
    return circ


def _extract_flags(circ: stim.Circuit, flags_to_insert: list[tuple[int, int]], basis: Literal["X", "Z"]) -> None:
    for q, f in flags_to_insert:
        close_flag_qubit(circ, basis, f, q)


def _merge_measurement_circuits(meas_circs: list[stim.Circuit], stabs_qubits: list[list[int]],
                                ticks: list[list[tuple[int, int]]]) -> stim.Circuit:
    merger = MeasurementCircuitMerger(meas_circs, stabs_qubits)
    return merger.merge(ticks)


def synthesize_and_merge_layer(
    previous_circ: stim.Circuit,
    stabs_qubits: list[list[int]],
    ticks: list[list[tuple[int, int]]],
    t: int,
    ancilla_start: int,
    basis: Literal["X", "Z"],
    layer_violations: list[dict] | None = None
) -> stim.Circuit:
    if not stabs_qubits:
        return previous_circ

    layer_violations = layer_violations or []

    ordered_qubits = _extract_ordered_qubits(stabs_qubits, ticks)
    meas_circs, current_ancilla = _generate_measurement_circuits(ordered_qubits, t, ancilla_start, basis)
    flags_to_insert, current_ancilla = _determine_flags_to_insert(layer_violations, current_ancilla)

    previous_circ = _inject_flags(previous_circ, flags_to_insert, basis)
    merged = _merge_measurement_circuits(meas_circs, stabs_qubits, ticks)
    _extract_flags(merged, flags_to_insert, basis)

    return previous_circ + merged


if __name__ == '__main__':
    stabs = ['01001100000010000', '10001000111010101']

import argparse
import logging
import pprint
import numpy as np
from mqt.qecc.circuit_synthesis import CNOTCircuit
from mqt.qecc.circuit_synthesis.faults import PureFaultSet
from spiderstate.optimize_parity_matrix import optimize_fault_tolerant_matrix
from spiderstate.utils import load_qecc
from spiderstate.verification import find_lookahead_verification_stabilizers

def compute_unitary_fault_set_1(cnots: list[tuple[int, int]], num_qubits: int, kind: str = "X"):
    circ = CNOTCircuit()
    seen = set()
    for (c, n) in cnots:
        if c not in seen:
            seen.add(c)
            circ.initialize_qubit(c, "X")
        if n not in seen:
            seen.add(n)
            circ.initialize_qubit(n, "Z")
    for rem in set(range(num_qubits)) - seen:
        circ.initialize_qubit(rem, "Z")
    circ.add_cnots(cnots)

    single_faults = PureFaultSet.from_cnot_circuit(circ, kind=kind)
    single_faults.remove_zero_rows()
    single_faults.remove_duplicates()
    return single_faults

def main():
    parser = argparse.ArgumentParser(description="Run lookahead SAT verification on a given QECC.")
    parser.add_argument("--code", type=str, default="31_1_7", help="Name of the code (e.g. 17_1_5)")
    parser.add_argument("--basis", type=str, default="FAO", help="Basis or layout type (e.g. FAO)")
    parser.add_argument("--max-col-ops", type=int, default=5, help="Maximum number of column operations (CNOTs)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of covers to evaluate in lookahead")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Print verbose progress and matrices")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    print(f"Loading QECC: {args.code} in basis {args.basis}")
    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(args.code, args.basis)
    t = d // 2

    print(f"Optimizing parity matrix (max_col_ops={args.max_col_ops})...")
    row_M, final_M, col_ops = optimize_fault_tolerant_matrix(H_x, t=t, max_col_ops=args.max_col_ops, max_basis_tries=10_000)
    
    if args.verbose:
        print("\nOriginal H_x matrix:")
        print(H_x)
        print("\nOptimized final_M matrix:")
        print(final_M)
        print("\nColumn operations (CNOTs):")
        pprint.pprint(col_ops)
        print()
        
    print(f"Found {len(col_ops)} column operations (CNOTs).")
    
    print("Computing initial single faults...")
    single_faults_x = compute_unitary_fault_set_1(col_ops, num_qubits=H_x.shape[1], kind="X")
    single_faults_z = compute_unitary_fault_set_1(col_ops, num_qubits=H_x.shape[1], kind="Z")
    
    stabs_x = np.concatenate((H_z, L_z))
    stabs_z = np.concatenate((H_x, L_x))
    
    print(f"\nRunning lookahead verification stabilizers search for t={t} layers (top_n={args.top_n})...")
    
    print("\n--- X Faults Verification ---")
    ver_x_stabs_layers = find_lookahead_verification_stabilizers(
        single_faults=single_faults_x,
        stabs=stabs_x,
        H_filter=H_x,
        t=t,
        top_n=args.top_n,
        verbose=args.verbose
    )
    
    for layer_idx, layer in enumerate(ver_x_stabs_layers):
        print(f"\n Layer {layer_idx + 1}:")
        if not layer:
            print("  No stabilizers needed.")
        else:
            for stab_idx, stab in enumerate(layer):
                print(f"  Stabilizer {stab_idx} weight: {np.sum(stab)}, checks: {np.where(stab)[0]}")
                
    print("\n--- Z Faults Verification ---")
    ver_z_stabs_layers = find_lookahead_verification_stabilizers(
        single_faults=single_faults_z,
        stabs=stabs_z,
        H_filter=np.concatenate((H_z, L_z)),
        t=t,
        top_n=args.top_n,
        verbose=args.verbose
    )
    
    for layer_idx, layer in enumerate(ver_z_stabs_layers):
        print(f"\n Layer {layer_idx + 1}:")
        if not layer:
            print("  No stabilizers needed.")
        else:
            for stab_idx, stab in enumerate(layer):
                print(f"  Stabilizer {stab_idx} weight: {np.sum(stab)}, checks: {np.where(stab)[0]}")

if __name__ == "__main__":
    main()

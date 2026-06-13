import argparse
import logging
import pprint
import numpy as np
from mqt.qecc.circuit_synthesis import CNOTCircuit
from mqt.qecc.circuit_synthesis.faults import PureFaultSet
from spiderstate.optimize_parity_matrix import optimize_fault_tolerant_matrix, cnot_cost
from spiderstate.utils import load_qecc
from spiderstate.verification import find_lookahead_verification_stabilizers, compute_unitary_fault_set_1
from spiderstate.cat_at_origin import cat_at_origin_with_verification



def main():
    parser = argparse.ArgumentParser(description="Run lookahead SAT verification on a given QECC.")
    parser.add_argument("--code", type=str, default="17_1_5", help="QECC code name (e.g., 17_1_5, 8_3_2, etc.)")
    parser.add_argument("--basis", type=str, default="MQT", help="Basis or layout type (e.g. FAO)")
    parser.add_argument("--max-col-ops", type=int, default=50, help="Maximum number of column operations (CNOTs)")
    parser.add_argument("--top-n", type=int, default=50, help="Number of covers to evaluate in lookahead")
    parser.add_argument("--state", type=str, choices=["0", "+"], default="0", help="Logical state to prepare ('0' or '+')")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Print verbose progress and matrices")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    print(f"Loading QECC: {args.code} in basis {args.basis}")
    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(args.code, args.basis)

    final_circ = cat_at_origin_with_verification(
        H_x=H_x, H_z=H_z, L_x=L_x, L_z=L_z, d=d,
        state=args.state, max_col_ops=args.max_col_ops, top_n=args.top_n, max_basis_tries=10_000, verbose=args.verbose
    )

    print("\n--- Final Fault Tolerant Verification Circuit ---")
    print(f"Total Qubits: {final_circ.num_qubits}")
    print(f"Total instructions: {len(final_circ)}")
    print("Circuit output omitted to avoid terminal spam. You can export or print `final_circ`.")

if __name__ == "__main__":
    main()

import numpy as np
from spiderstate.utils import load_qecc
from mqt.qecc.circuit_synthesis.faults import PureFaultSet
from spiderstate.verification import find_low_weight_verification_stabilizers

is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc("17_1_5", "FAO")

# Fake fault set
fs = PureFaultSet(17)
fs.add_fault(np.array([1]*17))
fs.remove_equivalent(np.concatenate((H_x, L_x)))

stabs = np.concatenate((H_z, L_z))
print("Running find_low_weight_verification_stabilizers...")
res = find_low_weight_verification_stabilizers([fs], stabs)
print("Result layers:", len(res))
if res and len(res[0]) > 0:
    print("Stabilizers in first layer:", len(res[0]))
    for s in res[0]:
        print("Weight:", np.sum(s))
else:
    print("No stabilizers found or needed.")

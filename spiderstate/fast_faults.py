import numpy as np
from mqt.qecc.circuit_synthesis.faults import PureFaultSet

class FastFaultSet(PureFaultSet):
    """
    A subclass of PureFaultSet that overrides coset_leader and filter_by_weight
    using a fast NumPy-based precomputed lookup table instead of a Z3 solver.
    This works best when the number of stabilizers is small (<= 16).
    """

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self._stabs_combinations_cache = {}

    @classmethod
    def from_fault_array(cls, faults: np.ndarray) -> "FastFaultSet":
        num_qubits = faults.shape[1] if faults.ndim == 2 else faults.shape[0]
        fs = cls(num_qubits)
        fs.faults = faults.copy()
        return fs

    @classmethod
    def from_cnot_circuit(cls, circ, kind="X") -> "FastFaultSet":
        # First use the base class method to generate the faults
        fs_base = PureFaultSet.from_cnot_circuit(circ, kind=kind)
        # Then convert to FastFaultSet
        return cls.from_fault_array(fs_base.faults)

    def _get_stabs_combinations(self, H_filter: np.ndarray):
        key = H_filter.tobytes()
        if key not in self._stabs_combinations_cache:
            num_stabs = H_filter.shape[0]
            if num_stabs <= 16:
                indices = np.arange(1 << num_stabs)[:, None]
                masks = (indices >> np.arange(num_stabs)) & 1
                self._stabs_combinations_cache[key] = (masks @ H_filter) % 2
            else:
                self._stabs_combinations_cache[key] = None
        return self._stabs_combinations_cache[key]

    def faults_to_coset_leaders(self, generators: np.ndarray) -> None:
        """Override using fast precomputed lookup if generators are small enough."""
        stabs_combinations = self._get_stabs_combinations(generators)
        if stabs_combinations is not None and len(self.faults) > 0:
            new_faults = []
            for f in self.faults:
                coset = (f ^ stabs_combinations)
                weights = np.sum(coset, axis=1)
                best_idx = np.argmin(weights)
                new_faults.append(coset[best_idx])
            self.faults = np.array(new_faults, dtype=np.int8)
        else:
            # Fall back to base class Z3 implementation
            super().faults_to_coset_leaders(generators)

    def filter_by_weight_at_least(self, t: int, H_filter: np.ndarray) -> None:
        """Override to avoid instantiating Z3 solver for small H_filter."""
        stabs_combinations = self._get_stabs_combinations(H_filter)
        if stabs_combinations is not None and len(self.faults) > 0:
            filtered = []
            for f in self.faults:
                coset = (f ^ stabs_combinations)
                min_weight = np.min(np.sum(coset, axis=1))
                if min_weight >= t:
                    filtered.append(f)
            self.faults = np.array(filtered, dtype=np.int8) if filtered else np.zeros((0, self.num_qubits), dtype=np.int8)
        else:
            super().filter_by_weight_at_least(t, H_filter)

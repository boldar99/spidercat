import numpy as np

def fast_greedy_set_cover(
    coverage: np.ndarray, 
    costs: np.ndarray, 
    initial_uncovered: np.ndarray = None, 
    target_coverage: int = 1,
    candidate_stabs: np.ndarray = None,
    selected_stabs_indices: list[int] = None
) -> list[int]:
    """
    Vectorized greedy set cover algorithm.
    coverage: (num_candidates, num_faults) boolean array
    costs: (num_candidates,) array
    """
    num_candidates, num_faults = coverage.shape
    if initial_uncovered is None:
        remaining_coverage = np.full(num_faults, target_coverage, dtype=int)
    else:
        remaining_coverage = initial_uncovered.astype(int)
        
    selected_idx = []
    ratios = np.full(num_candidates, np.inf)
    
    while np.any(remaining_coverage > 0):
        # Count newly covered faults for each candidate
        active_faults = remaining_coverage > 0
        covered_counts = np.sum(coverage[:, active_faults], axis=1)
        
        valid = covered_counts > 0
        
        # Enforce overlap <= 2 constraint if candidate_stabs is provided
        if candidate_stabs is not None:
            current_selected = selected_idx.copy()
            if selected_stabs_indices is not None:
                current_selected = selected_stabs_indices + current_selected
                
            for s_idx in current_selected:
                stab = candidate_stabs[s_idx]
                overlaps = np.sum(candidate_stabs & stab, axis=1)
                valid[overlaps > 2] = False
                
        if not valid.any():
            break
            
        ratios.fill(np.inf)
        ratios[valid] = costs[valid] / covered_counts[valid]
        
        best_idx = np.argmin(ratios)
        selected_idx.append(best_idx)
        
        # Update remaining_coverage
        remaining_coverage[coverage[best_idx]] -= 1
        
    return selected_idx

def generate_all_combinations(stabs: np.ndarray) -> np.ndarray:
    """
    Generates the full subspace spanned by the rows of stabs, INCLUDING the zero vector.
    """
    if stabs is None:
        return None
    k = stabs.shape[0]
    combos = np.zeros((1 << k, stabs.shape[1]), dtype=np.int8)
    for i in range(1 << k):
        comb = [j for j in range(k) if (i >> j) & 1]
        if comb:
            combos[i] = np.bitwise_xor.reduce(stabs[comb], axis=0)
    return combos

class TrueBackwardTracker:
    def __init__(self, t: int, candidate_stabs_X, candidate_stabs_Z, H_filter_X, H_filter_Z, costs_X, costs_Z):
        self.c = H_filter_X.shape[1] if H_filter_X is not None else (H_filter_Z.shape[1] if H_filter_Z is not None else 0)
        self.t = t
        
        self.U_X = np.eye(self.c, dtype=np.int8)
        self.U_Z = np.eye(self.c, dtype=np.int8)
        
        self.cnot_faults_X = np.zeros((0, self.c), dtype=np.int8)
        self.cnot_faults_Z = np.zeros((0, self.c), dtype=np.int8)
        
        self.candidate_stabs_X = candidate_stabs_X
        self.candidate_stabs_Z = candidate_stabs_Z
        self.H_filter_X = H_filter_X
        self.H_filter_Z = H_filter_Z
        self.costs_X = costs_X
        self.costs_Z = costs_Z
        
        self.has_X_cands = candidate_stabs_X is not None and len(candidate_stabs_X) > 0
        self.has_Z_cands = candidate_stabs_Z is not None and len(candidate_stabs_Z) > 0
        
    def copy(self):
        new_obj = TrueBackwardTracker.__new__(TrueBackwardTracker)
        new_obj.c = self.c
        new_obj.t = self.t
        new_obj.U_X = self.U_X.copy()
        new_obj.U_Z = self.U_Z.copy()
        new_obj.cnot_faults_X = self.cnot_faults_X.copy()
        new_obj.cnot_faults_Z = self.cnot_faults_Z.copy()
        
        new_obj.candidate_stabs_X = self.candidate_stabs_X
        new_obj.candidate_stabs_Z = self.candidate_stabs_Z
        new_obj.H_filter_X = self.H_filter_X
        new_obj.H_filter_Z = self.H_filter_Z
        new_obj.costs_X = self.costs_X
        new_obj.costs_Z = self.costs_Z
        new_obj.has_X_cands = self.has_X_cands
        new_obj.has_Z_cands = self.has_Z_cands
        return new_obj

    def update_cnot(self, source, target):
        new_x = np.zeros((3, self.c), dtype=np.int8)
        new_x[0, source] = 1; new_x[1, target] = 1; new_x[2, source] = 1; new_x[2, target] = 1
        
        new_z = np.zeros((3, self.c), dtype=np.int8)
        new_z[0, source] = 1; new_z[1, target] = 1; new_z[2, source] = 1; new_z[2, target] = 1
        
        propagated_new_x = (new_x @ self.U_X) % 2
        propagated_new_z = (new_z @ self.U_Z) % 2
        
        self.cnot_faults_X = np.vstack((self.cnot_faults_X, propagated_new_x))
        self.cnot_faults_Z = np.vstack((self.cnot_faults_Z, propagated_new_z))
        
        self.U_Z[target] ^= self.U_Z[source]
        self.U_X[source] ^= self.U_X[target]

    def evaluate_cost(self):
        verif_cost = 0
        
        full_F_X = np.vstack((self.U_X, self.cnot_faults_X))
        full_F_Z = np.vstack((self.U_Z, self.cnot_faults_Z))
        
        if self.has_Z_cands:
            if self.H_filter_X is not None:
                S_X = (full_F_X @ self.H_filter_X.T) % 2
                faults_active_X = np.sum(S_X, axis=1) >= 2
            else:
                faults_active_X = np.any(full_F_X, axis=1)
                
            if np.any(faults_active_X):
                cov_X = ((self.candidate_stabs_Z @ full_F_X.T) % 2).astype(bool)
                valid_cov = cov_X[:, faults_active_X]
                
                coverable = np.any(valid_cov, axis=0)
                valid_cov = valid_cov[:, coverable]
                
                uncoverable = np.sum(~coverable)
                verif_cost += uncoverable * 1000
                
                if valid_cov.shape[1] > 0:
                    chosen_idx = fast_greedy_set_cover(valid_cov, self.costs_Z)
                    verif_cost += np.sum(self.costs_Z[chosen_idx])
                
        if self.has_X_cands:
            if self.H_filter_Z is not None:
                S_Z = (full_F_Z @ self.H_filter_Z.T) % 2
                faults_active_Z = np.sum(S_Z, axis=1) >= 2
            else:
                faults_active_Z = np.any(full_F_Z, axis=1)
                
            if np.any(faults_active_Z):
                cov_Z = ((self.candidate_stabs_X @ full_F_Z.T) % 2).astype(bool)
                valid_cov = cov_Z[:, faults_active_Z]
                
                coverable = np.any(valid_cov, axis=0)
                valid_cov = valid_cov[:, coverable]
                
                uncoverable = np.sum(~coverable)
                verif_cost += uncoverable * 1000
                
                if valid_cov.shape[1] > 0:
                    chosen_idx = fast_greedy_set_cover(valid_cov, self.costs_X)
                    verif_cost += np.sum(self.costs_X[chosen_idx])
                
        return verif_cost

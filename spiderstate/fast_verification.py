import numpy as np

def fast_greedy_set_cover(coverage: np.ndarray, costs: np.ndarray, initial_uncovered: np.ndarray = None, target_coverage: int = 1) -> list[int]:
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
        if not valid.any():
            break
            
        ratios.fill(np.inf)
        ratios[valid] = costs[valid] / covered_counts[valid]
        
        best_idx = np.argmin(ratios)
        selected_idx.append(best_idx)
        
        # Update remaining_coverage
        remaining_coverage[coverage[best_idx]] -= 1
        
    return selected_idx

class DynamicCoverageTracker:
    def __init__(self, F_X: np.ndarray, F_Z: np.ndarray, candidate_stabs_X: np.ndarray, candidate_stabs_Z: np.ndarray, cnot_cost_fn, H_filter_X: np.ndarray = None, H_filter_Z: np.ndarray = None):
        self.F_X = F_X.copy()
        self.F_Z = F_Z.copy()
        self.c = F_X.shape[1]
        
        self.candidate_stabs_X = candidate_stabs_X
        self.candidate_stabs_Z = candidate_stabs_Z
        self.cnot_cost_fn = cnot_cost_fn
        self.H_filter_X = H_filter_X
        self.H_filter_Z = H_filter_Z
        
        if self.H_filter_X is not None:
            self.S_X = (self.F_X @ self.H_filter_X.T) % 2
        else:
            self.S_X = None
            
        if self.H_filter_Z is not None:
            self.S_Z = (self.F_Z @ self.H_filter_Z.T) % 2
        else:
            self.S_Z = None
        
        self.has_X_cands = candidate_stabs_X is not None and len(candidate_stabs_X) > 0
        self.has_Z_cands = candidate_stabs_Z is not None and len(candidate_stabs_Z) > 0
        
        if self.has_X_cands:
            self.weights_X = np.sum(candidate_stabs_X, axis=1)
            self.costs_X = np.array([self.cnot_cost_fn(np.ones((1, w), dtype=int)) + 1 for w in self.weights_X])
            self.coverage_Z = ((candidate_stabs_X @ self.F_Z.T) % 2).astype(bool)
        else:
            self.coverage_Z = None
            
        if self.has_Z_cands:
            self.weights_Z = np.sum(candidate_stabs_Z, axis=1)
            self.costs_Z = np.array([self.cnot_cost_fn(np.ones((1, w), dtype=int)) + 1 for w in self.weights_Z])
            self.coverage_X = ((candidate_stabs_Z @ self.F_X.T) % 2).astype(bool)
        else:
            self.coverage_X = None
            
    def copy(self):
        new_obj = DynamicCoverageTracker.__new__(DynamicCoverageTracker)
        new_obj.F_X = self.F_X.copy()
        new_obj.F_Z = self.F_Z.copy()
        new_obj.c = self.c
        
        new_obj.candidate_stabs_X = self.candidate_stabs_X
        new_obj.candidate_stabs_Z = self.candidate_stabs_Z
        new_obj.cnot_cost_fn = self.cnot_cost_fn
        new_obj.H_filter_X = self.H_filter_X
        new_obj.H_filter_Z = self.H_filter_Z
        
        if self.S_X is not None:
            new_obj.S_X = self.S_X.copy()
        else:
            new_obj.S_X = None
            
        if self.S_Z is not None:
            new_obj.S_Z = self.S_Z.copy()
        else:
            new_obj.S_Z = None
            
        new_obj.has_X_cands = self.has_X_cands
        new_obj.has_Z_cands = self.has_Z_cands
        
        if self.has_X_cands:
            new_obj.weights_X = self.weights_X
            new_obj.costs_X = self.costs_X
            new_obj.coverage_Z = self.coverage_Z.copy()
        else:
            new_obj.coverage_Z = None
            
        if self.has_Z_cands:
            new_obj.weights_Z = self.weights_Z
            new_obj.costs_Z = self.costs_Z
            new_obj.coverage_X = self.coverage_X.copy()
        else:
            new_obj.coverage_X = None
            
        return new_obj

    def update_cnot(self, source, target):
        # 1. Commutation Flip
        if self.has_Z_cands:
            active_faults_X = self.F_X[:, source] == 1
            active_candidates_Z = self.candidate_stabs_Z[:, target] == 1
            self.coverage_X ^= np.outer(active_candidates_Z, active_faults_X)
            
        if self.has_X_cands:
            active_faults_Z = self.F_Z[:, target] == 1
            active_candidates_X = self.candidate_stabs_X[:, source] == 1
            self.coverage_Z ^= np.outer(active_candidates_X, active_faults_Z)
            
        # 2. Propagate existing faults
        if self.H_filter_X is not None:
            active_X = self.F_X[:, source] == 1
            self.S_X[active_X] ^= self.H_filter_X[:, target]
            
        self.F_X[:, target] ^= self.F_X[:, source]
        
        if self.H_filter_Z is not None:
            active_Z = self.F_Z[:, target] == 1
            self.S_Z[active_Z] ^= self.H_filter_Z[:, source]
            
        self.F_Z[:, source] ^= self.F_Z[:, target]
        
        # 3. Inject new faults
        new_x = np.zeros((3, self.c), dtype=int)
        new_x[0, source] = 1
        new_x[1, target] = 1
        new_x[2, source] = 1; new_x[2, target] = 1
        self.F_X = np.vstack((self.F_X, new_x))
        if self.H_filter_X is not None:
            new_s_x = (new_x @ self.H_filter_X.T) % 2
            self.S_X = np.vstack((self.S_X, new_s_x))
        
        new_z = np.zeros((3, self.c), dtype=int)
        new_z[0, source] = 1
        new_z[1, target] = 1
        new_z[2, source] = 1; new_z[2, target] = 1
        self.F_Z = np.vstack((self.F_Z, new_z))
        if self.H_filter_Z is not None:
            new_s_z = (new_z @ self.H_filter_Z.T) % 2
            self.S_Z = np.vstack((self.S_Z, new_s_z))
        
        # 4. Coverage for new faults
        if self.has_Z_cands:
            new_cov_X = ((self.candidate_stabs_Z @ new_x.T) % 2).astype(bool)
            self.coverage_X = np.hstack((self.coverage_X, new_cov_X))
            
        if self.has_X_cands:
            new_cov_Z = ((self.candidate_stabs_X @ new_z.T) % 2).astype(bool)
            self.coverage_Z = np.hstack((self.coverage_Z, new_cov_Z))

    def evaluate_cost(self):
        verif_cost = 0
        
        if self.has_Z_cands:
            if self.H_filter_X is not None:
                faults_active_X = np.sum(self.S_X, axis=1) >= 2
            else:
                faults_active_X = np.any(self.F_X, axis=1)
                
            valid_cov = self.coverage_X[:, faults_active_X]
            
            coverable = np.any(valid_cov, axis=0)
            valid_cov = valid_cov[:, coverable]
            
            uncoverable = np.sum(~coverable)
            verif_cost += uncoverable * 1000
            
            if valid_cov.shape[1] > 0:
                chosen_idx = fast_greedy_set_cover(valid_cov, self.costs_Z)
                verif_cost += np.sum(self.costs_Z[chosen_idx])
                
        if self.has_X_cands:
            if self.H_filter_Z is not None:
                faults_active_Z = np.sum(self.S_Z, axis=1) >= 2
            else:
                faults_active_Z = np.any(self.F_Z, axis=1)
                
            valid_cov = self.coverage_Z[:, faults_active_Z]
            
            coverable = np.any(valid_cov, axis=0)
            valid_cov = valid_cov[:, coverable]
            
            uncoverable = np.sum(~coverable)
            verif_cost += uncoverable * 1000
            
            if valid_cov.shape[1] > 0:
                chosen_idx = fast_greedy_set_cover(valid_cov, self.costs_X)
                verif_cost += np.sum(self.costs_X[chosen_idx])
                
        return verif_cost

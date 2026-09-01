import numpy as np
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds
try:
    from ldpc import mod2
    from bposd import bposd_decoder
except ImportError:
    mod2 = None
    bposd_decoder = None

def get_rref(M):
    M = M.copy()
    rows, cols = M.shape
    pivots = []
    r = 0
    for c in range(cols):
        pivot = np.argmax(M[r:, c]) + r
        if M[pivot, c] == 0:
            continue
        M[[r, pivot]] = M[[pivot, r]]
        pivots.append((r, c))
        for i in range(rows):
            if i != r and M[i, c] == 1:
                M[i] = (M[i] + M[r]) % 2
        r += 1
        if r == rows:
            break
    return M, pivots

def is_in_row_space_fast(v, M_rref):
    v = v.copy()
    for row in M_rref:
        pivot_cols = np.where(row == 1)[0]
        if len(pivot_cols) > 0:
            c = pivot_cols[0]
            if v[c] == 1:
                v = (v + row) % 2
    return np.all(v == 0)

def build_syndrome_table(H, max_w):
    n = H.shape[1]
    table = {}
    for w in range(max_w):
        for err_idx in itertools.combinations(range(n), w):
            e = np.zeros(n, dtype=int)
            e[list(err_idx)] = 1
            s = tuple((H @ e) % 2)
            if s not in table:
                table[s] = w
    return table

def solve_milp_coset(v, M, n):
    k = M.shape[0] if M is not None and len(M) > 0 else 0
    if k == 0:
        return np.sum(v)
        
    num_vars = n + k + n
    c = np.zeros(num_vars)
    c[:n] = 1.0 
    
    integrality = np.ones(num_vars, dtype=int)
    
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    lb[-n:] = -max(1, k)
    ub[-n:] = max(1, k)
    
    bounds = Bounds(lb, ub)
    
    A = np.zeros((n, num_vars))
    lb_A = np.zeros(n)
    ub_A = np.zeros(n)
    
    for i in range(n):
        A[i, i] = 1
        for j in range(k):
            A[i, n + j] = -M[j, i]
        A[i, n + k + i] = 2
        lb_A[i] = v[i]
        ub_A[i] = v[i]
        
    constraints = LinearConstraint(A, lb_A, ub_A)
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if res.success:
        return int(round(res.fun))
    else:
        raise ValueError("MILP failed to solve.")


class LookupStrategy:
    def __init__(self, M_prep, t):
        self.M_prep = M_prep
        self.t = t
        if len(M_prep) > 0:
            self.H = mod2.nullspace(M_prep).toarray()
        else:
            n = M_prep.shape[1]
            self.H = np.eye(n, dtype=int)
        self.table = build_syndrome_table(self.H, 2 * t)
        
    def check_tier1(self, E, threshold=1):
        s = tuple((self.H @ E) % 2)
        return self.table.get(s, np.inf) <= threshold
        
    def check_tier3(self, E, L_cosets, threshold=None):
        if threshold is None: threshold = 2 * self.t
        for L_c in L_cosets:
            v = (E + L_c) % 2
            s = tuple((self.H @ v) % 2)
            w = self.table.get(s, np.inf)
            if w < threshold:
                return False
        return True


class TieredStrategy:
    def __init__(self, M_prep, t):
        self.M_prep = M_prep
        self.M_rref, _ = get_rref(M_prep)
        self.t = t
        if len(M_prep) > 0:
            self.H = mod2.nullspace(M_prep).toarray()
        else:
            self.H = np.eye(M_prep.shape[1], dtype=int)
            
        if bposd_decoder is not None:
            self.bposd = bposd_decoder(
                self.H,
                error_rate=0.01,
                max_iter=10,
                bp_method="ms",
                osd_method="osd0",
                osd_order=0
            )
        else:
            self.bposd = None
        
    def check_tier1(self, E, threshold=1):
        # RREF fast check only works easily for threshold=0 or 1
        # For threshold > 1, we fallback to MILP or just do full iteration if small
        if threshold == 1:
            if is_in_row_space_fast(E, self.M_rref): return True
            n = len(E)
            for i in range(n):
                e1 = np.zeros(n, dtype=int)
                e1[i] = 1
                if is_in_row_space_fast((E + e1) % 2, self.M_rref): return True
            return False
        else:
            n = len(E)
            w = solve_milp_coset(E, self.M_prep, n)
            return w <= threshold
        
    def check_tier3(self, E, L_cosets, threshold=None):
        if threshold is None: threshold = 2 * self.t
        n = len(E)
        for L_c in L_cosets:
            v = (E + L_c) % 2
            
            # Tier 2: BP-OSD
            if self.bposd is not None:
                s = (self.H @ v) % 2
                self.bposd.decode(s)
                guess = self.bposd.osdw_decoding
                w_heuristic = np.sum((v + guess) % 2)
                
                if w_heuristic < threshold:
                    return False
                    
            # Tier 3: Exact MILP
            w_exact = solve_milp_coset(v, self.M_prep, n)
            if w_exact < threshold:
                return False
                
        return True


class AlgebraicStrategy:
    """
    A strictly polynomial, decoder-independent strategy.
    It builds a lookup table of syndromes for all errors up to a fixed max_weight.
    This guarantees w_min* <= k algebraically without using BP-OSD or ILP.
    Perfect for scaling to high distances where perfect decoding is not assumed.
    """
    def __init__(self, S_prep, t, max_weight=1):
        self.S_prep = S_prep
        self.t = t
        self.n = S_prep.shape[1]
        self.max_weight = max_weight
        
        # Build syndrome lookup table for errors up to max_weight
        self.syndromes = {}
        import galois
        GF = galois.GF(2)
        H_check = GF(self.S_prep).null_space()
        
        def backtrack(idx, current_weight, current_err):
            if current_weight <= self.max_weight:
                syn = tuple((H_check @ current_err).tolist())
                if syn not in self.syndromes or current_weight < self.syndromes[syn]:
                    self.syndromes[syn] = current_weight
                    
            if current_weight == self.max_weight or idx == self.n:
                return
                
            backtrack(idx + 1, current_weight, current_err)
            current_err[idx] = 1
            backtrack(idx + 1, current_weight + 1, current_err)
            current_err[idx] = 0

        backtrack(0, 0, GF.Zeros(self.n))
        self.H_check = H_check
        self.GF = galois.GF(2)

    def check_tier1(self, E, threshold=1):
        """Checks if w_min*(E) <= threshold using the syndrome lookup table."""
        e_galois = self.GF(E)
        syn = tuple((self.H_check @ e_galois).tolist())
        return syn in self.syndromes and self.syndromes[syn] <= threshold
        
    def check_tier3(self, E, L_cosets):
        # We strictly rely on Tier 1 for algebraic safety.
        return False


class MILPStrategy:
    def __init__(self, M_prep, t):
        self.M_prep = M_prep
        self.t = t
        
    def check_tier1(self, E, threshold=1):
        n = len(E)
        w = solve_milp_coset(E, self.M_prep, n)
        return w <= threshold
        
    def check_tier3(self, E, L_cosets, threshold=None):
        if threshold is None: threshold = 2 * self.t
        n = len(E)
        for L_c in L_cosets:
            v = (E + L_c) % 2
            w = solve_milp_coset(v, self.M_prep, n)
            if w < threshold:
                return False
        return True

class HeuristicOnlyStrategy:
    """
    Uses Tier 1 (GF2 RREF) and Tier 2 (High-Order BP-OSD) but completely skips 
    the exact MILP solver. Highly scalable but trades exact certainty for a tight heuristic bound.
    """
    def __init__(self, M_prep, t, osd_order=10):
        self.M_prep = M_prep
        self.M_rref, _ = get_rref(M_prep)
        self.t = t
        if len(M_prep) > 0:
            self.H = mod2.nullspace(M_prep).toarray()
        else:
            self.H = np.eye(M_prep.shape[1], dtype=int)
            
        if bposd_decoder is not None:
            self.bposd = bposd_decoder(
                self.H,
                error_rate=0.01,
                max_iter=50,
                bp_method="ms",
                osd_method="osd_cs",
                osd_order=osd_order
            )
        else:
            raise ImportError("bposd is required for HeuristicOnlyStrategy")

    def check_tier1(self, E, threshold=1):
        if threshold == 1:
            if is_in_row_space_fast(E, self.M_rref):
                return True
            n = len(E)
            for i in range(n):
                e1 = np.zeros(n, dtype=int)
                e1[i] = 1
                if is_in_row_space_fast((E + e1) % 2, self.M_rref):
                    return True
            return False
        else:
            # Fallback for higher thresholds in heuristic mode:
            # We can use bposd to find the min weight of the syndrome
            s = (self.H @ E) % 2
            self.bposd.decode(s)
            guess = self.bposd.osdw_decoding
            w_heuristic = np.sum((E + guess) % 2)
            return w_heuristic <= threshold
        
    def check_tier3(self, E, L_cosets, threshold=None):
        if threshold is None: threshold = 2 * self.t
        for L_c in L_cosets:
            v = (E + L_c) % 2
            s = (self.H @ v) % 2
            self.bposd.decode(s)
            guess = self.bposd.osdw_decoding
            w_heuristic = np.sum((v + guess) % 2)
            
            # Since this is HeuristicOnly, we completely trust the heuristic bound.
            if w_heuristic < threshold:
                return False
        return True

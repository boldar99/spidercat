import numpy as np
import json

def rref_gf2(A):
    """Computes the reduced row echelon form of A over GF(2)."""
    A = np.array(A, dtype=int) % 2
    rows, cols = A.shape
    r = 0
    pivots = []
    for c in range(cols):
        if r >= rows:
            break
        # Find pivot in this column
        pivot_row = r
        while pivot_row < rows and A[pivot_row, c] == 0:
            pivot_row += 1
        if pivot_row == rows:
            continue
        # Swap rows
        A[[r, pivot_row]] = A[[pivot_row, r]]
        pivots.append(c)
        # Eliminate other rows
        for i in range(rows):
            if i != r and A[i, c] == 1:
                A[i] = (A[i] + A[r]) % 2
        r += 1
    return A, pivots

def row_basis(A):
    """Returns a basis for the row space of A."""
    A_rref, _ = rref_gf2(A)
    # Filter out zero rows
    basis = A_rref[np.any(A_rref, axis=1)]
    return basis

def nullspace_gf2(A):
    """Returns a basis for the nullspace of A over GF(2)."""
    A_rref, pivots = rref_gf2(A)
    rows, cols = A.shape
    rank = len(pivots)
    free_vars = [c for c in range(cols) if c not in pivots]
    
    N = np.zeros((len(free_vars), cols), dtype=int)
    for i, free_var in enumerate(free_vars):
        N[i, free_var] = 1
        for r_idx, p_col in enumerate(pivots):
            N[i, p_col] = A_rref[r_idx, free_var]
    return N

def extend_basis(basis, superset):
    """
    Given a basis of a subspace, and a superset of vectors that span a larger space,
    returns the vectors from the superset that extend the basis to a basis of the larger space.
    """
    extended = []
    current_basis = list(basis)
    for v in superset:
        # Check if v is linearly independent from current_basis
        test_matrix = np.vstack(current_basis + [v])
        _, pivots = rref_gf2(test_matrix)
        if len(pivots) > len(current_basis):
            extended.append(v)
            current_basis.append(v)
    return np.array(extended)

def inverse_gf2(A):
    """Returns the inverse of A over GF(2)."""
    n = A.shape[0]
    augmented = np.hstack((A, np.eye(n, dtype=int)))
    rref_aug, _ = rref_gf2(augmented)
    return rref_aug[:, n:] % 2

def find_css_logicals(Hx, Hz):
    Hx = np.array(Hx, dtype=int) % 2
    Hz = np.array(Hz, dtype=int) % 2
    
    Hx_basis = row_basis(Hx)
    Hz_basis = row_basis(Hz)
    
    Nx = nullspace_gf2(Hz)
    Nz = nullspace_gf2(Hx)
    
    tilde_Lx = extend_basis(Hx_basis, Nx)
    tilde_Lz = extend_basis(Hz_basis, Nz)
    
    if len(tilde_Lx) == 0:
        return [], []
        
    M = (tilde_Lx @ tilde_Lz.T) % 2
    Minv = inverse_gf2(M)
    
    Lx = tilde_Lx
    Lz = (Minv.T @ tilde_Lz) % 2
    
    return Lx.tolist(), Lz.tolist()

if __name__ == "__main__":
    code = {
      "name": "bcc_code",
      "n": 22,
      "k": 8,
      "d": 4,
      "is_self_dual": False,
    "H_x": [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    ],
    "H_z": [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    ],
    }
    
    Lx, Lz = find_css_logicals(code["H_x"], code["H_z"])
    
    print("Found Logicals:")
    print(json.dumps({"L_x": Lx, "L_z": Lz}))
    
    # Validation
    Hx = np.array(code["H_x"])
    Hz = np.array(code["H_z"])
    Lx_arr = np.array(Lx)
    Lz_arr = np.array(Lz)
    
    print("\nValidations:")
    print("Lx commutes with Hz:", np.all((Lx_arr @ Hz.T) % 2 == 0))
    print("Lz commutes with Hx:", np.all((Lz_arr @ Hx.T) % 2 == 0))
    print("Lx anti-commutes with Lz to form pairs:", np.all((Lx_arr @ Lz_arr.T) % 2 == np.eye(len(Lx))))

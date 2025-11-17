import os
import pickle as pkl
import numpy as np
import sympy as sp
from sympy import Symbol, Matrix, sqrt, eye, simplify
from sympy.physics.quantum import TensorProduct

pkl_path = "src/2xd_GlobalME_eq.pkl"

d = 7

# Qubit Operator
sigma_p = Matrix([[0, 0], [1, 0]])  # |1><0| (sigma plus)
sigma_m = Matrix([[0, 1], [0, 0]])  # |0><1| (sigma minus)

# Oscillator Operator (dim = d) 
def sympy_annihilation_operator(d):
    """
    Construct the d-dimensional truncated harmonic oscillator 
    annihilation operator a and creation operator a_dag in SymPy.
    """
    a = sp.zeros(d, d)
    for n in range(1, d):
        a[n-1, n] = sp.sqrt(n)   # a |n> = sqrt(n) |n-1>
    a_dag = a.H                 # Hermitian conjugate

    return a, a_dag

a, a_dag = sympy_annihilation_operator(d)

# System Hamiltonian
g = Symbol('g', real=True)
Omega = Symbol('Omega', real=True)
H_S = Omega * (TensorProduct(sigma_p * sigma_m, eye(d))+ TensorProduct(eye(2), a_dag*a)) + \
    g * (TensorProduct(sigma_p, a) + TensorProduct(sigma_m, a_dag))

# Transform into Hs eigenbasis
P, D = H_S.diagonalize()

normalized_eigenvectors = []
for i in range(P.shape[1]): 
    col = P.col(i) 
    normalized_col = col.normalized()
    normalized_eigenvectors.append(normalized_col)

U = Matrix.hstack(*normalized_eigenvectors) 

# System-Environment interaction operators, in Hs eigenbasis
sigma_m_subsys1 = TensorProduct(sigma_m, eye(d))
sigma_m_system_eigenbase = U.H * sigma_m_subsys1 * U
sigma_p_system_eigenbase = sigma_m_system_eigenbase.H

a_subsys2 = TensorProduct(eye(2), a)
a_system_eigenbase = U.H * a_subsys2 * U
a_dag_system_eigenbase = a_system_eigenbase.H

op_index = {
    0: ("sigma_m", sigma_m_system_eigenbase),
    1: ("sigma_p", sigma_p_system_eigenbase),
    2: ("a",      a_system_eigenbase),
    3: ("a_dag",   a_dag_system_eigenbase),
}

# Spectrum decomposition of operator S
def is_sympy_zero(expr):
    """Robust zero test for SymPy expressions."""
    return bool(expr.equals(0) or expr.is_zero is True)

def decompose_by_frequency(D_diag, S_eig):
    """
    Decompose an operator S (already in H's eigenbasis) into {S(ω)}
    such that [H, S(ω)] = -ω S(ω).
    H's eigenvalues are the diagonal entries of D_diag.
    Returns:
        omega_list: sorted list of all ω that actually appear for S
        S_omega: dict mapping ω -> matrix S(ω) in the eigenbasis
    """

    E = list(D_diag.diagonal())         # eigenvalues E_k
    S = sp.Matrix(S_eig)
    dim = S.shape[0]

    # 1. Collect all ω = E_n - E_m for which S[m,n] != 0
    omegas = set()
    for m in range(dim):
        for n in range(dim):
            elem = sp.simplify(S[m, n])
            if not is_sympy_zero(elem):
                omegas.add(sp.simplify(E[n] - E[m]))

    omega_list = sorted(omegas, key=sp.default_sort_key)

    # 2. Build S(ω) by selecting matrix elements with the matching energy gap
    S_omega = {}
    for w in omega_list:
        Sw = sp.zeros(dim, dim)
        for m in range(dim):
            for n in range(dim):
                if sp.simplify(E[n] - E[m] - w).equals(0):
                    Sw[m, n] = S[m, n]
        S_omega[w] = sp.simplify(Sw)

    return omega_list, S_omega

omega_list = {}        # omega_list[i] = [ω_0, ω_1, ...] 
S_omega_eig = {}       # S_omega_eig[i] = {ω -> S_i(ω)}

for i, (name, S_eig) in op_index.items():
    wlist, Sdict = decompose_by_frequency(D, S_eig)
    omega_list[i] = wlist
    S_omega_eig[i] = Sdict

# Find terms after rotating wave approximation
omega_sets = {i: set(S_omega_eig[i].keys()) for i in S_omega_eig}
"""
Secular (ω = ω′) retained triples (k, l, ω)
   Keep ω that is present for both operator k and l 
"""
retained_triples = []
for k in omega_sets:
    for l in omega_sets:
        common = omega_sets[k] & omega_sets[l]
        for w in sorted(common, key=sp.default_sort_key):
            retained_triples.append((k, l, w))

retained_triples_self = [(k, l, w) for (k, l, w) in retained_triples if k == l]

def _to_np(x):
    if isinstance(x, sp.MatrixBase):
        return np.array(x, dtype=complex)
    return np.asarray(x, dtype=complex)


S_omega_eig_np = {}  # S_omega_eig entries are numpy arrays
for i in S_omega_eig:
    S_omega_eig_np[i] = {}
    for w, mat in S_omega_eig[i].items():
        # Ensure that key w is sympy object.
        if isinstance(w, str):
            w_sym = sp.sympify(w)
        elif not isinstance(w, sp.Basic):
            w_sym = sp.sympify(str(w))
        else:
            w_sym = w
        S_omega_eig_np[i][w_sym] = _to_np(mat)


# --- assemble payload ---
export_payload = {
    "U": _to_np(U),
    "S_omega_eig": S_omega_eig_np,
    "retained_triples_self": retained_triples_self,
    "Hs_eig": sp.Matrix(D),  
}

# --- write to pkl ---
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        all_data = pkl.load(f)
    if not isinstance(all_data, dict):
        raise TypeError(f"{pkl_path} has incorrect data structure. ")
else:
    all_data = {}

all_data[d] = export_payload

with open(pkl_path, "wb") as f:
    pkl.dump(all_data, f)

print(f"Saved data for d = {d} into {pkl_path}")
import pickle as pkl
import numpy as np
import sympy as sp
import qutip as qt
import src.utilities as ut
from functools import partial

pkl_path = "src/2xd_GlobalME_eq.pkl"

def load_globalME_data(pkl_path, d):
    with open(pkl_path, "rb") as f:
        all_data = pkl.load(f)
    return all_data[d]

def eval_sym(expr: sp.Basic, params: dict[str, float]) -> float:
    """Evaluate sympy scalar by substituting parameters by name."""
    if not isinstance(expr, sp.Basic):
        return float(expr)

    expr_sub = expr.xreplace({s: params[s.name]
                              for s in expr.free_symbols
                              if s.name in params})
    val = complex(sp.N(expr_sub))
    return float(np.real_if_close(val))

# Lindblad Eq structure
def _rate_from_index(k: int, w_val: float, Omega_val: float,
                     kappa_h, kappa_c, nB_h, nB_c) -> float:
    """Return Γ_k(ω) according to channel index."""
    arg = Omega_val + w_val
    if k == 0:
        return kappa_h(arg) * (nB_h(arg) + 1.0)
    elif k == 1:
        return kappa_h(arg) * nB_h(arg)
    elif k == 2:
        return kappa_c(arg) * (nB_c(arg) + 1.0)
    elif k == 3:
        return kappa_c(arg) * nB_c(arg)
    else:
        raise ValueError(f"Unknown k={k}")

def build_Lops_from_pkl(
    d, 
    kappa_h_fun,
    kappa_c_fun,
    nB_h_fun, 
    nB_c_fun,
    *,
    param_values: dict[str, float],
) -> list[qt.Qobj]:
    """
    Build collapse operators L = sqrt(Γ_k(ω)) * S_k(ω).
    Returns list of QuTiP Qobj operators.
    """
    data = load_globalME_data(pkl_path, d)
    Omega_val = eval_sym(sp.sympify("Omega"), param_values)
    S_omega_eig = data["S_omega_eig"]
    retained_triples_self = data["retained_triples_self"]

    Lops = []
    for (k, _, w) in retained_triples_self:
        w_val = eval_sym(w, param_values)
        rate = _rate_from_index(k, w_val, Omega_val,
                                kappa_h_fun, kappa_c_fun, nB_h_fun, nB_c_fun)
        S_eig = np.asarray(S_omega_eig[k][w], dtype=complex)
        Lops.append(qt.Qobj(np.sqrt(rate) * S_eig))
    return Lops

def build_Hs_matrix_from_pkl(d, param_values: dict[str, float]) -> np.ndarray:
    """Build numeric D matrix from symbolic D_expr stored in .pkl."""
    data = load_globalME_data(pkl_path, d)
    Hs_eig: sp.Matrix = data["Hs_eig"]
    Hs_eval = Hs_eig.xreplace({s: param_values[s.name]
                              for s in Hs_eig.free_symbols
                              if s.name in param_values})
    Hs_num = np.array(Hs_eval.evalf(), dtype=complex)
    return Hs_num

def to_computational_basis(d, op_eig: np.ndarray) -> np.ndarray: 
    data = load_globalME_data(pkl_path, d) 
    U = data["U"]
    Udag = U.conj().T
    return U @ op_eig @ Udag

def calculate_steadystate_sol(d, Omega0, g, \
                                T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c):
    
    param_values = {"Omega": Omega0, "g": g} 

    nB_h_fun = partial(ut.n_B, T=T_h) 
    nB_c_fun = partial(ut.n_B, T=T_c) 
    kappa_h_fun = partial(ut.kappa_Ohmic, eta=eta_h, omega_c=omega_c_h) 
    kappa_c_fun = partial(ut.kappa_Ohmic, eta=eta_c, omega_c=omega_c_c) 

    # Note that Lops and Hs are both in Hs eigenbasis. 
    Lops = build_Lops_from_pkl(
        d = d,
        kappa_h_fun=kappa_h_fun, 
        kappa_c_fun=kappa_c_fun, 
        nB_h_fun=nB_h_fun, 
        nB_c_fun=nB_c_fun, 
        param_values=param_values, 
    )

    Hs_num = build_Hs_matrix_from_pkl(d, param_values) 
    Hs = qt.Qobj(np.array(Hs_num, dtype=complex)) 

    rho_ss = qt.steadystate(Hs, Lops)

    rho_ss = qt.Qobj(to_computational_basis(d, rho_ss.full()), \
                    dims=[[2,d],[2,d]])

    return rho_ss

def calculate_steady_negativity(d, Omega0, g, \
                                T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c):
    
    param_values = {"Omega": Omega0, "g": g} 

    nB_h_fun = partial(ut.n_B, T=T_h) 
    nB_c_fun = partial(ut.n_B, T=T_c) 
    kappa_h_fun = partial(ut.kappa_Ohmic, eta=eta_h, omega_c=omega_c_h) 
    kappa_c_fun = partial(ut.kappa_Ohmic, eta=eta_c, omega_c=omega_c_c) 

    # Note that Lops and Hs are both in Hs eigenbasis. 
    Lops = build_Lops_from_pkl(
        d = d,
        kappa_h_fun=kappa_h_fun, 
        kappa_c_fun=kappa_c_fun, 
        nB_h_fun=nB_h_fun, 
        nB_c_fun=nB_c_fun, 
        param_values=param_values, 
    )

    Hs_num = build_Hs_matrix_from_pkl(d, param_values) 
    Hs = qt.Qobj(np.array(Hs_num, dtype=complex)) 

    rho_ss = qt.steadystate(Hs, Lops)

    rho_ss = qt.Qobj(to_computational_basis(d, rho_ss.full()), \
                    dims=[[2,d],[2,d]])
    # rho_ss = (rho_ss + rho_ss.dag())/2
    N_ss = qt.negativity(rho_ss, subsys=0)

    return N_ss
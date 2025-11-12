from __future__ import annotations
import numpy as np
import pickle
import sympy as sp
from qutip import Qobj

pkl_path = "QubitQutrit_GlobalME_data.pkl"

# ============================================================
# Basic utilities
# ============================================================

def load_export(pkl_path: str):
    """Load exported .pkl file exported from notebook."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for key in ("U", "S_omega_eig", "retained_triples_self", "Hs_eig"):
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {pkl_path}")
    return data


def eval_sym(expr: sp.Basic, params: dict[str, float]) -> float:
    """Evaluate sympy scalar by substituting parameters by name."""
    if not isinstance(expr, sp.Basic):
        return float(expr)

    expr_sub = expr.xreplace({s: params[s.name]
                              for s in expr.free_symbols
                              if s.name in params})
    val = complex(sp.N(expr_sub))
    return float(np.real_if_close(val))


# ============================================================
# Lindblad part
# ============================================================

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
    kappa_h_fun,
    kappa_c_fun,
    nB_h_fun, 
    nB_c_fun,
    *,
    param_values: dict[str, float],
) -> list[Qobj]:
    """
    Build collapse operators L = sqrt(Γ_k(ω)) * S_k(ω).
    Returns list of QuTiP Qobj operators.
    """
    data = load_export(pkl_path)
    Omega_val = eval_sym(sp.sympify("Omega"), param_values)
    S_omega_eig = data["S_omega_eig"]
    retained_triples_self = data["retained_triples_self"]

    Lops = []
    for (k, _, w) in retained_triples_self:
        w_val = eval_sym(w, param_values)
        rate = _rate_from_index(k, w_val, Omega_val,
                                kappa_h_fun, kappa_c_fun, nB_h_fun, nB_c_fun)
        S_eig = np.asarray(S_omega_eig[k][w], dtype=complex)
        Lops.append(Qobj(np.sqrt(rate) * S_eig))
    return Lops


# ============================================================
# Hamiltonian (D matrix) part
# ============================================================

def build_Hs_matrix_from_pkl(param_values: dict[str, float]) -> np.ndarray:
    """Build numeric D matrix from symbolic D_expr stored in .pkl."""
    data = load_export(pkl_path)
    Hs_eig: sp.Matrix = data["Hs_eig"]
    Hs_eval = Hs_eig.xreplace({s: param_values[s.name]
                              for s in Hs_eig.free_symbols
                              if s.name in param_values})
    Hs_num = np.array(Hs_eval.evalf(), dtype=complex)
    return Hs_num

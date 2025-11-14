import qutip as qt
import numpy as np

import src.utilities as ut

def generate_qubit_oscillator_operators(d: int):
    # --- Qubit sector (dim = 2) ---
    id_qubit = qt.qeye(2)
    sigma_m = qt.sigmam()
    sigma_p = qt.sigmap()
    P_qubit = [qt.fock_dm(2, 0), qt.fock_dm(2, 1)]

    # --- Oscillator sector (dim = d) ---
    id_osc = qt.qeye(d)

    # annihilation operator in truncated Fock basis
    a_mat = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a_mat[n-1, n] = np.sqrt(n)     # |n-1><n| term

    a = qt.Qobj(a_mat)
    adag = a.dag()
    N = adag * a

    # population projectors for oscillator
    P_osc = [qt.fock_dm(d, n) for n in range(d)]

    return {
        "id_qubit" : id_qubit,
        "sigma_m"  : sigma_m,
        "sigma_p"  : sigma_p,
        "P_qubit"  : P_qubit,

        "id_osc"   : id_osc,
        "a"        : a,
        "adag"     : adag,
        "N"        : N,
        "P_osc"    : P_osc
    }

# --- Extend operators to Hs space (Qubit ⊗ d-dim truncated harmonic oscillator) ---
def OpQubit_to_OpS(op, basic_ops):
    return qt.tensor(op, basic_ops["id_osc"])

def OpOsc_to_OpS(op, basic_ops):
    return qt.tensor(basic_ops["id_qubit"], op)

# --- Construct System Hamiltonian --- 
def construct_Hs(g, basic_ops): 
    # In rotating frame, system Hamiltonian only contains interaction between Qubit and d-dim truncated harmonic oscillator
    # Hs = g(sigma_p*a + sigma_m*a_dag)
    return g * (OpQubit_to_OpS(basic_ops["sigma_p"], basic_ops) * OpOsc_to_OpS(basic_ops["a"], basic_ops) 
                + OpQubit_to_OpS(basic_ops["sigma_m"], basic_ops) * OpOsc_to_OpS(basic_ops["adag"], basic_ops))

# --- Construct Lindblad jump operators, ---
# --- Qubit-d-dim truncated harmonic oscillator system coupling to Hot and Cold Bosinic bath with Ohmic spectrum ----
def construct_L_ops(basic_ops, Omega0, T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c):
    # Dissipation coefficients
    # Qubit (coupling to hot bath)
    Gamma_h = ut.kappa_Ohmic(Omega0, eta_h, omega_c_h) * ut.n_B(Omega0, T_h)
    Gammabar_h = ut.kappa_Ohmic(Omega0, eta_h, omega_c_h) * (ut.n_B(Omega0, T_h) + 1) 

    # d-dim truncated harmonic oscillator (coupling to cold bath) 
    Gamma_c = ut.kappa_Ohmic(Omega0, eta_c, omega_c_c) * ut.n_B(Omega0, T_c)
    Gammabar_c = ut.kappa_Ohmic(Omega0, eta_c, omega_c_c) * (ut.n_B(Omega0, T_c) + 1)

    # Lindblad operators: sqrt(rate) * operator
    L_ops = [] 
    L_ops.append(np.sqrt(Gamma_h) * OpQubit_to_OpS(basic_ops["sigma_p"], basic_ops))
    L_ops.append(np.sqrt(Gammabar_h) * OpQubit_to_OpS(basic_ops["sigma_m"], basic_ops))
    L_ops.append(np.sqrt(Gamma_c) * OpOsc_to_OpS(basic_ops["adag"], basic_ops))
    L_ops.append(np.sqrt(Gammabar_c) * OpOsc_to_OpS(basic_ops["a"], basic_ops))

    return L_ops

def calculate_steadystate_sol(basic_ops, Omega0, g, \
                                T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c):
    Hs = construct_Hs(g, basic_ops)
    L_ops = construct_L_ops(basic_ops, Omega0, T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c)
    rho_ss = qt.steadystate(Hs, L_ops)
    return rho_ss

# --- Entanglement anylysis, negativity from steadty state ---
def calculate_steady_negativity(basic_ops, Omega0, g, \
                                T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c):
    Hs = construct_Hs(g, basic_ops)
    L_ops = construct_L_ops(basic_ops, Omega0, T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c)
    rho_ss = qt.steadystate(Hs, L_ops)
    N_ss = qt.negativity(rho_ss, 1)
    return N_ss
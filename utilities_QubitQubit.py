import qutip as qt
import numpy as np

import utilities as ut

# --- Operators for sub-system ---
# Qubit Operators (dim=2)
id_2d = qt.qeye(2)
sigma_m = qt.sigmam()           # |0><1|
sigma_p = qt.sigmap()           # |1><0|
# Populations for Qubit
P0_2d = qt.fock_dm(2, 0)
P1_2d = qt.fock_dm(2, 1)

# --- Extend operators to Hs space (Qubit ⊗ Qubit) ---
def Op2d1_to_OpS(op):
    # Extend the operator on the 1st Qubit to the Qubit ⊗ Qubit system space
    return qt.tensor(op, id_2d)

def Op2d2_to_OpS(op):
    # Extend the operator on the 2nd Qubit to the Qubit ⊗ Qubit system space
    return qt.tensor(id_2d, op)

# --- Construct System Hamiltonian --- 
def construct_Hs(g): 
    # In rotating frame, system Hamiltonian only contains interaction between Qubit and Qutrit
    # Hs = g(sigma_p*sigma_m + sigma_m*sigma_p)
    return g * (Op2d1_to_OpS(sigma_p) * Op2d2_to_OpS(sigma_m) 
                + Op2d1_to_OpS(sigma_m) * Op2d2_to_OpS(sigma_p))

# --- Construct Lindblad jump operators, ---
# --- Qubit-Qubit system coupling to Hot and Cold Bosinic bath with Ohmic spectrum ----
def construct_L_ops(Omega0, T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c):
    # Dissipation coefficients
    # 1st Qubit (coupling to hot bath)
    Gamma_h = ut.kappa_Ohmic(Omega0, eta_h, omega_c_h) * ut.n_B(Omega0, T_h)
    Gammabar_h = ut.kappa_Ohmic(Omega0, eta_h, omega_c_h) * (ut.n_B(Omega0, T_h) + 1) 

    # 2nd Qubit (coupling to cold bath) 
    Gamma_c = ut.kappa_Ohmic(Omega0, eta_c, omega_c_c) * ut.n_B(Omega0, T_c)
    Gammabar_c = ut.kappa_Ohmic(Omega0, eta_c, omega_c_c) * (ut.n_B(Omega0, T_c) + 1)

    # Lindblad operators: sqrt(rate) * operator
    L_ops = [] 
    L_ops.append(np.sqrt(Gamma_h) * Op2d1_to_OpS(sigma_p))
    L_ops.append(np.sqrt(Gammabar_h) * Op2d1_to_OpS(sigma_m))
    L_ops.append(np.sqrt(Gamma_c) * Op2d2_to_OpS(sigma_p))
    L_ops.append(np.sqrt(Gammabar_c) * Op2d2_to_OpS(sigma_m))

    return L_ops

# --- Entanglement anylysis, negativity from steadty state ---
def calculate_steady_negativity(Omega0, g, \
                                T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c):
    Hs = construct_Hs(g)
    L_ops = construct_L_ops(Omega0, T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c)
    rho_ss = qt.steadystate(Hs, L_ops)
    N_ss = qt.negativity(rho_ss, 1)
    return N_ss
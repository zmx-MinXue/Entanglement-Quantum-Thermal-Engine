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

# Qutrit Operators (dim=3, truncated harmonic oscillator)
id_3d = qt.qeye(3)
# annihilation operator: a = |0><1| + sqrt(2)|1><2|
a_3d = qt.Qobj([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]])
adag_3d = a_3d.dag()
N_3d = adag_3d * a_3d 
# Populations for Qutrit
P0_3d = qt.fock_dm(3, 0)
P1_3d = qt.fock_dm(3, 1)
P2_3d = qt.fock_dm(3, 2)

# --- Extend operators to Hs space (Qubit ⊗ Qutrit) ---
def Op2d_to_OpS(op):
    return qt.tensor(op, id_3d)

def Op3d_to_OpS(op):
    return qt.tensor(id_2d, op)

# --- Construct System Hamiltonian --- 
def construct_Hs(g): 
    # In rotating frame, system Hamiltonian only contains interaction between Qubit and Qutrit
    # Hs = g(sigma_p*a + sigma_m*a_dag)
    return g * (Op2d_to_OpS(sigma_p) * Op3d_to_OpS(a_3d) 
                + Op2d_to_OpS(sigma_m) * Op3d_to_OpS(adag_3d))

# --- Construct Lindblad jump operators, ---
# --- Qubit-Qutrit system coupling to Hot and Cold Bosinic bath with Ohmic spectrum ----
def construct_L_ops(Omega0, T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c):
    # Dissipation coefficients
    # Qubit (coupling to hot bath)
    Gamma_h = ut.kappa_Ohmic(Omega0, eta_h, omega_c_h) * ut.n_B(Omega0, T_h)
    Gammabar_h = ut.kappa_Ohmic(Omega0, eta_h, omega_c_h) * (ut.n_B(Omega0, T_h) + 1) 

    # Qutrit (coupling to cold bath) 
    Gamma_c = ut.kappa_Ohmic(Omega0, eta_c, omega_c_c) * ut.n_B(Omega0, T_c)
    Gammabar_c = ut.kappa_Ohmic(Omega0, eta_c, omega_c_c) * (ut.n_B(Omega0, T_c) + 1)

    # Lindblad operators: sqrt(rate) * operator
    L_ops = [] 
    L_ops.append(np.sqrt(Gamma_h) * Op2d_to_OpS(sigma_p))
    L_ops.append(np.sqrt(Gammabar_h) * Op2d_to_OpS(sigma_m))
    L_ops.append(np.sqrt(Gamma_c) * Op3d_to_OpS(adag_3d))
    L_ops.append(np.sqrt(Gammabar_c) * Op3d_to_OpS(a_3d))

    return L_ops

# --- Entanglement anylysis, negativity from steadty state ---
def calculate_steady_negativity(Omega0, g, \
                                T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c):
    Hs = construct_Hs(g)
    L_ops = construct_L_ops(Omega0, T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c)
    rho_ss = qt.steadystate(Hs, L_ops)
    N_ss = qt.negativity(rho_ss, 1)
    return N_ss
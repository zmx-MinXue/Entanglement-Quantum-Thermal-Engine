import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

import utilities_QubitQutrit as u23

# --- Parameters ---
# Frequency & temperature units are chosen such that ħ = k_B = 1. 

# System parameters 
Omega0 = 1.0
g = 1e-2                # g << Omega0 for Local ME validity

# Hot thermal reservoir (h), coupling to Qubit 
T_h = 5
eta_h = 2e-4            # Coupling strength to Ohmic bath, dimensionless
omega_c_h = 50.0        # Cut off frequency for Ohmic bath (omega_c >> Omega0)

# Cold thermal reservoir (h), coupling to Qutrit 
T_c = 0.5
eta_c = 2e-4
omega_c_c = 50.0


# System Hamiltonian
Hs = u23.construct_Hs(g)
# Lindblad Jump Operactors (including dissipation rates)
L_ops = u23.construct_L_ops(Omega0, T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c)


# --- Solving Lindblad Master Equation ---

# Steady State Solution
print("Starting steady state calculation...\n")
rho_ss = qt.steadystate(Hs, L_ops)
print("Steady state calculation completed.\n")

# Time Evolution
# Initial state: Qubit in state |0>, Qutrit in state |0>
psi_h0 = qt.basis(2, 0)  # |0>
psi_c0 = qt.basis(3, 0)  # |0>
rho0 = qt.tensor(qt.ket2dm(psi_h0), qt.ket2dm(psi_c0))

# Time span
tlist = np.linspace(0, 10.0 / g, 200)  

# Expectation values to record (Populations here)
e_ops = [
    u23.Op2d_to_OpS(u23.P0_2d),
    u23.Op2d_to_OpS(u23.P1_2d),
    u23.Op3d_to_OpS(u23.P0_3d),
    u23.Op3d_to_OpS(u23.P1_3d),
    u23.Op3d_to_OpS(u23.P2_3d)
]

print("Starting time evolution calculation...")
options = qt.Options(store_states=True)
result = qt.mesolve(Hs, rho0, tlist, L_ops, e_ops = e_ops, options=options)
print("Time evolution calculation completed.")


# --- Results --- 

# Steady rate solution
print("--- Steady state solution ---")
print(f"Qubit (coupling to hot bath) |0> population: \
      {qt.expect(u23.Op2d_to_OpS(u23.P0_2d), rho_ss):.4f}")
print(f"Qubit (coupling to hot bath) |1> population: \
      {qt.expect(u23.Op2d_to_OpS(u23.P1_2d), rho_ss):.4f}")
print(f"Qutrit (coupling to cold bath) |0> population: \
      {qt.expect(u23.Op3d_to_OpS(u23.P0_3d), rho_ss):.4f}")
print(f"Qutrit (coupling to cold bath) |1> population: \
      {qt.expect(u23.Op3d_to_OpS(u23.P1_3d), rho_ss):.4f}")
print(f"Qutrit (coupling to cold bath) |2> population: \
      {qt.expect(u23.Op3d_to_OpS(u23.P2_3d), rho_ss):.4f}")
print(f"Qutrit (coupling to cold bath) <n>: \
      {qt.expect(u23.Op3d_to_OpS(u23.N_3d), rho_ss):.4f}")

# Time evolution
plt.figure(figsize=(10, 6))
plt.plot(tlist, result.expect[0], label="Qubit <|0><0|>")
plt.plot(tlist, result.expect[1], label="Qubit <|1><1|>")
plt.plot(tlist, result.expect[2], label="Qutrit <|0><0|>", linestyle='--')
plt.plot(tlist, result.expect[3], label="Qutrit <|1><1|>", linestyle='--')
plt.plot(tlist, result.expect[4], label="Qutrit <|2><2|>", linestyle='--')
plt.title("Time evolution for Qubit-Qutrit System, Local Master Equation")
plt.xlabel("t")
plt.ylabel("Populations")
plt.legend()
plt.grid(True)
plt.show()

# --- Analysis ---
# Calculating negativity
print("Calculating negativity...")
negativity_over_time = [qt.negativity(rho_t, 1) for rho_t in result.states]
print("Negativity calculation completed.")

# Plot negativity
plt.figure(figsize=(10, 6))
plt.plot(tlist, negativity_over_time)
plt.title("Time Evolution of Negativity")
plt.xlabel("t")
plt.ylabel("Negativity")
plt.grid(True)
plt.show()
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os
sys.path.append(os.getcwd())

import src.utilities_2xd_LocalME as u2dl

# --- Fixed Parameters ---
d = 3

g_fixed = 0.03
eta_c_fixed = 2e-2  

Omega0 = 1.0

T_h = 10.0
T_c = 0.1
omega_c_h = 50.0
omega_c_c = 50.0

basic_ops = u2dl.generate_qubit_oscillator_operators(d)

# --- 1D scan setup ---
N_points_1D = 80
eta_h_range = np.linspace(1e-5, 1e-3, N_points_1D) 

print("1D parameter sweeping for Negativity and heat currents vs eta_h")

neg_data = np.zeros(N_points_1D)

JH_TD = np.zeros(N_points_1D)
JC_TD = np.zeros(N_points_1D)
JH_HS = np.zeros(N_points_1D)
JC_HS = np.zeros(N_points_1D)

for i, eta_h in enumerate(tqdm(eta_h_range, desc="Scanning eta_h")):
    rho_ss = u2dl.calculate_steadystate_sol(
        basic_ops, Omega0, g_fixed,
        T_h, T_c, eta_h, eta_c_fixed, omega_c_h, omega_c_c,
        lab_frame=True
    )

    neg_data[i] = qt.negativity(rho_ss, 1)

    # Heat currents with H_TD
    JH_TD[i], JC_TD[i] = u2dl.calculate_thermo_heat_current(
        rho_ss, basic_ops, Omega0, g_fixed,
        T_h, T_c, eta_h, eta_c_fixed, omega_c_h, omega_c_c,
        use_Hs=False
    )

    # Heat currents with Hs_lab
    JH_HS[i], JC_HS[i] = u2dl.calculate_thermo_heat_current(
        rho_ss, basic_ops, Omega0, g_fixed,
        T_h, T_c, eta_h, eta_c_fixed, omega_c_h, omega_c_c,
        use_Hs=True
    )

Jsum_TD = JH_TD + JC_TD
Jsum_HS = JH_HS + JC_HS

print("max |J_h+J_c| using H_TD =", np.max(np.abs(Jsum_TD)))
print("max |J_h+J_c| using Hs   =", np.max(np.abs(Jsum_HS)))

print("1D sweep completed.")

# --- Find max negativity point ---
imax = int(np.argmax(neg_data))
eta_h_maxN = eta_h_range[imax]
N_max = neg_data[imax]

# --- Find max heat-current point (choose |J_h| maximum) ---
jmax = int(np.argmax(np.abs(JH_TD)))
eta_h_maxJ = eta_h_range[jmax]
Jh_max = JH_TD[jmax]
Jc_at_maxJ = JC_TD[jmax]

print("1D sweep completed.")
print(rf"Max negativity = {N_max:.6f} at eta_h = {eta_h_maxN:.2e}")
print(rf"Max |J_h| at eta_h = {eta_h_maxJ:.2e}: J_h = {Jh_max:.6e}, J_c = {Jc_at_maxJ:.6e}, J_h+J_c = {(Jh_max+Jc_at_maxJ):.3e}")

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(8, 5))

# Negativity (keep default blue)
ax1.plot(eta_h_range, neg_data, color="black", label=r'Negativity')
ax1.set_xlabel(r'$\eta_h$ (hot-bath coupling)')
ax1.set_ylabel(r'Negativity')
ax1.set_title(
    rf"($\eta_c$ = {eta_c_fixed:.1e}, $g$ = {g_fixed})"
)

# Mark max negativity
ax1.plot(eta_h_maxN, N_max, marker='o', markersize=6, color="black")
ax1.axvline(eta_h_maxN, linewidth=1, linestyle=':', alpha=0.8, color="black")
ax1.annotate(
    rf"max Neg. @ $\eta_h={eta_h_maxN:.2e}$",
    xy=(eta_h_maxN, N_max),
    xytext=(-20, -20),
    textcoords="offset points",
    color="black"
)

# Heat currents (all dashed; colors not overlapping with negativity blue)
ax2 = ax1.twinx()
ax2.plot(eta_h_range, JH_TD, linestyle="--", color="C1", label=r'$J_h$')
ax2.plot(eta_h_range, JC_TD, linestyle="--", color="C0", label=r'$J_c$')
ax2.plot(eta_h_range, JH_TD + JC_TD, linestyle="--", color="C2", label=r'$J_h + J_c$')
ax2.set_ylabel(r'Heat current')

# Mark max heat current (|J_h|)
ax2.plot(eta_h_maxJ, Jh_max, marker='*', markersize=10, color="C1")
ax2.axvline(eta_h_maxJ, linewidth=1, linestyle=':', alpha=0.8, color="C1")
ax2.annotate(
    rf"max $|J_h|$ @ $\eta_h={eta_h_maxJ:.2e}$",
    xy=(eta_h_maxJ, Jh_max),
    xytext=(10, 15),
    textcoords="offset points",
    color="C1"
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.show()



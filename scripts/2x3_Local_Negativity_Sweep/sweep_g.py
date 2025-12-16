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
basic_ops = u2dl.generate_qubit_oscillator_operators(d)

Omega0 = 1.0

T_h = 10.0
T_c = 0.1
omega_c_h = 50.0
omega_c_c = 50.0

# Fix bath couplings
eta_h_fixed = 4e-4
eta_c_fixed = 2e-2

# --- 1D scan setup: scan g ---
N_points_1D = 80
g_range = np.linspace(1e-4, 8e-2, N_points_1D)  # adjust upper bound if you want

print("1D parameter sweeping for Negativity and heat currents vs g")

neg_data = np.zeros(N_points_1D)

JH_TD = np.zeros(N_points_1D)
JC_TD = np.zeros(N_points_1D)
JH_HS = np.zeros(N_points_1D)
JC_HS = np.zeros(N_points_1D)

for i, g in enumerate(tqdm(g_range, desc="Scanning g")):
    rho_ss = u2dl.calculate_steadystate_sol(
        basic_ops, Omega0, g,
        T_h, T_c, eta_h_fixed, eta_c_fixed, omega_c_h, omega_c_c,
        lab_frame=True
    )

    neg_data[i] = qt.negativity(rho_ss, 1)

    # Heat currents with H_TD
    JH_TD[i], JC_TD[i] = u2dl.calculate_thermo_heat_current(
        rho_ss, basic_ops, Omega0, g,
        T_h, T_c, eta_h_fixed, eta_c_fixed, omega_c_h, omega_c_c,
        use_Hs=False
    )

    # Heat currents with Hs_lab
    JH_HS[i], JC_HS[i] = u2dl.calculate_thermo_heat_current(
        rho_ss, basic_ops, Omega0, g,
        T_h, T_c, eta_h_fixed, eta_c_fixed, omega_c_h, omega_c_c,
        use_Hs=True
    )

Jsum_TD = JH_TD + JC_TD
Jsum_HS = JH_HS + JC_HS

print("max |J_h+J_c| using H_TD =", np.max(np.abs(Jsum_TD)))
print("max |J_h+J_c| using Hs   =", np.max(np.abs(Jsum_HS)))
print("1D sweep completed.")

# --- Find max negativity point ---
imax = int(np.argmax(neg_data))
g_maxN = g_range[imax]
N_max = neg_data[imax]

# --- Find max heat-current point (choose |J_h| maximum, using H_TD definition) ---
jmax = int(np.argmax(np.abs(JH_TD)))
g_maxJ = g_range[jmax]
Jh_max = JH_TD[jmax]
Jc_at_maxJ = JC_TD[jmax]

print(rf"Max negativity = {N_max:.6f} at g = {g_maxN:.4f}")
print(rf"Max |J_h| at g = {g_maxJ:.4f}: "
      rf"J_h = {Jh_max:.6e}, J_c = {Jc_at_maxJ:.6e}, "
      rf"J_h+J_c = {(Jh_max+Jc_at_maxJ):.3e}")

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(8, 5))

# Negativity (black)
ax1.plot(g_range, neg_data, color="black", linewidth=2, label=r'Negativity')
ax1.set_xlabel(r'$g$ (qubit--oscillator coupling)')
ax1.set_ylabel(r'Negativity')
ax1.set_title(rf"($\eta_h$={eta_h_fixed:.1e}, $\eta_c$={eta_c_fixed:.1e}, $\Omega$={Omega0}, $T_h$={T_h}, $T_c$={T_c})")

# Mark max negativity
ax1.plot(g_maxN, N_max, marker='o', markersize=6, color="black")
ax1.axvline(g_maxN, linewidth=1, linestyle=':', alpha=0.8, color="black")
ax1.annotate(
    rf"max Neg. @ $g={g_maxN:.3f}$",
    xy=(g_maxN, N_max),
    xytext=(-20, -20),
    textcoords="offset points",
    color="black"
)

# Heat currents (all dashed)
ax2 = ax1.twinx()
ax2.plot(g_range, JH_TD, linestyle="--", color="C1", label=r'$J_h$')
ax2.plot(g_range, JC_TD, linestyle="--", color="C0", label=r'$J_c$')
ax2.plot(g_range, JH_TD + JC_TD, linestyle="--", color="C2", label=r'$J_h + J_c$')
ax2.set_ylabel(r'Heat current')

# Mark max heat current (|J_h|)
ax2.plot(g_maxJ, Jh_max, marker='*', markersize=10, color="C1")
ax2.axvline(g_maxJ, linewidth=1, linestyle=':', alpha=0.8, color="C1")
ax2.annotate(
    rf"max $|J_h|$ @ $g={g_maxJ:.3f}$",
    xy=(g_maxJ, Jh_max),
    xytext=(10, 15),
    textcoords="offset points",
    color="C1"
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.show()

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os
sys.path.append(os.getcwd())

import utilities_QubitQutrit as u23

# --- Fixed Parameters ---
Omega0 = 1.0
# g_fixed = 0.1

T_h = 10.0
T_c = 0.1
omega_c_h = 50.0
omega_c_c = 50.0

N_points_2D = 40

eta_h_fixed = 1.44e-3
g_range = np.linspace(1e-4, 0.2, N_points_2D)
# eta_h_range = np.linspace(1e-4, 3e-3, N_points_2D)
eta_c_range = np.linspace(0.01, 0.15, N_points_2D)

# --- Parameter Sweeping for eta_h, eta_c ---
print("2D parameter sweeping for N(eta_h, eta_c)")
N_data = np.zeros((N_points_2D, N_points_2D))
for i, g in enumerate(tqdm(g_range, desc="Scanning g")):
    for j, eta_c in enumerate(eta_c_range):
        N_data[i, j] = u23.calculate_steady_negativity(Omega0, g, \
                                T_h, T_c, eta_h_fixed, eta_c, omega_c_h, omega_c_c)
print("2D parameter sweeping completed.")


# --- Identifying maximum point ---
max_neg_2D = np.max(N_data)
max_idx = np.unravel_index(np.argmax(N_data), N_data.shape)

print(f"Maximum Negativity {max_neg_2D:.6f}, \
        at g = {g_range[max_idx[0]]:.2e} \
        and eta_c = {eta_c_range[max_idx[1]]:.2e}")

# --- Plotting Colormap --- 
plt.figure(figsize=(8, 6))
X, Y = np.meshgrid(eta_c_range, g_range)
plt.pcolormesh(X, Y, N_data, shading='auto', cmap='viridis', 
               vmax=max_neg_2D, vmin=0.0)
plt.colorbar(label='Negativity')
# Marking maximum point
plt.plot(eta_c_range[max_idx[1]], g_range[max_idx[0]], \
            'r*', markersize=10, label=f'Max Negativity = {max_neg_2D:.4f}')
plt.title(f'Negativity N for steady state at fixed eta_h = {eta_h_fixed}')
plt.xlabel('eta_c (Cold Bath Coupling)')
plt.ylabel('g (Qubit-Qutrit Coupling)')
plt.legend()
plt.show()
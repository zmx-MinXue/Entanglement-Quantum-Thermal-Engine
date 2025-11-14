import pickle
import time
from scipy.optimize import differential_evolution

import src.utilities_2xd_LocalME as u2dL

output_pkl = "2xd_Local_Negativity_scan_d.pkl"

# --- Fixed Parameters (same for all d) ---
Omega0 = 1.0
T_h = 10.0
T_c = 0.1
omega_c_h = 50.0
omega_c_c = 50.0

# --- Dimension d for scan ---
d_list = [2, 3, 4, 5, 6, 7] 

# --- Boundaries for optimization parameters ---
bounds = [
    (1e-5, 1e-2),   # eta_h 
    (1e-3, 0.1),    # eta_c 
    (1e-3, 0.05)    # g 
]

all_results = []

for d in d_list:
    print(f"\n============================")
    print(f"   Optimizing for d = {d}")
    print(f"============================")

    basic_ops = u2dL.generate_qubit_oscillator_operators(d)

    # --- Objective Function for d ---
    def objective_function(params):
        eta_h, eta_c, g = params
        negativity = u2dL.calculate_steady_negativity(
            basic_ops,
            Omega0, g,
            T_h, T_c,
            eta_h, eta_c,
            omega_c_h, omega_c_c
        )
        return -negativity 

    result = differential_evolution(
        objective_function,
        bounds,
        disp=True,
        polish=True
    )

    if result.success:
        optimal_params = result.x
        max_negativity = -result.fun

        print(f"\n--- Optimization Successful for d = {d} ---")
        print(f"Maximum Negativity: {max_negativity:.6f}")
        print("Optimal parameters:")
        print(f"  eta_h = {optimal_params[0]:.5e}")
        print(f"  eta_c = {optimal_params[1]:.5e}")
        print(f"      g = {optimal_params[2]:.5e}")

        all_results.append({
            "d": d,
            "max_neg": max_negativity,
            "eta_h": optimal_params[0],
            "eta_c": optimal_params[1],
            "g": optimal_params[2],
            "rho_ss": u2dL.calculate_steadystate_sol(basic_ops, Omega0, g=optimal_params[2], \
                                T_h=T_h, T_c=T_c, eta_h=optimal_params[0], eta_c=optimal_params[1], \
                                    omega_c_h=omega_c_h, omega_c_c=omega_c_c)
        })
    else:
        print(f"\n--- Optimization failure for d = {d} ---")
        print(result.message)


with open(output_pkl, "wb") as f:
    pickle.dump(all_results, f)

print(f"\nResults saved to {output_pkl}")
import time
import os, sys
import pickle as pkl
import numpy as np
from scipy.optimize import differential_evolution

import src.utilities_2xd_GlobalME as u2dg

result_path = "results/data/2xd_Global_Negativity_scan_d.pkl"

# --- Fixed Parameters ---
d = 2 

Omega0 = 1.0
T_h = 10.0
T_c = 0.1
omega_c_h = 50.0
omega_c_c = 50.0

# --- Objective Function ---
def objective_function(params):
    eta_h, eta_c, g = params
    
    negativity = u2dg.calculate_steady_negativity(d, Omega0, g, \
                            T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c)
    
    return -negativity


# --- Boundary for optimiaztion parameters ---
bounds = [
    (1e-10, 1e-2),       # eta_h 
    (1e-9, 0.5),        # eta_c 
    (1e-5, 5)         # g 
]

# --- Initial Guess ---
initial_guess = [
    np.random.uniform(bounds[0][0], bounds[0][1]),
    np.random.uniform(bounds[1][0], bounds[1][1]),
    np.random.uniform(bounds[2][0], bounds[2][1])
]

print("Optimizing negativity, params (eta_h, eta_c, g)...")
print(f"Initial guess: eta_h={initial_guess[0]:.2e}, eta_c={initial_guess[1]:.2e}, g={initial_guess[2]:.2f}")
print(f"Boundary: eta_h={bounds[0]}, eta_c={bounds[1]}, g={bounds[2]}")

start_time = time.time()

# --- Optimizer ---
result = differential_evolution(
    objective_function, 
    bounds, 
    disp=True, 
    polish=True
)

end_time = time.time()
print(f"Optimizing time usage: {end_time - start_time:.2f} s")

# --- Results ---
if result.success:
    optimal_params = result.x
    max_negativity = -result.fun  

    print("\n--- Optimization Successful. ---")
    print(f"Maximum Negativity: {max_negativity:.6f}")
    print("Parameters:")
    print(f"  eta_h = {optimal_params[0]:.5e}")
    print(f"  eta_c = {optimal_params[1]:.5e}")
    print(f"      g = {optimal_params[2]:.5e}")

    if os.path.exists(result_path):
        with open(result_path, "rb") as f:
            all_data = pkl.load(f)
        if not isinstance(all_data, dict):
            raise TypeError(f"{result_path} has incorrect data structure. ")
    else:
        all_data = {}

    all_data[d] = {
        "max_neg": max_negativity,
        "g": optimal_params[2],
        "eta_h": optimal_params[0],
        "eta_c": optimal_params[1],
        "rho_ss": u2dg.calculate_steadystate_sol(d, Omega0, g=optimal_params[2], \
                                T_h=T_h, T_c=T_c, eta_h=optimal_params[0], eta_c=optimal_params[1], \
                                    omega_c_h=omega_c_h, omega_c_c=omega_c_c), 
    }

    with open(result_path, "wb") as f:
        pkl.dump(all_data, f)

    print(f"Saved data for d = {d} into {result_path}")

else:
    print("\n--- Optimization failure. ---")
    print(result.message)
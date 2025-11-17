import qutip as qt
import numpy as np
import numdifftools as nd
from scipy.optimize import differential_evolution
import time

import sys
import os
sys.path.append(os.getcwd())

import utilities_QubitQutrit_GlobalME as u23g

# --- Fixed Parameters ---
Omega0 = 1.0
T_h = 10.0
T_c = 0.1
omega_c_h = 50.0
omega_c_c = 50.0

# --- Objective Function ---
def objective_function(params):
    eta_h, eta_c, g = params
    negativity = u23g.calculate_steady_negativity(Omega0, g, \
                            T_h, T_c, eta_h, eta_c, omega_c_h, omega_c_c)
    
    return -negativity


# --- Boundary for optimiaztion parameters ---
bounds = [
    (1e-10, 1e-2),       # eta_h 
    (1e-9, 1),        # eta_c 
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

else:
    print("\n--- Optimization failure. ---")
    print(result.message)
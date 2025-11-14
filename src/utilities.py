import numpy as np

# --- Dissipation ---
# Ohmic spectrum: 
# J(omega) = eta * omega * exp(-omega/omega_c)
# kappa(omega) = 2 * pi * J(omega)
def kappa_Ohmic(omega, eta, omega_c): 
    return 2 * np.pi * eta * omega * np.exp(- omega / omega_c)

# Bose-Einstein distribution: n_B(w, T)
def n_B(omega, T):
    return 1.0 / (np.exp(omega/T) - 1.0)

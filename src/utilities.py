import numpy as np

# --- Dissipation ---
# Ohmic spectrum: 
# J(omega) = eta * omega * exp(-omega/omega_c)
# kappa(omega) = 2 * pi * J(omega)
def kappa_Ohmic(omega, eta, omega_c): 
    w = abs(omega)
    return 2 * np.pi * eta * w * np.exp(- w / omega_c)

# Bose-Einstein distribution: n_B(w, T)
def n_B(omega, T):
    w = abs(omega)
    return 1.0 / (np.exp(w/T) - 1.0)

import utilities as ut

Omega0 = 1.0
T_h = 10.0
T_c = 0.1
omega_c_h = 50.0
omega_c_c = 50.0

eta_h = 1.51e-03
eta_c = 6.74e-02
g = 9.75e-02

Gamma_h = ut.kappa_Ohmic(Omega0, eta_h, omega_c_h) * ut.n_B(Omega0, T_h)
Gammabar_h = ut.kappa_Ohmic(Omega0, eta_h, omega_c_h) * (ut.n_B(Omega0, T_h) + 1) 

Gamma_c = ut.kappa_Ohmic(Omega0, eta_c, omega_c_c) * ut.n_B(Omega0, T_c)
Gammabar_c = ut.kappa_Ohmic(Omega0, eta_c, omega_c_c) * (ut.n_B(Omega0, T_c) + 1)

print("Gamma_h: "+str(Gamma_h))
print("Gammabar_h: "+str(Gammabar_h))
print("Gamma_c: "+str(Gamma_c))
print("Gammabar_c: "+str(Gammabar_c))
print("g: "+str(g))
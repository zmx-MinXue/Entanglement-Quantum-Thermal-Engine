import numpy as np
import qutip as qt
import utilities as ut
import utilities_QubitQutrit_GlobalME as u23g
from functools import partial

T_h = 10.0
T_c = 0.1
omega_c_h = 50.0
omega_c_c = 50.0

eta_h = 1.51e-03
eta_c = 6.74e-02
g = 9.75e-02

param_values = {"Omega": 1.0, "g": g} 

nB_h_fun = partial(ut.n_B, T=T_h) 
nB_c_fun = partial(ut.n_B, T=T_c) 
kappa_h_fun = partial(ut.kappa_Ohmic, eta=eta_h, omega_c=omega_c_h) 
kappa_c_fun = partial(ut.kappa_Ohmic, eta=eta_c, omega_c=omega_c_c) 

# Note that Lops and Hs are both in Hs eigenbasis. 
Lops = u23g.build_Lops_from_pkl(
    kappa_h_fun=kappa_h_fun, 
    kappa_c_fun=kappa_c_fun, 
    nB_h_fun=nB_h_fun, 
    nB_c_fun=nB_c_fun, 
    param_values=param_values, 
)

Hs_num = u23g.build_Hs_matrix_from_pkl(param_values) 
Hs = qt.Qobj(np.array(Hs_num, dtype=complex)) 

rho_ss = qt.steadystate(Hs, Lops)
print(u23g.to_computational_basis(rho_ss.full()))

rho_ss = qt.Qobj(u23g.to_computational_basis(rho_ss.full()), \
                 dims=[[2,3],[2,3]])
N = qt.negativity(rho_ss, subsys=0)
print("Negativity =", N)
print(qt.negativity(rho_ss, subsys=1))

# rho0 = np.zeros((6,6), dtype=complex)
# rho0[0,0] = 1.0
# print(u23g.to_computational_basis(rho0))
# rho0=qt.Qobj(rho0)


# tlist = np.linspace(0, 20, 201)
# result = qt.mesolve(H, rho0, tlist, Lops)

# # 选前3个能级的投影
# proj_list = [qt.ket2dm(qt.basis(6, n)) for n in range(3)]
# labels = [f"p{n}" for n in range(len(proj_list))]

# # 用“列表”作为 e_ops，并在末尾追加 Tr 与 <H>
# e_ops = proj_list + [qt.qeye(6), H]   # 顺序：p0, p1, p2, Tr, <H>

# result = qt.mesolve(H, rho0, tlist, Lops, e_ops=e_ops)

# # 拆出结果：前 len(proj_list) 个是各占据，倒数第二是 Tr，倒数第一是 <H>
# pop_traces = result.expect[:len(proj_list)]
# trace_vals = result.expect[-2]
# energy_vals = result.expect[-1]

# # 画占据 + 迹
# plt.figure(figsize=(8,5))
# for lab, y in zip(labels, pop_traces):
#     plt.plot(tlist, y, label=lab, linewidth=1.8)
# plt.plot(tlist, trace_vals, "--", label="Tr(rho)", linewidth=1.2)
# plt.xlabel("t"); plt.ylabel("population / trace")
# plt.legend(); plt.tight_layout(); plt.show()

# # 画能量期望
# plt.figure(figsize=(7,4))
# plt.plot(tlist, energy_vals, label="<H>", linewidth=1.8)
# plt.xlabel("t"); plt.ylabel("Energy expectation")
# plt.legend(); plt.tight_layout(); plt.show()
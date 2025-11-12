import numpy as np
from qutip import Qobj, mesolve
from qutip import basis, ket2dm, expect, qeye
from functools import partial
import matplotlib.pyplot as plt
from utilities_QubitQutrit_GlobalME import build_Lops_from_pkl, build_Hs_matrix_from_pkl
from utilities import kappa_Ohmic, n_B

T_h = 10.0
T_c = 0.1
omega_c_h = 50.0
omega_c_c = 50.0

eta_h = 1.51e-03
eta_c = 6.74e-02
g = 9.75e-02

nB_h_fun     = partial(n_B, T=T_h)                 
nB_c_fun     = partial(n_B, T=T_c) 
kappa_h_fun = partial(kappa_Ohmic, eta=eta_h, omega_c=omega_c_h)
kappa_c_fun = partial(kappa_Ohmic, eta=eta_c, omega_c=omega_c_c)

param_values = {"Omega": 1.0, "g": 9.75e-02}
Lops = build_Lops_from_pkl(
    kappa_h_fun=kappa_h_fun,
    kappa_c_fun=kappa_c_fun,
    nB_h_fun=nB_h_fun,
    nB_c_fun=nB_c_fun,
    param_values=param_values,
)
print(f"# of Lops = {len(Lops)}")

# 准备 H 与 ρ0（必须与 Lops 在同一基）
# 例如：在本征基下，H = D（对角），ρ0 任意
D_num = build_Hs_matrix_from_pkl(param_values)
H = Qobj(np.array(D_num, dtype=complex))   
rho0 = np.zeros((6,6), dtype=complex)
rho0[0,0] = 1.0
rho0=Qobj(rho0)


tlist = np.linspace(0, 20, 201)
# result = mesolve(H, rho0, tlist, Lops)

# 选前3个能级的投影
proj_list = [ket2dm(basis(6, n)) for n in range(3)]
labels = [f"p{n}" for n in range(len(proj_list))]

# 用“列表”作为 e_ops，并在末尾追加 Tr 与 <H>
e_ops = proj_list + [qeye(6), H]   # 顺序：p0, p1, p2, Tr, <H>

result = mesolve(H, rho0, tlist, Lops, e_ops=e_ops)

# 拆出结果：前 len(proj_list) 个是各占据，倒数第二是 Tr，倒数第一是 <H>
pop_traces = result.expect[:len(proj_list)]
trace_vals = result.expect[-2]
energy_vals = result.expect[-1]

# 画占据 + 迹
plt.figure(figsize=(8,5))
for lab, y in zip(labels, pop_traces):
    plt.plot(tlist, y, label=lab, linewidth=1.8)
plt.plot(tlist, trace_vals, "--", label="Tr(rho)", linewidth=1.2)
plt.xlabel("t"); plt.ylabel("population / trace")
plt.legend(); plt.tight_layout(); plt.show()

# 画能量期望
plt.figure(figsize=(7,4))
plt.plot(tlist, energy_vals, label="<H>", linewidth=1.8)
plt.xlabel("t"); plt.ylabel("Energy expectation")
plt.legend(); plt.tight_layout(); plt.show()
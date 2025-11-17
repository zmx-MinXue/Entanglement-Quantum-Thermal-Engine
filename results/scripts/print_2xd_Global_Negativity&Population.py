import pickle
import numpy as np
import qutip as qt

pkl_file = "results/data/2xd_Global_Negativity_scan_d.pkl"

with open(pkl_file, "rb") as f:
    all_data = pickle.load(f)

# all_data is a dict with key d： d -> {...}
d_list = sorted(all_data.keys())

# --- Negativity ---
neg_table = {d: all_data[d]["max_neg"] for d in d_list}

# --- Compute population matrix ---
max_d = max(d_list)
pop_table = []

for d in d_list:
    entry = all_data[d]
    rho_ss = entry["rho_ss"]

    if isinstance(rho_ss, np.ndarray):
        rho_ss = qt.Qobj(rho_ss, dims=[[2, d], [2, d]])
    elif isinstance(rho_ss, qt.Qobj):
        rho_ss.dims = [[2, d], [2, d]]
    else:
        raise TypeError(f"Unsupported type for rho_ss at d={d}: {type(rho_ss)}")

    row_pop = []
    for n in range(d):
        Pn_osc = qt.fock_dm(d, n)
        Pn_full = qt.tensor(qt.qeye(2), Pn_osc)
        pop_n = (Pn_full * rho_ss).tr().real
        row_pop.append(pop_n)

    while len(row_pop) < max_d:
        row_pop.append("")

    pop_table.append((d, row_pop))

# --- Print table ---
col_width = 12

# --------- header：d | Negativity | n=0 n=1 ... ----------
header_cells = ["d", "Negativity"] + [f"n={n}" for n in range(max_d)]
header_str = (
    f"{header_cells[0]:>4} | "
    f"{header_cells[1]:>{col_width}} |"
    + " ".join(f"{c:>{col_width}}" for c in header_cells[2:])
)
print(header_str)

print("-" * len(header_str))

# --------- row for d：d, Neg, populations ----------
for d, row in pop_table:
    # Negativity
    neg = neg_table.get(d, None)
    if isinstance(neg, float):
        neg_str = f"{neg:>{col_width}.5e}"
    else:
        neg_str = " " * col_width

    # Populations
    row_str = []
    for x in row:
        if isinstance(x, float):
            row_str.append(f"{x:>{col_width}.5e}")
        else:
            row_str.append(" " * col_width)

    line = f"{d:>4} | {neg_str} |" + " ".join(row_str)
    print(line)
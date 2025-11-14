import pickle
import numpy as np
import qutip as qt

pkl_file = "results/data/2xd_Local_Negativity_scan_d.pkl"

with open(pkl_file, "rb") as f:
    results = pickle.load(f)

results = sorted(results, key=lambda x: x["d"])

# --- Negativity ---
neg_table = {item["d"]: item["max_neg"] for item in results}

# --- Compute population matrix ---
max_d = max(r["d"] for r in results)
pop_table = []

for item in results:
    d = item["d"]
    rho_ss = item["rho_ss"]

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

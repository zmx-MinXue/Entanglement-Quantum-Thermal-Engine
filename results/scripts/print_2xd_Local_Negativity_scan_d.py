import pickle

pkl_file = "results/data/2xd_Local_Negativity_scan_d.pkl"

with open(pkl_file, "rb") as f:
    results = pickle.load(f)

print("\n========== Loaded Results ==========\n")

for item in results:
    d = item["d"]
    max_neg = item["max_neg"]
    eta_h = item["eta_h"]
    eta_c = item["eta_c"]
    g = item["g"]

    print(f"--- d = {d} ---")
    print(f"Max Negativity = {max_neg:.6f}")
    print(f"eta_h = {eta_h:.5e}")
    print(f"eta_c = {eta_c:.5e}")
    print(f"    g = {g:.5e}")
    print()

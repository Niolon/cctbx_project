"""Debug script to compare U_eq values with reference."""
import json
import numpy as np
import os

FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_150k_low10"

# Load reference
ref_path = os.path.join(FIXTURE_DIR, "ref_results.json")
with open(ref_path, 'r') as f:
    ref_data = json.load(f)

u_cart = np.array(ref_data['u_cart'])
print("Reference U_cart shape:", u_cart.shape)
print("Reference U_cart first 5 atoms:")
for i in range(min(5, len(u_cart))):
    u = u_cart[i]
    u_eq = (u[0,0] + u[1,1] + u[2,2]) / 3
    print(f"  Atom {i}: U_eq = {u_eq:.6f} Å²")
    print(f"           U11={u[0,0]:.6f}, U22={u[1,1]:.6f}, U33={u[2,2]:.6f}")

print("\nOther reference data:")
print(f"  Temperature: {ref_data['temperature']} K")
print(f"  Medium limit: {ref_data['limits']['medium_freq_limit']} cm^-1")
print(f"  High limit: {ref_data['limits']['high_freq_limit']} cm^-1")

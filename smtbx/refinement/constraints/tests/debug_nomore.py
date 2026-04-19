"""Debug script to compare constants and computation."""
import scipy.constants as const
import numpy as np

# Calculate precise constants
K_TO_CM1 = const.k / (const.h * const.c * 100)
U_CART_FACTOR_CM1 = const.hbar * 1e20 / (const.atomic_mass * 2 * const.pi * const.c * 100)

print("=== PHYSICAL CONSTANTS ===")
print(f"K_TO_CM1 (scipy): {K_TO_CM1}")
print(f"K_TO_CM1 (mine):  0.6950348004")
print(f"U_CART_FACTOR_CM1 (scipy): {U_CART_FACTOR_CM1}")
print(f"U_CART_FACTOR_CM1 (mine):  16.8576303")

# Load fixture and check weights
import json
import os

fixture_dir = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_150k_low10"
npz_path = os.path.join(fixture_dir, "phonon_data.npz")

if os.path.exists(npz_path):
    print("\n=== PHONON DATA ===")
    data = np.load(npz_path, allow_pickle=False)
    print("Keys:", list(data.keys()))
    print("Weights:", data['weights'][:10], "...")
    print("Weights sum:", np.sum(data['weights']))
    print("Frequencies:", data['frequencies_cm1'][:10], "...")
    print("Q-points:", data['q_points'] if 'q_points' in data else "N/A")

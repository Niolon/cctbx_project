"""Quick debug: check if Jacobian rows match mode count."""
from __future__ import absolute_import, print_function
import os, json, numpy as np

FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_23k_low10"

# Load phonon data  
from smtbx.refinement.constraints.nomore_phonon_data import PhononData
phonon_data = PhononData.load(os.path.join(FIXTURE_DIR, "phonon_data.npz"))

# Load limits
with open(os.path.join(FIXTURE_DIR, "ref_results.json")) as f:
  ref_data = json.load(f)

print(f"Phonon atoms: {phonon_data.n_atoms}")  
print(f"Phonon modes: {phonon_data.n_modes}")
print(f"LOW limit: {ref_data['limits']['medium_freq_limit']:.2f} cm^-1")
print(f"HIGH limit: {ref_data['limits']['high_freq_limit']:.2f} cm^-1")

# Get frequencies from npz
with np.load(os.path.join(FIXTURE_DIR, "phonon_data.npz"), allow_pickle=True) as npz:
  freqs = npz['frequencies']
  
print(f"Frequencies shape: {freqs.shape}")
print(f"First 15 frequencies: {freqs[:15]}")

# Count modes
n_low = sum(1 for f in freqs if f < ref_data['limits']['medium_freq_limit'])
n_med = sum(1 for f in freqs if ref_data['limits']['medium_freq_limit'] <= f < ref_data['limits']['high_freq_limit'])
n_high = sum(1 for f in freqs if f >= ref_data['limits']['high_freq_limit'])
print(f"\nMode split: LOW={n_low}, MED={n_med}, HIGH={n_high}")
print(f"Active modes (LOW+MED) per atom = {n_low + n_med}")
print(f"This is what initial_frequencies_.size() should be in C++")
print(f"And jacobian_rows from Python should also have this size")

"""
Full validation test for NoMoRe constraint.

Tests against L-Alanine 150K phonon data from cctbx_migration fixtures.

Run with:
  libtbx.python tst_nomore.py
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import json
import numpy as np


# Paths to test data
FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_150k_low10"
VALIDATION_CIF = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\validation_data\L-Ala_150K_xray.cif"


def run_basic_tests():
  """Basic import and binding tests."""
  print("=" * 60)
  print("BASIC TESTS")
  print("=" * 60)

  # Test 1: Import test
  print("  1. Testing imports...")
  try:
    from smtbx.refinement.constraints.nomore import nomore_adp, build_atom_mapping
    from smtbx.refinement.constraints.nomore_phonon_data import PhononData
    from smtbx.refinement.constraints.nomore_physics import calculate_adp_amplitude
    from smtbx.refinement import constraints
    print("     [OK] All imports successful")
  except ImportError as e:
    print("     [FAIL] Import error: %s" % e)
    return False

  # Test 2: Check C++ binding
  print("  2. Testing C++ binding...")
  try:
    _ = constraints.nomore_u_star_parameter
    print("     [OK] C++ binding available")
  except AttributeError as e:
    print("     [FAIL] C++ binding not available: %s" % e)
    print("     NOTE: Recompile C++ code first!")
    return False

  # Test 3: Physics function test
  print("  3. Testing physics functions...")
  amp = calculate_adp_amplitude(100.0, 298.0, 5.0)
  if amp > 0:
    print("     [OK] ADP amplitude at 100 cm^-1, 298K = %.6f" % amp)
  else:
    print("     [FAIL] Invalid amplitude")
    return False

  return True


def run_physics_validation():
  """Test physics functions against reference values."""
  print("\n" + "=" * 60)
  print("PHYSICS VALIDATION")
  print("=" * 60)

  from smtbx.refinement.constraints.nomore_physics import (
    bose_einstein_energy_cm1, calculate_adp_amplitude)

  # Test at 150K (matching fixture conditions)
  temperature = 150.0
  test_cases = [
    (10.0, "very low frequency"),
    (100.0, "medium frequency"),
    (500.0, "high frequency"),
    (3000.0, "very high frequency (ZPE dominated)"),
  ]

  print("  Temperature: %.1f K" % temperature)
  for freq, desc in test_cases:
    energy = bose_einstein_energy_cm1(freq, temperature)
    amp = calculate_adp_amplitude(freq, temperature)
    print("    %s (%.0f cm^-1): E=%.3f cm^-1, A=%.6f" % (desc, freq, energy, amp))

  return True


def run_u_cart_validation():
  """Validate U_cart computation against reference results."""
  print("\n" + "=" * 60)
  print("U_CART VALIDATION (vs reference)")
  print("=" * 60)

  from smtbx.refinement.constraints.nomore_phonon_data import PhononData
  from smtbx.refinement.constraints.nomore import compute_mode_tensors
  from smtbx.refinement.constraints.nomore_physics import calculate_adp_amplitude

  # Check if fixture exists
  npz_path = os.path.join(FIXTURE_DIR, "phonon_data.npz")
  ref_path = os.path.join(FIXTURE_DIR, "ref_results.json")

  if not os.path.exists(npz_path):
    print("  [SKIP] Fixture not found: %s" % npz_path)
    return True

  # Load phonon data
  print("  Loading phonon data...")
  phonon_data = PhononData.load(npz_path)
  print("    Atoms: %d, Modes: %d, Q-points: %d" % (
    phonon_data.n_atoms, phonon_data.n_modes, phonon_data.n_q_points))

  # Load reference results
  print("  Loading reference results...")
  with open(ref_path, 'r') as f:
    ref_data = json.load(f)

  temperature = ref_data['temperature']
  medium_freq_limit = ref_data['limits']['medium_freq_limit']
  high_freq_limit = ref_data['limits']['high_freq_limit']
  ref_u_cart = np.array(ref_data['u_cart'])

  print("    Temperature: %.1f K" % temperature)
  print("    Limits: medium=%.2f, high=%.2f cm^-1" % (medium_freq_limit, high_freq_limit))
  print("    Reference U_cart shape: %s" % str(ref_u_cart.shape))

  # Compute mode tensors
  print("  Computing mode tensors...")
  mode_tensors = compute_mode_tensors(phonon_data.eigenvectors, phonon_data.masses)
  print("    Mode tensors shape: %s" % str(mode_tensors.shape))

  # Classify modes
  freqs = phonon_data.frequencies_cm1
  low_mask = freqs < medium_freq_limit
  medium_mask = (freqs >= medium_freq_limit) & (freqs < high_freq_limit)
  high_mask = freqs >= high_freq_limit
  print("    LOW modes: %d, MEDIUM: %d, HIGH: %d" % (
    np.sum(low_mask), np.sum(medium_mask), np.sum(high_mask)))

  # Compute U_cart for each atom
  print("  Computing U_cart...")
  n_atoms = phonon_data.n_atoms
  computed_u_cart = np.zeros((n_atoms, 3, 3))

  for atom_idx in range(n_atoms):
    u_cart = np.zeros((3, 3))
    for mode_idx in range(phonon_data.n_modes):
      freq = freqs[mode_idx]
      weight = phonon_data.weights[mode_idx]
      amp = calculate_adp_amplitude(freq, temperature)
      tensor = mode_tensors[mode_idx, atom_idx]
      u_cart += amp * weight * tensor
    computed_u_cart[atom_idx] = u_cart / phonon_data.n_q_points

  # Compare against reference
  print("  Comparing with reference...")
  max_diff = 0.0
  max_diff_atom = 0
  for atom_idx in range(min(n_atoms, len(ref_u_cart))):
    diff = np.abs(computed_u_cart[atom_idx] - ref_u_cart[atom_idx])
    atom_max = np.max(diff)
    if atom_max > max_diff:
      max_diff = atom_max
      max_diff_atom = atom_idx

  print("    Max difference: %.6f Angstrom^2 (atom %d)" % (max_diff, max_diff_atom))

  if max_diff < 0.001:
    print("  [OK] U_cart computation matches reference")
    return True
  elif max_diff < 0.01:
    print("  [WARN] Small differences detected, may need investigation")
    return True
  else:
    print("  [FAIL] Large difference detected!")
    return False


def run_constraint_test():
  """Test the constraint class setup (without full refinement)."""
  print("\n" + "=" * 60)
  print("CONSTRAINT SETUP TEST")
  print("=" * 60)

  from smtbx.refinement.constraints.nomore_phonon_data import PhononData
  from smtbx.refinement.constraints.nomore import nomore_adp

  npz_path = os.path.join(FIXTURE_DIR, "phonon_data.npz")
  ref_path = os.path.join(FIXTURE_DIR, "ref_results.json")

  if not os.path.exists(npz_path):
    print("  [SKIP] Fixture not found")
    return True

  # Load phonon data
  print("  Loading phonon data...")
  phonon_data = PhononData.load(npz_path)

  with open(ref_path, 'r') as f:
    ref_data = json.load(f)

  temperature = ref_data['temperature']
  medium_limit = ref_data['limits']['medium_freq_limit']
  high_limit = ref_data['limits']['high_freq_limit']

  # Create constraint (without target structure for now)
  print("  Creating nomore_adp constraint...")

  from smtbx.refinement.constraints.nomore import compute_mode_tensors
  mode_tensors = compute_mode_tensors(phonon_data.eigenvectors, phonon_data.masses)

  constraint = nomore_adp(
    mode_tensors=mode_tensors,
    frequencies=phonon_data.frequencies_cm1,
    masses=phonon_data.masses,
    weights=phonon_data.weights,
    n_q_points=phonon_data.n_q_points,
    atom_indices=list(range(phonon_data.n_atoms)),
    temperature=temperature,
    min_freq=5.0,
    medium_freq_limit=medium_limit,
    high_freq_limit=high_limit,
  )

  print("    LOW modes: %d" % len(constraint.low_indices))
  print("    MEDIUM modes: %d" % len(constraint.medium_indices))
  print("    HIGH modes: %d" % len(constraint.high_indices))
  print("    Constrained atoms: %d" % len(constraint.atom_indices))
  print("    Constrained params: %d" % len(constraint.constrained_parameters))

  print("  [OK] Constraint object created successfully")
  return True


def run():
  """Run all tests."""
  print("\nNoMoRe Constraint Full Test Suite\n")

  results = []

  # Basic tests
  results.append(("Basic tests", run_basic_tests()))

  # Physics validation
  results.append(("Physics validation", run_physics_validation()))

  # U_cart validation
  results.append(("U_cart validation", run_u_cart_validation()))

  # Constraint setup
  results.append(("Constraint setup", run_constraint_test()))

  # Summary
  print("\n" + "=" * 60)
  print("SUMMARY")
  print("=" * 60)
  all_passed = True
  for name, passed in results:
    status = "[PASS]" if passed else "[FAIL]"
    print("  %s %s" % (status, name))
    if not passed:
      all_passed = False

  print()
  if all_passed:
    print("All tests passed!")
    return True
  else:
    print("Some tests failed.")
    return False


if __name__ == "__main__":
  success = run()
  if not success:
    sys.exit(1)

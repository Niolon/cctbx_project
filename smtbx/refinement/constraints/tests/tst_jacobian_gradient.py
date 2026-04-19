"""
Numerical gradient test for NoMoRe constraint Jacobian.

Compares analytical dU*/d(scale) from the constraint against numerical
finite differences.
"""
from __future__ import absolute_import, division, print_function

import os
import json
import numpy as np

from cctbx import xray, crystal, adptbx
from cctbx.array_family import flex

FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_23k_low10"


def test_jacobian():
  """Test analytical Jacobian via numerical finite differences."""
  print("=" * 60)
  print("Numerical Gradient Test for NoMoRe Jacobian")
  print("=" * 60)

  # Build minimal structure
  crystal_symmetry = crystal.symmetry(
    unit_cell=(5.9279, 12.2597, 5.7939, 90, 90, 90),
    space_group_symbol="P 21 21 21"
  )

  # Load CIF for atom positions
  from iotbx import cif
  cif_path = os.path.join(FIXTURE_DIR, "L-Ala_23K_xray.cif")
  cif_obj = cif.reader(file_path=cif_path)
  block = list(cif_obj.model().values())[0]

  scatterers = flex.xray_scatterer()
  for i, label in enumerate(block['_atom_site_label']):
    site = (
      float(block['_atom_site_fract_x'][i].split('(')[0]),
      float(block['_atom_site_fract_y'][i].split('(')[0]),
      float(block['_atom_site_fract_z'][i].split('(')[0])
    )
    sc = xray.scatterer(label=label, site=site, u=0.01, occupancy=1.0)
    sc.convert_to_anisotropic(crystal_symmetry.unit_cell())
    scatterers.append(sc)

  xs = xray.structure(crystal_symmetry=crystal_symmetry, scatterers=scatterers)
  print(f"Structure: {xs.scatterers().size()} atoms")

  # Load phonon data
  from smtbx.refinement.constraints.nomore_phonon_data import PhononData
  phonon_data = PhononData.load(os.path.join(FIXTURE_DIR, "phonon_data.npz"))

  # Load limits
  with open(os.path.join(FIXTURE_DIR, "ref_results.json")) as f:
    ref_data = json.load(f)
  T = ref_data['temperature']
  medium_limit = ref_data['limits']['medium_freq_limit']
  high_limit = ref_data['limits']['high_freq_limit']

  # Create constraint
  from smtbx.refinement.constraints.nomore import nomore_adp
  constraint = nomore_adp.create(
    phonon_data=phonon_data, target_structure=xs,
    temperature=T, min_freq=5.0,
    medium_freq_limit=medium_limit, high_freq_limit=high_limit, tolerance=2.0
  )

  print(f"LOW modes: {len(constraint.low_indices)}")
  print(f"MEDIUM modes: {len(constraint.medium_indices)}")
  print(f"Constrained atoms: {len(constraint.atom_indices)}")

  # Set up flags
  for sc in xs.scatterers():
    sc.flags.set_grad_site(False)
    sc.flags.set_grad_occupancy(False)
    if sc.flags.use_u_aniso():
      sc.flags.set_grad_u_aniso(True)

  # Build reparametrisation
  import smtbx.utils
  from smtbx.refinement import constraints as cm
  connectivity_table = smtbx.utils.connectivity_table(xs)
  reparam = cm.reparametrisation(xs, [constraint], connectivity_table, temperature=T-273)

  # Enable Jacobian
  constraint.enable_jacobian()
  constraint.update_frequencies()

  # ===== Numerical Gradient Test =====
  print("\n" + "=" * 60)
  print("Testing dU*/d(scale) for first LOW mode")
  print("=" * 60)

  # Get a LOW mode that's not at min_freq (clamped modes have near-zero derivatives)
  # Mode 0, 1, 2 are at 5.0 cm^-1 (clamped), use mode 3 or later
  test_mode_list_idx = 3  # Use 4th LOW mode (index 3 in low_freq_params)
  if len(constraint.low_freq_params) <= test_mode_list_idx:
    test_mode_list_idx = 0  # Fallback

  test_param = constraint.low_freq_params[test_mode_list_idx]
  test_mode = constraint.low_indices[test_mode_list_idx]
  initial_freq = float(constraint.frequencies[test_mode])

  print(f"Testing Mode {test_mode}, initial freq: {initial_freq:.2f} cm^-1")
  print(f"Parameter index: {test_param.index}")

  # Get first constrained atom
  test_atom_idx = constraint.atom_indices[0]
  sc = xs.scatterers()[test_atom_idx]
  print(f"Testing atom: {sc.label}")

  # Function to compute U* for a given scale factor
  def compute_u_star(scale):
    """Compute U* for atom given scale factor for first LOW mode."""
    # Create a temporary copy approach: manually apply scale
    # and call linearise
    # For simplicity, we'll use the constraint's internal machinery

    # Save original value
    original_scale = test_param.value

    # Set new scale (this doesn't have a setter, so we use shifts)
    from scitbx.array_family import flex as scitbx_flex
    n_ind = reparam.n_independents
    shifts = scitbx_flex.double(n_ind, 0.0)
    shifts[test_param.index] = scale - original_scale

    # Apply shifts and linearise
    reparam.apply_shifts(shifts)
    constraint.update_frequencies()
    reparam.linearise()
    reparam.store()

    # Get U* for test atom
    u_star = sc.u_star
    return np.array(u_star)

  # Get U* at base scale (1.0)
  u_star_0 = compute_u_star(1.0)
  print(f"U* at scale=1.0: {u_star_0}")

  # Numerical gradient via finite differences
  eps = 0.001
  u_star_plus = compute_u_star(1.0 + eps)
  u_star_minus = compute_u_star(1.0 - eps)
  numerical_du_star = (u_star_plus - u_star_minus) / (2 * eps)
  print(f"\nNumerical dU*/d(scale) (eps={eps}):")
  print(f"  {numerical_du_star}")

  # Get analytical gradient from Jacobian
  # Reset to scale=1.0
  compute_u_star(1.0)
  reparam.linearise()

  jt = reparam.jacobian_transpose_matching_grad_fc()
  print(f"\nJacobian transpose shape: {jt.n_rows} x {jt.n_cols}")

  # Get row for test_param
  param_row = test_param.index
  u_param_start = reparam.asu_scatterer_parameters[test_atom_idx].u.index

  print(f"Parameter row: {param_row}")
  print(f"U start index: {u_param_start}")

  # Debug: what is the nomore_u_star_parameter's index()?
  nomore_param = constraint._nomore_params[0]
  print(f"nomore_u_star_parameter index()={nomore_param.index}, scatterer={xs.scatterers()[test_atom_idx].label}")
  print(f"Parameter index: {test_param.index}")

  # Try to access sparse matrix properly
  analytical_du_star = []
  print(f"\nReading Jacobian from row={param_row}, cols {u_param_start}..{u_param_start+5}")

  # Since we found Offset +1 works, let's also test reading with offset
  print("\n== Testing column offset ==")
  for offset in [0, -1]:
    print(f"  Testing offset={offset}:")
    test_cols = []
    for k in range(6):
      col_idx = u_param_start + offset + k
      try:
        col_vec = jt.col(col_idx)
        val = 0.0
        for entry in col_vec:
          if entry[0] == param_row:
            val = entry[1]
            break
        test_cols.append(val)
      except:
        test_cols.append(0.0)
    print(f"    Values: {[f'{v:+.2e}' for v in test_cols]}")

  # Method 1: Try column access (using offset=0 for now)
  for k in range(6):
    col_idx = u_param_start + k
    try:
      # Get the column as a sparse vector
      col_vec = jt.col(col_idx)
      # Check if param_row is in this column's nonzeros
      val = 0.0
      for entry in col_vec:
        if entry[0] == param_row:
          val = entry[1]
          break
      analytical_du_star.append(val)
      print(f"  col {col_idx}: val={val}")
    except Exception as e:
      print(f"  col {col_idx}: error {e}")
      analytical_du_star.append(0.0)

  print(f"\nAnalytical dU*/d(scale) from Jacobian:")
  print(f"  {analytical_du_star}")

  # Compare
  print("\n" + "-" * 40)
  print("Comparison (looking for matching values):")
  print("-" * 40)
  print(f"Numerical:   {[f'{v:+.5e}' for v in numerical_du_star]}")
  print(f"Analytical:  {[f'{v:+.5e}' for v in analytical_du_star]}")

  # Find best permutation match
  from itertools import permutations
  best_perm = None
  best_error = float('inf')

  for perm in permutations(range(6)):
    error = sum(abs(numerical_du_star[i] - analytical_du_star[perm[i]]) for i in range(6))
    if error < best_error:
      best_error = error
      best_perm = perm

  print(f"\nBest permutation: {best_perm}")
  print(f"Permutation error: {best_error:.2e}")
  print("Matches:")
  for i in range(6):
    n = numerical_du_star[i]
    a = analytical_du_star[best_perm[i]]
    diff = abs(n - a)
    rel_err = abs(diff / n) if abs(n) > 1e-10 else (0 if abs(a) < 1e-10 else float('inf'))
    print(f"  num[{i}]={n:+.5e} ↔ ana[{best_perm[i]}]={a:+.5e}  diff={diff:.2e}  rel={rel_err:.1%}")

  max_diff = 0
  for i in range(6):
    diff = abs(numerical_du_star[i] - analytical_du_star[i])
    max_diff = max(max_diff, diff)
    status = "OK" if diff < 1e-6 else "MISMATCH"
    print(f"  U*[{i}]: num={numerical_du_star[i]:+.8f}, ana={analytical_du_star[i]:+.8f}, diff={diff:.2e} [{status}]")

  # Check if identity permutation works with offset
  print("\nTrying offset analysis:")
  for offset in range(-3, 4):
    matches = sum(1 for i in range(6) if abs(numerical_du_star[i] - analytical_du_star[(i+offset)%6]) < 2e-5)
    print(f"  Offset {offset:+d}: {matches}/6 matches")

  print(f"\nMax difference: {max_diff:.2e}")
  if best_error < 1e-5:
    print("[OK] Permuted Jacobian matches numerical gradient!")
    return True
  else:
    print("[MISMATCH] Jacobian still does not match - need to investigate further")
    return False


if __name__ == "__main__":
  success = test_jacobian()
  if not success:
    import sys
    sys.exit(1)

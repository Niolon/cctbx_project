"""Quick test to check if Cholesky failure still happens after excluding clamped modes."""
from __future__ import absolute_import, division, print_function

import os
import json
import sys

FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_23k_low10"

def test():
  from cctbx import xray, crystal, adptbx
  from cctbx.array_family import flex
  from iotbx import cif

  print("Loading structure...")
  crystal_symmetry = crystal.symmetry(
    unit_cell=(5.9279, 12.2597, 5.7939, 90, 90, 90),
    space_group_symbol="P 21 21 21"
  )

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
  print(f"LOW freqs: {[float(constraint.frequencies[i]) for i in constraint.low_indices]}")

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

  # Check which params are variable
  print("Frequency params:")
  for i, param in enumerate(constraint.low_freq_params):
    is_var = param.is_variable
    print(f"  Mode {constraint.low_indices[i]}: variable={is_var}")

  # Enable Jacobian
  constraint.enable_jacobian()
  constraint.update_frequencies()

  print(f"\nNumber of independents: {reparam.n_independents}")

  # Try one linearise to see if it works
  try:
    reparam.linearise()
    print("[OK] linearise() succeeded")
  except Exception as e:
    print(f"[ERROR] linearise() failed: {e}")
    import traceback
    traceback.print_exc()
    return False

  # Try actual refinement cycle
  import smtbx.refinement.least_squares as ls
  obs, weights = get_fo2_obs(xs)
  print(f"\nAttempting refinement with {obs.size()} reflections...")

  try:
    ls_engine = ls.normal_equations(
      fo_sq=obs,
      xray_structure=xs,
      reparametrisation=reparam,
      weighting_scheme=ls.unit_weighting(),
      scale_factor=1.0
    )
    print("[OK] normal_equations object created")

    # Try to solve
    ls_engine.solve()
    print("[OK] solve() succeeded!")
  except Exception as e:
    print(f"[ERROR] Refinement failed: {e}")
    import traceback
    traceback.print_exc()
    return False

  return True


if __name__ == "__main__":
  success = test()
  print("\n" + ("SUCCESS" if success else "FAILED"))
  sys.exit(0 if success else 1)

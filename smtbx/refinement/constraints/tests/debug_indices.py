"""Debug script to check frequency parameter indices."""
from __future__ import absolute_import, division, print_function

import os
import json

FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_23k_low10"


def debug():
  from cctbx import xray, crystal
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

  # AFTER finalize, check frequency param indices
  print("\nFrequency parameters (AFTER finalize):")
  print(f"  LOW mode indices: {list(constraint.low_indices)}")
  print(f"  LOW freqs: {[float(constraint.frequencies[i]) for i in constraint.low_indices]}")

  for i, param in enumerate(constraint.low_freq_params):
    mode_idx = constraint.low_indices[i]
    freq = float(constraint.frequencies[mode_idx])
    is_var = param.is_variable
    idx = param.index if is_var else "N/A"
    print(f"  Mode {mode_idx}: freq={freq:.2f}, variable={is_var}, index={idx}")

  if constraint.mfsf_param:
    print(f"  MFSF: variable={constraint.mfsf_param.is_variable}, index={constraint.mfsf_param.index}")

  # Enable Jacobian
  constraint.enable_jacobian()
  constraint.update_frequencies()

  # Check nomore params
  print(f"\nNoMoRe params created: {len(constraint._nomore_params)}")
  for i, param in enumerate(constraint._nomore_params[:3]):  # First 3
    print(f"  Atom {i}: index={param.index()}, n_active_modes={param.n_active_modes()}")

  print("\n[OK] Debug complete")


if __name__ == "__main__":
  debug()

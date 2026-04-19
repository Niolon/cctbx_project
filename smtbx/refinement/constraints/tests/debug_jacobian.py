"""Debug script to check if Jacobian is being filled for NoMoRe frequencies."""
from __future__ import absolute_import, division, print_function

import os
import json
import numpy as np

from cctbx import xray, crystal
from cctbx.array_family import flex

FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_23k_low10"


def main():
  """Check if Jacobian matrix has entries for frequency parameters."""
  from iotbx import cif

  # Load structure
  cif_path = os.path.join(FIXTURE_DIR, "L-Ala_23K_xray.cif")
  cif_obj = cif.reader(file_path=cif_path)
  cif_model = cif_obj.model()
  block = list(cif_model.values())[0]

  crystal_symmetry = crystal.symmetry(
    unit_cell=(5.9279, 12.2597, 5.7939, 90, 90, 90),
    space_group_symbol="P 21 21 21"
  )

  scatterers = flex.xray_scatterer()
  for i, label in enumerate(block['_atom_site_label']):
    site = (
      float(block['_atom_site_fract_x'][i].split('(')[0]),
      float(block['_atom_site_fract_y'][i].split('(')[0]),
      float(block['_atom_site_fract_z'][i].split('(')[0])
    )
    u_eq = float(block['_atom_site_U_iso_or_equiv'][i].split('(')[0])
    symbol = block['_atom_site_type_symbol'][i]
    sc = xray.scatterer(label=label, site=site, u=u_eq, occupancy=1.0)
    sc.convert_to_anisotropic(crystal_symmetry.unit_cell())
    scatterers.append(sc)

  xs = xray.structure(crystal_symmetry=crystal_symmetry, scatterers=scatterers)
  print(f"Structure: {xs.scatterers().size()} atoms")

  # Load phonon data
  from smtbx.refinement.constraints.nomore_phonon_data import PhononData
  phonon_data = PhononData.load(os.path.join(FIXTURE_DIR, "phonon_data.npz"))

  # Load reference for limits
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
  print(f"LOW modes: {len(constraint.low_indices)}, MEDIUM: {len(constraint.medium_indices)}")

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
  print(f"n_independents: {reparam.n_independents}")

  # Enable Jacobian
  constraint.enable_jacobian()
  constraint.update_frequencies()

  # Linearise to fill Jacobian
  reparam.linearise()

  # Check frequency parameter indices
  print("\nFrequency parameter indices:")
  for i, param in enumerate(constraint.low_freq_params):
    print(f"  Mode {constraint.low_indices[i]}: index = {param.index}, value = {param.value:.2f}")

  if constraint.mfsf_param:
    print(f"  MFSF: index = {constraint.mfsf_param.index}, value = {constraint.mfsf_param.value:.4f}")

  # Check Jacobian transpose for frequency parameter rows
  print("\nChecking Jacobian transpose for frequency parameter contributions:")
  jt = reparam.jacobian_transpose_matching_grad_fc()
  print(f"  Jacobian transpose shape: {jt.n_rows} x {jt.n_cols}")

  # Check if frequency param rows have any non-zero entries
  for i, param in enumerate(constraint.low_freq_params[:3]):  # Check first 3 LOW freqs
    row = param.index
    nnz_in_row = 0
    for col in range(jt.n_cols):
      try:
        val = jt(row, col)
        if abs(val) > 1e-12:
          nnz_in_row += 1
      except:
        pass
    print(f"  Row {row} (Mode {constraint.low_indices[i]}): {nnz_in_row} non-zero entries")

  print("\n[Done]")


if __name__ == "__main__":
  main()

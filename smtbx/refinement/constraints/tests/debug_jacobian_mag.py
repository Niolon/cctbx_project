"""Check Jacobian magnitude for frequency parameters vs U* parameters."""
from __future__ import absolute_import, division, print_function

import os
import json
import numpy as np

FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_23k_low10"


def check_jacobian():
  from cctbx import xray, crystal
  from cctbx.array_family import flex
  from iotbx import cif

  print("Loading structure and constraint...")
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

  # Set up flags - set grad_u_aniso=True THEN let constraint handle it
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

  # Linearise to fill Jacobian
  reparam.linearise()

  # Get Jacobian transpose
  jt = reparam.jacobian_transpose_matching_grad_fc()
  n_rows = jt.n_rows
  n_cols = jt.n_cols
  print(f"Jacobian transpose shape: {n_rows} x {n_cols}")

  print(f"\nRow analysis (first 10 and last 10 rows):")
  for row in list(range(min(10, n_rows))) + list(range(max(n_rows-10, 10), n_rows)):
    max_val = 0.0
    count = 0
    for col in range(n_cols):
      val = jt(row, col)
      if abs(val) > 1e-20:
        count += 1
        if abs(val) > abs(max_val):
          max_val = val
    print(f"  Row {row:2d}: {count:4d} nonzeros, max_abs={abs(max_val):.2e}")

  print("\n[OK] Done")


if __name__ == "__main__":
  check_jacobian()

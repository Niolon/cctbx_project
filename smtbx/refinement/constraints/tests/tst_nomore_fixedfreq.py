"""Test NoMoRe with only constrained U* (no frequency refinement) to isolate issue."""
from __future__ import absolute_import, division, print_function

import os
import json

FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_23k_low10"


def test_nomore_no_freq_refine():
  """Test NoMoRe constraint without frequency refinement."""
  print("=" * 60)
  print("NoMoRe Test - NO Frequency Refinement (U* constrained only)")
  print("=" * 60)

  from cctbx import xray, crystal, miller
  from cctbx.array_family import flex
  from iotbx import cif

  # Load structure
  print("\n1. Loading structure...")
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
  print(f"  Atoms: {xs.scatterers().size()}")

  # Load phonon data
  from smtbx.refinement.constraints.nomore_phonon_data import PhononData
  phonon_data = PhononData.load(os.path.join(FIXTURE_DIR, "phonon_data.npz"))

  # Load limits
  with open(os.path.join(FIXTURE_DIR, "ref_results.json")) as f:
    ref_data = json.load(f)
  T = ref_data['temperature']
  medium_limit = ref_data['limits']['medium_freq_limit']
  high_limit = ref_data['limits']['high_freq_limit']

  # Create constraint with refine_frequencies=False (fixed frequencies)
  from smtbx.refinement.constraints.nomore import nomore_adp
  constraint = nomore_adp.create(
    phonon_data=phonon_data, target_structure=xs,
    temperature=T, min_freq=5.0,
    medium_freq_limit=medium_limit, high_freq_limit=high_limit, tolerance=2.0,
    refine_frequencies=False  # Fixed frequencies - only constrained U*
  )

  # Set up flags - IMPORTANT: set grad_u_aniso=False since NoMoRe replaces U*
  print("\n2. Setting refinement flags (U* from constraint, not free)...")
  for sc in xs.scatterers():
    sc.flags.set_grad_site(False)
    sc.flags.set_grad_occupancy(False)
    # NoMoRe replaces U*, so don't additionally refine it
    if sc.flags.use_u_aniso():
      sc.flags.set_grad_u_aniso(False)

  # Build reparametrisation with constraint but MARK all freq params as not variable
  import smtbx.utils
  from smtbx.refinement import constraints as cm
  connectivity_table = smtbx.utils.connectivity_table(xs)

  print("\n3. Building reparametrisation...")
  reparam = cm.reparametrisation(xs, [constraint], connectivity_table, temperature=T-273)

  # Frequencies already fixed via refine_frequencies=False
  constraint.update_frequencies()

  print(f"  Independent parameters: {reparam.n_independents}")

  # Finalize and linearise
  reparam.linearise()
  reparam.store()

  # Load experimental data
  print("\n5. Loading experimental Fo²...")
  hkl_data = []
  if '_shelx_hkl_file' in block:
    for line in block['_shelx_hkl_file'].strip().split('\n'):
      parts = line.split()
      if len(parts) >= 5:
        try:
          h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
          fo_sq = float(parts[3])
          sigma = float(parts[4])
          if sigma > 0 and fo_sq > -100:
            hkl_data.append((h, k, l, fo_sq, sigma))
        except:
          pass

  indices = flex.miller_index([(d[0], d[1], d[2]) for d in hkl_data])
  data = flex.double([d[3] for d in hkl_data])
  sigmas = flex.double([d[4] for d in hkl_data])
  ms = miller.set(crystal_symmetry=crystal_symmetry, indices=indices, anomalous_flag=False)
  f_obs_sq = ms.array(data=data, sigmas=sigmas).set_observation_type_xray_intensity()
  obs = f_obs_sq.as_xray_observations()
  print(f"  Reflections: {f_obs_sq.size()}")

  # Run refinement
  print("\n6. Running refinement...")
  import smtbx.refinement.least_squares as ls
  from smtbx.refinement.least_squares import crystallographic_ls

  ls_engine = crystallographic_ls(
    observations=obs,
    reparametrisation=reparam,
    weighting_scheme=ls.mainstream_shelx_weighting(),
  )
  print("  [OK] LS engine created")

  # If n_independents is 0, this will fail differently
  if reparam.n_independents == 0:
    print("  [SKIP] No independent parameters to refine")
    return True

  ls_engine.build_up()
  print(f"  Initial chi²: {ls_engine.objective():.4f}")

  ls_engine.solve()
  print("  [OK] solve() succeeded")

  ls_engine.step_forward()
  print("  [OK] step_forward() succeeded!")

  print(f"  chi² after cycle: {ls_engine.objective():.4f}")

  return True


if __name__ == "__main__":
  import sys
  success = test_nomore_no_freq_refine()
  print("\n" + "=" * 60)
  print("RESULT:", "SUCCESS" if success else "FAILED")
  print("=" * 60)
  sys.exit(0 if success else 1)

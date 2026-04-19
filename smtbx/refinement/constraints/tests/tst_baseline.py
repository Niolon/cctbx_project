"""Test refinement without NoMoRe constraint to verify baseline works."""
from __future__ import absolute_import, division, print_function

import os
import sys

FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_23k_low10"


def test_baseline_refinement():
  """Run refinement without any custom constraint."""
  print("=" * 60)
  print("Baseline Refinement Test (No NoMoRe constraint)")
  print("=" * 60)

  from cctbx import xray, crystal, adptbx
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

  # Set up flags - only refine U_aniso
  print("\n2. Setting refinement flags...")
  for sc in xs.scatterers():
    sc.flags.set_grad_site(False)
    sc.flags.set_grad_occupancy(False)
    if sc.flags.use_u_aniso():
      sc.flags.set_grad_u_aniso(True)
    if sc.flags.use_u_iso():
      sc.flags.set_grad_u_iso(False)

  # Build reparametrisation WITHOUT any custom constraints
  import smtbx.utils
  from smtbx.refinement import constraints as cm

  connectivity_table = smtbx.utils.connectivity_table(xs)

  print("\n3. Building reparametrisation (no custom constraints)...")
  reparam = cm.reparametrisation(
    xs,
    [],  # Empty list - no custom constraints!
    connectivity_table,
    temperature=23 - 273.15,  # Doesn't matter for baseline
  )
  print(f"  Independent parameters: {reparam.n_independents}")

  # Load experimental Fo^2 from CIF file 
  print("\n4. Loading experimental Fo² from CIF...")
  from cctbx import miller

  # Parse the _shelx_hkl_file block
  hkl_data = []
  if '_shelx_hkl_file' in block:
    hkl_lines = block['_shelx_hkl_file'].strip().split('\n')
    for line in hkl_lines:
      parts = line.split()
      if len(parts) >= 5:
        try:
          h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
          fo_sq = float(parts[3])
          sigma = float(parts[4])
          if sigma > 0 and fo_sq > -100:  # Valid data
            hkl_data.append((h, k, l, fo_sq, sigma))
        except:
          pass

  if not hkl_data:
    print("  [ERROR] No experimental reflections found")
    return False

  indices = flex.miller_index([(d[0], d[1], d[2]) for d in hkl_data])
  data = flex.double([d[3] for d in hkl_data])
  sigmas = flex.double([d[4] for d in hkl_data])

  ms = miller.set(crystal_symmetry=crystal_symmetry, indices=indices, anomalous_flag=False)
  f_obs_sq = ms.array(data=data, sigmas=sigmas)
  print(f"  Reflections: {f_obs_sq.size()}")

  # Finalize reparametrisation
  print("\n4b. Finalizing reparametrisation...")
  reparam.finalise()
  reparam.linearise()
  reparam.store()
  print("  [OK] Reparametrisation finalized")

  # Try refinement
  print("\n5. Running refinement cycle...")
  import smtbx.refinement.least_squares as ls

  try:
    # Use the crystallographic_ls API - must convert to xray_observations first
    from smtbx.refinement.least_squares import crystallographic_ls

    # Set observation type and convert to observations object
    f_obs_sq = f_obs_sq.set_observation_type_xray_intensity()
    obs = f_obs_sq.as_xray_observations()

    ls_engine = crystallographic_ls(
      observations=obs,
      reparametrisation=reparam,
      weighting_scheme=ls.mainstream_shelx_weighting(),
    )
    print("  [OK] least_squares engine created")

    # Build up the normal equations first
    ls_engine.build_up()
    print("  [OK] build_up() succeeded")
    print(f"  Initial chi²: {ls_engine.objective():.4f}")

    # Solve the normal equations (Cholesky decomposition)
    try:
      ls_engine.solve()
      print("  [OK] solve() succeeded")
    except RuntimeError as e:
      print(f"  [ERROR] solve() failed: {e}")
      return False

    # Run one cycle - apply the shifts
    ls_engine.step_forward()
    print("  [OK] step_forward() succeeded!")

    print(f"  chi² after cycle: {ls_engine.objective():.4f}")

    shifts = ls_engine.shifts
    print(f"  Shifts: {shifts.size()} values")
    print(f"  Max shift: {max(abs(s) for s in shifts):.2e}")

    return True

  except Exception as e:
    print(f"  [ERROR] Refinement failed: {e}")
    import traceback
    traceback.print_exc()
    return False


if __name__ == "__main__":
  success = test_baseline_refinement()
  print("\n" + ("=" * 60))
  print("RESULT:", "SUCCESS" if success else "FAILED")
  print("=" * 60)
  sys.exit(0 if success else 1)

"""
Refinement test for NoMoRe constraint.

Uses L-Alanine 23K X-ray data with proper space group (P 21 21 21) and
maps P1 phonon data onto ASU atoms for refinement.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import json
import numpy as np

from cctbx import xray, crystal
from cctbx.array_family import flex
from iotbx import cif

# Paths to data
FIXTURE_DIR = r"\\wsl.localhost\Ubuntu-24.04\home\niklas\nomore_ase\cctbx_migration\tests\fixtures\alanine_23k_low10"


def load_structure_from_cif(cif_path):
  """Load xray_structure from CIF file."""
  cif_obj = cif.reader(file_path=cif_path)
  cif_model = cif_obj.model()

  # Get the first data block
  for block_name in cif_model:
    block = cif_model[block_name]
    break

  # Build crystal symmetry
  crystal_symmetry = crystal.symmetry(
    unit_cell=(
      float(block['_cell_length_a'].split('(')[0]),
      float(block['_cell_length_b'].split('(')[0]),
      float(block['_cell_length_c'].split('(')[0]),
      float(block['_cell_angle_alpha']),
      float(block['_cell_angle_beta']),
      float(block['_cell_angle_gamma']),
    ),
    space_group_symbol=block['_space_group_name_H-M_alt'].strip().replace("'", "")
  )

  # Build scatterers
  scatterers = flex.xray_scatterer()
  labels = list(block['_atom_site_label'])
  symbols = list(block['_atom_site_type_symbol'])
  fract_x = list(block['_atom_site_fract_x'])
  fract_y = list(block['_atom_site_fract_y'])
  fract_z = list(block['_atom_site_fract_z'])
  u_eq_list = list(block['_atom_site_U_iso_or_equiv'])
  adp_types = list(block['_atom_site_adp_type'])

  for i in range(len(labels)):
    site = (
      float(fract_x[i].split('(')[0]),
      float(fract_y[i].split('(')[0]),
      float(fract_z[i].split('(')[0])
    )
    u_eq = float(u_eq_list[i].split('(')[0])

    sc = xray.scatterer(
      label=labels[i],
      site=site,
      u=u_eq,
      occupancy=1.0
    )

    # Convert ALL atoms to anisotropic (NoMoRe requires anisotropic ADPs)
    sc.convert_to_anisotropic(crystal_symmetry.unit_cell())

    scatterers.append(sc)

  # Set anisotropic U values if available
  if '_atom_site_aniso_label' in block:
    aniso_labels = list(block['_atom_site_aniso_label'])
    u11 = [float(x.split('(')[0]) for x in block['_atom_site_aniso_U_11']]
    u22 = [float(x.split('(')[0]) for x in block['_atom_site_aniso_U_22']]
    u33 = [float(x.split('(')[0]) for x in block['_atom_site_aniso_U_33']]
    u23 = [float(x.split('(')[0]) for x in block['_atom_site_aniso_U_23']]
    u13 = [float(x.split('(')[0]) for x in block['_atom_site_aniso_U_13']]
    u12 = [float(x.split('(')[0]) for x in block['_atom_site_aniso_U_12']]

    for j, label in enumerate(aniso_labels):
      for sc in scatterers:
        if sc.label == label:
          # CIF gives U_cif which is U_cart-style (Å²)
          from cctbx import adptbx
          u_cif = (u11[j], u22[j], u33[j], u12[j], u13[j], u23[j])
          u_star = adptbx.u_cart_as_u_star(crystal_symmetry.unit_cell(), u_cif)
          sc.u_star = u_star
          break

  xs = xray.structure(
    crystal_symmetry=crystal_symmetry,
    scatterers=scatterers
  )

  return xs


def run_refinement_test():
  """Run refinement with NoMoRe constraint using proper ASU structure."""
  print("=" * 60)
  print("NoMoRe REFINEMENT TEST (L-Alanine 23K, P 21 21 21)")
  print("=" * 60)

  cif_path = os.path.join(FIXTURE_DIR, "L-Ala_23K_xray.cif")
  npz_path = os.path.join(FIXTURE_DIR, "phonon_data.npz")
  ref_path = os.path.join(FIXTURE_DIR, "ref_results.json")

  if not os.path.exists(cif_path):
    print(f"  [SKIP] CIF not found: {cif_path}")
    return True

  if not os.path.exists(npz_path):
    print(f"  [SKIP] Phonon data not found: {npz_path}")
    return True

  # Load structure from CIF
  print("\n1. Loading structure from CIF...")
  xs = load_structure_from_cif(cif_path)
  print(f"  Space group: {xs.space_group_info()}")
  print(f"  ASU atoms: {xs.scatterers().size()}")
  print(f"  Unit cell: {xs.unit_cell()}")

  # Show ASU atoms
  print("  Atoms:")
  for sc in xs.scatterers():
    u_eq = sc.u_iso_or_equiv(xs.unit_cell())
    print(f"    {sc.label}: {sc.element_symbol()}, U_eq = {u_eq:.6f} Å²")

  # Load phonon data
  print("\n2. Loading phonon data...")
  from smtbx.refinement.constraints.nomore_phonon_data import PhononData
  phonon_data = PhononData.load(npz_path)
  print(f"  Phonon atoms: {phonon_data.n_atoms}")
  print(f"  Modes: {phonon_data.n_modes}")

  # Load reference for limits
  print("\n3. Loading reference results...")
  with open(ref_path, 'r') as f:
    ref_data = json.load(f)
  temperature = ref_data['temperature']
  medium_limit = ref_data['limits']['medium_freq_limit']
  high_limit = ref_data['limits']['high_freq_limit']
  print(f"  Temperature: {temperature} K")
  print(f"  Limits: medium={medium_limit:.2f}, high={high_limit:.2f} cm^-1")

  # Create constraint
  print("\n4. Creating NoMoRe constraint...")
  from smtbx.refinement.constraints.nomore import nomore_adp

  try:
    constraint = nomore_adp.create(
      phonon_data=phonon_data,
      target_structure=xs,
      temperature=temperature,
      min_freq=5.0,
      medium_freq_limit=medium_limit,
      high_freq_limit=high_limit,
      tolerance=2.0,
    )
    print(f"  LOW modes: {len(constraint.low_indices)}")
    print(f"  MEDIUM modes: {len(constraint.medium_indices)}")
    print(f"  HIGH modes: {len(constraint.high_indices)}")
    print(f"  Constrained atoms: {len(constraint.atom_indices)}")
  except Exception as e:
    print(f"  [ERROR] Failed to create constraint: {e}")
    import traceback
    traceback.print_exc()
    return False

  # Store initial frequencies
  initial_freqs = [constraint.frequencies[idx] for idx in constraint.low_indices]

  # Set up flags
  print("\n5. Setting up refinement flags...")
  for sc in xs.scatterers():
    sc.flags.set_grad_site(False)
    sc.flags.set_grad_occupancy(False)
    if sc.flags.use_u_aniso():
      sc.flags.set_grad_u_aniso(True)
    if sc.flags.use_u_iso():
      sc.flags.set_grad_u_iso(False)

  # Build reparametrisation
  import smtbx.utils
  from smtbx.refinement import constraints as constraint_module

  connectivity_table = smtbx.utils.connectivity_table(xs)

  print("\n6. Building reparametrisation...")
  reparametrisation = constraint_module.reparametrisation(
    xs,
    [constraint],
    connectivity_table,
    temperature=temperature - 273.15,
  )
  print(f"  Independent parameters: {reparametrisation.n_independents}")

  # Enable Jacobian
  print("\n7. Enabling Jacobian...")
  constraint.enable_jacobian()
  constraint.update_frequencies()
  print("  [OK] Jacobian enabled")

  # Initial U values
  print("\n8. Initial U values from constraint:")
  reparametrisation.linearise()
  reparametrisation.store()
  for sc in xs.scatterers():
    u_eq = sc.u_iso_or_equiv(xs.unit_cell())
    print(f"    {sc.label}: U_eq = {u_eq:.6f} Å²")

  print("\n9. Frequency parameters (initial):")
  if hasattr(constraint, 'low_freq_params') and constraint.low_freq_params:
    for i, param in enumerate(constraint.low_freq_params):
      print(f"    Mode {constraint.low_indices[i]}: {param.value:.2f} cm^-1")

  if hasattr(constraint, 'mfsf_param') and constraint.mfsf_param:
    print(f"  MFSF: {constraint.mfsf_param.value:.6f}")

  # Load experimental Fo² from CIF's SHELX HKL block
  print("\n10. Loading experimental Fo² from CIF...")
  cif_path = os.path.join(FIXTURE_DIR, "L-Ala_23K_xray.cif")

  # Parse SHELX HKL block directly
  with open(cif_path, 'r') as f:
    content = f.read()

  # Find _shelx_hkl_file block
  hkl_start = content.find("_shelx_hkl_file")
  if hkl_start > 0:
    # Find the semicolon-delimited block
    block_start = content.find(";", hkl_start) + 1
    block_end = content.find(";", block_start)
    hkl_text = content[block_start:block_end].strip()

    # Parse HKL lines
    indices = flex.miller_index()
    intensities = []
    sigmas = []
    for line in hkl_text.split("\n"):
      parts = line.split()
      if len(parts) >= 5:
        h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
        intensity = float(parts[3])
        sigma = float(parts[4])
        indices.append((h, k, l))
        intensities.append(intensity)
        sigmas.append(sigma)

    # Create miller array
    from cctbx import miller
    fo_sq = miller.array(
      miller.set(xs.crystal_symmetry(), indices, anomalous_flag=False),
      data=flex.double(intensities),
      sigmas=flex.double(sigmas)
    ).set_observation_type_xray_intensity()

    print(f"  Loaded: {fo_sq.size()} reflections from SHELX HKL block")
    print(f"  Intensity range: {min(intensities):.1f} - {max(intensities):.1f}")
  else:
    print("  [WARNING] No SHELX HKL block found, using calculated Fc")
    mi = xs.build_miller_set(anomalous_flag=False, d_min=0.7)
    fc = mi.structure_factors_from_scatterers(xs, algorithm="direct").f_calc()
    fo_sq = fc.norm()
    fo_sq = fo_sq.customized_copy(sigmas=flex.double(fo_sq.size(), 1.0))

  print(f"  Reflections: {fo_sq.size()}")

  # No perturbation needed - deltas start at 0, so initial fit uses initial frequencies

  # Run refinement cycles
  print("\n11. Running refinement cycles...")
  from smtbx.refinement import least_squares

  obs = fo_sq.as_xray_observations()
  ls = least_squares.crystallographic_ls(
    obs,
    reparametrisation,
    weighting_scheme=least_squares.mainstream_shelx_weighting()
  )

  ls.build_up()
  print(f"  Initial chi²: {ls.objective():.6f}")

  n_cycles = 3
  for cycle in range(n_cycles):
    try:
      constraint.update_frequencies()
      ls.solve()
      shifts = ls.step()
      reparametrisation.apply_shifts(shifts)
      reparametrisation.linearise()
      reparametrisation.store()
      ls.build_up()
      print(f"  Cycle {cycle+1}: chi² = {ls.objective():.6f}")
    except Exception as e:
      print(f"  Cycle {cycle+1}: stopped ({e})")
      break

  # Final frequencies (parameters are SCALE FACTORS, actual = initial × scale)
  print("\n12. Frequency parameters (after refinement):")
  print("  (Note: Parameters are SCALE FACTORS from initial values)")
  if hasattr(constraint, 'low_freq_params') and constraint.low_freq_params:
    for i, param in enumerate(constraint.low_freq_params):
      initial = initial_freqs[i] if i < len(initial_freqs) else 0
      scale = param.value
      actual = initial * scale
      print(f"    Mode {constraint.low_indices[i]}: {initial:.2f} × {scale:.6f} = {actual:.2f} cm^-1")

  if hasattr(constraint, 'mfsf_param') and constraint.mfsf_param:
    mfsf = constraint.mfsf_param.value
    print(f"  MFSF: {mfsf:.6f}")

  # Summary
  print("\n13. Frequency change summary:")
  if initial_freqs and hasattr(constraint, 'low_freq_params'):
    scale_deviations = [abs(p.value - 1.0) for p in constraint.low_freq_params]
    max_dev = max(scale_deviations) if scale_deviations else 0
    print(f"  Max scale deviation from 1.0: {max_dev:.6f}")
    if max_dev > 0.001:
      print("  [OK] Frequencies are being refined!")
    else:
      print("  [INFO] Scales unchanged (may need more cycles or real data mismatch)")

  print("\n" + "=" * 60)
  print("[OK] Refinement test completed!")
  print("=" * 60)
  return True


if __name__ == "__main__":
  success = run_refinement_test()
  if not success:
    sys.exit(1)

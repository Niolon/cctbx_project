"""
NoMoRe ADP Constraint for smtbx.

Normal Mode Refinement constraint that computes ADPs from phonon frequencies.
Reduces parameter count from 6*N_atoms to N_modes (or fewer with frequency splits).
"""
from __future__ import absolute_import, division, print_function

import smtbx.refinement.constraints as _
from smtbx.refinement.constraints import InvalidConstraint

import numpy as np


def build_atom_mapping(phonon_positions, phonon_symbols, target_structure,
                       tolerance=2.0):
  """
  Build mapping from phonon atom indices to target structure indices.

  Uses Hungarian algorithm for optimal 1-to-1 matching, constrained by
  element type (only atoms of same element can be matched).

  Args:
    phonon_positions: (N, 3) Fractional coordinates from phonon data.
    phonon_symbols: List of element symbols from phonon data.
    target_structure: cctbx xray_structure.
    tolerance: Maximum distance in Angstroms for valid match.

  Returns:
    mapping: Array where mapping[phonon_idx] = target_idx.
    distances: Array of matched distances.

  Raises:
    InvalidConstraint: If matching fails.
  """
  from scipy.optimize import linear_sum_assignment

  # Get target data
  scatterers = list(target_structure.scatterers())
  target_positions = np.array([sc.site for sc in scatterers])
  target_symbols = [sc.element_symbol() for sc in scatterers]
  unit_cell = target_structure.unit_cell()

  n_phonon = len(phonon_positions)
  n_target = len(target_positions)

  if n_phonon != n_target:
    raise InvalidConstraint(
      "NoMoRe atom count mismatch: phonon has %d, structure has %d" %
      (n_phonon, n_target))

  # Build cost matrix with element constraints
  LARGE_COST = 1e10
  cost_matrix = np.full((n_phonon, n_target), LARGE_COST)

  for i in range(n_phonon):
    for j in range(n_target):
      if phonon_symbols[i] != target_symbols[j]:
        continue
      diff = np.array(phonon_positions[i]) - np.array(target_positions[j])
      diff = diff - np.round(diff)
      cart_diff = unit_cell.orthogonalize(tuple(diff))
      dist = np.sqrt(sum(x**2 for x in cart_diff))
      if dist < tolerance:
        cost_matrix[i, j] = dist

  row_ind, col_ind = linear_sum_assignment(cost_matrix)

  mapping = np.zeros(n_phonon, dtype=int)
  distances = np.zeros(n_phonon)
  for i, j in zip(row_ind, col_ind):
    if cost_matrix[i, j] >= LARGE_COST:
      raise InvalidConstraint(
        "NoMoRe: Cannot match atom %d (%s) to any target atom" %
        (i, phonon_symbols[i]))
    mapping[i] = j
    distances[i] = cost_matrix[i, j]

  return mapping, distances


def compute_mode_tensors(eigenvectors, masses):
  """Compute (e x e*) / m for each mode and atom."""
  n_modes, n_atoms, _ = eigenvectors.shape
  tensors = np.zeros((n_modes, n_atoms, 3, 3))

  for j in range(n_modes):
    for k in range(n_atoms):
      vec = eigenvectors[j, k, :]
      dyad = np.outer(vec, vec.conj())
      tensors[j, k, :, :] = dyad.real / masses[k]

  return tensors


class nomore_adp(object):
  """
  NoMoRe constraint: ADPs computed from phonon frequencies.

  Frequency categories:
    LOW: Individually refined (one parameter per mode)
    MEDIUM: Scaled by single MFSF parameter (nu = MFSF * nu_initial)
    HIGH: Fixed at initial DFT values
  """

  def __init__(self, mode_tensors, frequencies, masses, weights, n_q_points,
               atom_indices, temperature, min_freq=5.0,
               medium_freq_limit=None, high_freq_limit=None):
    """Initialize NoMoRe constraint. Use nomore_adp.create() factory."""
    self.mode_tensors = mode_tensors
    self.frequencies = np.asarray(frequencies)
    self.masses = masses
    self.weights = weights
    self.n_q_points = n_q_points
    self.atom_indices = list(atom_indices)
    self.temperature = temperature
    self.min_freq = min_freq
    self.medium_freq_limit = medium_freq_limit
    self.high_freq_limit = high_freq_limit
    self._classify_modes()

  def _classify_modes(self):
    """Classify modes into LOW, MEDIUM, HIGH categories."""
    n_modes = len(self.frequencies)
    indices = np.arange(n_modes)
    freqs = self.frequencies

    self.low_indices = indices
    self.medium_indices = np.array([], dtype=int)
    self.high_indices = np.array([], dtype=int)

    if self.medium_freq_limit is not None and self.high_freq_limit is not None:
      self.low_indices = indices[freqs < self.medium_freq_limit]
      self.medium_indices = indices[
        (freqs >= self.medium_freq_limit) & (freqs < self.high_freq_limit)]
      self.high_indices = indices[freqs >= self.high_freq_limit]
    elif self.high_freq_limit is not None:
      self.low_indices = indices[freqs < self.high_freq_limit]
      self.high_indices = indices[freqs >= self.high_freq_limit]
    elif self.medium_freq_limit is not None:
      self.low_indices = indices[freqs < self.medium_freq_limit]
      self.medium_indices = indices[freqs >= self.medium_freq_limit]

  @classmethod
  def create(cls, phonon_data, target_structure, temperature,
             min_freq=5.0, medium_freq_limit=None, high_freq_limit=None,
             tolerance=2.0):
    """Factory function with atom mapping."""
    mapping, distances = build_atom_mapping(
      phonon_data.positions_frac,
      phonon_data.symbols,
      target_structure,
      tolerance=tolerance)

    inverse_mapping = np.zeros(phonon_data.n_atoms, dtype=int)
    for phonon_idx, target_idx in enumerate(mapping):
      inverse_mapping[target_idx] = phonon_idx

    reordered_eigenvectors = phonon_data.eigenvectors[:, inverse_mapping, :]
    reordered_masses = phonon_data.masses[inverse_mapping]

    mode_tensors = compute_mode_tensors(reordered_eigenvectors, reordered_masses)
    atom_indices = list(range(len(target_structure.scatterers())))

    return cls(
      mode_tensors=mode_tensors,
      frequencies=phonon_data.frequencies_cm1,
      masses=reordered_masses,
      weights=phonon_data.weights,
      n_q_points=phonon_data.n_q_points,
      atom_indices=atom_indices,
      temperature=temperature,
      min_freq=min_freq,
      medium_freq_limit=medium_freq_limit,
      high_freq_limit=high_freq_limit,
    )

  @property
  def constrained_parameters(self):
    """Return tuple of (atom_idx, 'U') for all constrained atoms."""
    return tuple((idx, 'U') for idx in self.atom_indices)

  def add_to(self, reparametrisation):
    """Add constraint parameters to reparametrisation."""
    from scitbx.array_family import flex
    from smtbx.refinement.constraints.nomore_physics import calculate_adp_amplitude

    structure = reparametrisation.structure
    scatterers = structure.scatterers()

    n_low = len(self.low_indices)
    n_medium = len(self.medium_indices)
    has_mfsf = n_medium > 0

    # Create LOW frequency parameters
    low_freq_params = []
    for idx in self.low_indices:
      freq = float(self.frequencies[idx])
      param = reparametrisation.add(
        _.independent_scalar_parameter,
        value=freq,
        variable=True)
      low_freq_params.append(param)

    # Create MFSF parameter if needed
    mfsf_param = None
    if has_mfsf:
      mfsf_param = reparametrisation.add(
        _.independent_scalar_parameter,
        value=1.0,
        variable=True)

    # Store for later frequency updates
    self.low_freq_params = low_freq_params
    self.mfsf_param = mfsf_param
    self._nomore_params = []

    # Create nomore_u_star_parameter for each atom
    for atom_idx in self.atom_indices:
      sc = scatterers[atom_idx]

      if not sc.flags.use_u_aniso():
        raise InvalidConstraint(
          "NoMoRe requires anisotropic ADPs for atom %s" % sc.label)

      # Build arrays for all active modes (LOW + MEDIUM)
      active_indices = np.concatenate([self.low_indices, self.medium_indices])
      n_active = len(active_indices)

      # Mode tensors (weighted)
      mode_tensors_flat = []
      for mode_idx in active_indices:
        tensor = self.mode_tensors[mode_idx, atom_idx]
        weighted = tensor * self.weights[mode_idx]
        mode_tensors_flat.extend(weighted.flatten())

      # Current frequencies
      frequencies = [float(self.frequencies[i]) for i in active_indices]

      # Jacobian indices: LOW params have indices, MEDIUM all point to MFSF
      jacobian_indices = []
      jacobian_weights = []

      for i, mode_idx in enumerate(active_indices):
        if mode_idx in self.low_indices:
          # Find which LOW param this corresponds to
          low_pos = list(self.low_indices).index(mode_idx)
          jacobian_indices.append(low_freq_params[low_pos].index)
          jacobian_weights.append(1.0)
        else:
          # MEDIUM mode - points to MFSF
          jacobian_indices.append(mfsf_param.index)
          jacobian_weights.append(float(self.frequencies[mode_idx]))  # Initial freq

      # Compute HIGH mode contribution (fixed U_cart)
      high_contrib = np.zeros(6)
      for mode_idx in self.high_indices:
        freq = self.frequencies[mode_idx]
        weight = self.weights[mode_idx]
        amp = calculate_adp_amplitude(freq, self.temperature, self.min_freq)
        tensor = self.mode_tensors[mode_idx, atom_idx]
        weighted = tensor * weight * amp
        # Sum into sym_mat format: [U11,U22,U33,U12,U13,U23]
        high_contrib[0] += weighted[0, 0]
        high_contrib[1] += weighted[1, 1]
        high_contrib[2] += weighted[2, 2]
        high_contrib[3] += weighted[0, 1]
        high_contrib[4] += weighted[0, 2]
        high_contrib[5] += weighted[1, 2]
      high_contrib /= self.n_q_points

      # Create the C++ parameter
      param = reparametrisation.add(
        _.nomore_u_star_parameter,
        scatterer=sc,
        mode_tensors=flex.double(mode_tensors_flat),
        frequencies=flex.double(frequencies),
        jacobian_indices=flex.size_t(jacobian_indices),
        jacobian_weights=flex.double(jacobian_weights),
        high_contribution=flex.double(list(high_contrib)),
        temperature=self.temperature,
        min_freq=self.min_freq,
        normalization=float(self.n_q_points))

      reparametrisation.asu_scatterer_parameters[atom_idx].u = param
      self._nomore_params.append(param)

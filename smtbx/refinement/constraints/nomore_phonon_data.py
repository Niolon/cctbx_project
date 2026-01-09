"""
PhononData: Container for phonon data used by NoMoRe constraint.

This module handles loading and storing phonon information from NPZ files.
"""
from __future__ import absolute_import, division, print_function

import json


class PhononData(object):
  """
  Phonon data for NoMoRe ADP constraint.

  All frequencies are in cm^-1. This class serves as the interface between
  external phonon calculations and the NoMoRe constraint.

  Attributes:
    frequencies_cm1: (N_modes,) Mode frequencies in cm^-1.
    eigenvectors: (N_modes, N_atoms, 3) Eigenvector displacements (complex).
    masses: (N_atoms,) Atomic masses in AMU.
    weights: (N_modes,) Q-point multiplicity weights for each mode.
    degeneracy_groups: List of lists of degenerate mode indices.
    positions_frac: (N_atoms, 3) Fractional coordinates.
    symbols: List of atomic symbols.
    n_atoms: Number of atoms.
    n_modes: Total number of modes.
    n_q_points: Number of unique q-points.
  """

  def __init__(self, frequencies_cm1, eigenvectors, masses, weights,
               degeneracy_groups, positions_frac, symbols, n_atoms,
               n_q_points):
    """Initialize PhononData with arrays."""
    import numpy as np
    self.frequencies_cm1 = np.asarray(frequencies_cm1)
    self.eigenvectors = np.asarray(eigenvectors)
    self.masses = np.asarray(masses)
    self.weights = np.asarray(weights)
    self.degeneracy_groups = degeneracy_groups
    self.positions_frac = np.asarray(positions_frac)
    self.symbols = list(symbols)
    self.n_atoms = int(n_atoms)
    self.n_q_points = int(n_q_points)

  @property
  def n_modes(self):
    """Total number of modes."""
    return len(self.frequencies_cm1)

  @classmethod
  def load(cls, path):
    """
    Load PhononData from NPZ file.

    Args:
      path: Path to .npz file.

    Returns:
      PhononData instance.
    """
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
      # Reconstruct complex eigenvectors from packed format
      eigenvectors_packed = data['eigenvectors_packed']
      eigenvectors = eigenvectors_packed[..., 0] + 1j * eigenvectors_packed[..., 1]

      # Parse JSON-encoded lists
      degeneracy_groups = json.loads(str(data['degeneracy_groups_json']))
      symbols = json.loads(str(data['symbols_json']))

      return cls(
        frequencies_cm1=data['frequencies_cm1'],
        eigenvectors=eigenvectors,
        masses=data['masses'],
        weights=data['weights'],
        degeneracy_groups=degeneracy_groups,
        positions_frac=data['positions_frac'],
        symbols=symbols,
        n_atoms=int(data['n_atoms']),
        n_q_points=len(data['q_points']),
      )

  def save(self, path):
    """
    Save PhononData to NPZ file.

    Args:
      path: Output path for .npz file.
    """
    import numpy as np
    # Pack complex eigenvectors
    eigenvectors_packed = np.stack([
      self.eigenvectors.real,
      self.eigenvectors.imag
    ], axis=-1)

    np.savez_compressed(
      path,
      frequencies_cm1=self.frequencies_cm1,
      eigenvectors_packed=eigenvectors_packed,
      masses=self.masses,
      weights=self.weights,
      degeneracy_groups_json=json.dumps(self.degeneracy_groups),
      positions_frac=self.positions_frac,
      symbols_json=json.dumps(self.symbols),
      n_atoms=self.n_atoms,
      q_points=np.zeros((self.n_q_points, 3)),  # Placeholder
    )

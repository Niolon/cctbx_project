"""
PhononData: Transfer object for phonon data across unit boundary.

This dataclass holds all phonon information needed by the refinement layer,
with frequencies in cm⁻¹ (crystallography standard). It serves as the single
communication interface between the ASE phonon layer and the NoMoRe refinement layer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import numpy as np
import json


@dataclass
class PhononData:
    """
    Phonon data for NoMoRe refinement.
    
    All frequencies are in cm⁻¹. This is the unit boundary - ASE-side code
    uses eV internally, but this object exposes cm⁻¹ to downstream code.
    
    Phonon Properties:
        frequencies_cm1: (N_modes,) Mode frequencies in cm⁻¹.
        eigenvectors: (N_modes, N_atoms, 3) Eigenvector displacements (complex).
        q_points: (N_q, 3) Unique q-points in fractional coordinates.
        mode_q_indices: (N_modes,) Index into q_points for each mode.
        weights: Optional (N_modes,) mode weights. DEPRECATED - must be all 1.0 if provided.
                 Non-unit weights are no longer supported (raise error).
        degeneracy_groups: List of lists of degenerate mode indices.
        band_indices: Optional (N_modes,) Band assignment for band scaling (None if not computed).
        
    Geometry (optimized structure used for phonon calculation):
        positions_frac: (N_atoms, 3) Fractional coordinates.
        cell: (3, 3) Unit cell vectors in Å.
        symbols: List of atomic symbols.
        masses: (N_atoms,) Atomic masses in amu.
        supercell: (3,) Supercell dimensions used for phonon calculation.
        
    Derived:
        n_atoms: Number of atoms in the primitive cell.
    """
    # Phonon properties (required)
    frequencies_cm1: np.ndarray
    eigenvectors: np.ndarray
    q_points: np.ndarray
    mode_q_indices: np.ndarray
    
    # Geometry (required)
    positions_frac: np.ndarray
    cell: np.ndarray
    symbols: List[str]
    masses: np.ndarray
    supercell: tuple
    
    # Derived (required)
    n_atoms: int
    
    # Optional fields (with defaults - must come last)
    weights: Optional[np.ndarray] = None  # Deprecated, for external compatibility only
    degeneracy_groups: Optional[List[List[int]]] = None
    band_indices: Optional[np.ndarray] = None  # Band assignment for band scaling
    
    def __post_init__(self):
        """Validate shapes and types."""
        import warnings
        
        n_modes = len(self.frequencies_cm1)
        
        # Ensure supercell is a tuple
        if not isinstance(self.supercell, tuple):
            self.supercell = tuple(self.supercell)
        
        # Validate phonon arrays
        assert self.eigenvectors.shape[0] == n_modes, \
            f"eigenvectors shape mismatch: {self.eigenvectors.shape[0]} vs {n_modes}"
        assert len(self.mode_q_indices) == n_modes, \
            f"mode_q_indices length mismatch: {len(self.mode_q_indices)} vs {n_modes}"
        assert self.eigenvectors.shape[1] == self.n_atoms, \
            f"eigenvectors atom dimension mismatch: {self.eigenvectors.shape[1]} vs {self.n_atoms}"
        
        # Validate weights if provided (external compatibility)
        if self.weights is not None:
            assert len(self.weights) == n_modes, \
                f"weights length mismatch: {len(self.weights)} vs {n_modes}"
            
            # Check if all weights are unit (1.0)
            if np.allclose(self.weights, 1.0):
                warnings.warn(
                    "PhononData.weights is deprecated. After the IBZ→FBZ expansion fix, "
                    "all mode weights are 1.0 and the field is no longer used internally. "
                    "Please remove 'weights' from your PhononData construction.",
                    DeprecationWarning,
                    stacklevel=3
                )
            else:
                # Non-unit weights are not supported anymore
                raise ValueError(
                    "Non-unit mode weights are no longer supported. After the IBZ→FBZ expansion fix, "
                    "all modes are explicitly expanded to the full Brillouin zone with proper symmetry "
                    "rotations applied to eigenvectors. Weighted IBZ sums do not correctly handle tensor "
                    "quantities like ADPs. Your code must be updated to use full BZ expansion."
                )
        
        # Validate geometry arrays
        assert len(self.masses) == self.n_atoms, \
            f"masses length mismatch: {len(self.masses)} vs {self.n_atoms}"
        assert self.positions_frac.shape == (self.n_atoms, 3), \
            f"positions_frac shape mismatch: {self.positions_frac.shape} vs ({self.n_atoms}, 3)"
        assert self.cell.shape == (3, 3), \
            f"cell shape mismatch: {self.cell.shape} vs (3, 3)"
        assert len(self.symbols) == self.n_atoms, \
            f"symbols length mismatch: {len(self.symbols)} vs {self.n_atoms}"
        assert len(self.supercell) == 3, \
            f"supercell length mismatch: {len(self.supercell)} vs 3"
    
    @property
    def n_modes(self) -> int:
        """Total number of modes."""
        return len(self.frequencies_cm1)
    
    @property
    def n_q_points(self) -> int:
        """Number of unique q-points."""
        return len(self.q_points)
    
    def get_q_point_for_mode(self, mode_idx: int) -> np.ndarray:
        """Get the q-point (fractional) for a given mode."""
        return self.q_points[self.mode_q_indices[mode_idx]]

    # =========================================================================
    # Serialization: NPZ format (compact binary)
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary suitable for np.savez_compressed().
        
        Complex arrays are stored as real arrays with shape (..., 2) for real/imag.
        Lists are JSON-encoded for storage in NPZ.
        """
        # Handle complex eigenvectors: store as (N, atoms, 3, 2) for real/imag
        eigenvectors_packed = np.stack([
            self.eigenvectors.real, 
            self.eigenvectors.imag
        ], axis=-1)
        
        return {
            # Phonon data
            'frequencies_cm1': self.frequencies_cm1,
            'eigenvectors_packed': eigenvectors_packed,
            'q_points': self.q_points,
            'mode_q_indices': self.mode_q_indices,
            'degeneracy_groups_json': json.dumps(self.degeneracy_groups),
            'band_indices': self.band_indices if self.band_indices is not None else np.array([]),
            # Geometry
            'positions_frac': self.positions_frac,
            'cell': self.cell,
            'symbols_json': json.dumps(self.symbols),
            'masses': self.masses,
            'supercell': np.array(self.supercell),
            # Metadata
            'n_atoms': np.array(self.n_atoms),
            '_version': np.array(1),  # Format version for future compatibility
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhononData':
        """
        Reconstruct PhononData from dictionary (e.g., loaded from NPZ).
        """
        # Reconstruct complex eigenvectors
        eigenvectors_packed = data['eigenvectors_packed']
        eigenvectors = eigenvectors_packed[..., 0] + 1j * eigenvectors_packed[..., 1]
        
        # Parse JSON-encoded lists
        degeneracy_groups = json.loads(str(data['degeneracy_groups_json']))
        symbols = json.loads(str(data['symbols_json']))
        
        # Handle band_indices (may not be present in older files)
        band_indices = data.get('band_indices', np.array([]))
        band_indices = band_indices if len(band_indices) > 0 else None
        
        return cls(
            frequencies_cm1=np.array(data['frequencies_cm1']),
            eigenvectors=eigenvectors,
            q_points=np.array(data['q_points']),
            mode_q_indices=np.array(data['mode_q_indices']),
            degeneracy_groups=degeneracy_groups,
            band_indices=band_indices,
            positions_frac=np.array(data['positions_frac']),
            cell=np.array(data['cell']),
            symbols=symbols,
            masses=np.array(data['masses']),
            supercell=tuple(data['supercell']),
            n_atoms=int(data['n_atoms']),
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save to NPZ file (compact binary format)."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.npz')
        np.savez_compressed(path, **self.to_dict())
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PhononData':
        """Load from NPZ file."""
        with np.load(path, allow_pickle=False) as data:
            return cls.from_dict(dict(data))

    # =========================================================================
    # Serialization: CIF format (human-readable, crystallographic standard)
    # =========================================================================
    
    def to_cif(self, path: Union[str, Path], data_block_name: str = "phonon_data") -> None:
        """
        Write complete PhononData to CIF file.
        
        The CIF contains all information needed to reconstruct the PhononData,
        ensuring reproducibility.
        
        Args:
            path: Output CIF file path.
            data_block_name: CIF data block name.
        """
        path = Path(path)
        
        with open(path, 'w') as f:
            f.write(f"data_{data_block_name}\n\n")
            
            # Cell parameters
            a, b, c = np.linalg.norm(self.cell, axis=1)
            alpha = np.degrees(np.arccos(np.dot(self.cell[1], self.cell[2]) / (b * c)))
            beta = np.degrees(np.arccos(np.dot(self.cell[0], self.cell[2]) / (a * c)))
            gamma = np.degrees(np.arccos(np.dot(self.cell[0], self.cell[1]) / (a * b)))
            
            f.write("# Cell parameters\n")
            f.write(f"_cell_length_a    {a:.6f}\n")
            f.write(f"_cell_length_b    {b:.6f}\n")
            f.write(f"_cell_length_c    {c:.6f}\n")
            f.write(f"_cell_angle_alpha {alpha:.4f}\n")
            f.write(f"_cell_angle_beta  {beta:.4f}\n")
            f.write(f"_cell_angle_gamma {gamma:.4f}\n\n")
            
            # Supercell info
            f.write("# Phonon calculation parameters\n")
            f.write(f"_nomore_supercell_a {self.supercell[0]}\n")
            f.write(f"_nomore_supercell_b {self.supercell[1]}\n")
            f.write(f"_nomore_supercell_c {self.supercell[2]}\n\n")
            
            # Atom sites
            f.write("# Atomic positions (fractional)\n")
            f.write("loop_\n")
            f.write("_atom_site_label\n")
            f.write("_atom_site_type_symbol\n")
            f.write("_atom_site_fract_x\n")
            f.write("_atom_site_fract_y\n")
            f.write("_atom_site_fract_z\n")
            f.write("_atom_site_mass\n")
            for i, (sym, pos, mass) in enumerate(zip(self.symbols, self.positions_frac, self.masses)):
                label = f"{sym}{i+1}"
                f.write(f"{label:6s} {sym:2s} {pos[0]:12.8f} {pos[1]:12.8f} {pos[2]:12.8f} {mass:10.6f}\n")
            f.write("\n")
            
            # Q-points
            f.write("# Irreducible q-points (fractional)\n")
            f.write("loop_\n")
            f.write("_nomore_qpoint_id\n")
            f.write("_nomore_qpoint_fract_x\n")
            f.write("_nomore_qpoint_fract_y\n")
            f.write("_nomore_qpoint_fract_z\n")
            for i, q in enumerate(self.q_points):
                f.write(f"{i:4d} {q[0]:12.8f} {q[1]:12.8f} {q[2]:12.8f}\n")
            f.write("\n")
            
            # Phonon modes (frequencies and metadata)
            f.write("# Phonon modes\n")
            f.write("loop_\n")
            f.write("_nomore_mode_id\n")
            f.write("_nomore_mode_frequency_cm1\n")
            f.write("_nomore_mode_qpoint_id\n")
            for i in range(self.n_modes):
                f.write(f"{i:4d} {self.frequencies_cm1[i]:12.4f} {self.mode_q_indices[i]:4d}\n")
            f.write("\n")
            
            # Degeneracy groups (JSON in CIF field)
            f.write("# Degeneracy groups (JSON format)\n")
            f.write(f"_nomore_degeneracy_groups '{json.dumps(self.degeneracy_groups)}'\n\n")
            
            # Band indices (optional)
            if self.band_indices is not None:
                f.write("# Band assignment (optional)\n")
                f.write("loop_\n")
                f.write("_nomore_band_mode_id\n")
                f.write("_nomore_band_index\n")
                for i, band_id in enumerate(self.band_indices):
                    f.write(f"{i:4d} {band_id:4d}\n")
                f.write("\n")
            
            # Eigenvectors (one block per mode, stored as real/imag pairs)
            f.write("# Eigenvectors (real and imaginary parts)\n")
            f.write("loop_\n")
            f.write("_nomore_eigenvector_mode_id\n")
            f.write("_nomore_eigenvector_atom_id\n")
            f.write("_nomore_eigenvector_x_real\n")
            f.write("_nomore_eigenvector_x_imag\n")
            f.write("_nomore_eigenvector_y_real\n")
            f.write("_nomore_eigenvector_y_imag\n")
            f.write("_nomore_eigenvector_z_real\n")
            f.write("_nomore_eigenvector_z_imag\n")
            for mode_id in range(self.n_modes):
                for atom_id in range(self.n_atoms):
                    ev = self.eigenvectors[mode_id, atom_id]
                    f.write(f"{mode_id:4d} {atom_id:4d} "
                           f"{ev[0].real:14.10f} {ev[0].imag:14.10f} "
                           f"{ev[1].real:14.10f} {ev[1].imag:14.10f} "
                           f"{ev[2].real:14.10f} {ev[2].imag:14.10f}\n")
    
    @classmethod
    def from_cif(cls, path: Union[str, Path]) -> 'PhononData':
        """
        Load PhononData from CIF file.
        
        Args:
            path: Path to CIF file with phonon data.
            
        Returns:
            PhononData reconstructed from CIF.
        """
        path = Path(path)
        
        # Parse CIF manually (simple parser for our specific format)
        cell_params = {}
        supercell = [1, 1, 1]
        atoms = []
        q_points = []
        modes = []
        eigenvectors_raw = []
        degeneracy_groups = []
        band_indices_raw = []  # For band assignment
        
        with open(path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Cell parameters
            if line.startswith('_cell_length_a'):
                cell_params['a'] = float(line.split()[1])
            elif line.startswith('_cell_length_b'):
                cell_params['b'] = float(line.split()[1])
            elif line.startswith('_cell_length_c'):
                cell_params['c'] = float(line.split()[1])
            elif line.startswith('_cell_angle_alpha'):
                cell_params['alpha'] = float(line.split()[1])
            elif line.startswith('_cell_angle_beta'):
                cell_params['beta'] = float(line.split()[1])
            elif line.startswith('_cell_angle_gamma'):
                cell_params['gamma'] = float(line.split()[1])
            
            # Supercell
            elif line.startswith('_nomore_supercell_a'):
                supercell[0] = int(line.split()[1])
            elif line.startswith('_nomore_supercell_b'):
                supercell[1] = int(line.split()[1])
            elif line.startswith('_nomore_supercell_c'):
                supercell[2] = int(line.split()[1])
            
            # Degeneracy groups
            elif line.startswith('_nomore_degeneracy_groups'):
                # Extract JSON from single quotes
                json_str = line.split("'")[1]
                degeneracy_groups = json.loads(json_str)
            
            # Loop blocks
            elif line == 'loop_':
                i += 1
                headers = []
                while i < len(lines) and lines[i].strip().startswith('_'):
                    headers.append(lines[i].strip())
                    i += 1
                
                # Read loop data
                data_rows = []
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('_') and not lines[i].strip().startswith('#') and lines[i].strip() != 'loop_':
                    data_rows.append(lines[i].strip().split())
                    i += 1
                
                # Process based on loop type
                if '_atom_site_label' in headers:
                    for row in data_rows:
                        atoms.append({
                            'symbol': row[1],
                            'frac': [float(row[2]), float(row[3]), float(row[4])],
                            'mass': float(row[5])
                        })
                elif '_nomore_qpoint_id' in headers:
                    for row in data_rows:
                        q_points.append([float(row[1]), float(row[2]), float(row[3])])
                elif '_nomore_mode_id' in headers and '_nomore_mode_frequency_cm1' in headers:
                    for row in data_rows:
                        modes.append({
                            'freq': float(row[1]),
                            'q_id': int(row[2])
                            # Note: weight column removed as of 2026-02-15
                        })
                elif '_nomore_eigenvector_mode_id' in headers:
                    for row in data_rows:
                        eigenvectors_raw.append({
                            'mode_id': int(row[0]),
                            'atom_id': int(row[1]),
                            'x': complex(float(row[2]), float(row[3])),
                            'y': complex(float(row[4]), float(row[5])),
                            'z': complex(float(row[6]), float(row[7]))
                        })
                elif '_nomore_band_mode_id' in headers:
                    for row in data_rows:
                        band_indices_raw.append(int(row[1]))
                
                continue  # Don't increment i again
            
            i += 1
        
        # Build cell matrix from parameters
        a, b, c = cell_params['a'], cell_params['b'], cell_params['c']
        alpha = np.radians(cell_params['alpha'])
        beta = np.radians(cell_params['beta'])
        gamma = np.radians(cell_params['gamma'])
        
        # Standard crystallographic convention for cell vectors
        cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
        sin_gamma = np.sin(gamma)
        
        cell = np.array([
            [a, 0, 0],
            [b * cos_gamma, b * sin_gamma, 0],
            [c * cos_beta, 
             c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
             c * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2*cos_alpha*cos_beta*cos_gamma) / sin_gamma]
        ])
        
        # Build arrays
        n_atoms = len(atoms)
        n_modes = len(modes)
        
        positions_frac = np.array([atom['frac'] for atom in atoms])
        symbols = [atom['symbol'] for atom in atoms]
        masses = np.array([atom['mass'] for atom in atoms])
        
        frequencies_cm1 = np.array([m['freq'] for m in modes])
        mode_q_indices = np.array([m['q_id'] for m in modes])
        q_points_arr = np.array(q_points)
        
        # Reconstruct eigenvectors
        eigenvectors = np.zeros((n_modes, n_atoms, 3), dtype=complex)
        for ev in eigenvectors_raw:
            eigenvectors[ev['mode_id'], ev['atom_id']] = [ev['x'], ev['y'], ev['z']]
        
        # Reconstruct band_indices if present
        band_indices = None
        if band_indices_raw:
            band_indices = np.array(band_indices_raw, dtype=int)
        
        return cls(
            frequencies_cm1=frequencies_cm1,
            eigenvectors=eigenvectors,
            q_points=q_points_arr,
            mode_q_indices=mode_q_indices,
            degeneracy_groups=degeneracy_groups,
            positions_frac=positions_frac,
            cell=cell,
            symbols=symbols,
            masses=masses,
            supercell=tuple(supercell),
            n_atoms=n_atoms,
            band_indices=band_indices
        )

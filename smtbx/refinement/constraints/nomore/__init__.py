"""
NoMoRe ADP constraint and refinement workflow for smtbx.

Provides two public classes:

    PhononADPConstraint
        Self-contained smtbx-compatible ADP constraint.  Expresses all ASU
        ADPs as functions of partition-group scale factors via the thermal ADP
        equation.  Conforms to the smtbx Python constraint protocol so it can
        be registered with reparametrisation and, eventually, contributed to
        smtbx as a standalone plugin.

    NoMoReRefinement
        Workflow class that wires PhononADPConstraint to smtbx's
        crystallographic_ls and exposes compute_gradient() in partition-
        parameter (scale-factor) space directly.

Neither class imports from NoMoReCalculator or ObjectiveFunctionBuilder;
all thermal ADP mathematics is reimplemented self-containedly.
"""

from __future__ import annotations

import numpy as np
import scipy.constants as const
from typing import Optional

from cctbx import adptbx
import smtbx.refinement.constraints as _sc
import smtbx.refinement.least_squares
from scitbx import matrix
from scitbx.array_family import flex

from .phonon_data import PhononData
from .frequency_partition import (
    FrequencyPartitionStrategy, SensitivityBasedStrategy, ThermalCutoffStrategy
)
from .geometry import build_atom_mapping_periodic


# ---------------------------------------------------------------------------
# Physical constant: ADP amplitude factor for cm⁻¹ arithmetic
# U[Å²] = (E_cm1 / ω_cm1²) × _U_FACTOR / m_amu
# Derived from: U = ħ/(m ω) × E/ħω = ħ E / (m ω²)
_U_FACTOR = const.hbar * 1e20 / (const.atomic_mass * 2 * const.pi * const.c * 100)
_K_TO_CM1 = const.k / (const.h * const.c * 100)   # Boltzmann in cm⁻¹/K
_MIN_FREQ = 5.0     # cm⁻¹ — hard floor to avoid singularity
_ACOUSTIC_INIT = 10.0  # cm⁻¹ — initial value for near-zero acoustic branches


# ---------------------------------------------------------------------------

class PhononADPConstraint:
    """
    smtbx ADP constraint: U_asu = U(partition_scale_factors, phonon_data).

    All ASU atom ADPs are expressed as functions of N_groups scale factors
    s_g via the thermal displacement equation:

        ω_m(s) = ω_init_m × s_{group(m)}     (fixed modes keep ω_init)
        A_m    = C × E(ω_m) / ω_m²           (Bose-Einstein amplitude)
        U_cart[k] = Σ_m A_m × T_mk / N_q

    where T_mk = (1/count_k) Σ_{i→k} R_i^T Re(e_mi ⊗ e_mi†)/m_i R_i is the
    pre-computed ASU mode tensor (rotation-averaged over symmetry copies).

    Conforms to the smtbx Python constraint protocol (constrained_parameters
    + add_to) so it can be passed to reparametrisation as-is.

    Self-contained: the only crystallographic dependency is a plain cctbx
    xray.structure (no SmtbxAdapter or custom adapter needed).

    Args:
        phonon_data:         PhononData (eigenvectors, masses, frequencies_cm1, …)
        partition_strategy:  FrequencyPartitionStrategy that assigns modes to
                             refinement groups.

    Note:
        Temperature and xray_structure are obtained from the reparametrisation
        object in add_to().  Call add_to() before using update_structure() or
        any method that computes ADPs.
    """

    def __init__(
        self,
        phonon_data: PhononData,
        partition_strategy: FrequencyPartitionStrategy,
    ) -> None:
        self.phonon_data = phonon_data
        self.partition_strategy = partition_strategy

        frequencies = phonon_data.frequencies_cm1.copy()
        frequencies[frequencies < 10.0] = 10.0
        self._initial_frequencies_cm1 = frequencies
        self.n_modes = len(self._initial_frequencies_cm1)
        self.n_q = len(phonon_data.q_points)

        # Populated in add_to()
        self._scale_params: list = []

        # Deferred until add_to() — set to None so callers get a clear error
        self.temperature = None
        self._structure = None
        self._unit_cell = None
        self._n_asu_atoms = None
        self._p1_to_asu = None
        self._phonon_to_p1 = None
        self._mode_tensors = None

    # ------------------------------------------------------------------
    # smtbx constraint protocol
    # ------------------------------------------------------------------

    @property
    def constrained_parameters(self) -> tuple:
        """All ASU atom U parameters are governed by this constraint."""
        if self._n_asu_atoms is None:
            return ()
        return tuple((i, 'U') for i in range(self._n_asu_atoms))

    def add_to(self, reparametrisation) -> None:
        """
        Register in the reparametrisation.

        Obtains temperature and xray_structure from the reparametrisation,
        builds all crystallographic geometry mappings, pre-computes ASU mode
        tensors, updates scatterer U_star values from initial frequencies, and
        registers one independent_scalar_parameter per active partition group.

        Note: U_star parameters remain independent in smtbx (variable=True)
        because we apply the freq→U Jacobian chain rule in Python after
        build_up().  The scale-factor parameters are registered for
        bookkeeping and future C++ integration.
        """
        xray_structure = reparametrisation.structure
        self.temperature = reparametrisation.temperature
        self._structure = xray_structure
        self._unit_cell = xray_structure.unit_cell()
        self._n_asu_atoms = xray_structure.scatterers().size()

        for sc in reparametrisation.structure.scatterers():
            sc.flags.set_use_u_aniso(True)
            sc.flags.set_grad_u_aniso(True)
            sc.flags.set_use_u_iso(False)
            sc.flags.set_grad_u_iso(False)

        # Build crystallographic geometry mappings
        p1_structure = xray_structure.expand_to_p1(sites_mod_positive=True)
        self._p1_to_asu = self._build_p1_to_asu(xray_structure, p1_structure)
        self._phonon_to_p1 = self._build_phonon_to_p1(self.phonon_data, p1_structure)

        # Pre-compute mode tensors directly in ASU space: (N_modes, N_asu, 3, 3).
        self._mode_tensors = self._precompute_asu_mode_tensors(
            self.phonon_data.eigenvectors, self.phonon_data.masses
        )

        # Compute partition groups.
        # SensitivityBasedStrategy and ThermalCutoffStrategy accept temperature
        # as their second positional arg; all other strategies use the base
        # signature compute_groups(phonon_data, pre_groups=None).
        if isinstance(self.partition_strategy,
                      (SensitivityBasedStrategy, ThermalCutoffStrategy)):
            self._groups = self.partition_strategy.compute_groups(
                self.phonon_data, self.temperature + 273.15)
        else:
            self._groups = self.partition_strategy.compute_groups(self.phonon_data)
        # Sorted unique active group IDs → determines parameter ordering
        self._active_group_ids = np.array(
            sorted(set(self._groups.group_ids[self._groups.group_ids >= 0]))
        )
        self.n_parameters = len(self._active_group_ids)

        # Current scale factors (mutable state, start at 1.0)
        self.current_scales = np.ones(self.n_parameters)

        self._scale_params = []
        for _, value in zip(self._active_group_ids, self.current_scales):
            param = reparametrisation.add(
                _sc.independent_scalar_parameter,
                value=value,
                variable=True
            )
            self._scale_params.append(param)
        
        param = reparametrisation.add(
            _sc.nomore_u_star,
            scatterers= tuple(reparametrisation.structure.scatterers()),
            scale_params = tuple(self._scale_params),
            mode_tensors_ustar = flex.double(self._mode_tensors.tolist()),
            initial_frequencies = flex.double(self._initial_frequencies_cm1.tolist()),
            group_ids = flex.int(self._groups.group_ids.tolist()),
            temperature = reparametrisation.temperature + 273.15,
            n_modes=len(self._initial_frequencies_cm1),
            n_q = self.n_q
        )
        
        for i_sc in range(self._n_asu_atoms):
            reparametrisation.asu_scatterer_parameters[i_sc].u = param
            reparametrisation.shared_Us[i_sc] = param

    def scales_to_frequencies(self, scale_factors: np.ndarray) -> np.ndarray:
        """Convert N_groups scale factors to N_modes absolute frequencies."""
        freqs = self._initial_frequencies_cm1.copy()
        for param_idx, group_id in enumerate(self._active_group_ids):
            mode_indices = self._groups.get_group_modes(group_id)
            freqs[mode_indices] = (
                self._initial_frequencies_cm1[mode_indices] * scale_factors[param_idx]
            )
        return freqs

    def current_frequencies(self) -> np.ndarray:
        """Return current N_modes frequencies implied by current_scales."""
        return self.scales_to_frequencies(self.current_scales)

    # ------------------------------------------------------------------
    # Internal helpers (self-contained thermal ADP math)
    # ------------------------------------------------------------------

    def _precompute_asu_mode_tensors(
        self,
        eigenvectors: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute mode tensors directly in ASU space.  Shape: (N_modes, N_asu, 3, 3).

        Folds three steps into one precomputation:
          1. phonon atom order → CCTBX P1 order  (_phonon_to_p1)
          2. symmetry rotation to ASU frame       (R^T T R for each P1 atom)
          3. averaging over symmetry-equivalent P1 copies of each ASU atom

        At runtime, _compute_u_cart_asu and _compute_jacobian_wrt_scales
        can then work entirely in ASU space with no P1 intermediate.
        """
        n_modes, n_phonon, _ = eigenvectors.shape
        n_asu = self._n_asu_atoms

        # Build per-phonon-atom lookup arrays (independent of mode)
        asu_indices = np.empty(n_phonon, dtype=int)
        R_mats = np.empty((n_phonon, 3, 3))
        counts = np.zeros(n_asu)
        for ph_idx in range(n_phonon):
            p1_idx = self._phonon_to_p1[ph_idx] if self._phonon_to_p1 is not None else ph_idx
            asu_idx, r_cart = self._p1_to_asu[p1_idx]
            asu_indices[ph_idx] = asu_idx
            R_mats[ph_idx] = r_cart
            counts[asu_idx] += 1

        # T_p[m,k] = Re(e_mk ⊗ e_mk†) / m_k  — shape (N_modes, N_phonon, 3, 3)
        T_p = np.einsum('mki,mkj->mkij', eigenvectors, eigenvectors.conj()).real
        T_p /= masses[np.newaxis, :, np.newaxis, np.newaxis]

        mat_f = np.array(matrix.sqr(self._unit_cell.fractionalization_matrix())).reshape(3,3)
        rf_matrix = np.einsum('ab, kbc -> kac', mat_f, R_mats)

        # Rotate to ASU frame and convert to U*: T_r[m,k] = F^T R_k^T @ T_p[m,k] @ R_k F
        T_r = np.einsum('kba,mkbc,kcd->mkad', rf_matrix, T_p, rf_matrix)

        # Scatter-add contributions to ASU atoms, then average
        tensors = np.zeros((n_modes, n_asu, 3, 3))
        for ph_idx, k_asu in enumerate(asu_indices):
            tensors[:, k_asu] += T_r[:, ph_idx]
        tensors /= counts[np.newaxis, :, np.newaxis, np.newaxis]

        i = [0, 1, 2, 0, 0, 1]
        j = [0, 1, 2, 1, 2, 2]
        #i = [0, 0, 0, 1, 1, 2]
        #j = [0, 1, 2, 1, 2, 2]

        return np.ascontiguousarray(tensors[:,:,i,j].ravel())

    @staticmethod
    def _build_p1_to_asu(xray_structure, p1_structure) -> list:
        """
        Build list[(asu_idx, r_cart)] mapping each CCTBX P1 atom to its
        ASU origin atom and the Cartesian rotation that relates them.
        """
        asu_sites = xray_structure.sites_frac()
        p1_sites = p1_structure.sites_frac()
        unit_cell = xray_structure.unit_cell()
        space_group = xray_structure.space_group()
        orth = np.array(unit_cell.orthogonalization_matrix()).reshape(3, 3)
        frac = np.array(unit_cell.fractionalization_matrix()).reshape(3, 3)

        mapping = []
        for p1_site in p1_sites:
            found = False
            for j, asu_site in enumerate(asu_sites):
                for op in space_group.all_ops():
                    equiv = op * asu_site
                    diff = [p - e for p, e in zip(p1_site, equiv)]
                    diff = [d - round(d) for d in diff]
                    if unit_cell.length(diff) ** 2 < 1e-4:
                        r_frac = np.array(op.r().as_double()).reshape(3, 3)
                        mapping.append((j, orth @ r_frac @ frac))
                        found = True
                        break
                if found:
                    break
            if not found:
                print("P1 atom could not be mapped to ASU; using identity fallback")
                mapping.append((0, np.eye(3)))
        return mapping

    @staticmethod
    def _build_phonon_to_p1(phonon_data: PhononData, p1_structure) -> Optional[np.ndarray]:
        """
        Map phonon atom order → CCTBX P1 atom order.
        Returns None when sizes don't match (P1 structure assumed already ordered).
        """

        p1_scatterers = list(p1_structure.scatterers())
        if len(phonon_data.positions_frac) != len(p1_scatterers):
            print(
                f"Phonon atoms ({len(phonon_data.positions_frac)}) != P1 atoms "
                f"({len(p1_scatterers)}). Skipping phonon→P1 mapping."
            )
            return None

        p1_positions = np.array([sc.site for sc in p1_scatterers])
        p1_elements = np.array([sc.element_symbol() for sc in p1_scatterers])
        phonon_elements = np.array(phonon_data.symbols)
        unit_cell = p1_structure.unit_cell()

        indices, _ = build_atom_mapping_periodic(
            phonon_data.positions_frac, p1_positions, unit_cell,
            source_elements=phonon_elements, target_elements=p1_elements,
        )
        return np.array(indices, dtype=int)

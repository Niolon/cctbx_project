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

        # Build crystallographic geometry mappings
        p1_structure = xray_structure.expand_to_p1(sites_mod_positive=True)
        self._p1_to_asu = self._build_p1_to_asu(xray_structure, p1_structure)
        self._phonon_to_p1 = self._build_phonon_to_p1(self.phonon_data, p1_structure)

        # Pre-compute mode tensors directly in ASU space: (N_modes, N_asu, 3, 3).
        self._mode_tensors = self._precompute_asu_mode_tensors(
            self.phonon_data.eigenvectors, self.phonon_data.masses
        )

        # Compute partition groups
        self._groups = self.partition_strategy.compute_groups(self.phonon_data, self.temperature)
        # Sorted unique active group IDs → determines parameter ordering
        self._active_group_ids = np.array(
            sorted(set(self._groups.group_ids[self._groups.group_ids >= 0]))
        )
        self.n_parameters = len(self._active_group_ids)

        # Current scale factors (mutable state, start at 1.0)
        self.current_scales = np.ones(self.n_parameters)

        # Update structure to match initial frequencies (scale = 1.0)
        self.update_structure(self.current_scales)

        # Register one scalar per group
        self._scale_params = []
        for _ in self._active_group_ids:
            param = reparametrisation.add(
                _sc.independent_scalar_parameter,
                value=1.0,
                variable=True,
            )
            self._scale_params.append(param)

    # ------------------------------------------------------------------
    # Structure update
    # ------------------------------------------------------------------

    def update_structure(self, scale_factors: np.ndarray) -> None:
        """
        Update ASU scatterer U_star values from scale factors.

        Does NOT reinitialise the smtbx least-squares object; the caller
        (NoMoReRefinement) is responsible for that.

        Args:
            scale_factors: (N_groups,) scale factors for active groups.
        """
        u_cart_asu = self._compute_u_cart_asu(scale_factors)

        for i, sc in enumerate(self._structure.scatterers()):
            u = u_cart_asu[i]
            u_tuple = (u[0, 0], u[1, 1], u[2, 2], u[0, 1], u[0, 2], u[1, 2])
            sc.u_star = adptbx.u_cart_as_u_star(self._unit_cell, u_tuple)

        self.current_scales = np.asarray(scale_factors, dtype=float).copy()

    # ------------------------------------------------------------------
    # Gradient chain rule
    # ------------------------------------------------------------------

    def gradient_wrt_scales(self, grad_u_cart_asu_flat: np.ndarray) -> np.ndarray:
        """
        Transform dχ²/dU_cart_asu (from smtbx) to dχ²/d(scale_factors).

        Applies the chain rule:
            dχ²/ds_g = Σ_{i,j,k} (dχ²/dU_cart_asu_ijk) × (∂U_cart_asu_ijk/∂s_g)

        Args:
            grad_u_cart_asu_flat: (N_asu_atoms × 9,) gradient of χ² w.r.t.
                                  U_cart for each ASU atom (row-major 3×3).

        Returns:
            grad_scales: (N_groups,) gradient in scale-factor space.
        """
        jac_asu = self._compute_jacobian_wrt_scales(self.current_scales)
        return grad_u_cart_asu_flat @ jac_asu

    # ------------------------------------------------------------------
    # Scale ↔ frequency mapping
    # ------------------------------------------------------------------

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

        # Rotate to ASU frame: T_r[m,k] = R_k^T @ T_p[m,k] @ R_k
        T_r = np.einsum('kba,mkbc,kcd->mkad', R_mats, T_p, R_mats)

        # Scatter-add contributions to ASU atoms, then average
        tensors = np.zeros((n_modes, n_asu, 3, 3))
        for ph_idx, k_asu in enumerate(asu_indices):
            tensors[:, k_asu] += T_r[:, ph_idx]
        tensors /= counts[np.newaxis, :, np.newaxis, np.newaxis]

        return tensors

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
                logger.warning("P1 atom could not be mapped to ASU; using identity fallback")
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
            logger.warning(
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

    def _safe_frequencies(self, scale_factors: np.ndarray) -> np.ndarray:
        """Return clamped working frequencies (handle acoustic / low-freq modes)."""
        freqs = self.scales_to_frequencies(scale_factors)
        working = freqs.copy()
        working[working < 1.0] = _ACOUSTIC_INIT
        return np.maximum(working, _MIN_FREQ)

    def _bose_einstein(
        self, freqs_cm1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bose-Einstein mode energy E = ω(0.5 + n) and its derivative dE/dω.

        Returns:
            E      (N_modes,)  mode energy in cm⁻¹
            dE_dω  (N_modes,)  ∂E/∂ω
        """
        kt = max(self.temperature * _K_TO_CM1, 1e-6)
        x = freqs_cm1 / kt
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            ex = np.exp(np.minimum(x, 500.0))   # clamp to avoid inf overflow
            n = 1.0 / (ex - 1.0)
            dn_dx = -ex / (ex - 1.0) ** 2

        small = (x > 0) & (x < 1e-4)
        n[small] = 1.0 / x[small] - 0.5
        dn_dx[small] = -1.0 / x[small] ** 2

        # High-T frozen limit: x >> 1 → n ≈ 0, dn/dx ≈ 0
        large = x >= 500.0
        n[large] = 0.0
        dn_dx[large] = 0.0

        zero = x <= 0
        n[zero] = 0.0
        dn_dx[zero] = 0.0

        E = freqs_cm1 * (0.5 + n)
        dE_dω = (0.5 + n) + freqs_cm1 * dn_dx / kt
        return E, dE_dω

    def _compute_u_cart_asu(self, scale_factors: np.ndarray) -> np.ndarray:
        """Compute U_cart for all ASU atoms.  Shape: (N_asu, 3, 3)."""
        ν = self._safe_frequencies(scale_factors)
        E, _ = self._bose_einstein(ν)
        amplitudes = E / ν ** 2 * _U_FACTOR       # (N_modes,)
        u_cart = np.tensordot(amplitudes, self._mode_tensors, axes=([0], [0]))
        return u_cart / self.n_q

    def _compute_jacobian_wrt_scales(self, scale_factors: np.ndarray) -> np.ndarray:
        """
        Compute ∂U_cart_asu/∂s_g directly in ASU space.
        Shape: (N_asu × 9, N_groups).

        For each active group g:
            ∂U_cart_asu[k]/∂s_g = Σ_{m ∈ group_g} (dA_m/dω_m) × ω_init_m × T_mk_asu / N_q

        where dA/dω = _U_FACTOR × (dE/dω × ω − 2E) / ω³.
        """
        ν = self._safe_frequencies(scale_factors)
        E, dE_dω = self._bose_einstein(ν)

        # dA/dω = C × (dE/dω × ω − 2E) / ω³
        dA_dν = _U_FACTOR * (dE_dω * ν - 2 * E) / ν ** 3   # (N_modes,)

        n_asu = self._mode_tensors.shape[1]
        tensors_flat = self._mode_tensors.reshape(self.n_modes, n_asu * 9)

        jac = np.zeros((n_asu * 9, self.n_parameters))
        for param_idx, group_id in enumerate(self._active_group_ids):
            mode_indices = self._groups.get_group_modes(group_id)
            # ∂ω_m/∂s_g = ω_init_m  (scale parameterisation)
            sensitivity = dA_dν[mode_indices] * self._initial_frequencies_cm1[mode_indices]
            jac[:, param_idx] = tensors_flat[mode_indices].T @ sensitivity / self.n_q

        return jac

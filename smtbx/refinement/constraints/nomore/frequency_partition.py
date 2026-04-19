"""
Frequency partition strategies for NoMoRe refinement.

Implements strategy pattern for assigning phonon modes to refinement groups.
All strategies return RefinementGroups with group_ids array mapping each mode
to a refinement group (individual, shared MFSF, band-based, or fixed).

Resolves Q-018 (Extract FrequencyPartitionStrategy) and integrates ADR-013
(sensitivity-based classification).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import numpy as np


@dataclass
class RefinementGroups:
    """
    Refinement group assignments for phonon modes.
    
    Attributes:
        group_ids: (n_modes,) array assigning each mode to a refinement group.
            - group_ids[i] = k (k >= 0): mode i is in refinement group k
            - group_ids[i] = -1: mode i is FIXED (not refined)
            - Modes with same group_id share a scaling factor
            - Unique positive group_ids = number of refinement parameters
        
        group_metadata: Metadata dictionary keyed by group_id.
            Contains info like 'type' (individual/mfsf/band/fixed),
            'n_modes', 'freq_range', etc.
    
    Conventions:
        - Individual refinement: Each mode has unique group_id
        - MFSF refinement: Multiple modes share same group_id
        - Band scaling: group_id = band index
        - Fixed modes: group_id = -1
    """
    group_ids: np.ndarray
    group_metadata: Dict[int, Dict[str, Any]]
    
    def __post_init__(self):
        """Validate refinement groups."""
        if not isinstance(self.group_ids, np.ndarray):
            self.group_ids = np.array(self.group_ids, dtype=int)
        
        # Ensure integer type
        if self.group_ids.dtype != np.int64 and self.group_ids.dtype != np.int32:
            self.group_ids = self.group_ids.astype(int)
    
    def n_parameters(self) -> int:
        """Number of refinement parameters (excludes fixed modes)."""
        return len([g for g in np.unique(self.group_ids) if g >= 0])
    
    def get_group_modes(self, group_id: int) -> np.ndarray:
        """Get mode indices for a specific group."""
        return np.where(self.group_ids == group_id)[0]
    
    def get_fixed_modes(self) -> np.ndarray:
        """Get indices of fixed modes (group_id = -1)."""
        return np.where(self.group_ids == -1)[0]
    
    def get_refined_modes(self) -> np.ndarray:
        """Get indices of refined modes (group_id >= 0)."""
        return np.where(self.group_ids >= 0)[0]


class FrequencyPartitionStrategy(ABC):
    """
    Abstract base class for frequency partition strategies.
    
    All strategies must implement compute_groups() with PhononData.
    Strategy-specific parameters are set in __init__().
    """
    
    @abstractmethod
    def compute_groups(self, phonon_data, pre_groups: Optional[List[List[int]]] = None) -> RefinementGroups:
        """
        Assign phonon modes to refinement groups.
        
        Args:
            phonon_data: PhononData object containing all phonon information
            pre_groups: Optional list of pre-existing groups (e.g., from band_indices or degeneracy).
                Each element is a list of mode indices that should be grouped together.
                Strategy will compute group-averaged criteria and apply thresholds to groups.
        
        Returns:
            RefinementGroups with group_ids and metadata
        
        Raises:
            ValueError: If required data not available in phonon_data
        """
        pass


class FixedThresholdStrategy(FrequencyPartitionStrategy):
    """
    Partition modes using fixed frequency thresholds.
    
    Two modes of operation:
    
    1. Three-tier with medium_limit and high_limit:
       - LOW (< medium_limit): Individual mode refinement
       - MEDIUM (medium_limit <= ω < high_limit): Shared MFSF scaling
       - HIGH (>= high_limit): Fixed at initial values
    
    2. Fixed count with n_refined and high_limit:
       - N lowest modes (sorted by frequency): Individual refinement
       - Modes between n-th and high_limit: Shared MFSF scaling
       - Modes >= high_limit: Fixed at initial values
    
    Default thresholds from ADR-013: 200 cm⁻¹ and 1000 cm⁻¹.
    """
    
    def __init__(
        self, 
        medium_limit: Optional[float] = None, 
        high_limit: float = 1000.0,
        n_refined: Optional[int] = None
    ):
        """
        Initialize with fixed frequency thresholds or fixed count.
        
        Args:
            medium_limit: LOW/MEDIUM boundary in cm⁻¹ (three-tier mode)
            high_limit: MEDIUM/HIGH boundary in cm⁻¹
            n_refined: Number of lowest modes to refine (fixed count mode)
        
        Raises:
            ValueError: If both or neither of medium_limit and n_refined are provided
            ValueError: If medium_limit >= high_limit
        """
        if (medium_limit is None) == (n_refined is None):
            raise ValueError(
                "Must provide exactly one of: medium_limit (three-tier) or n_refined (fixed count)"
            )
        
        if medium_limit is not None and medium_limit >= high_limit:
            raise ValueError(
                f"medium_limit ({medium_limit}) must be < high_limit ({high_limit})"
            )
        
        if n_refined is not None and n_refined < 1:
            raise ValueError(f"n_refined must be >= 1, got {n_refined}")
        
        self.medium_limit = medium_limit if medium_limit is not None else 200.0
        self.high_limit = high_limit
        self.n_refined = n_refined
    
    def compute_groups(self, phonon_data, pre_groups: Optional[List[List[int]]] = None) -> RefinementGroups:
        """
        Assign groups based on fixed thresholds or fixed count.
        
        If pre_groups provided, applies thresholds to group-averaged frequencies.
        """
        frequencies = phonon_data.frequencies_cm1
        n_modes = len(frequencies)
        
        # Determine LOW/MEDIUM/HIGH split
        if self.n_refined is not None:
            # Fixed count mode: n-lowest modes are LOW, then MEDIUM up to high_limit
            if pre_groups is None:
                # Sort by frequency to find lowest n_refined
                sorted_indices = np.argsort(frequencies)
                low_mask = np.zeros(n_modes, dtype=bool)
                low_mask[sorted_indices[:self.n_refined]] = True
                medium_mask = (~low_mask) & (frequencies < self.high_limit)
                high_mask = frequencies >= self.high_limit
            else:
                # With pre_groups: rank by mean frequency, take lowest n groups as LOW
                group_mean_freqs = []
                for pre_group in pre_groups:
                    mean_freq = np.mean(frequencies[pre_group])
                    group_mean_freqs.append((mean_freq, pre_group))
                
                # Sort pre_groups by mean frequency
                group_mean_freqs.sort(key=lambda x: x[0])
                
                # Take n_refined groups (or all if n_refined >= n_groups) as LOW
                n_groups_low = min(self.n_refined, len(pre_groups))
                low_groups = [g for _, g in group_mean_freqs[:n_groups_low]]
                
                low_mask = np.zeros(n_modes, dtype=bool)
                for g in low_groups:
                    low_mask[g] = True
                
                # MEDIUM: remaining modes below high_limit
                medium_mask = (~low_mask) & (frequencies < self.high_limit)
                high_mask = frequencies >= self.high_limit
        else:
            # Three-tier mode: use medium_limit threshold
            low_mask = frequencies < self.medium_limit
            medium_mask = (frequencies >= self.medium_limit) & (frequencies < self.high_limit)
            high_mask = frequencies >= self.high_limit
        
        if pre_groups is None:
            # No pre-grouping: each mode considered individually
            group_ids = np.full(n_modes, -1, dtype=int)
            next_group_id = 0
            mfsf_group_id = None
            
            for i in range(n_modes):
                if high_mask[i]:
                    # HIGH: fixed (not refined)
                    group_ids[i] = -1
                elif low_mask[i]:
                    # LOW: individual refinement (unique group per mode)
                    group_ids[i] = next_group_id
                    next_group_id += 1
                elif medium_mask[i]:
                    # MEDIUM: shared MFSF (all modes get same group ID)
                    if mfsf_group_id is None:
                        mfsf_group_id = next_group_id
                        next_group_id += 1
                    group_ids[i] = mfsf_group_id
        else:
            # Apply thresholds to pre-grouped modes (e.g., bands or degenerate modes)
            group_ids = np.full(n_modes, -1, dtype=int)
            next_group_id = 0
            mfsf_group_id = None
            
            for pre_group in pre_groups:
                # Determine tier for this pre-group
                if np.all(high_mask[pre_group]):
                    # HIGH: entire group is fixed
                    group_ids[pre_group] = -1
                elif np.any(low_mask[pre_group]):
                    # LOW: this entire pre-group gets individual refinement
                    group_ids[pre_group] = next_group_id
                    next_group_id += 1
                else:
                    # MEDIUM: assign to shared MFSF group
                    if mfsf_group_id is None:
                        mfsf_group_id = next_group_id
                        next_group_id += 1
                    group_ids[pre_group] = mfsf_group_id
        
        # Build metadata
        metadata = {}
        
        # Add MFSF group metadata if any modes in MEDIUM range
        if np.any(medium_mask) and mfsf_group_id is not None:
            medium_freqs = frequencies[medium_mask]
            metadata[mfsf_group_id] = {
                'type': 'mfsf',
                'n_modes': int(np.sum(medium_mask)),
                'freq_range': (float(np.min(medium_freqs)), float(np.max(medium_freqs))),
                'strategy': 'fixed_threshold' if self.n_refined is None else 'fixed_count'
            }
        
        # Add info about fixed modes
        if np.any(high_mask):
            metadata[-1] = {
                'type': 'fixed',
                'n_modes': int(np.sum(high_mask)),
                'min_freq': float(np.min(frequencies[high_mask])) if np.any(high_mask) else None
            }
        
        return RefinementGroups(group_ids=group_ids, group_metadata=metadata)


class ThermalCutoffStrategy(FixedThresholdStrategy):
    """
    Compute frequency thresholds from kT, then use parent's grouping logic.
    
    Extends FixedThresholdStrategy with temperature-dependent limits:
    - medium_limit = medium_factor * kT
    - high_limit = high_factor * kT
    
    Default factors: 1.5 kT and 2.0 kT (conservative, per ADR-013).
    """
    
    def __init__(self, medium_factor: float = 1.5, high_factor: float = 2.0):
        """
        Initialize with kT factors.
        
        Args:
            medium_factor: Multiple of kT for LOW/MEDIUM boundary
            high_factor: Multiple of kT for MEDIUM/HIGH boundary
        
        Raises:
            ValueError: If medium_factor >= high_factor
        """
        if medium_factor >= high_factor:
            raise ValueError(
                f"medium_factor ({medium_factor}) must be < high_factor ({high_factor})"
            )
        self.medium_factor = medium_factor
        self.high_factor = high_factor
        self.n_refined = None
        # Don't set limits yet - need temperature from compute_groups()
    
    def compute_groups(self, phonon_data, pre_groups: Optional[List[List[int]]] = None) -> RefinementGroups:
        """
        Compute thermal limits and assign groups.
        
        If pre_groups provided, applies thermal thresholds to group-averaged frequencies.
        """
        # Get temperature from phonon_data or raise error
        temperature = getattr(phonon_data, 'temperature', None)
        if temperature is None:
            raise ValueError("ThermalCutoffStrategy requires temperature in phonon_data")
        
        from nomore_ase.utils import units
        
        # Compute frequency limits from kT
        self.medium_limit = units.thermal_cutoff_cm1(temperature, self.medium_factor)
        self.high_limit = units.thermal_cutoff_cm1(temperature, self.high_factor)
        
        # Call parent's grouping logic with pre_groups
        groups = super().compute_groups(phonon_data, pre_groups)
        
        # Add thermal metadata
        groups.group_metadata['temperature'] = temperature
        groups.group_metadata['thermal_factors'] = (self.medium_factor, self.high_factor)
        
        return groups


class SensitivityBasedStrategy(FrequencyPartitionStrategy):
    """
    Data-driven partitioning based on cumulative ADP sensitivity.
    
    Implements ADR-013 approach: modes are ordered by frequency, then
    cumulative ADP sensitivity is computed. Thresholds define:
    - LOW: modes contributing first low_threshold of total sensitivity
    - MEDIUM: modes from low_threshold to high_threshold
    - HIGH: remaining modes (low sensitivity, poorly constrained)
    
    Default: 90% threshold for LOW, 99% for MEDIUM (from ADR-013 analysis).
    """
    
    def __init__(self, low_threshold: float = 0.90, high_threshold: float = 0.99):
        """
        Initialize with cumulative sensitivity thresholds.
        
        Args:
            low_threshold: Cumulative sensitivity for LOW/MEDIUM boundary (0-1)
            high_threshold: Cumulative sensitivity for MEDIUM/HIGH boundary (0-1)
        
        Raises:
            ValueError: If thresholds not in valid range
        """
        if not (0 < low_threshold < high_threshold < 1):
            raise ValueError(
                f"Must have 0 < low_threshold < high_threshold < 1, "
                f"got {low_threshold}, {high_threshold}"
            )
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def compute_groups(self, phonon_data, pre_groups: Optional[List[List[int]]] = None) -> RefinementGroups:
        """
        Assign groups based on ADP sensitivity analysis.
        
        If pre_groups provided, computes group-averaged sensitivities.
        """
        # Extract required data
        frequencies = phonon_data.frequencies_cm1
        eigenvectors = phonon_data.eigenvectors
        masses = phonon_data.masses
        temperature = getattr(phonon_data, 'temperature', None)
        weights = phonon_data.weights
        
        # Validate required inputs
        if eigenvectors is None or masses is None or temperature is None:
            raise ValueError(
                "SensitivityBasedStrategy requires eigenvectors, masses, and temperature"
            )
        
        from nomore_ase.core.sensitivity_utils import compute_mode_sensitivity
        
        # Compute per-mode ADP sensitivity
        sensitivities = compute_mode_sensitivity(
            frequencies, eigenvectors, masses, temperature, weights
        )
        
        if pre_groups is None:
            # Original behavior: sort individual modes by sensitivity
            sorted_idx = np.argsort(frequencies)
            sorted_sensitivities = sensitivities[sorted_idx]
        else:
            # Compute group-averaged sensitivities
            n_groups = len(pre_groups)
            group_sensitivities = np.zeros(n_groups)
            group_mean_freqs = np.zeros(n_groups)
            
            for i, pre_group in enumerate(pre_groups):
                group_sensitivities[i] = np.mean(sensitivities[pre_group])
                group_mean_freqs[i] = np.mean(frequencies[pre_group])
            
            # Sort groups by frequency
            sorted_idx = np.argsort(group_mean_freqs)
            sorted_sensitivities = group_sensitivities[sorted_idx]
        cumulative = np.cumsum(sorted_sensitivities)
        cumulative /= cumulative[-1]  # Normalize to [0, 1]
        
        # Find threshold indices
        low_idx = np.searchsorted(cumulative, self.low_threshold)
        high_idx = np.searchsorted(cumulative, self.high_threshold)
        
        # Assign groups
        n_modes = len(frequencies)
        group_ids = np.full(n_modes, -1, dtype=int)  # Default: fixed
        
        if pre_groups is None:
            # LOW modes: individual refinement (unique group per mode)
            for rank, mode_idx in enumerate(sorted_idx[:low_idx]):
                group_ids[mode_idx] = rank
            
            # MEDIUM modes: shared MFSF group
            mfsf_group_id = low_idx  # Use low_idx as unique MFSF group ID
            for mode_idx in sorted_idx[low_idx:high_idx]:
                group_ids[mode_idx] = mfsf_group_id
        else:
            # Apply to pre-groups
            next_group_id = 0
            mfsf_group_id = None
            
            for rank in range(len(pre_groups)):
                group_idx = sorted_idx[rank]
                pre_group = pre_groups[group_idx]
                
                if rank < low_idx:
                    # LOW: individual refinement for this pre-group
                    group_ids[pre_group] = next_group_id
                    next_group_id += 1
                elif rank < high_idx:
                    # MEDIUM: shared MFSF
                    if mfsf_group_id is None:
                        mfsf_group_id = next_group_id
                        next_group_id += 1
                    group_ids[pre_group] = mfsf_group_id
                # else: HIGH, remains -1
        
        # Build metadata
        metadata = {
            mfsf_group_id: {
                'type': 'mfsf',
                'n_modes': high_idx - low_idx,
                'sensitivity_threshold': (self.low_threshold, self.high_threshold),
                'cumulative_sensitivity_range': (float(cumulative[low_idx-1] if low_idx > 0 else 0),
                                                  float(cumulative[high_idx-1])),
                'strategy': 'sensitivity'
            }
        }
        
        # Store cumulative for diagnostics
        metadata['cumulative_sensitivity'] = cumulative
        
        return RefinementGroups(group_ids=group_ids, group_metadata=metadata)


class ManualStrategy(FrequencyPartitionStrategy):
    """
    Explicitly provide refinement group assignments.
    
    Used for:
    - Custom grouping schemes
    - Backward compatibility with explicit limits
    - Testing and debugging
    """
    
    def __init__(self, group_ids: np.ndarray, metadata: Optional[Dict] = None):
        """
        Initialize with explicit group assignments.
        
        Args:
            group_ids: Pre-computed group assignments, shape (n_modes,)
            metadata: Optional metadata dictionary
        """
        self.group_ids = np.array(group_ids, dtype=int)
        self.metadata = metadata if metadata is not None else {}
    
    def compute_groups(self, phonon_data, pre_groups: Optional[List[List[int]]] = None) -> RefinementGroups:
        """
        Return pre-set group assignments.
        
        Validates that group_ids matches phonon_data length.
        pre_groups parameter ignored for manual strategy.
        """
        # Validate group_ids matches frequencies length
        if len(self.group_ids) != len(phonon_data.frequencies_cm1):
            raise ValueError(
                f"group_ids length ({len(self.group_ids)}) != "
                f"frequencies length ({len(phonon_data.frequencies_cm1)})"
            )
        
        return RefinementGroups(
            group_ids=self.group_ids.copy(),
            group_metadata=self.metadata.copy()
        )


class BandScalingStrategy(FrequencyPartitionStrategy):
    """
    Wrapper strategy that applies band-based pre-grouping to another strategy.
    
    This strategy:
    1. Creates pre_groups from band_indices in phonon_data
    2. Passes these pre_groups to the base_strategy
    3. Base strategy applies its criteria (thermal/fixed/sensitivity) to band-averaged values
    
    Example:
        # Apply thermal cutoff to band-averaged frequencies
        strategy = BandScalingStrategy(ThermalCutoffStrategy(1.5, 3.0))
    """
    
    def __init__(self, base_strategy: FrequencyPartitionStrategy):
        """
        Initialize with a base strategy to apply to bands.
        
        Args:
            base_strategy: Strategy that will receive band-based pre_groups
        """
        self.base_strategy = base_strategy
    
    def compute_groups(self, phonon_data, pre_groups: Optional[List[List[int]]] = None) -> RefinementGroups:
        """
        Create band-based pre_groups and pass to base strategy.
        
        Args:
            phonon_data: Must contain band_indices
            pre_groups: If provided, will be merged with band grouping
        
        Returns:
            RefinementGroups from base strategy applied to bands
        """
        if phonon_data.band_indices is None:
            raise ValueError("BandScalingStrategy requires band_indices in phonon_data")
        
        # Create pre_groups from bands
        band_indices = phonon_data.band_indices
        unique_bands = np.unique(band_indices[band_indices >= 0])
        
        band_pre_groups = []
        for band_id in unique_bands:
            band_mask = band_indices == band_id
            band_modes = np.where(band_mask)[0].tolist()
            band_pre_groups.append(band_modes)
        
        # If we already have pre_groups (e.g., from degeneracy), merge them with bands
        if pre_groups is not None:
            # Need to merge: within each band, keep degeneracy groups together
            # For now, use band groups (degeneracy within bands stays grouped)
            # This is conservative - degenerate modes in same band stay together
            pass  # band_pre_groups already handles this
        
        # Apply base strategy with band pre_groups
        return self.base_strategy.compute_groups(phonon_data, band_pre_groups)


def create_pre_groups(phonon_data) -> Optional[List[List[int]]]:
    """
    Create pre-groups from band_indices or degeneracy_groups in phonon_data.
    
    Pre-groups combine modes that should always be refined together (e.g., degenerate modes
    or modes in the same phonon band). Strategies then apply their criteria to group-averaged
    quantities.
    
    Args:
        phonon_data: PhononData object
        
    Returns:
        List of lists, where each inner list contains mode indices that belong together.
        Returns None if no pre-grouping needed (each mode is its own group).
    """
    band_indices = phonon_data.band_indices
    degeneracy_groups = phonon_data.degeneracy_groups
    
    # Start with degeneracy groups or individual modes
    if degeneracy_groups is not None and len(degeneracy_groups) > 0:
        # Use degeneracy groups as starting point
        pre_groups = [list(group) for group in degeneracy_groups]
    else:
        # Each mode is its own group
        pre_groups = [[i] for i in range(len(phonon_data.frequencies_cm1))]
    
    # If band_indices provided, merge degenerate groups within same band
    if band_indices is not None:
        # Group by band
        unique_bands = np.unique(band_indices[band_indices >= 0])
        band_groups = []
        
        for band_id in unique_bands:
            band_mask = band_indices == band_id
            band_mode_indices = np.where(band_mask)[0]
            
            # Collect all pre_groups that belong to this band
            modes_in_band = []
            for mode_idx in band_mode_indices:
                # Find which pre_group contains this mode
                for pre_group in pre_groups:
                    if mode_idx in pre_group:
                        modes_in_band.extend(pre_group)
                        break
            
            # Remove duplicates and create band group
            band_groups.append(sorted(set(modes_in_band)))
        
        pre_groups = band_groups
    
    # If all groups are singletons, return None (no pre-grouping needed)
    if all(len(g) == 1 for g in pre_groups):
        return None
    
    return pre_groups


# ---------------------------------------------------------------------------
# Helpers for eigenvector-based mode / subspace matching
# ---------------------------------------------------------------------------

def _subspace_overlap(evecs1_group: np.ndarray, evecs2_group: np.ndarray) -> float:
    """
    Compute the subspace overlap between two degenerate mode groups.

    For a k-fold degenerate subspace, individual eigenvectors are defined only
    up to an arbitrary unitary rotation within the subspace.  The physically
    meaningful quantity is therefore the overlap between the *subspaces* spanned
    by each group, not between individual vectors.

    The overlap is computed as::

        overlap = ‖U† V‖_F² / k

    where U and V are (k × 3·N_atoms) matrices of the (flattened) eigenvectors
    and ‖·‖_F denotes the Frobenius norm.  This equals 1 when the two subspaces
    are identical and 0 when they are orthogonal.  For k=1 it reduces to
    |⟨u|v⟩|², which is identical to the scalar eigenvector overlap used
    elsewhere.

    Args:
        evecs1_group: (k, N_atoms, 3) complex eigenvectors of the first group
        evecs2_group: (k, N_atoms, 3) complex eigenvectors of the second group

    Returns:
        Subspace overlap in [0, 1].

    Raises:
        ValueError: If the two groups have different sizes.
    """
    k1 = evecs1_group.shape[0]
    k2 = evecs2_group.shape[0]
    if k1 != k2:
        raise ValueError(
            f"Subspace overlap requires equal-sized groups, got {k1} vs {k2}"
        )
    k = k1
    # Flatten to (k, 3·N_atoms)
    u = evecs1_group.reshape(k, -1)
    v = evecs2_group.reshape(k, -1)
    # Cross-Gram matrix (k × k)
    M = u.conj() @ v.T
    return float(np.linalg.norm(M, "fro") ** 2 / k)


def _get_degeneracy_groups_at_q(
    degeneracy_groups: List[List[int]],
    q_mode_indices: np.ndarray,
) -> List[List[int]]:
    """
    Filter global degeneracy groups to those modes present at a specific q-point.

    Args:
        degeneracy_groups: Global list of degenerate mode groups (from PhononData).
        q_mode_indices: 1-D array of global mode indices belonging to this q-point.

    Returns:
        List of lists of *global* mode indices, one list per degenerate group at
        this q-point.  Modes that appear in no degeneracy group (or in a singleton
        group) are returned as singleton lists.
    """
    q_set = set(q_mode_indices.tolist())

    # Build a lookup: mode_idx → group_members (only the subset that live at this q)
    seen: set[int] = set()
    groups: List[List[int]] = []

    for group in degeneracy_groups:
        members_at_q = [m for m in group if m in q_set]
        if not members_at_q:
            continue
        # Add as a group (possibly partial if degeneracy_groups span q-points, but
        # typically they won't for per-q extracted modes)
        for m in members_at_q:
            if m not in seen:
                groups.append(members_at_q)
                seen.update(members_at_q)
                break

    # Any mode not yet assigned → singleton
    for idx in q_mode_indices:
        if idx not in seen:
            groups.append([int(idx)])
            seen.add(int(idx))

    return groups


# ---------------------------------------------------------------------------
# TransferStrategy
# ---------------------------------------------------------------------------


class TransferStrategy(FrequencyPartitionStrategy):
    """
    Transfer a refinement strategy from a source PhononData to a target PhononData.

    Use this when the same compound is measured at slightly different conditions
    (e.g. different temperatures or pressures) so that cell parameters shift but
    the atomic connectivity and q-mesh are identical.  A strategy optimised for
    the source structure can be transferred to the target by matching phonon modes
    via eigenvector overlap.

    **Degeneracy handling**:

    Individual eigenvectors within a degenerate subspace are arbitrary (any
    unitary rotation is equally valid), so per-mode overlap is not meaningful
    there.  Instead, *subspace* overlap ``‖U†V‖_F² / k`` is used, which is
    invariant to unitary rotations within the subspace.  Matching is then done
    group-by-group (same degeneracy size) using the Hungarian algorithm.

    Args:
        source_phonon_data: PhononData from which the strategy was derived.
        source_groups: RefinementGroups computed for source_phonon_data.
        min_overlap: Minimum acceptable subspace/eigenvector overlap for a
            match.  If any mode or group pair falls below this threshold,
            ``compute_groups`` raises ``ValueError``.  Default 0.5.
    """

    def __init__(
        self,
        source_phonon_data,
        source_groups: RefinementGroups,
        min_overlap: float = 0.5,
    ):
        if not (0.0 < min_overlap <= 1.0):
            raise ValueError(
                f"min_overlap must be in (0, 1], got {min_overlap}"
            )
        self.source_phonon_data = source_phonon_data
        self.source_group_ids = source_groups.group_ids.copy()
        self.source_metadata = source_groups.group_metadata.copy()
        self.min_overlap = min_overlap

    def compute_groups(
        self,
        phonon_data,
        pre_groups: Optional[List[List[int]]] = None,
    ) -> RefinementGroups:
        """
        Transfer group assignments from the source to *phonon_data* (target).

        The ``pre_groups`` parameter is accepted for API compatibility but is
        silently ignored — the transfer is authoritative.

        Args:
            phonon_data: Target PhononData (same q-mesh as source).
            pre_groups: Ignored.

        Returns:
            RefinementGroups with group_ids transferred from source.

        Raises:
            ValueError: If q-points or mode count differ between source and target,
                if degeneracy sizes are incompatible at any q-point, or if any
                matched pair has subspace overlap below ``min_overlap``.
        """
        from nomore_ase.optimization.band_assignment import match_modes_hungarian

        src = self.source_phonon_data
        tgt = phonon_data

        # ---- Structural compatibility checks --------------------------------
        if src.q_points.shape != tgt.q_points.shape:
            raise ValueError(
                f"q_points shape mismatch: source {src.q_points.shape} vs "
                f"target {tgt.q_points.shape}. "
                "TransferStrategy requires identical q-meshes."
            )
        if not np.allclose(src.q_points, tgt.q_points, atol=1e-6):
            raise ValueError(
                "q_points arrays differ between source and target. "
                "TransferStrategy requires identical q-meshes."
            )
        if len(src.frequencies_cm1) != len(tgt.frequencies_cm1):
            raise ValueError(
                f"Mode count mismatch: source {len(src.frequencies_cm1)} vs "
                f"target {len(tgt.frequencies_cm1)}."
            )

        n_modes = len(tgt.frequencies_cm1)
        new_group_ids = np.full(n_modes, -1, dtype=int)
        transfer_diagnostics: Dict[int, Dict[str, Any]] = {}

        unique_q_indices = np.unique(src.mode_q_indices)

        for q_i in unique_q_indices:
            src_mask = src.mode_q_indices == q_i
            tgt_mask = tgt.mode_q_indices == q_i

            src_global = np.where(src_mask)[0]
            tgt_global = np.where(tgt_mask)[0]

            src_evecs = src.eigenvectors[src_global]  # (M, N_atoms, 3)
            tgt_evecs = tgt.eigenvectors[tgt_global]  # (M, N_atoms, 3)

            # Degeneracy groups at this q-point
            src_deg_groups = _get_degeneracy_groups_at_q(
                src.degeneracy_groups, src_global
            )
            tgt_deg_groups = _get_degeneracy_groups_at_q(
                tgt.degeneracy_groups, tgt_global
            )

            # Partition groups by size
            def _by_size(groups: List[List[int]]) -> Dict[int, List[List[int]]]:
                d: Dict[int, List[List[int]]] = {}
                for g in groups:
                    d.setdefault(len(g), []).append(g)
                return d

            src_by_size = _by_size(src_deg_groups)
            tgt_by_size = _by_size(tgt_deg_groups)

            worst_overlap_at_q = 1.0
            q_i_int = int(q_i)

            for k, src_groups_k in src_by_size.items():
                if k not in tgt_by_size:
                    raise ValueError(
                        f"q-point index {q_i}: source has {len(src_groups_k)} "
                        f"group(s) of size {k} but target has none.  "
                        "Degeneracy structure is incompatible."
                    )
                tgt_groups_k = tgt_by_size[k]
                if len(src_groups_k) != len(tgt_groups_k):
                    raise ValueError(
                        f"q-point index {q_i}: source has {len(src_groups_k)} "
                        f"group(s) of size {k} but target has {len(tgt_groups_k)}.  "
                        "Degeneracy structure is incompatible."
                    )

                n_groups = len(src_groups_k)

                # Build subspace overlap matrix between source and target groups
                # overlap_mat[i, j] = subspace overlap of src_groups_k[i] vs tgt_groups_k[j]
                overlap_mat = np.zeros((n_groups, n_groups))
                for i, sg in enumerate(src_groups_k):
                    # Convert global indices → local slice indices within src_global
                    src_local = np.array(
                        [np.where(src_global == gi)[0][0] for gi in sg]
                    )
                    u = src_evecs[src_local]  # (k, N_atoms, 3)
                    for j, tg in enumerate(tgt_groups_k):
                        tgt_local = np.array(
                            [np.where(tgt_global == gi)[0][0] for gi in tg]
                        )
                        v = tgt_evecs[tgt_local]  # (k, N_atoms, 3)
                        overlap_mat[i, j] = _subspace_overlap(u, v)

                # Hungarian matching: maximise total subspace overlap
                assignment = match_modes_hungarian(overlap_mat)
                # assignment[i] = j means src_groups_k[i] → tgt_groups_k[j]

                for i, sg in enumerate(src_groups_k):
                    j = assignment[i]
                    tg = tgt_groups_k[j]
                    ov = overlap_mat[i, j]

                    if ov < self.min_overlap:
                        raise ValueError(
                            f"q-point index {q_i}: matched group pair has subspace "
                            f"overlap {ov:.3f} < min_overlap={self.min_overlap:.3f}.  "
                            "The structures may be too dissimilar for strategy transfer.  "
                            "Lower min_overlap or derive a fresh strategy for the target."
                        )

                    worst_overlap_at_q = min(worst_overlap_at_q, ov)

                    # Look up the source group_id (all members of sg share same id)
                    src_group_id = int(self.source_group_ids[sg[0]])

                    # Assign to target modes
                    for tgt_mode in tg:
                        new_group_ids[tgt_mode] = src_group_id

            transfer_diagnostics[q_i_int] = {"worst_overlap": worst_overlap_at_q}

        # Build metadata: carry over source metadata, add transfer info
        metadata = {k: dict(v) for k, v in self.source_metadata.items()
                    if isinstance(v, dict)}
        metadata["transfer"] = {
            "source_supercell": src.supercell,
            "target_supercell": tgt.supercell,
            "min_overlap_threshold": self.min_overlap,
            "per_q_diagnostics": transfer_diagnostics,
            "worst_overlap_global": min(
                d["worst_overlap"] for d in transfer_diagnostics.values()
            ) if transfer_diagnostics else 1.0,
        }

        return RefinementGroups(group_ids=new_group_ids, group_metadata=metadata)


def create_strategy(strategy_name: str, strategy_params: Optional[Dict[str, Any]] = None) -> FrequencyPartitionStrategy:
    """
    Factory function to create frequency partition strategies.
    
    This centralizes strategy creation and provides a clean interface
    for the workflow layer.
    
    Args:
        strategy_name: Name of strategy ('thermal', 'fixed', 'sensitivity', 'band', 'manual')
        strategy_params: Parameters for the chosen strategy (optional)
            - thermal: {'medium_factor': 1.5, 'high_factor': 3.0}
            - fixed: {'medium_limit': 200.0, 'high_limit': 1000.0}
            - sensitivity: {'low_threshold': 0.90, 'high_threshold': 0.99}
            - band: {'fixed_bands_limit': 1500.0}
            - manual: {'group_ids': array, 'metadata': dict}
    
    Returns:
        FrequencyPartitionStrategy instance
        
    Raises:
        ValueError: If strategy_name is unknown
    """
    if strategy_params is None:
        strategy_params = {}
    
    # Apply defaults for each strategy
    if strategy_name == 'thermal':
        defaults = {'medium_factor': 1.5, 'high_factor': 3.0}
        params = {**defaults, **strategy_params}
        return ThermalCutoffStrategy(**params)
    
    elif strategy_name == 'fixed':
        defaults = {'medium_limit': 200.0, 'high_limit': 1000.0}
        params = {**defaults, **strategy_params}
        return FixedThresholdStrategy(**params)
    
    elif strategy_name == 'sensitivity':
        defaults = {'low_threshold': 0.90, 'high_threshold': 0.99}
        params = {**defaults, **strategy_params}
        return SensitivityBasedStrategy(**params)
    
    elif strategy_name == 'manual':
        if 'group_ids' not in strategy_params:
            raise ValueError("Manual strategy requires 'group_ids' in strategy_params")
        return ManualStrategy(**strategy_params)

    elif strategy_name == 'transfer':
        required = ('source_phonon_data', 'source_groups')
        for key in required:
            if key not in strategy_params:
                raise ValueError(
                    f"Transfer strategy requires '{key}' in strategy_params"
                )
        return TransferStrategy(**strategy_params)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Valid options: 'thermal', 'fixed', 'sensitivity', 'manual', 'transfer'"
        )

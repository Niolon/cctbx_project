"""Geometry utilities for periodic boundary handling."""

import numpy as np
from typing import Tuple
from scipy.optimize import linear_sum_assignment

def wrap_to_unit_cell(positions_frac: np.ndarray) -> np.ndarray:
    """
    Wrap fractional coordinates to the [0, 1) range.
    
    Args:
        positions_frac: (N, 3) array of fractional coordinates.
        
    Returns:
        Wrapped coordinates in [0, 1) range.
    """
    return positions_frac - np.floor(positions_frac)


def periodic_diff(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """
    Compute the difference between two fractional positions accounting for
    periodic boundary conditions.
    
    The result is wrapped to [-0.5, 0.5] range to find the minimum image.
    
    Args:
        pos1: First position (fractional coordinates).
        pos2: Second position (fractional coordinates).
        
    Returns:
        Difference vector wrapped to [-0.5, 0.5].
    """
    diff = pos1 - pos2
    return diff - np.round(diff)


def find_closest_atom_periodic(
    query_pos: np.ndarray,
    reference_positions: np.ndarray,
    unit_cell,
    tolerance: float = 0.5
) -> Tuple[int, float]:
    """
    Find the closest atom in reference_positions to query_pos, accounting for
    periodic boundary conditions.
    
    Both positions should be in fractional coordinates. The query position
    is automatically wrapped to [0, 1) before comparison.
    
    Args:
        query_pos: (3,) fractional coordinates of the query point.
        reference_positions: (N, 3) fractional coordinates of reference atoms.
        unit_cell: CCTBX or ASE unit cell object with a `length()` method that
                   computes cartesian distance from fractional difference vector.
        tolerance: Maximum allowed distance in Å for a valid match.
        
    Returns:
        Tuple of (best_index, distance_in_angstrom).
        Returns (-1, inf) if no match found within tolerance.
    """
    # Wrap query position to [0, 1)
    query_wrapped = wrap_to_unit_cell(query_pos)
    
    best_match = -1
    best_dist = float('inf')
    
    for j, ref_pos in enumerate(reference_positions):
        diff = periodic_diff(query_wrapped, ref_pos)
        
        # Compute cartesian distance
        # Handle both CCTBX and numpy interfaces
        if hasattr(unit_cell, 'length'):
            # CCTBX unit_cell
            dist = unit_cell.length(tuple(diff))
        else:
            # numpy/ASE - assume unit_cell is a (3,3) matrix
            cart_diff = diff @ unit_cell
            dist = np.linalg.norm(cart_diff)
        
        if dist < best_dist:
            best_dist = dist
            best_match = j
    
    if best_dist > tolerance:
        best_match = -1
        
    return best_match, best_dist


def build_atom_mapping_periodic(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    unit_cell,
    tolerance: float = 2.0,
    source_elements: np.ndarray = None,
    target_elements: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a mapping from source atom indices to target atom indices,
    accounting for periodic boundaries using optimal bipartite matching.
    
    Uses the Hungarian algorithm (linear_sum_assignment) to find the globally
    optimal 1-to-1 mapping that minimizes total distance.
    
    If element types are provided, matching is done separately for each element
    type (C matches C, O matches O, etc.) which is more efficient and prevents
    cross-element matching.
    
    Args:
        source_positions: (N, 3) fractional coordinates of source atoms.
        target_positions: (M, 3) fractional coordinates of target atoms.
        unit_cell: Unit cell object (CCTBX or 3x3 array).
        tolerance: Maximum distance in Å for valid match (warning threshold).
        source_elements: Optional (N,) array of element symbols/atomic numbers.
        target_elements: Optional (M,) array of element symbols/atomic numbers.
        
    Returns:
        Tuple of (indices, distances) where:
        - indices[i] = index in target_positions matched to source_positions[i]
        - distances[i] = distance to matched target in Å.
    """    
    n_source = len(source_positions)
    indices = np.zeros(n_source, dtype=int)
    distances = np.zeros(n_source)
    
    # If elements provided, match per-element groups
    if source_elements is not None and target_elements is not None:
        unique_elements = np.unique(source_elements)
        
        for elem in unique_elements:
            # Get indices for this element in source and target
            src_mask = (source_elements == elem)
            tgt_mask = (target_elements == elem)
            src_indices = np.where(src_mask)[0]
            tgt_indices = np.where(tgt_mask)[0]
            
            if len(src_indices) != len(tgt_indices):
                raise ValueError(
                    f"Element count mismatch for {elem}: "
                    f"{len(src_indices)} in source, {len(tgt_indices)} in target"
                )
            
            if len(src_indices) == 0:
                continue
            
            # Build cost matrix for this element subset
            cost_matrix = _build_distance_matrix(
                source_positions[src_mask], 
                target_positions[tgt_mask], 
                unit_cell
            )
            
            # Optimal matching for this element
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Map back to original indices
            for r, c in zip(row_ind, col_ind):
                orig_src_idx = src_indices[r]
                orig_tgt_idx = tgt_indices[c]
                indices[orig_src_idx] = orig_tgt_idx
                distances[orig_src_idx] = cost_matrix[r, c]
    else:
        # No elements - match all atoms together
        cost_matrix = _build_distance_matrix(source_positions, target_positions, unit_cell)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for i, j in zip(row_ind, col_ind):
            indices[i] = j
            distances[i] = cost_matrix[i, j]
        
    return indices, distances


def _build_distance_matrix(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    unit_cell
) -> np.ndarray:
    """Build a distance matrix between source and target positions."""
    n_source = len(source_positions)
    n_target = len(target_positions)
    cost_matrix = np.zeros((n_source, n_target))
    
    for i, src_pos in enumerate(source_positions):
        src_wrapped = wrap_to_unit_cell(src_pos)
        for j, tgt_pos in enumerate(target_positions):
            diff = periodic_diff(src_wrapped, tgt_pos)
            
            if hasattr(unit_cell, 'length'):
                dist = unit_cell.length(tuple(diff))
            else:
                cart_diff = diff @ unit_cell
                dist = np.linalg.norm(cart_diff)
            
            cost_matrix[i, j] = dist
    
    return cost_matrix
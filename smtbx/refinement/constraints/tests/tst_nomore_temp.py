"""
Temporary diagnostic file — NOT part of the permanent test suite.
Remove after the amplitude unit issue is resolved.

Sections:
  1. Sentinel test — verifies store() writes to all ASU scatterers
  2. Script A  — amplitude vs physical formula (sweep freq × T)
  3. Script B  — amplitude needed vs amplitude produced
  4. Script C  — candidate corrected formula
  5. Script D  — corrected formula vs physical (should give ratios ≈ 1)

Run:
  libtbx.python tst_nomore_temp.py
"""
import math
import os
import numpy as np
import scipy.constants as sc_const

import smtbx.refinement.constraints as _sc
from smtbx.refinement.constraints.nomore.phonon_data import PhononData
from smtbx.refinement.constraints.nomore.frequency_partition import (
    FrequencyPartitionStrategy, RefinementGroups,
)
from smtbx.refinement.constraints.nomore import PhononADPConstraint
from cctbx import crystal, xray, adptbx
from cctbx.array_family import flex

# ---------------------------------------------------------------------------
# Physical constants — mirrors nomore_adp.cpp exactly
# ---------------------------------------------------------------------------
_C_CMS         = 29979245800.0
_HBAR          = 1.054571817e-34
_HBAR_C        = _HBAR * _C_CMS
_KB            = 1.380649e-23
_AMU           = 1.66053906660e-27
_HBAR_C_DIV_KB = _HBAR_C / _KB                          # old (wrong) ħ·c/k_B
_H_C_DIV_KB    = 2.0 * math.pi * _HBAR_C / _KB          # correct h·c/k_B = 2π·ħ·c/k_B
_U_FACTOR      = _HBAR * 1e20 / (_AMU * 2.0 * math.pi * _C_CMS)


def _cpp_amplitude_old(freq_cm1, temperature):
    """OLD C++ amplitude formula (wrong — kept for ratio comparison in Script D)."""
    ex = math.exp(_HBAR_C_DIV_KB * freq_cm1 / temperature)
    n  = 1.0 / (ex - 1.0)
    e  = _HBAR_C * freq_cm1 * (0.5 + n)
    return e / (freq_cm1 * freq_cm1) * _U_FACTOR


def _cpp_amplitude(freq_cm1, temperature):
    """CORRECTED C++ amplitude formula: U_FACTOR*(0.5+n)/freq with x=h·c·freq/(k_B·T)."""
    x  = _H_C_DIV_KB * freq_cm1 / temperature
    ex = math.exp(x)
    n  = 1.0 / (ex - 1.0)
    return _U_FACTOR * (0.5 + n) / freq_cm1


# ---------------------------------------------------------------------------
# Helpers shared across sections
# ---------------------------------------------------------------------------

def _make_structure():
    cs = crystal.symmetry((10, 10, 10, 90, 90, 90), 'P 1')
    sc = xray.scatterer('C', site=(0.0, 0.0, 0.0), u=0.05)
    sc.flags.set_use_u_aniso(True)
    sc.flags.set_grad_u_aniso(True)
    sc.flags.set_use_u_iso(False)
    return xray.structure(cs.special_position_settings(),
                          flex.xray_scatterer([sc]))


class _AllInOneStrategy(FrequencyPartitionStrategy):
    def compute_groups(self, phonon_data, pre_groups=None):
        n = len(phonon_data.frequencies_cm1)
        return RefinementGroups(np.zeros(n, dtype=int), {0: {}})


# ============================================================================
# SECTION 1 — Sentinel test
# Requires a recompile of nomore_adp.cpp with the sentinel substitution:
#
#   // TODO Remove again, diagnose store()
#   for (int i_asu=0; i_asu < n_asu; ++i_asu)
#     for (int i_uij=0; i_uij < 6; ++i_uij)
#       u_stars_[i_asu][i_uij] = 0.01 * i_asu + 1e-3 * i_uij;
#   // TODO until here — original accumulation loop below, commented out
#
# If sentinel values appear in scatterer u_star → store() is fine.
# ============================================================================

_LALA_CIF = r'D:\datasets\LAla_nomore\L-Ala_23K_xray.cif'
_LALA_NPZ = r'D:\datasets\LAla_nomore\Lala23.npz'


def test_sentinel_store():
    """Verify store() writes to all ASU scatterers (requires sentinel C++ build)."""
    if not (os.path.exists(_LALA_CIF) and os.path.exists(_LALA_NPZ)):
        print("SKIP test_sentinel_store (data files not found)")
        return

    import iotbx.cif
    xs = list(iotbx.cif.reader(file_path=_LALA_CIF)
              .build_crystal_structures().values())[0]
    for sc in xs.scatterers():
        sc.flags.set_use_u_aniso(True)
        sc.flags.set_grad_u_aniso(True)
        sc.flags.set_use_u_iso(False)
        sc.flags.set_grad_u_iso(False)

    pd      = PhononData.load(_LALA_NPZ)
    n_modes = len(pd.frequencies_cm1)
    n_q     = len(pd.q_points)
    unit_cell = xs.unit_cell()
    n_asu   = xs.scatterers().size()

    p1 = xs.expand_to_p1(sites_mod_positive=True)
    c  = PhononADPConstraint(pd, _AllInOneStrategy())
    c._unit_cell    = unit_cell
    c._n_asu_atoms  = n_asu
    c._phonon_to_p1 = PhononADPConstraint._build_phonon_to_p1(pd, p1)
    c._p1_to_asu    = PhononADPConstraint._build_p1_to_asu(xs, p1)
    mt = c._precompute_asu_mode_tensors(pd.eigenvectors, pd.masses)

    reparam = _sc.ext.reparametrisation(unit_cell)
    sp = reparam.add(_sc.independent_scalar_parameter, value=1.0, variable=True)
    reparam.add(
        _sc.nomore_u_star,
        scatterers         = tuple(xs.scatterers()),
        scale_params       = (sp,),
        mode_tensors_ustar = flex.double(mt.tolist()),
        initial_frequencies= flex.double(c._initial_frequencies_cm1.tolist()),
        group_ids          = flex.int([0] * n_modes),
        temperature        = 23.0,
        n_modes            = n_modes,
        n_q                = n_q,
    )
    reparam.finalise()
    reparam.linearise()
    reparam.store()

    print("\n  Sentinel check (expecting u_star[i_asu][i_uij] == 0.01*i_asu + 1e-3*i_uij):")
    all_ok = True
    for i_asu, sc in enumerate(xs.scatterers()):
        for i_uij in range(6):
            expected = 0.01 * i_asu + 1e-3 * i_uij
            got      = sc.u_star[i_uij]
            ok       = abs(got - expected) < 1e-10
            if not ok:
                all_ok = False
                print(f"  FAIL atom {i_asu:2d} comp {i_uij}: expected {expected:.4f}, got {got:.6e}")

    if all_ok:
        print("  All sentinel values correct — store() writes to all scatterers.")
        print("PASS test_sentinel_store")
    else:
        print("FAIL test_sentinel_store — store() did NOT write sentinel values correctly.")


# ============================================================================
# SECTION 2 — Script A: amplitude vs physical formula
# No recompile needed.
# ============================================================================

def script_a_amplitude_vs_physical():
    """
    For eigenvec=(1,0,0), C atom, cubic 10 Å cell:
      physical U*_11 = hbar*(0.5+n)/(m_kg*omega) * 1e20 / a²
      code     U*_11 = _cpp_amplitude(freq, T) * T_ustar_11
                     where T_ustar_11 = (1/a)²/m_AMU = (0.1)²/12
    Prints ratio physical/code for a sweep of freq and T.
    """
    T_ustar_11 = 1.0 / (12.0 * 100.0)   # (1/a)² / m_AMU = 8.333e-4
    a = 10.0  # Å

    print("\n=== Script A: amplitude × T_ustar vs physical U*_11 ===")
    print(f"  {'freq':>6}  {'T':>6}  {'U*_phys':>12}  {'U*_code':>12}  {'ratio':>10}")
    for freq in [10., 100., 200., 1000., 3000.]:
        for T in [23., 300.]:
            omega   = 2 * math.pi * sc_const.c * 100.0 * freq   # rad/s (c in m/s, freq in cm⁻¹ → ×100 for m⁻¹)
            m_kg    = 12.0 * sc_const.atomic_mass
            x       = sc_const.hbar * omega / (sc_const.k * T)
            n       = 1.0 / (math.exp(x) - 1.0)
            U_cart_phys = sc_const.hbar * (0.5 + n) / (m_kg * omega) * 1e20  # Å²
            U_star_phys = U_cart_phys / (a * a)                                # dimensionless

            U_star_code = _cpp_amplitude(freq, T) * T_ustar_11
            ratio = U_star_phys / U_star_code if U_star_code != 0 else float('nan')
            print(f"  {freq:6.0f}  {T:6.0f}  {U_star_phys:12.4e}  {U_star_code:12.4e}  {ratio:10.4f}")


# ============================================================================
# SECTION 3 — Script B: amplitude needed vs amplitude produced
# ============================================================================

def script_b_amplitude_needed():
    """
    For freq=200 cm⁻¹, T=300 K:
      amplitude_needed = U*_11 / T_ustar_11
                       = hbar*(0.5+n)/(m_kg*omega)*1e20/a² / [(1/a)²/m_AMU]
                       = hbar*(0.5+n)*m_AMU/(m_kg*omega)*1e20
                       = hbar*(0.5+n)/(AMU_kg*omega)*1e20   (since m_AMU*AMU_kg = m_kg)
    Compares to _cpp_amplitude.
    """
    print("\n=== Script B: needed amplitude vs produced amplitude ===")
    for freq in [10., 200., 1000.]:
        for T in [23., 300.]:
            omega = 2 * math.pi * sc_const.c * 100.0 * freq
            x     = sc_const.hbar * omega / (sc_const.k * T)
            n     = 1.0 / (math.exp(x) - 1.0)
            # amplitude such that amplitude * T_ustar_11 = U*_11
            # U*_11 = hbar*(0.5+n)/(m_kg*omega)*1e20/a²,  T_ustar_11=(1/a²)/m_AMU
            # → amplitude = U*_11/T_ustar_11 = hbar*(0.5+n)/(m_kg*omega)*1e20*m_AMU
            #             = hbar*(0.5+n)/(AMU_kg*omega)*1e20
            amp_needed = sc_const.hbar * (0.5 + n) / (sc_const.atomic_mass * omega) * 1e20
            amp_code   = _cpp_amplitude(freq, T)
            ratio = amp_needed / amp_code if amp_code != 0 else float('nan')
            print(f"  freq={freq:6.0f}  T={T:6.0f}K  amp_needed={amp_needed:.6e}  "
                  f"amp_code={amp_code:.6e}  ratio={ratio:.6f}")


# ============================================================================
# SECTION 4 — Script C: candidate corrected formula
# ============================================================================

def _amplitude_corrected(freq_cm1, temperature):
    """
    Candidate fix with BOTH corrections:
      1. exponent uses h·c/k_B (not ħ·c/k_B)
      2. amplitude = U_FACTOR*(0.5+n)/freq
    This is the same as _cpp_amplitude after the fix.
    """
    x  = _H_C_DIV_KB * freq_cm1 / temperature
    ex = math.exp(x)
    n  = 1.0 / (ex - 1.0)
    return _U_FACTOR * (0.5 + n) / freq_cm1


def script_c_corrected_formula():
    """Compare _amplitude_corrected × T_ustar to physical U*_11."""
    T_ustar_11 = 1.0 / (12.0 * 100.0)
    a = 10.0
    print("\n=== Script C: corrected formula vs physical U*_11 ===")
    print(f"  {'freq':>6}  {'T':>6}  {'U*_phys':>12}  {'U*_corr':>12}  {'ratio':>10}")
    for freq in [10., 100., 200., 1000., 3000.]:
        for T in [23., 300.]:
            omega       = 2 * math.pi * sc_const.c * 100.0 * freq
            m_kg        = 12.0 * sc_const.atomic_mass
            x           = sc_const.hbar * omega / (sc_const.k * T)
            n           = 1.0 / (math.exp(x) - 1.0)
            U_star_phys = sc_const.hbar * (0.5 + n) / (m_kg * omega) * 1e20 / (a * a)
            U_star_corr = _amplitude_corrected(freq, T) * T_ustar_11
            ratio = U_star_phys / U_star_corr if U_star_corr != 0 else float('nan')
            print(f"  {freq:6.0f}  {T:6.0f}  {U_star_phys:12.4e}  {U_star_corr:12.4e}  {ratio:10.4f}")


# ============================================================================
# SECTION 5 — Script D: current vs corrected amplitude ratio
# Quick sanity: corrected / current should equal a simple constant factor
# ============================================================================

def script_d_ratio_current_vs_corrected():
    """Show corrected/old ratio (constant = 1/HBAR_C, confirming the spurious factor)."""
    print("\n=== Script D: corrected / OLD amplitude ratio (expect constant ≈ 3.163e23) ===")
    for freq in [10., 100., 200., 1000., 3000.]:
        for T in [23., 300.]:
            amp_old  = _cpp_amplitude_old(freq, T)
            amp_corr = _amplitude_corrected(freq, T)
            ratio = amp_corr / amp_old if amp_old != 0 else float('nan')
            print(f"  freq={freq:6.0f}  T={T:6.0f}K  ratio={ratio:.6e}")


def script_e_corrected_vs_physical():
    """
    FINAL CHECK: corrected formula vs physical U*_11.
    All ratios should be ≈ 1.0 across freq and T.
    """
    T_ustar_11 = 1.0 / (12.0 * 100.0)
    a = 10.0
    print("\n=== Script E: CORRECTED formula vs physical U*_11 (ratios should be ≈ 1.0) ===")
    print(f"  {'freq':>6}  {'T':>6}  {'U*_phys':>12}  {'U*_corr':>12}  {'ratio':>10}")
    all_close = True
    for freq in [10., 100., 200., 1000., 3000.]:
        for T in [23., 300.]:
            omega       = 2 * math.pi * sc_const.c * 100.0 * freq
            m_kg        = 12.0 * sc_const.atomic_mass
            x           = sc_const.hbar * omega / (sc_const.k * T)
            n           = 1.0 / (math.exp(x) - 1.0)
            U_star_phys = sc_const.hbar * (0.5 + n) / (m_kg * omega) * 1e20 / (a * a)
            U_star_corr = _amplitude_corrected(freq, T) * T_ustar_11
            ratio = U_star_phys / U_star_corr if U_star_corr != 0 else float('nan')
            ok    = abs(ratio - 1.0) < 0.001
            if not ok:
                all_close = False
            flag = "OK" if ok else "*** FAIL ***"
            print(f"  {freq:6.0f}  {T:6.0f}  {U_star_phys:12.4e}  {U_star_corr:12.4e}  {ratio:10.6f}  {flag}")
    if all_close:
        print("  All ratios ≈ 1.0 — corrected formula is physically correct.")


# ============================================================================
# Runner
# ============================================================================

def run():
    print("=" * 60)
    print("tst_nomore_temp.py — diagnostic run")
    print("=" * 60)

    print("\n--- Section 1: Sentinel test (needs sentinel C++ build) ---")
    test_sentinel_store()

    print("\n--- Section 2: Script A (no recompile needed) ---")
    script_a_amplitude_vs_physical()

    print("\n--- Section 3: Script B ---")
    script_b_amplitude_needed()

    print("\n--- Section 4: Script C ---")
    script_c_corrected_formula()

    print("\n--- Section 5: Script D ---")
    script_d_ratio_current_vs_corrected()

    print("\n--- Section 6: Script E (corrected formula validation) ---")
    script_e_corrected_vs_physical()

    print("\n" + "=" * 60)
    print("Diagnostic run complete.")
    print("=" * 60)


if __name__ == '__main__':
    run()

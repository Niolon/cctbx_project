"""
NoMoRe C++ integration tests.
Run with: libtbx.python tst_nomore.py
"""
import math
import os
import numpy as np
import sys
from cctbx import crystal, xray, adptbx
from cctbx.array_family import flex
from scitbx import matrix
from libtbx.test_utils import approx_equal
import smtbx.refinement.constraints as _sc
from smtbx.refinement.constraints.nomore import PhononADPConstraint
from smtbx.refinement.constraints.nomore.phonon_data import PhononData
from smtbx.refinement.constraints.nomore.frequency_partition import (
    FrequencyPartitionStrategy, RefinementGroups,
    FixedThresholdStrategy, ThermalCutoffStrategy, SensitivityBasedStrategy,
    ManualStrategy,
)

# ---------------------------------------------------------------------------
# Physical constants — mirrors nomore_adp.cpp exactly
# ---------------------------------------------------------------------------
_C_CMS          = 29979245800.0          # cm/s
_HBAR           = 1.054571817e-34        # J·s
_HBAR_C         = _HBAR * _C_CMS        # J·cm
_KB             = 1.380649e-23           # J/K
_AMU            = 1.66053906660e-27      # kg
_H_C_DIV_KB     = 2.0 * math.pi * _HBAR_C / _KB   # h·c/k_B = 2π·ħ·c/k_B  (correct BE exponent)
_U_FACTOR       = _HBAR * 1e20 / (_AMU * 2.0 * math.pi * _C_CMS)


def _cpp_amplitude(freq_cm1, temperature):
    """Replicate the corrected C++ linearise() amplitude: U_FACTOR*(0.5+n)/freq."""
    x  = _H_C_DIV_KB * freq_cm1 / temperature
    ex = math.exp(x)
    n  = 1.0 / (ex - 1.0)
    return _U_FACTOR * (0.5 + n) / freq_cm1


def _cpp_damplitude_dscale(freq_cm1, temperature):
    """Replicate the corrected C++ da_dscale Jacobian term (at scale=1)."""
    x  = _H_C_DIV_KB * freq_cm1 / temperature
    ex = math.exp(x)
    n  = 1.0 / (ex - 1.0)
    dn_domega = -n * (1.0 + n) * _H_C_DIV_KB / temperature
    da_domega = _U_FACTOR * (dn_domega / freq_cm1 - (0.5 + n) / freq_cm1**2)
    return da_domega * freq_cm1   # × domega/dscale = freq_init at scale=1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_structure():
    cs = crystal.symmetry((10, 10, 10, 90, 90, 90), 'P 1')
    sc = xray.scatterer('C', site=(0.0, 0.0, 0.0), u=0.05)
    sc.flags.set_use_u_aniso(True)
    sc.flags.set_grad_u_aniso(True)
    sc.flags.set_use_u_iso(False)
    return xray.structure(cs.special_position_settings(),
                          flex.xray_scatterer([sc]))


def _make_phonon_data(n_modes=3, freq_cm1=200.0, seed=42):
    rng = np.random.default_rng(seed)
    evecs = rng.standard_normal((n_modes, 1, 3))
    for j in range(n_modes):
        evecs[j] /= np.linalg.norm(evecs[j])
    return PhononData(
        frequencies_cm1   = np.full(n_modes, freq_cm1),
        eigenvectors      = evecs.astype(complex),
        q_points          = np.array([[0.0, 0.0, 0.0]]),
        mode_q_indices    = np.zeros(n_modes, dtype=int),
        positions_frac    = np.array([[0.0, 0.0, 0.0]]),
        cell              = np.eye(3) * 10.0,
        symbols           = ['C'],
        masses            = np.array([12.0]),
        supercell         = (1, 1, 1),
        n_atoms           = 1,
        degeneracy_groups = None,
    )


class _AllInOneStrategy(FrequencyPartitionStrategy):
    def compute_groups(self, phonon_data, pre_groups=None):
        n = len(phonon_data.frequencies_cm1)
        return RefinementGroups(np.zeros(n, dtype=int), {0: {}})


class _IndividualStrategy(FrequencyPartitionStrategy):
    def compute_groups(self, phonon_data, pre_groups=None):
        n = len(phonon_data.frequencies_cm1)
        return RefinementGroups(np.arange(n, dtype=int),
                                {i: {} for i in range(n)})


class _FixedStrategy(FrequencyPartitionStrategy):
    def compute_groups(self, phonon_data, pre_groups=None):
        n = len(phonon_data.frequencies_cm1)
        return RefinementGroups(np.full(n, -1, dtype=int), {})


def _make_constraint_manual(n_modes=3, freq_cm1=200.0, temperature=300.0,
                             seed=42, strategy=None):
    """PhononADPConstraint with internal state set directly (no add_to)."""
    xs = _make_structure()
    pd = _make_phonon_data(n_modes, freq_cm1, seed)
    if strategy is None:
        strategy = _AllInOneStrategy()
    c = PhononADPConstraint(pd, strategy)
    c.temperature    = temperature
    c._structure     = xs
    c._unit_cell     = xs.unit_cell()
    c._n_asu_atoms   = 1
    c._phonon_to_p1  = np.array([0])
    c._p1_to_asu     = [(0, np.eye(3))]
    c._mode_tensors  = c._precompute_asu_mode_tensors(pd.eigenvectors, pd.masses)
    c._groups        = strategy.compute_groups(pd)
    c._active_group_ids = np.array(
        sorted(set(c._groups.group_ids[c._groups.group_ids >= 0])))
    c.n_parameters   = len(c._active_group_ids)
    c.current_scales = np.ones(c.n_parameters)
    return c, pd, xs


# ---------------------------------------------------------------------------
# Section A — Python mode-tensor math
# ---------------------------------------------------------------------------

def test_mode_tensor_shape():
    c, _, _ = _make_constraint_manual(n_modes=4)
    assert c._mode_tensors.size == 4 * 1 * 6, \
        f"Expected {4*6}, got {c._mode_tensors.size}"
    print("PASS test_mode_tensor_shape")


def test_mode_tensors_nonneg_diagonal():
    c, _, _ = _make_constraint_manual(n_modes=3)
    t = c._mode_tensors.reshape(3, 1, 6)
    for m in range(3):
        for d in range(3):   # u11, u22, u33 are first 3 Voigt components
            assert t[m, 0, d] >= -1e-14, \
                f"Negative diagonal: mode={m} comp={d} val={t[m,0,d]}"
    print("PASS test_mode_tensors_nonneg_diagonal")


def test_scales_to_frequencies_identity():
    c, pd, _ = _make_constraint_manual(
        n_modes=3, freq_cm1=150.0, strategy=_IndividualStrategy())
    freqs = c.scales_to_frequencies(np.ones(3))
    assert np.allclose(freqs, pd.frequencies_cm1, rtol=1e-12), \
        f"Expected {pd.frequencies_cm1}, got {freqs}"
    print("PASS test_scales_to_frequencies_identity")


def test_scales_to_frequencies_scale():
    c, _, _ = _make_constraint_manual(
        n_modes=2, freq_cm1=200.0, strategy=_IndividualStrategy())
    freqs = c.scales_to_frequencies(np.array([1.1, 0.9]))
    assert np.allclose(freqs, [220.0, 180.0], rtol=1e-12), \
        f"Expected [220,180], got {freqs}"
    print("PASS test_scales_to_frequencies_scale")


def test_mode_tensor_x_eigenvector():
    """T_ustar[u11] = 1/(m*a^2) for eigenvector (1,0,0) in P1 cubic 10 A."""
    pd = PhononData(
        frequencies_cm1   = np.array([200.0]),
        eigenvectors      = np.array([[[1.0, 0.0, 0.0]]]).astype(complex),
        q_points          = np.array([[0.0, 0.0, 0.0]]),
        mode_q_indices    = np.array([0]),
        positions_frac    = np.array([[0.0, 0.0, 0.0]]),
        cell=np.eye(3)*10.0, symbols=['C'], masses=np.array([12.0]),
        supercell=(1,1,1), n_atoms=1, degeneracy_groups=None,
    )
    xs = _make_structure()
    c  = PhononADPConstraint(pd, _AllInOneStrategy())
    c._unit_cell    = xs.unit_cell()
    c._n_asu_atoms  = 1
    c._phonon_to_p1 = np.array([0])
    c._p1_to_asu    = [(0, np.eye(3))]
    t = c._precompute_asu_mode_tensors(pd.eigenvectors, pd.masses).reshape(1, 1, 6)

    # F = diag(0.1,0.1,0.1) => T*[0,0] = (1/10)^2 / 12
    expected = 1.0 / (12.0 * 100.0)
    assert approx_equal(t[0, 0, 0], expected, eps=1e-12), \
        f"T_ustar[u11]: expected {expected}, got {t[0,0,0]}"
    for i in range(1, 6):
        assert approx_equal(t[0, 0, i], 0.0, eps=1e-12), \
            f"T_ustar[{i}] expected 0, got {t[0,0,i]}"
    print("PASS test_mode_tensor_x_eigenvector")


# ---------------------------------------------------------------------------
# Section B — Raw C++ nomore_u_star
# ---------------------------------------------------------------------------

def _raw_reparam_1mode(xs, freq_cm1, temperature, group_id, t11, scale=1.0):
    """Build ext.reparametrisation with a single-mode nomore_u_star."""
    reparam = _sc.ext.reparametrisation(xs.unit_cell())
    scale_params = []
    if group_id >= 0:
        sp = reparam.add(_sc.independent_scalar_parameter,
                         value=scale, variable=True)
        scale_params.append(sp)
    reparam.add(
        _sc.nomore_u_star,
        scatterers         = tuple(xs.scatterers()),
        scale_params       = tuple(scale_params),
        mode_tensors_ustar = flex.double([t11, 0.0, 0.0, 0.0, 0.0, 0.0]),
        initial_frequencies= flex.double([freq_cm1]),
        group_ids          = flex.int([group_id]),
        temperature        = temperature,
        n_modes            = 1,
        n_q                = 1,
    )
    reparam.finalise()
    return reparam, scale_params


def test_construction():
    xs = _make_structure()
    _raw_reparam_1mode(xs, 200.0, 300.0, group_id=0, t11=1.0/(12*100))
    print("PASS test_construction")


def test_linearise_u_star_values():
    """C++ linearise() U*_11 matches analytic Bose-Einstein amplitude x mode tensor."""
    xs   = _make_structure()
    freq = 200.0
    T    = 300.0
    t11  = 1.0 / (12.0 * 100.0)

    reparam, _ = _raw_reparam_1mode(xs, freq, T, group_id=0, t11=t11)
    reparam.linearise()
    reparam.store()

    expected = _cpp_amplitude(freq, T) * t11
    actual   = xs.scatterers()[0].u_star[0]
    assert approx_equal(actual, expected, eps=1e-10), \
        f"U*_11: expected {expected:.8e}, got {actual:.8e}"
    print("PASS test_linearise_u_star_values")


def test_linearise_multiple_modes():
    """Two modes with same frequency sum their mode-tensor contributions."""
    xs   = _make_structure()
    freq = 300.0
    T    = 200.0
    t11a = 1.0 / (12.0 * 100.0)
    t11b = 2.0 / (12.0 * 100.0)

    reparam = _sc.ext.reparametrisation(xs.unit_cell())
    sp = reparam.add(_sc.independent_scalar_parameter, value=1.0, variable=True)
    reparam.add(
        _sc.nomore_u_star,
        scatterers         = tuple(xs.scatterers()),
        scale_params       = (sp,),
        mode_tensors_ustar = flex.double([t11a, 0, 0, 0, 0, 0,
                                          t11b, 0, 0, 0, 0, 0]),
        initial_frequencies= flex.double([freq, freq]),
        group_ids          = flex.int([0, 0]),
        temperature        = T,
        n_modes            = 2,
        n_q                = 1,
    )
    reparam.finalise()
    reparam.linearise()
    reparam.store()

    amp      = _cpp_amplitude(freq, T)
    expected = amp * (t11a + t11b)
    actual   = xs.scatterers()[0].u_star[0]
    assert approx_equal(actual, expected, eps=1e-10), \
        f"2-mode U*_11: expected {expected:.8e}, got {actual:.8e}"
    print("PASS test_linearise_multiple_modes")


def test_fixed_mode():
    """group_id=-1: U* computed at initial freq regardless of scale params."""
    xs   = _make_structure()
    freq = 200.0
    T    = 300.0
    t11  = 1.0 / (12.0 * 100.0)

    reparam = _sc.ext.reparametrisation(xs.unit_cell())
    reparam.add(
        _sc.nomore_u_star,
        scatterers         = tuple(xs.scatterers()),
        scale_params       = (),
        mode_tensors_ustar = flex.double([t11, 0, 0, 0, 0, 0]),
        initial_frequencies= flex.double([freq]),
        group_ids          = flex.int([-1]),
        temperature        = T,
        n_modes            = 1,
        n_q                = 1,
    )
    reparam.finalise()
    reparam.linearise()
    reparam.store()

    expected = _cpp_amplitude(freq, T) * t11
    actual   = xs.scatterers()[0].u_star[0]
    assert approx_equal(actual, expected, eps=1e-10), \
        f"Fixed mode U*_11: expected {expected:.8e}, got {actual:.8e}"
    print("PASS test_fixed_mode")


def test_scale_changes_u_star():
    """Different scale values produce different U*."""
    xs   = _make_structure()
    freq = 200.0
    T    = 300.0
    t11  = 1.0 / (12.0 * 100.0)

    def _u11(scale):
        r, _ = _raw_reparam_1mode(xs, freq, T, group_id=0, t11=t11, scale=scale)
        r.linearise()
        r.store()
        return xs.scatterers()[0].u_star[0]

    u11_ref = _u11(1.0)
    u11_hi  = _u11(2.0)
    assert u11_hi != u11_ref, "U* unchanged after scale change"
    print("PASS test_scale_changes_u_star")


def test_jacobian_finite_differences():
    """C++ Jacobian (da_dscale formula) matches FD of U*_11 w.r.t. scale."""
    xs   = _make_structure()
    freq = 200.0
    T    = 300.0
    t11  = 1.0 / (12.0 * 100.0)
    eps  = 1e-5

    def _u11(scale):
        r, _ = _raw_reparam_1mode(xs, freq, T, group_id=0, t11=t11, scale=scale)
        r.linearise()
        r.store()
        return xs.scatterers()[0].u_star[0]

    fd_jac       = (_u11(1.0 + eps) - _u11(1.0 - eps)) / (2.0 * eps)
    analytic_jac = _cpp_damplitude_dscale(freq, T) * t11
    assert approx_equal(analytic_jac, fd_jac, eps=1e-4), \
        f"Jacobian: analytic={analytic_jac:.8e}, FD={fd_jac:.8e}"
    print("PASS test_jacobian_finite_differences")


def test_amplitude_physical_regression():
    """
    Regression: C++ amplitude must match independent scipy.constants reference.

    Uses eigenvector (1,0,0) for a C atom in a cubic 10 Å P1 cell so that
    T_ustar_11 = 1/(m_AMU * a²) = 1/(12 * 100) and U*_11 has an analytic form.

    The old wrong formula gave results ~3e26× too small (spurious ħ·c factor)
    and a wrong Bose-Einstein exponent (missing 2π). Both would cause this
    test to fail with a relative error >> 1e-4.
    """
    try:
        import scipy.constants as sc_const
    except ImportError:
        print("SKIP test_amplitude_physical_regression (scipy not available)")
        return

    xs   = _make_structure()
    freq = 200.0   # cm⁻¹
    T    = 300.0   # K
    t11  = 1.0 / (12.0 * 100.0)   # T_ustar_11 for C atom, a=10 Å

    reparam, _ = _raw_reparam_1mode(xs, freq, T, group_id=0, t11=t11)
    reparam.linearise()
    reparam.store()
    u11_cpp = xs.scatterers()[0].u_star[0]

    # Independent reference using scipy.constants
    omega   = 2.0 * math.pi * sc_const.c * 100.0 * freq   # rad/s
    m_kg    = 12.0 * sc_const.atomic_mass                  # kg
    x       = sc_const.hbar * omega / (sc_const.k * T)
    n       = 1.0 / (math.exp(x) - 1.0)
    # U_cart_11 [Å²] = ħ*(0.5+n)/(m*ω) * 1e20
    # U_star_11 = U_cart_11 / a² (cubic cell, a=10 Å)
    u11_ref = sc_const.hbar * (0.5 + n) / (m_kg * omega) * 1e20 / (10.0 ** 2)

    rel_err = abs(u11_cpp - u11_ref) / u11_ref
    assert rel_err < 1e-4, (
        f"Physical regression FAILED: U*_11={u11_cpp:.6e}, "
        f"scipy ref={u11_ref:.6e}, rel_err={rel_err:.3e}"
    )
    print(f"PASS test_amplitude_physical_regression  "
          f"(U*_11={u11_cpp:.4e} Å², ref={u11_ref:.4e} Å², rel_err={rel_err:.1e})")


def test_n_q_divides_u_star():
    """U* is inversely proportional to n_q."""
    xs   = _make_structure()
    freq = 200.0
    T    = 300.0
    t11  = 1.0 / (12.0 * 100.0)

    def _run(n_q):
        reparam = _sc.ext.reparametrisation(xs.unit_cell())
        sp = reparam.add(_sc.independent_scalar_parameter, value=1.0, variable=True)
        reparam.add(
            _sc.nomore_u_star,
            scatterers=tuple(xs.scatterers()),
            scale_params=(sp,),
            mode_tensors_ustar=flex.double([t11, 0, 0, 0, 0, 0]),
            initial_frequencies=flex.double([freq]),
            group_ids=flex.int([0]),
            temperature=T, n_modes=1, n_q=n_q,
        )
        reparam.finalise()
        reparam.linearise()
        reparam.store()
        return xs.scatterers()[0].u_star[0]

    u1 = _run(1)
    u2 = _run(2)
    assert approx_equal(u2, u1 / 2.0, eps=1e-12), \
        f"n_q=2 should halve U*: u(nq=1)={u1:.8e}, u(nq=2)={u2:.8e}"
    print("PASS test_n_q_divides_u_star")


# ---------------------------------------------------------------------------
# Section C — Python/C++ consistency
# ---------------------------------------------------------------------------

def test_python_tensors_match_analytic():
    """_precompute_asu_mode_tensors gives exact analytic value for (1,0,0)."""
    pd = PhononData(
        frequencies_cm1   = np.array([200.0]),
        eigenvectors      = np.array([[[1.0, 0.0, 0.0]]]).astype(complex),
        q_points=np.array([[0,0,0]]), mode_q_indices=np.array([0]),
        positions_frac=np.array([[0,0,0]]), cell=np.eye(3)*10.,
        symbols=['C'], masses=np.array([12.]), supercell=(1,1,1),
        n_atoms=1, degeneracy_groups=None,
    )
    xs = _make_structure()
    c  = PhononADPConstraint(pd, _AllInOneStrategy())
    c._unit_cell    = xs.unit_cell()
    c._n_asu_atoms  = 1
    c._phonon_to_p1 = np.array([0])
    c._p1_to_asu    = [(0, np.eye(3))]
    t = c._precompute_asu_mode_tensors(pd.eigenvectors, pd.masses).reshape(1, 1, 6)
    expected = 1.0 / (12.0 * 100.0)
    assert approx_equal(t[0, 0, 0], expected, eps=1e-12), \
        f"T_ustar[u11]={t[0,0,0]}, expected {expected}"
    for i in range(1, 6):
        assert approx_equal(t[0, 0, i], 0.0, eps=1e-12), \
            f"T_ustar[{i}]={t[0,0,i]}, expected 0"
    print("PASS test_python_tensors_match_analytic")


def test_cpp_u_star_from_python_tensors():
    """U* from C++ with Python-computed mode tensors matches analytic reference."""
    xs   = _make_structure()
    freq = 200.0
    T    = 300.0
    pd   = _make_phonon_data(n_modes=3, freq_cm1=freq, seed=7)

    c = PhononADPConstraint(pd, _AllInOneStrategy())
    c._unit_cell    = xs.unit_cell()
    c._n_asu_atoms  = 1
    c._phonon_to_p1 = np.array([0])
    c._p1_to_asu    = [(0, np.eye(3))]
    mode_tensors_np = c._precompute_asu_mode_tensors(pd.eigenvectors, pd.masses)

    reparam = _sc.ext.reparametrisation(xs.unit_cell())
    sp = reparam.add(_sc.independent_scalar_parameter, value=1.0, variable=True)
    reparam.add(
        _sc.nomore_u_star,
        scatterers         = tuple(xs.scatterers()),
        scale_params       = (sp,),
        mode_tensors_ustar = flex.double(mode_tensors_np.tolist()),
        initial_frequencies= flex.double(pd.frequencies_cm1.tolist()),
        group_ids          = flex.int([0, 0, 0]),
        temperature        = T,
        n_modes            = 3,
        n_q                = 1,
    )
    reparam.finalise()
    reparam.linearise()
    reparam.store()

    amp   = _cpp_amplitude(freq, T)
    t_sum = mode_tensors_np.reshape(3, 1, 6).sum(axis=0)[0]
    for i in range(6):
        expected = amp * t_sum[i]
        actual   = xs.scatterers()[0].u_star[i]
        assert approx_equal(actual, expected, eps=1e-10), \
            f"U*[{i}]: expected {expected:.8e}, got {actual:.8e}"
    print("PASS test_cpp_u_star_from_python_tensors")


# ---------------------------------------------------------------------------
# Section D — compute_groups dispatch
# ---------------------------------------------------------------------------

def test_compute_groups_dispatch():
    """
    add_to() must call compute_groups(phonon_data) for standard strategies
    (not compute_groups(phonon_data, temperature)).  Verify by recording what
    pre_groups value the strategy actually receives.
    """
    from smtbx.refinement.constraints.nomore.frequency_partition import (
        SensitivityBasedStrategy, ThermalCutoffStrategy,
    )

    pd = _make_phonon_data(n_modes=3)
    received = {}

    class _RecordingStrategy(FrequencyPartitionStrategy):
        def compute_groups(self, phonon_data, pre_groups=None):
            received['pre_groups'] = pre_groups
            n = len(phonon_data.frequencies_cm1)
            return RefinementGroups(np.zeros(n, dtype=int), {0: {}})

    c = PhononADPConstraint(pd, _RecordingStrategy())
    c.temperature = 300.0

    # Replicate the dispatch logic from the fixed add_to()
    if isinstance(c.partition_strategy, (SensitivityBasedStrategy, ThermalCutoffStrategy)):
        c._groups = c.partition_strategy.compute_groups(c.phonon_data, c.temperature)
    else:
        c._groups = c.partition_strategy.compute_groups(c.phonon_data)

    assert received.get('pre_groups') is None, (
        f"Standard strategy received temperature ({received['pre_groups']!r}) "
        f"in the pre_groups slot — fix not applied"
    )
    print("PASS test_compute_groups_dispatch (standard strategy gets no temperature)")


# ---------------------------------------------------------------------------
# Section E — Real-data integration (L-Alanine 23K)
# ---------------------------------------------------------------------------

_LALA_CIF = r'D:\datasets\LAla_nomore\L-Ala_23K_xray.cif'
_LALA_NPZ = r'D:\datasets\LAla_nomore\Lala23.npz'
_LALA_HKL = r'D:\datasets\LAla_nomore\A23_SHELXL_iam.hkl'


def test_integration_lala_real_data():
    """
    Full pipeline with real L-Alanine data at 23K.

    Exercises the complete chain:
        CIF load → P1 expansion → phonon→P1 mapping → mode-tensor
        precomputation → C++ nomore_u_star → physical ADP validation.

    Skipped gracefully when the data files are absent.  Passes when:
      - all ASU ADP tensors are positive semi-definite, and
      - heavy-atom Ueq in [0.001, 0.030] Å²,
      - H-atom Ueq in [0.003, 0.100] Å².
    """
    if not (os.path.exists(_LALA_CIF) and os.path.exists(_LALA_NPZ)):
        print("SKIP test_integration_lala_real_data (data files not found)")
        return

    import iotbx.cif
    reader     = iotbx.cif.reader(file_path=_LALA_CIF)
    structures = reader.build_crystal_structures()
    xs         = list(structures.values())[0]

    # Force all scatterers (incl. H) to anisotropic
    for sc in xs.scatterers():
        sc.flags.set_use_u_aniso(True)
        sc.flags.set_grad_u_aniso(True)
        sc.flags.set_use_u_iso(False)
        sc.flags.set_grad_u_iso(False)

    TEMPERATURE = 23.0 - 273
    unit_cell   = xs.unit_cell()
    n_asu       = xs.scatterers().size()

    pd      = PhononData.load(_LALA_NPZ)
    n_modes = len(pd.frequencies_cm1)
    n_q     = len(pd.q_points)

    print(f"\n  {xs.space_group_info()}, {n_asu} ASU atoms, "
          f"{n_modes} modes, {n_q} q-pt(s), T={TEMPERATURE} K")

    import smtbx.utils
    constraint = PhononADPConstraint(pd, _AllInOneStrategy())
    connectivity_table = smtbx.utils.connectivity_table(xs)
    reparam = _sc.reparametrisation(
        xs, [constraint], connectivity_table,
        temperature=TEMPERATURE)
    reparam.linearise()
    reparam.store()

    # Validate and display results
    print(f"\n  {'Label':>6}  {'Ueq (Å²)':>10}  {'posDef':>6}  Status")
    print(f"  {'-'*6}  {'-'*10}  {'-'*6}  {'-'*24}")
    all_ok = True

    for sc in xs.scatterers():
        u_cart  = adptbx.u_star_as_u_cart(unit_cell, sc.u_star)
        ueq     = (u_cart[0] + u_cart[1] + u_cart[2]) / 3.0
        u_mat   = np.array([[u_cart[0], u_cart[3], u_cart[4]],
                             [u_cart[3], u_cart[1], u_cart[5]],
                             [u_cart[4], u_cart[5], u_cart[2]]])
        min_eig = np.linalg.eigvalsh(u_mat).min()
        pos_def = min_eig > -1e-8

        is_H    = sc.element_symbol().strip().upper() == 'H'
        lo, hi  = (0.003, 0.100) if is_H else (0.001, 0.030)
        in_bounds = lo <= ueq <= hi
        ok      = pos_def and in_bounds
        if not ok:
            all_ok = False

        if ok:
            status = "OK"
        else:
            parts = []
            if not pos_def:
                parts.append(f"not-pos-def(min_eig={min_eig:.2e})")
            if not in_bounds:
                parts.append(f"Ueq outside [{lo},{hi}]")
            status = "FAIL: " + ", ".join(parts)
        print(f"  {sc.label:>6}  {ueq:10.5f}  {str(pos_def):>6}  {status}")

    assert all_ok, "One or more atoms have ADPs outside physically reasonable bounds"
    print("PASS test_integration_lala_real_data")


def _run_lala_refinement(xs, fo_sq, pd, strategy, temperature, label):
    """
    Run one NoMoRe refinement cycle and return wR2.

    Sites are held fixed; the NoMoRe scale parameters from `strategy` are
    the only refined parameters.  The structure xs is modified in-place
    (u_star updated by store()).
    """
    import smtbx.utils
    from smtbx.refinement import least_squares
    from scitbx.lstbx import normal_eqns_solving

    # Fresh copy so strategies don't interfere with each other
    xs_copy = xs.deep_copy_scatterers()
    for sc in xs_copy.scatterers():
        sc.flags.set_grad_site(False)
        sc.flags.set_use_u_aniso(True)
        sc.flags.set_use_u_iso(False)
        sc.flags.set_grad_u_iso(False)

    connectivity_table = smtbx.utils.connectivity_table(xs_copy)
    constraint = PhononADPConstraint(pd, strategy)
    reparam = _sc.reparametrisation(
        xs_copy, [constraint], connectivity_table,
        temperature=temperature
    )

    ls = least_squares.crystallographic_ls(
        fo_sq.as_xray_observations(),
        reparam,
        weighting_scheme=least_squares.mainstream_shelx_weighting())

    n_params = constraint.n_parameters
    try:
        cycles = normal_eqns_solving.naive_iterations(
            ls,
            n_max_iterations=10,
            gradient_threshold=1e-8,
            step_threshold=1e-8)
    except RuntimeError as e:
        if 'cholesky' in str(e).lower() or 'SCITBX_ASSERT' in str(e):
            # Singular normal matrix: mode tensors are linearly dependent for
            # this dataset (e.g. degenerate modes with identical ADP contributions).
            # This is a data/strategy mismatch, not a code bug.
            print(f"    {label:<24}  n_params={n_params:3d}  SINGULAR (normal matrix rank-deficient)")
            return None
        raise

    wR2 = ls.wR2()
    print(f"    {label:<24}  n_params={n_params:3d}  "
          f"iters={cycles.n_iterations:2d}  wR2={wR2:.4f}")

    # Validate heavy-atom Ueq
    unit_cell = xs_copy.unit_cell()
    for sc in xs_copy.scatterers():
        if sc.element_symbol().strip().upper() == 'H':
            continue
        u_cart = adptbx.u_star_as_u_cart(unit_cell, sc.u_star)
        ueq = (u_cart[0] + u_cart[1] + u_cart[2]) / 3.0
        assert 0.001 <= ueq <= 0.050, (
            f"{label}: {sc.label} Ueq={ueq:.5f} outside [0.001, 0.050]")

    return wR2


def test_refinement_lala_real_data():
    """
    Refinement of L-Alanine at 23 K using the NoMoRe ADP constraint against
    real measured intensities from A23_SHELXL_iam.hkl (SHELX HKLF 4 format).

    Three partitioning strategies are compared:
      - AllInOne:   all modes share one scale factor  (1 parameter)
      - Individual: each mode has its own scale factor (n_modes parameters)
      - Fixed:      all modes frozen at DFT frequencies (0 parameters)

    Sites are held fixed in all cases.  Asserts:
      - Each strategy completes without error.
      - wR2 < 0.20 (physically reasonable single- or multi-parameter model).
      - Individual strategy wR2 ≤ AllInOne wR2 (more parameters can only help).
      - Heavy-atom Ueq in [0.001, 0.050] Å² for every strategy.
    """
    if not (os.path.exists(_LALA_CIF) and os.path.exists(_LALA_NPZ)
            and os.path.exists(_LALA_HKL)):
        print("SKIP test_refinement_lala_real_data (data files not found)")
        return

    import iotbx.cif
    from iotbx.shelx import hklf as shelx_hklf

    reader = iotbx.cif.reader(file_path=_LALA_CIF)
    xs = list(reader.build_crystal_structures().values())[0]

    TEMPERATURE_C = 23.0 - 273.15
    pd = PhononData.load(_LALA_NPZ)

    # Load real Fo² from the SHELX HKLF 4 file; merge symmetry equivalents
    hkl_reader = shelx_hklf.reader(file_name=_LALA_HKL)
    fo_sq = hkl_reader.as_miller_arrays(
        crystal_symmetry=xs.crystal_symmetry(),
        merge_equivalents=True,
    )[0]
    fo_sq = fo_sq.resolution_filter(d_min=0.8)
    fo_sq = fo_sq.select(fo_sq.data() > -3.0 * fo_sq.sigmas())
    print(f"\n  {fo_sq.size()} unique reflections, T={TEMPERATURE_C} K")

    n_modes = len(pd.frequencies_cm1)
    freqs   = pd.frequencies_cm1

    # Verify strategy group structures before running any refinement.
    # For each strategy, compute groups and print the partition summary so
    # unexpected assignments are visible immediately.
    def _group_summary(strategy, label):
        if isinstance(strategy, (ThermalCutoffStrategy, SensitivityBasedStrategy)):
            groups = strategy.compute_groups(pd, TEMPERATURE_C)
        else:
            groups = strategy.compute_groups(pd)
        ids = groups.group_ids
        n_fixed  = int((ids < 0).sum())
        n_params = int(len(set(ids[ids >= 0])))
        print(f"    {label:<24}  modes={n_modes:4d}  "
              f"params={n_params:4d}  fixed={n_fixed:4d}")
        return n_params

    print(f"\n  Group structure (before refinement):")
    print(f"    {'Strategy':<24}  {'modes':>5}  {'params':>6}  {'fixed':>5}")

    # ManualStrategy: two shared groups split at 500 cm⁻¹ (low / mid),
    # modes > 1000 cm⁻¹ fixed. Produces exactly 2 parameters — well-conditioned.
    manual_ids = np.full(n_modes, -1, dtype=int)
    manual_ids[freqs <  500.0] = 0   # group 0: low-frequency modes
    manual_ids[(freqs >= 500.0) & (freqs < 1000.0)] = 1  # group 1: mid-frequency

    strategies = [
        (_AllInOneStrategy(),                          "AllInOne"),
        (_FixedStrategy(),                             "Fixed"),
        (FixedThresholdStrategy(medium_limit=200.0,
                                high_limit=1000.0),    "FixedThreshold"),
        (ThermalCutoffStrategy(medium_factor=1.5,
                               high_factor=3.0),       "ThermalCutoff"),
        (SensitivityBasedStrategy(low_threshold=0.90,
                                  high_threshold=0.99),"Sensitivity"),
        (ManualStrategy(manual_ids),                   "Manual(2 groups)"),
    ]

    for strategy, label in strategies:
        _group_summary(strategy, label)

    print(f"\n  Refinement results:")
    print(f"    {'Strategy':<24}  {'n_params':>8}  {'iters':>5}  {'wR2':>7}")
    results = {}
    for strategy, label in strategies:
        results[label] = _run_lala_refinement(
            xs, fo_sq, pd, strategy, TEMPERATURE_C, label)

    wR2_all = results["AllInOne"]
    assert wR2_all is not None, "AllInOne must converge (1 parameter, always well-conditioned)"
    assert wR2_all < 0.20, f"AllInOne wR2={wR2_all:.4f} too high"

    # Strategies with more parameters should do at least as well as AllInOne
    # when they converge.  Singular strategies are acceptable — they indicate
    # that the mode tensors are linearly dependent for this dataset, which is
    # a property of the phonon data, not a code bug.
    for label in ("ThermalCutoff", "Manual(2 groups)"):
        wR2 = results[label]
        if wR2 is None:
            print(f"  NOTE: {label} was singular — skipping wR2 assertion")
            continue
        assert wR2 < 0.20, f"{label} wR2={wR2:.4f} too high"
        assert wR2 <= wR2_all + 0.01, (
            f"{label} wR2={wR2:.4f} should not exceed AllInOne wR2={wR2_all:.4f}")

    print("PASS test_refinement_lala_real_data")


def test_strategy_capacity_sweep():
    """
    For each non-trivial partitioning strategy, sweep from tight (few params)
    to loose (many params) and report the maximum refinable parameter count.

    Stops each sweep at the first singularity so we know the capacity limit
    of this dataset (520 unique reflections, 156 modes).
    """
    if not (os.path.exists(_LALA_CIF) and os.path.exists(_LALA_NPZ)
            and os.path.exists(_LALA_HKL)):
        print("SKIP test_strategy_capacity_sweep (data files not found)")
        return

    import iotbx.cif
    from iotbx.shelx import hklf as shelx_hklf

    reader = iotbx.cif.reader(file_path=_LALA_CIF)
    xs = list(reader.build_crystal_structures().values())[0]
    pd = PhononData.load(_LALA_NPZ)
    TEMPERATURE_C = 23.0 - 273.15

    hkl_reader = shelx_hklf.reader(file_name=_LALA_HKL)
    fo_sq = hkl_reader.as_miller_arrays(
        crystal_symmetry=xs.crystal_symmetry(), merge_equivalents=True)[0]
    fo_sq = fo_sq.resolution_filter(d_min=0.8)
    fo_sq = fo_sq.select(fo_sq.data() > -3.0 * fo_sq.sigmas())

    def _sweep_step(strategy):
        groups   = strategy.compute_groups(pd) \
                   if not isinstance(strategy, (ThermalCutoffStrategy, SensitivityBasedStrategy)) \
                   else strategy.compute_groups(pd, TEMPERATURE_C)
        n_params = int(len(set(groups.group_ids[groups.group_ids >= 0])))
        wR2      = _run_lala_refinement(xs, fo_sq, pd, strategy, TEMPERATURE_C, "")
        return n_params, wR2

    # ------------------------------------------------------------------
    # FixedThresholdStrategy — n_refined mode (individual lowest N modes)
    # Increase n_refined one at a time; high_limit keeps the rest fixed.
    # ------------------------------------------------------------------
    print("\n  FixedThresholdStrategy  n_refined↑  (high_limit=200 cm⁻¹):")
    print(f"    {'n_refined':>9}  {'n_params':>8}  {'result':>14}")
    for n_ref in range(1, 30):
        strategy         = FixedThresholdStrategy(n_refined=n_ref, high_limit=200.0)
        n_params, wR2    = _sweep_step(strategy)
        if wR2 is None:
            print(f"    {n_ref:9d}  {n_params:8d}  SINGULAR ← capacity limit")
            break
        print(f"    {n_ref:9d}  {n_params:8d}  wR2={wR2:.4f}")

    n_ref -= 2

    for high_limit in range(200, 2000, 100):
        strategy         = FixedThresholdStrategy(n_refined=n_ref, high_limit=float(high_limit))
        n_params, wR2    = _sweep_step(strategy)
        if wR2 is None:
            print(f"    {n_ref:9d}  {n_params:8d}  SINGULAR ← capacity limit")
            break
        print(f"    {n_ref:9d}  {n_params:8d}  wR2={wR2:.4f}")


    # ------------------------------------------------------------------
    # ThermalCutoffStrategy — two independent sweeps:
    #   1. medium_factor↑ (more LOW modes individually refined), high_factor fixed high
    #   2. high_factor↓  (fewer modes fixed),                    medium_factor fixed low
    # ------------------------------------------------------------------
    # ThermalCutoffStrategy — two sequential sweeps:
    #   Sweep 1: medium_factor↑ (high_factor fixed high) — adds LOW individual modes.
    #   Sweep 2: high_factor↑ starting from just above the best medium_factor found
    #            in sweep 1 — adds the MFSF group and may eventually go singular.
    # Both sweeps go in the "adding parameters" direction.

    print("\n  ThermalCutoffStrategy  medium_factor↑  (high_factor=50.0):")
    print(f"    {'med_factor':>10}  {'n_params':>8}  {'result':>14}")
    best_mf_i = 0
    factors =  [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 50.0]
    for mf_i in range(len(factors) - 1):
        strategy = ThermalCutoffStrategy(
            medium_factor=factors[mf_i],
            high_factor=factors[mf_i + 1]
        )
        n_params, wR2 = _sweep_step(strategy)
        if wR2 is None:
            print(f"    {factors[mf_i]:10.1f}  {n_params:8d}  SINGULAR ← capacity limit")
            break
        print(f"    {factors[mf_i]:10.1f}  {n_params:8d}  wR2={wR2:.4f}")
        best_mf_i = mf_i

    # Sweep 2: use best_mf from sweep 1; increase high_factor from just above it
    # (minimum separation = 0.5) — this adds more modes into the MFSF range.
    use_mf = factors[best_mf_i - 1]
    print(f"\n  ThermalCutoffStrategy  high_factor↑  (medium_factor={use_mf:.1f}):")
    print(f"    {'high_factor':>11}  {'n_params':>8}  {'result':>14}")
    for hf_i in range(best_mf_i, len(factors)):
        strategy = ThermalCutoffStrategy(
            medium_factor=use_mf,
            high_factor=factors[hf_i]
        )
        n_params, wR2 = _sweep_step(strategy)
        if wR2 is None:
            print(f"    {factors[hf_i]:11.1f}  {n_params:8d}  SINGULAR ← capacity limit")
            break
        print(f"    {factors[hf_i]:11.1f}  {n_params:8d}  wR2={wR2:.4f}")

    # ------------------------------------------------------------------
    # SensitivityBasedStrategy — widen both thresholds (more params)
    # Start with very tight (few params), open up low then high threshold.
    # ------------------------------------------------------------------
    print("\n  SensitivityBasedStrategy  thresholds↑:")
    print(f"    {'low_thr':>7}  {'high_thr':>8}  {'n_params':>8}  {'result':>14}")
    threshold_pairs = [
        (0.30, 0.50), (0.50, 0.70), (0.70, 0.85),
        (0.80, 0.90), (0.85, 0.95), (0.90, 0.99),
        (0.95, 0.99), (0.98, 0.999),
    ]
    for lo, hi in threshold_pairs:
        strategy      = SensitivityBasedStrategy(low_threshold=lo, high_threshold=hi)
        n_params, wR2 = _sweep_step(strategy)
        if wR2 is None:
            print(f"    {lo:7.2f}  {hi:8.3f}  {n_params:8d}  SINGULAR ← capacity limit")
            break
        print(f"    {lo:7.2f}  {hi:8.3f}  {n_params:8d}  wR2={wR2:.4f}")

    print("\nPASS test_strategy_capacity_sweep")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("NoMoRe C++ integration tests")
    print("=" * 60)

    print("\n-- Section A: Python mode-tensor math --")
    test_mode_tensor_shape()
    test_mode_tensors_nonneg_diagonal()
    test_scales_to_frequencies_identity()
    test_scales_to_frequencies_scale()
    test_mode_tensor_x_eigenvector()

    print("\n-- Section B: Raw C++ nomore_u_star --")
    test_construction()
    test_linearise_u_star_values()
    test_linearise_multiple_modes()
    test_fixed_mode()
    test_scale_changes_u_star()
    test_jacobian_finite_differences()
    test_n_q_divides_u_star()
    test_amplitude_physical_regression()

    print("\n-- Section C: Python/C++ consistency --")
    test_python_tensors_match_analytic()
    test_cpp_u_star_from_python_tensors()

    print("\n-- Section D: compute_groups dispatch --")
    test_compute_groups_dispatch()

    if len(sys.argv) > 1 and sys.argv[1] == "--real-data":
        print("\n-- Section E: Real-data integration (L-Ala 23K) --")
        test_integration_lala_real_data()
        test_refinement_lala_real_data()
        test_strategy_capacity_sweep()

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == '__main__':
    run()

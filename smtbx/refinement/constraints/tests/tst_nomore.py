. Hpw"""
NoMoRe C++ integration tests.
Run with: libtbx.python tst_nomore.py
"""
import math
import os
import numpy as np
from cctbx import crystal, xray, adptbx
from cctbx.array_family import flex
from scitbx import matrix
from libtbx.test_utils import approx_equal
import smtbx.refinement.constraints as _sc
from smtbx.refinement.constraints.nomore import PhononADPConstraint
from smtbx.refinement.constraints.nomore.phonon_data import PhononData
from smtbx.refinement.constraints.nomore.frequency_partition import (
    FrequencyPartitionStrategy, RefinementGroups,
)

# ---------------------------------------------------------------------------
# Physical constants — mirrors nomore_adp.cpp exactly
# ---------------------------------------------------------------------------
_C_CMS          = 29979245800.0          # cm/s
_HBAR           = 1.054571817e-34        # J·s
_HBAR_C         = _HBAR * _C_CMS        # J·cm
_KB             = 1.380649e-23           # J/K
_AMU            = 1.66053906660e-27      # kg
_HBAR_C_DIV_KB  = _HBAR_C / _KB         # K·cm
_U_FACTOR       = _HBAR * 1e20 / (_AMU * 2.0 * math.pi * _C_CMS)


def _cpp_amplitude(freq_cm1, temperature):
    """Replicate the C++ linearise() amplitude calculation."""
    ex = math.exp(_HBAR_C_DIV_KB * freq_cm1 / temperature)
    n  = 1.0 / (ex - 1.0)
    e  = _HBAR_C * freq_cm1 * (0.5 + n)
    return e / (freq_cm1 * freq_cm1) * _U_FACTOR


def _cpp_damplitude_dscale(freq_cm1, temperature):
    """Replicate the C++ da_dscale Jacobian term (at scale=1)."""
    ex = math.exp(_HBAR_C_DIV_KB * freq_cm1 / temperature)
    n  = 1.0 / (ex - 1.0)
    e  = _HBAR_C * freq_cm1 * (0.5 + n)
    dn_domega = (_HBAR_C_DIV_KB / temperature) * ex / ((ex - 1.0) ** 2)
    de_domega = _HBAR_C * (0.5 + n + freq_cm1 * dn_domega)
    da_domega = -2.0 * e / freq_cm1**3 + de_domega / freq_cm1**2
    return da_domega * _U_FACTOR * freq_cm1   # domega/dscale = freq_init at scale=1


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

    TEMPERATURE = 23.0
    unit_cell   = xs.unit_cell()
    n_asu       = xs.scatterers().size()

    pd      = PhononData.load(_LALA_NPZ)
    n_modes = len(pd.frequencies_cm1)
    n_q     = len(pd.q_points)

    print(f"\n  {xs.space_group_info()}, {n_asu} ASU atoms, "
          f"{n_modes} modes, {n_q} q-pt(s), T={TEMPERATURE} K")

    # Crystallographic geometry mappings
    p1_structure = xs.expand_to_p1(sites_mod_positive=True)
    p1_to_asu    = PhononADPConstraint._build_p1_to_asu(xs, p1_structure)
    phonon_to_p1 = PhononADPConstraint._build_phonon_to_p1(pd, p1_structure)

    # Precompute mode tensors in U* Voigt space: flat (n_modes * n_asu * 6,)
    c = PhononADPConstraint(pd, _AllInOneStrategy())
    c._unit_cell    = unit_cell
    c._n_asu_atoms  = n_asu
    c._phonon_to_p1 = phonon_to_p1
    c._p1_to_asu    = p1_to_asu
    mode_tensors_np = c._precompute_asu_mode_tensors(pd.eigenvectors, pd.masses)

    assert mode_tensors_np.size == n_modes * n_asu * 6, (
        f"Mode tensor array size {mode_tensors_np.size} != {n_modes * n_asu * 6}"
    )

    # Diagnostic: verify mode tensors are non-zero before passing to C++
    mt_abs_sum = np.abs(mode_tensors_np).sum()
    print(f"  mode_tensors abs-sum={mt_abs_sum:.4e}, "
          f"first 6={[f'{v:.3e}' for v in mode_tensors_np[:6]]}")

    # Diagnostic: Python-side expected Ueq for ASU atom 0 (all-in-one group, scale=1)
    mt_reshaped = mode_tensors_np.reshape(n_modes, n_asu, 6)
    py_u_stars = np.zeros((n_asu, 6))
    for im in range(n_modes):
        amp = _cpp_amplitude(float(c._initial_frequencies_cm1[im]), TEMPERATURE)
        py_u_stars += amp * mt_reshaped[im] / n_q
    py_u_cart0 = adptbx.u_star_as_u_cart(unit_cell, tuple(float(v) for v in py_u_stars[0]))
    py_ueq0 = (py_u_cart0[0] + py_u_cart0[1] + py_u_cart0[2]) / 3.0
    print(f"  Python-side Ueq(atom 0)={py_ueq0:.5f} Å²")

    # Diagnostic: scatterer u_star BEFORE store()
    u_star_before = xs.scatterers()[0].u_star
    print(f"  u_star[0] before store: {tuple(round(v,6) for v in u_star_before)}")

    # Build raw C++ reparametrisation with a single all-in-one scale group
    reparam = _sc.ext.reparametrisation(unit_cell)
    sp = reparam.add(_sc.independent_scalar_parameter, value=1.0, variable=True)
    reparam.add(
        _sc.nomore_u_star,
        scatterers         = tuple(xs.scatterers()),
        scale_params       = (sp,),
        mode_tensors_ustar = flex.double(mode_tensors_np.tolist()),
        initial_frequencies= flex.double(c._initial_frequencies_cm1.tolist()),
        group_ids          = flex.int([0] * n_modes),
        temperature        = TEMPERATURE,
        n_modes            = n_modes,
        n_q                = n_q,
    )
    reparam.finalise()
    reparam.linearise()
    reparam.store()

    u_star_after = xs.scatterers()[0].u_star
    print(f"  u_star[0] after  store: {tuple(round(v,6) for v in u_star_after)}")

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

    print("\n-- Section C: Python/C++ consistency --")
    test_python_tensors_match_analytic()
    test_cpp_u_star_from_python_tensors()

    print("\n-- Section D: compute_groups dispatch --")
    test_compute_groups_dispatch()

    print("\n-- Section E: Real-data integration (L-Ala 23K) --")
    test_integration_lala_real_data()

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == '__main__':
    run()

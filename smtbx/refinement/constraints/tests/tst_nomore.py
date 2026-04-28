from libtbx import easy_run
from cctbx import crystal, xray
from cctbx.array_family import flex
from scitbx import matrix
import smtbx.refinement.constraints as _sc
import smtbx.utils
import numpy as np

# 1. Minimal one-atom structure — carbon in P1
cs = crystal.symmetry(unit_cell=(10,10,10,90,90,90), 
                      space_group_symbol='P1')
sc = xray.scatterer('C', site=(0.1, 0.2, 0.3), u=0.05)
sc.flags.set_use_u_aniso(True)
sc.flags.set_grad_u_aniso(True)
sc.flags.set_use_u_iso(False)
xs = xray.structure(cs.special_position_settings(), 
                    flex.xray_scatterer([sc]))

# 2. Dummy phonon data — 3 modes, 1 q-point, 1 atom
n_modes = 3
n_asu = 1
n_q = 1

# Mode tensors: (n_modes * n_asu * 6) — all identical simple tensor
mode_tensors = flex.double([0.01, 0.01, 0.01, 0.0, 0.0, 0.0] * n_modes)

# Frequencies in cm-1
initial_frequencies = flex.double([100.0, 200.0, 300.0])

# Group ids: mode 0 -> group 0, mode 1 -> group 1, mode 2 -> fixed
group_ids = flex.int([0, 1, -1])

# Scale params — create reparametrisation first
connectivity = smtbx.utils.connectivity_table(xs)
reparam = _sc.ext.reparametrisation(xs.unit_cell())

scale0 = reparam.add(_sc.independent_scalar_parameter, value=1.0, variable=True)
scale1 = reparam.add(_sc.independent_scalar_parameter, value=1.0, variable=True)

#sc_list = tuple(xs.scatterers())
#print(sc_list[0])
print(scale0)

# 3. Add nomore_u_star
# param = reparam.add(
#     _sc.nomore_u_star,
#     scatterers=tuple(xs.scatterers()),
#     scale_params=(scale0, scale1),
#     mode_tensors_ustar=mode_tensors,
#     initial_frequencies=initial_frequencies,
#     group_ids=group_ids,
#     temperature=100.0,
#     n_modes=n_modes,
#     n_q=n_q
# )

param = reparam.add(
    _sc.nomore_u_star,
    xs.scatterers(),
    (scale0, scale1),
    mode_tensors,
    initial_frequencies,
    group_ids,
    100.0,
    n_modes,
    n_q
)
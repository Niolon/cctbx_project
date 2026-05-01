#include <smtbx/refinement/constraints/nomore_adp.h>
#include <scitbx/constants.h>

namespace smtbx { namespace refinement { namespace constraints {

void nomore_u_star::linearise(
  uctbx::unit_cell const &unit_cell,
  sparse_matrix_type *jacobian_transpose
) {
  // variables
  const double SPEED_OF_LIGHT = 29979245800 ; // cm s^-1
  const double PLANCK_REDUCED = 1.054571817e-34 ; // J s
  const double H_BAR_C = PLANCK_REDUCED * SPEED_OF_LIGHT ; // J cm
  const double BOLTZMANN = 1.380649e-23 ; // J K^-1
  const double H_BAR_C_DIV_KB = H_BAR_C / BOLTZMANN ; // J K
  const double AMU = 1.66053906660e-27;  // kg
  const double U_FACTOR = PLANCK_REDUCED * 1e20 / (AMU * 2.0 * scitbx::constants::pi * SPEED_OF_LIGHT);

  // H_C_DIV_KB = h*c/k_B = 2π*ħ*c/k_B  (correct Bose-Einstein exponent for wavenumber input)
  const double H_C_DIV_KB = 2.0 * scitbx::constants::pi * H_BAR_C / BOLTZMANN;

  int n_asu = scatterers_.size();
  for (int i_asu=0; i_asu < n_asu; ++i_asu)
    for (int i_uij=0; i_uij < 6; ++i_uij)
      u_stars_[i_asu][i_uij] = 0.0;

  for (int i_mode=0; i_mode < n_modes_; ++i_mode) {
    int i_group = group_ids_[i_mode];
    double freq;
    if (i_group >= 0) {
      freq = scale_params_[i_group]->value * initial_frequencies_[i_mode];
    } else {
      freq = initial_frequencies_[i_mode];
    }

    double ex        = std::exp(H_C_DIV_KB * freq / temperature_);
    double n         = 1.0 / (ex - 1.0);
    double amplitude = U_FACTOR * (0.5 + n) / freq;

    for (int i_asu=0; i_asu < n_asu; ++i_asu)
      for (int i_uij=0; i_uij < 6; ++i_uij) {
        int i_mode_tensor = 6 * i_asu + i_uij + i_mode * n_asu * 6;
        u_stars_[i_asu][i_uij] += amplitude * mode_tensors_ustar_[i_mode_tensor] / n_q_;
      }

    if (jacobian_transpose != NULL && i_group >= 0) {
      sparse_matrix_type &jt = *jacobian_transpose;
      double domega_dscale = initial_frequencies_[i_mode];
      // dn/domega = -n*(1+n) * H_C_DIV_KB / temperature_
      double dn_domega = -n * (1.0 + n) * H_C_DIV_KB / temperature_;
      // da/domega = U_FACTOR * ( dn/domega/freq - (0.5+n)/freq² )
      double da_domega = U_FACTOR * (dn_domega / freq - (0.5 + n) / (freq * freq));
      double da_dscale = da_domega * domega_dscale;
      for (int i_asu=0; i_asu < n_asu; ++i_asu)
        for (int i_uij=0; i_uij < 6; ++i_uij) {
          int i_mode_tensor = 6 * i_asu + i_uij + i_mode * n_asu * 6;
          jt(scale_params_[i_group]->index(), index() + i_asu*6 + i_uij) +=
            da_dscale * mode_tensors_ustar_[i_mode_tensor] / n_q_;
        }
    }
  }   
}

index_range nomore_u_star::component_indices_for(scatterer_type const *scatterer) const {
  for (int i=0; i < scatterers_.size(); i++) {
    if (scatterers_[i] == scatterer)
        return index_range(index() + i * 6, 6);
  }
  return index_range();
}

void nomore_u_star::write_component_annotations_for(
  scatterer_type const *scatterer,
  std::ostream &output
) const {
    for (int i=0; i < scatterers_.size(); i++)  {
      if( scatterers_[i] == scatterer )  {
        output << scatterers_[i]->label << ".u11,"
          << scatterers_[i]->label << ".u22,"
          << scatterers_[i]->label << ".u33,"
          << scatterers_[i]->label << ".u12,"
          << scatterers_[i]->label << ".u13,"
          << scatterers_[i]->label << ".u23,";
        return;
      }
    }
}

}}}
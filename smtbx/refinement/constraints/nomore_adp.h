#ifndef SMTBX_REFINEMENT_CONSTRAINTS_NOMORE_ADP_H
#define SMTBX_REFINEMENT_CONSTRAINTS_NOMORE_ADP_H

#include <smtbx/refinement/constraints/reparametrisation.h>
namespace smtbx { namespace refinement { namespace constraints {

class nomore_u_star : public asu_parameter
{
  const af::shared<independent_scalar_parameter*> scale_params_;
  const af::shared<double> mode_tensors_ustar_;
  const af::shared<double> initial_frequencies_;
  const af::shared<int> group_ids_;
  const double temperature_;
  const int n_modes_; 
  const int n_q_;
  af::shared<tensor_rank_2_t> u_stars_;
  af::shared<scatterer_type *> scatterers_;
public: 
  nomore_u_star(
    af::shared<scatterer_type *> const &scatterers,
    af::shared<independent_scalar_parameter*> const& scale_params,
    af::shared<double> const& mode_tensors_ustar,
    af::shared<double> const& initial_frequencies,
    af::shared<int> const& group_ids,
    double temperature,
    int n_modes,
    int n_q
  ) : parameter(scale_params.size()),
    scatterers_(scatterers),
    mode_tensors_ustar_(mode_tensors_ustar.begin(), mode_tensors_ustar.end()),
    initial_frequencies_(initial_frequencies.begin(), initial_frequencies.end()),
    group_ids_(group_ids.begin(), group_ids.end()),
    temperature_(temperature),
    n_modes_(n_modes),
    scale_params_(scale_params),
    n_q_(n_q)
  {
    for (int i=0; i < scale_params.size(); ++i) {
      set_argument(i, scale_params[i]);
    }
    u_stars_.resize(scatterers.size());
  };

  virtual void linearise(uctbx::unit_cell const &unit_cell,
                         sparse_matrix_type *jacobian_transpose);

  virtual scatterer_sequence_type scatterers() const {
    return scatterers_.const_ref();
  };

  virtual af::ref<double> components() {
    return af::ref<double>(u_stars_[0].begin(), 6*u_stars_.size());
  };

  virtual index_range component_indices_for(scatterer_type const *scatterer) const;
  virtual void write_component_annotations_for(scatterer_type const *scatterer,
                                              std::ostream &output) const;

  virtual void store(uctbx::unit_cell const &unit_cell) const {
    for (int i=0; i < scatterers_.size(); ++i) {
      scatterers_[i]->u_star = u_stars_[i];
    }
  };

};
  
}}}
#endif
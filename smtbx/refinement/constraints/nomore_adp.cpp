#include <smtbx/refinement/constraints/nomore_adp.h>

namespace smtbx { namespace refinement { namespace constraints {

void nomore_u_star::linearise(
  uctbx::unit_cell const &unit_cell,
  sparse_matrix_type *jacobian_transpose
) {
    
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
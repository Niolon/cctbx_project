#include <smtbx/refinement/constraints/nomore.h>
#include <cctbx/adptbx.h>

namespace smtbx { namespace refinement { namespace constraints {

void
nomore_u_star_parameter
::linearise(uctbx::unit_cell const &unit_cell,
            sparse_matrix_type *jacobian_transpose)
{
  std::size_t n_modes = frequencies_.size();

  // Initialize U_cart to zero
  scitbx::sym_mat3<double> u_cart(0, 0, 0, 0, 0, 0);

  // Storage for Jacobian: dU_cart/dnu for each mode
  std::vector<scitbx::sym_mat3<double> > du_cart_dfreq;
  if (jacobian_transpose) {
    du_cart_dfreq.resize(n_modes, scitbx::sym_mat3<double>(0,0,0,0,0,0));
  }

  // Compute contributions from each mode
  for (std::size_t m = 0; m < n_modes; ++m) {
    double freq = frequencies_[m];
    double amp = calculate_adp_amplitude(freq, temperature_, min_freq_);

    // Mode tensor at mode_tensors_[m*9 ... m*9+8] (row-major 3x3)
    std::size_t base = m * 9;
    double T00 = mode_tensors_[base + 0];
    double T11 = mode_tensors_[base + 4];
    double T22 = mode_tensors_[base + 8];
    double T01 = mode_tensors_[base + 1];
    double T02 = mode_tensors_[base + 2];
    double T12 = mode_tensors_[base + 5];

    // Accumulate U_cart
    u_cart[0] += amp * T00;
    u_cart[1] += amp * T11;
    u_cart[2] += amp * T22;
    u_cart[3] += amp * T01;
    u_cart[4] += amp * T02;
    u_cart[5] += amp * T12;

    // Jacobian: dU_cart/dnu
    if (jacobian_transpose) {
      double damp = calculate_amplitude_derivative(freq, temperature_, min_freq_);
      du_cart_dfreq[m][0] = damp * T00;
      du_cart_dfreq[m][1] = damp * T11;
      du_cart_dfreq[m][2] = damp * T22;
      du_cart_dfreq[m][3] = damp * T01;
      du_cart_dfreq[m][4] = damp * T02;
      du_cart_dfreq[m][5] = damp * T12;
    }
  }

  // Add HIGH mode contribution (fixed, pre-computed U_cart)
  if (high_contribution_.size() >= 6) {
    u_cart[0] += high_contribution_[0];
    u_cart[1] += high_contribution_[1];
    u_cart[2] += high_contribution_[2];
    u_cart[3] += high_contribution_[3];
    u_cart[4] += high_contribution_[4];
    u_cart[5] += high_contribution_[5];
  }

  // Apply normalization
  for (int k = 0; k < 6; ++k) {
    u_cart[k] /= normalization_;
  }
  if (jacobian_transpose) {
    for (std::size_t m = 0; m < n_modes; ++m) {
      for (int k = 0; k < 6; ++k) {
        du_cart_dfreq[m][k] /= normalization_;
      }
    }
  }

  // Convert U_cart to U*
  value = adptbx::u_cart_as_u_star(unit_cell, u_cart);

  // Fill Jacobian if requested
  if (!jacobian_transpose) return;
  sparse_matrix_type &jt = *jacobian_transpose;

  // For each mode, compute dU*/dnu and fill at correct Jacobian position
  for (std::size_t m = 0; m < n_modes; ++m) {
    // Skip if this mode has no associated parameter (e.g., HIGH modes
    // would have jacobian_indices_[m] set to a sentinel value or handled separately)
    if (m >= jacobian_indices_.size()) continue;
    std::size_t param_idx = jacobian_indices_[m];

    // Weight: for MFSF modes, dnu/dMFSF = nu_initial
    double weight = (m < jacobian_weights_.size()) ? jacobian_weights_[m] : 1.0;

    // Convert dU_cart/dnu to dU*/dnu
    scitbx::sym_mat3<double> du_cart_scaled = du_cart_dfreq[m];
    for (int k = 0; k < 6; ++k) {
      du_cart_scaled[k] *= weight;
    }
    scitbx::sym_mat3<double> du_star = adptbx::u_cart_as_u_star(
      unit_cell, du_cart_scaled);

    // Fill Jacobian: jt(param_row, u_component_col)
    for (int k = 0; k < 6; ++k) {
      jt(param_idx, index() + k) += du_star[k];
    }
  }
}

}}} // namespace smtbx::refinement::constraints

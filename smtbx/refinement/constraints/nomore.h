#ifndef SMTBX_REFINEMENT_CONSTRAINTS_NOMORE_H
#define SMTBX_REFINEMENT_CONSTRAINTS_NOMORE_H

#include <smtbx/refinement/constraints/reparametrisation.h>
#include <cctbx/adptbx.h>
#include <scitbx/array_family/shared.h>
#include <cmath>
#include <vector>

namespace smtbx { namespace refinement { namespace constraints {

// ============================================================================
// Physical Constants for NoMoRe ADP calculation
// All frequencies in cm^-1
// ============================================================================

namespace nomore_constants {
  // Boltzmann constant in cm^-1/K: k_B / (h * c * 100)
  const double K_TO_CM1 = 0.6950348004861275;

  // Factor for U_cart: hbar * 1e20 / (AMU * 2*pi * c * 100)
  // Derived from scipy.constants
  const double U_CART_FACTOR_CM1 = 33.71525833617554;
}

// ============================================================================
// Physics Functions
// ============================================================================

/// Calculate Bose-Einstein energy in cm^-1 units
inline double bose_einstein_energy_cm1(double freq_cm1, double temperature_K) {
  using namespace nomore_constants;
  double kt_cm1 = temperature_K * K_TO_CM1;
  if (kt_cm1 < 1e-6) kt_cm1 = 1e-6;

  double x = freq_cm1 / kt_cm1;
  double occupation;
  if (x < 1e-4 && x > 0) {
    occupation = 1.0/x - 0.5;
  } else if (x <= 0) {
    occupation = 0.0;
  } else {
    occupation = 1.0 / (std::exp(x) - 1.0);
  }
  return freq_cm1 * (0.5 + occupation);
}

/// Calculate ADP amplitude factor from frequency
inline double calculate_adp_amplitude(double freq_cm1, double temperature_K,
                                       double min_freq = 5.0) {
  using namespace nomore_constants;
  double safe_freq = (freq_cm1 < min_freq) ? min_freq : freq_cm1;
  double energy = bose_einstein_energy_cm1(safe_freq, temperature_K);
  return energy / (safe_freq * safe_freq) * U_CART_FACTOR_CM1;
}

/// Calculate derivative of amplitude w.r.t. frequency: dA/dnu
inline double calculate_amplitude_derivative(double freq_cm1, double temperature_K,
                                              double min_freq = 5.0) {
  using namespace nomore_constants;
  double nu = (freq_cm1 < min_freq) ? min_freq : freq_cm1;

  double kt_cm1 = temperature_K * K_TO_CM1;
  if (kt_cm1 < 1e-6) kt_cm1 = 1e-6;
  double x = nu / kt_cm1;

  double n, dn_dx;
  if (x < 1e-4 && x > 0) {
    n = 1.0/x - 0.5;
    dn_dx = -1.0/(x*x);
  } else if (x <= 0) {
    n = 0.0;
    dn_dx = 0.0;
  } else {
    double ex = std::exp(x);
    n = 1.0 / (ex - 1.0);
    dn_dx = -ex / ((ex - 1.0) * (ex - 1.0));
  }

  double dE_dnu = (0.5 + n) + nu * dn_dx * (1.0/kt_cm1);
  double E = nu * (0.5 + n);
  return U_CART_FACTOR_CM1 * (dE_dnu * nu - 2.0 * E) / (nu * nu * nu);
}

// ============================================================================
// NoMoRe U* Parameter Class (simplified design)
// ============================================================================

/// ADP from phonon frequencies - stores frequencies and indices for Jacobian
class nomore_u_star_parameter : public asu_u_star_parameter
{
public:
  /**
   * Constructor for NoMoRe U* parameter.
   *
   * All mode tensors and frequencies are stored internally.
   * Jacobian indices map frequency values to their positions in the
   * global independent parameter array.
   */
  nomore_u_star_parameter(
    scatterer_type *scatterer,
    af::shared<double> const &mode_tensors,      // (n_modes, 9) flattened
    af::shared<double> const &frequencies,        // Current frequency values
    af::shared<std::size_t> const &jacobian_indices, // Index in Jacobian for each freq
    af::shared<double> const &jacobian_weights,   // dnu/dparam for each freq
    af::shared<double> const &high_contribution,  // Fixed U_cart from HIGH modes
    double temperature,
    double min_freq,
    double normalization)
  : parameter(0),  // No direct arguments - we handle Jacobian manually
    single_asu_scatterer_parameter(scatterer),
    mode_tensors_(mode_tensors),
    frequencies_(frequencies),
    jacobian_indices_(jacobian_indices),
    jacobian_weights_(jacobian_weights),
    high_contribution_(high_contribution),
    temperature_(temperature),
    min_freq_(min_freq),
    normalization_(normalization)
  {}

  /// Update frequencies (called before linearise for each cycle)
  void set_frequencies(af::shared<double> const &freqs) {
    frequencies_ = freqs;
  }

  std::size_t n_modes() const { return frequencies_.size(); }

  virtual void linearise(uctbx::unit_cell const &unit_cell,
                         sparse_matrix_type *jacobian_transpose);

private:
  af::shared<double> mode_tensors_;
  af::shared<double> frequencies_;
  af::shared<std::size_t> jacobian_indices_;
  af::shared<double> jacobian_weights_;
  af::shared<double> high_contribution_;
  double temperature_;
  double min_freq_;
  double normalization_;
};

}}} // namespace smtbx::refinement::constraints

#endif // SMTBX_REFINEMENT_CONSTRAINTS_NOMORE_H

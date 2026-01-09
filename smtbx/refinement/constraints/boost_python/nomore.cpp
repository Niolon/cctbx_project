#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>

#include <smtbx/refinement/constraints/nomore.h>

namespace smtbx { namespace refinement { namespace constraints {
  namespace boost_python {

    struct nomore_u_star_parameter_wrapper {
      typedef nomore_u_star_parameter wt;

      static void wrap() {
        using namespace boost::python;

        class_<wt,
               bases<asu_u_star_parameter>,
               boost::shared_ptr<wt> >("nomore_u_star_parameter", no_init)
          .def(init<wt::scatterer_type*,
                    af::shared<double> const&,
                    af::shared<double> const&,
                    af::shared<std::size_t> const&,
                    af::shared<double> const&,
                    af::shared<double> const&,
                    double,
                    double,
                    double>
               ((arg("scatterer"),
                 arg("mode_tensors"),
                 arg("frequencies"),
                 arg("jacobian_indices"),
                 arg("jacobian_weights"),
                 arg("high_contribution"),
                 arg("temperature"),
                 arg("min_freq"),
                 arg("normalization"))))
          .def("set_frequencies", &wt::set_frequencies)
          .def("n_modes", &wt::n_modes)
          ;

        implicitly_convertible<boost::shared_ptr<wt>, boost::shared_ptr<parameter> >();
      }
    };

    void wrap_nomore() {
      nomore_u_star_parameter_wrapper::wrap();
    }

}}}} // namespace smtbx::refinement::constraints::boost_python

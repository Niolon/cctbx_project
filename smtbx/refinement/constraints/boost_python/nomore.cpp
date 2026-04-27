#include <boost/python/class.hpp>
#include <boost/python/implicit.hpp>
#include <scitbx/boost_python/container_conversions.h>

#include <smtbx/refinement/constraints/nomore_adp.h>
namespace smtbx { namespace refinement { namespace constraints {
  namespace boost_python {
    struct nomore_wrapper {
      typedef nomore_u_star wt;

      static void wrap() {
        using namespace boost::python;
        class_<
          wt,
          bases<asu_parameter>,
          boost::shared_ptr<wt>
        >("nomore_u_star", no_init)
        .def(
          init<
            af::shared<wt::scatterer_type *> const &,
            af::shared<independent_scalar_parameter*> const &, 
            af::shared<double> const &,
            af::shared<double> const &,
            af::shared<int> const &,
            double,
            int,
            int
          >
          ((
            arg("scatterers"),
            arg("scale_params"),
            arg("mode_tensors_ustar"),
            arg("initial_frequencies"),
            arg("group_ids"),
            arg("temperature"),
            arg("n_modes"),
            arg("n_q")
          ))
        );
        implicitly_convertible<boost::shared_ptr<wt>, boost::shared_ptr<parameter> >(); 
      }
    };
    void wrap_nomore() {
      nomore_wrapper::wrap();
    }
  }
}}}
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <dials/algorithms/scaling/scaling_helper.h>
#include <cctbx/miller.h>
#include <dials/array_family/scitbx_shared_and_versa.h>

namespace dials_scaling { namespace boost_python {

  using namespace boost::python;

  using scitbx::sparse::matrix;

  void export_determine_outlier_indices() {
    def("determine_outlier_indices",
        &determine_outlier_indices,
        (arg("h_index_matrix"), arg("z_scores"), arg("zmax")));
  }

  void export_elementwise_square() {
    def("elementwise_square", &elementwise_square, (arg("m")));
  }

  void export_calc_dIh_by_dpi() {
    def("calc_dIh_by_dpi",
        &calculate_dIh_by_dpi,
        (arg("a"), arg("sumgsq"), arg("h_index_mat"), arg("derivatives")));
  }

  void export_calc_jacobian() {
    def("calc_jacobian",
        &calc_jacobian,
        (arg("derivatives"),
         arg("h_index_mat"),
         arg("Ih"),
         arg("g"),
         arg("dIh"),
         arg("sumgsq")));
  }

  void export_sph_harm_table() {
    def("create_sph_harm_table",
        &create_sph_harm_table,
        (arg("s0_theta_phi"), arg("s1_theta_phi"), arg("lmax")));
  }

  void export_rotate_vectors_about_axis() {
    def("rotate_vectors_about_axis",
        &rotate_vectors_about_axis,
        (arg("rot_axis"), arg("vectors"), arg("angles")));
  }

  void export_calc_theta_phi() {
    def("calc_theta_phi", &calc_theta_phi, (arg("xyz")));
  }

  void export_calc_sigmasq() {
    def("calc_sigmasq", &calc_sigmasq, (arg("jacobian_transpose"), arg("var_cov")));
  }

  void export_row_multiply() {
    def("row_multiply", &row_multiply, (arg("m"), arg("v")));
  }

  void export_limit_outlier_weights() {
    def("limit_outlier_weights",
        &limit_outlier_weights,
        (arg("weights"), arg("h_index_mat")));
  }

  void export_calc_lookup_index() {
    def("calc_lookup_index",
        &calc_lookup_index,
        (arg("thetaphi"), arg("points_per_degree")));
  }

  void export_create_sph_harm_lookup_table() {
    def("create_sph_harm_lookup_table",
        &create_sph_harm_lookup_table,
        (arg("lmax"), arg("points_per_degree")));
  }

  void export_calculate_harmonic_tables_from_selections() {
    def("calculate_harmonic_tables_from_selections",
        &calculate_harmonic_tables_from_selections,
        (arg("s0_selection"), arg("s1_selection"), arg("coefficients_list")));
  }

  void export_gaussian_smoother_first_fixed() {
    class_<GaussianSmootherFirstFixed>("GaussianSmootherFirstFixed", no_init)
      .def(init<vec2<double>, std::size_t>((arg("x_range"), arg("num_intervals"))))
      .def("set_smoothing", &GaussianSmootherFirstFixed::set_smoothing)
      .def("num_values", &GaussianSmootherFirstFixed::num_values)
      .def("num_samples", &GaussianSmootherFirstFixed::num_samples)
      .def("num_average", &GaussianSmootherFirstFixed::num_average)
      .def("sigma", &GaussianSmootherFirstFixed::sigma)
      .def("spacing", &GaussianSmootherFirstFixed::spacing)
      .def("positions", &GaussianSmootherFirstFixed::positions)
      .def("value_weight", &GaussianSmootherFirstFixed::value_weight)
      .def("value_weight_first_fixed",
           &GaussianSmootherFirstFixed::value_weight_first_fixed)
      .def("multi_value_weight", &GaussianSmootherFirstFixed::multi_value_weight)
      .def("multi_value_weight_first_fixed",
           &GaussianSmootherFirstFixed::multi_value_weight_first_fixed);
  }

  void export_weighted_split_unmerged() {
    class_<weighted_split_unmerged>("weighted_split_unmerged", no_init)
      .def(init<scitbx::af::const_ref<cctbx::miller::index<> > const&,
                scitbx::af::const_ref<double> const&,
                scitbx::af::const_ref<double> const&,
                bool,
                unsigned>((arg("unmerged_indices"),
                           arg("unmerged_data"),
                           arg("unmerged_sigmas"),
                           arg("weighted") = true,
                           arg("seed") = 0)))
      .def("data", &weighted_split_unmerged::data);

    class_<ResultsStruct>("ResultsStruct", no_init)
      .def("get_data1", &ResultsStruct::get_data1)
      .def("get_data2", &ResultsStruct::get_data2)
      .def("get_sigma1", &ResultsStruct::get_sigma1)
      .def("get_sigma2", &ResultsStruct::get_sigma2)
      .def("get_indices", &ResultsStruct::get_indices);
  }

}}  // namespace dials_scaling::boost_python

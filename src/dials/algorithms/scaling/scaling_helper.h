#ifndef DIALS_SCALING_SCALING_HELPER_H
#define DIALS_SCALING_SCALING_HELPER_H

#include <dials/array_family/scitbx_shared_and_versa.h>
#include <scitbx/sparse/matrix.h>
#include <scitbx/math/zernike.h>
#include <scitbx/math/basic_statistics.h>
#include <dials/error.h>
#include <math.h>
#include <dials/algorithms/refinement/gaussian_smoother.h>
#include <scitbx/random.h>
#include <cctbx/miller.h>

typedef scitbx::sparse::matrix<double>::column_type col_type;

namespace dials_scaling {

using namespace boost::python;
using namespace dials::refinement;

class GaussianSmootherFirstFixed : public dials::refinement::GaussianSmoother {
public:
  GaussianSmootherFirstFixed(vec2<double> x_range, std::size_t num_intervals)
      : GaussianSmoother(x_range, num_intervals) {}

  dials::refinement::SingleValueWeights value_weight_first_fixed(
    double x,
    const scitbx::af::const_ref<double> values) {
    // use sparse storage as only naverage (default 3) values are non-zero
    vector<double> weight(nvalues - 1);

    // normalised coordinate
    double z = (x - x0) / spacing_;
    double sumwv = 0.0;
    double sumweight = 0.0;

    vec2<int> irange = idx_range(z);

    for (int i = irange[0]; i < irange[1]; ++i) {
      double ds = (z - positions_[i]) / sigma_;
      double w = exp(-ds * ds);
      if (i > 0) {
        weight[i - 1] = w;
      }
      weight[i] = w;
      sumwv += w * values[i];
      sumweight += w;
    }

    double value;
    if (sumweight > 0.0) {
      value = sumwv / sumweight;
    } else {
      value = 0.0;
    }

    return SingleValueWeights(value, weight, sumweight);
  }

  dials::refinement::MultiValueWeights multi_value_weight_first_fixed(
    const scitbx::af::const_ref<double> x,
    const scitbx::af::const_ref<double> values) {
    // Use sparse storage as only naverage (default 3) values per row are
    // non-zero
    std::size_t npoints = x.size();  //# data
    DIALS_ASSERT(npoints > 1);
    matrix<double> weight(npoints, nvalues - 1);

    // Allocate space for the interpolated values and sumweights, with raw
    // refs for fastest access (See Michael Hohn's notes)
    scitbx::af::shared<double> value(npoints, scitbx::af::init_functor_null<double>());
    scitbx::af::ref<double> value_ref = value.ref();
    scitbx::af::shared<double> sumweight(npoints,
                                         scitbx::af::init_functor_null<double>());
    scitbx::af::ref<double> sumweight_ref = sumweight.ref();

    for (std::size_t irow = 0; irow < npoints; ++irow) {
      // normalised coordinate
      double z = (x[irow] - x0) / spacing_;
      double sumw = 0.0;
      double sumwv = 0.0;

      vec2<int> irange = idx_range(z);

      for (int icol = irange[0]; icol < irange[1]; ++icol) {
        double ds = (z - positions_[icol]) / sigma_;
        double w = exp(-ds * ds);
        if (icol > 0) {
          weight(irow, icol - 1) = w;
        }
        sumw += w;
        sumwv += w * values[icol];
      }
      sumweight_ref[irow] = sumw;

      if (sumw > 0.0) {
        value_ref[irow] = sumwv / sumw;
      } else {
        value_ref[irow] = 0.0;
      }
    }

    return MultiValueWeights(value, weight, sumweight);
  }
};

/**
 * Elementwise squaring of a matrix
 */
scitbx::sparse::matrix<double> elementwise_square(scitbx::sparse::matrix<double> m) {
  scitbx::sparse::matrix<double> result(m.n_rows(), m.n_cols());

  // outer loop iterate over the columns
  for (std::size_t j = 0; j < m.n_cols(); j++) {
    // inner loop iterate over the non-zero elements of the column
    for (scitbx::sparse::matrix<double>::row_iterator p = m.col(j).begin();
         p != m.col(j).end();
         ++p) {
      std::size_t i = p.index();
      result(i, j) = *p * *p;
    }
  }
  return result;
}

scitbx::af::shared<double> limit_outlier_weights(
  scitbx::af::shared<double> weights,
  scitbx::sparse::matrix<double> h_index_mat) {
  scitbx::math::median_functor med;
  for (int i = 0; i < h_index_mat.n_cols(); ++i) {
    const col_type column = h_index_mat.col(i);
    scitbx::af::shared<double> theseweights;
    for (col_type::const_iterator it = column.begin(); it != column.end(); ++it) {
      int refl_idx = it.index();
      theseweights.push_back(weights[refl_idx]);
    }
    // now get the median
    double median = med(theseweights.ref());
    double ceil = 10.0 * median;
    for (col_type::const_iterator it = column.begin(); it != column.end(); ++it) {
      int refl_idx = it.index();
      if (weights[refl_idx] > ceil) {
        weights[refl_idx] = ceil;
      }
    }
  }
  return weights;
}

scitbx::sparse::matrix<double> calculate_dIh_by_dpi(
  scitbx::af::shared<double> dIh,
  scitbx::af::shared<double> sumgsq,
  scitbx::sparse::matrix<double> h_index_mat,
  scitbx::sparse::matrix<double> derivatives) {
  // derivatives is a matrix where rows are params and cols are reflections
  int n_params = derivatives.n_rows();
  int n_groups = h_index_mat.n_cols();
  scitbx::sparse::matrix<double> dIh_by_dpi(n_groups, n_params);

  for (int i = 0; i < h_index_mat.n_cols(); ++i) {
    const col_type column = h_index_mat.col(i);
    // loop over reflection groups
    column.compact();
    for (col_type::const_iterator it = column.begin(); it != column.end(); ++it) {
      // so it.index gives index of a refl from group i
      int refl_idx = it.index();
      const col_type dgidx_by_dpi = derivatives.col(refl_idx);
      // deriv of one refl wrt all params
      for (col_type::const_iterator dgit = dgidx_by_dpi.begin();
           dgit != dgidx_by_dpi.end();
           ++dgit) {
        // dgit.index indicates which params have nonzero derivs
        dIh_by_dpi(i, dgit.index()) += (dIh[refl_idx] * *dgit / sumgsq[i]);
      }
    }
  }
  dIh_by_dpi.compact();
  return dIh_by_dpi;
}

scitbx::sparse::matrix<double> calculate_dIh_by_dpi_transpose(
  scitbx::af::shared<double> dIh,
  scitbx::af::shared<double> sumgsq,
  scitbx::sparse::matrix<double> h_index_mat,
  scitbx::sparse::matrix<double> derivatives) {
  // derivatives is a matrix where rows are params and cols are reflections
  int n_params = derivatives.n_rows();
  int n_groups = h_index_mat.n_cols();
  scitbx::sparse::matrix<double> dIh_by_dpi(n_params, n_groups);

  for (int i = 0; i < h_index_mat.n_cols(); ++i) {
    const col_type column = h_index_mat.col(i);
    // first loop over h_idx to get indices for a reflection group
    column.compact();
    scitbx::sparse::vector<double> deriv_of_group_by_params(n_params);
    for (col_type::const_iterator it = column.begin(); it != column.end(); ++it) {
      // it.index gives index of a refl from group i
      int refl_idx = it.index();
      const col_type dgidx_by_dpi = derivatives.col(refl_idx);
      // deriv of one refl wrt all params
      for (col_type::const_iterator dgit = dgidx_by_dpi.begin();
           dgit != dgidx_by_dpi.end();
           ++dgit) {
        // dgit.index indicates which params have nonzero derivs
        dIh_by_dpi(dgit.index(), i) += (dIh[refl_idx] * *dgit / sumgsq[i]);
      }
    }
  }
  dIh_by_dpi.compact();
  return dIh_by_dpi;
}

scitbx::sparse::matrix<double> calc_jacobian(scitbx::sparse::matrix<double> derivatives,
                                             scitbx::sparse::matrix<double> h_index_mat,
                                             scitbx::af::shared<double> Ih,
                                             scitbx::af::shared<double> g,
                                             scitbx::af::shared<double> dIh,
                                             scitbx::af::shared<double> sumgsq) {
  // derivatives is a matrix where rows are params and cols are reflections
  int n_params = derivatives.n_rows();
  int n_refl = derivatives.n_cols();

  scitbx::sparse::matrix<double> dIhbydpiT =
    calculate_dIh_by_dpi_transpose(dIh, sumgsq, h_index_mat, derivatives);
  scitbx::sparse::matrix<double> Jacobian(n_refl, n_params);

  for (int i = 0; i < h_index_mat.n_cols(); ++i) {
    const col_type column = h_index_mat.col(i);
    // first loop over h_idx to get indices for a reflection group
    column.compact();
    for (col_type::const_iterator it = column.begin(); it != column.end(); ++it) {
      // it.index gives index of a refl from group i
      int refl_idx = it.index();
      const col_type dgidx_by_dpi = derivatives.col(refl_idx);
      // deriv of one refl wrt all params
      dgidx_by_dpi.compact();
      // loop over nonzero elements of dgidx by dpi
      for (col_type::const_iterator dgit = dgidx_by_dpi.begin();
           dgit != dgidx_by_dpi.end();
           ++dgit) {
        // dgit.index indicates which params have nonzero derivs
        Jacobian(refl_idx, dgit.index()) -= *dgit * Ih[refl_idx];
      }
      // now loop over nonzero elements of dIhbydpi
      // get col corresponding to group
      const col_type dIh_col = dIhbydpiT.col(i);
      // now loop over nonzero params
      for (col_type::const_iterator dIit = dIh_col.begin(); dIit != dIh_col.end();
           ++dIit) {
        Jacobian(refl_idx, dIit.index()) -= g[refl_idx] * *dIit;
      }
    }
  }
  Jacobian.compact();
  return Jacobian;
}

scitbx::sparse::matrix<double> row_multiply(scitbx::sparse::matrix<double> m,
                                            scitbx::af::const_ref<double> v) {
  DIALS_ASSERT(m.n_rows() == v.size());

  // call compact to ensure that each elt of the matrix is only defined once
  m.compact();

  scitbx::sparse::matrix<double> result(m.n_rows(), m.n_cols());

  // outer loop iterate over the columns
  for (std::size_t j = 0; j < m.n_cols(); j++) {
    // inner loop iterate over the non-zero elements of the column
    for (scitbx::sparse::matrix<double>::row_iterator p = m.col(j).begin();
         p != m.col(j).end();
         ++p) {
      std::size_t i = p.index();
      result(i, j) = *p * v[i];
    }
  }
  return result;
}

scitbx::af::shared<scitbx::vec2<double> > calc_theta_phi(
  scitbx::af::shared<scitbx::vec3<double> > xyz) {
  // physics conventions, phi from 0 to 2pi (xy plane, 0 along x axis), theta from 0 to
  // pi (angle from z)
  int n_obs = xyz.size();
  scitbx::af::shared<scitbx::vec2<double> > theta_phi(n_obs);
  for (int i = 0; i < n_obs; i++) {
    theta_phi[i] = scitbx::vec2<double>(
      std::atan2(pow(pow(xyz[i][1], 2) + pow(xyz[i][0], 2), 0.5), xyz[i][2]),
      fmod(std::atan2(xyz[i][1], xyz[i][0]) + (2.0 * scitbx::constants::pi),
           2.0 * scitbx::constants::pi));
    // atan2 returns in -pi to pi range, with 0 along x. We want to make it go from 0 to
    // 2pi
  }
  return theta_phi;
}

scitbx::af::shared<std::size_t> calc_lookup_index(
  scitbx::af::shared<scitbx::vec2<double> > theta_phi,
  double points_per_degree) {
  // theta index 0 to 2pi (shift by pi from input theta),
  scitbx::af::shared<std::size_t> lookup_index(theta_phi.size());
  for (int i = 0; i < theta_phi.size(); ++i) {
    lookup_index[i] =
      int(360.0 * points_per_degree
            * floor(theta_phi[i][0] * points_per_degree * 180.0 / scitbx::constants::pi)
          + floor(theta_phi[i][1] * 180.0 * points_per_degree / scitbx::constants::pi));
  }
  return lookup_index;
}

boost::python::tuple determine_outlier_indices(
  scitbx::sparse::matrix<double> h_index_mat,
  scitbx::af::shared<double> z_scores,
  double zmax) {
  scitbx::af::shared<std::size_t> outlier_indices;
  scitbx::af::shared<std::size_t> other_potential_outlier_indices;
  for (int i = 0; i < h_index_mat.n_cols(); ++i) {
    const col_type column = h_index_mat.col(i);
    column.compact();
    double max_z = zmax;  // copy value//
    int n_elem = 0;
    int index_of_max = 0;
    for (col_type::const_iterator it = column.begin(); it != column.end(); ++it) {
      double val = z_scores[it.index()];
      if (val > max_z) {
        max_z = val;
        index_of_max = it.index();
      }
      ++n_elem;
    }
    if (n_elem > 2 && max_z > zmax) {
      // want to get indices of other potential outliers too
      outlier_indices.push_back(index_of_max);
      for (col_type::const_iterator it = column.begin(); it != column.end(); ++it) {
        if (it.index() != index_of_max) {
          other_potential_outlier_indices.push_back(it.index());
        }
      }
    }
  }
  return boost::python::make_tuple(outlier_indices, other_potential_outlier_indices);
}

scitbx::af::shared<double> calc_sigmasq(
  scitbx::sparse::matrix<double> jacobian_transpose,
  scitbx::sparse::matrix<double> var_cov_matrix) {
  int n_cols = jacobian_transpose.n_cols();
  scitbx::af::shared<double> sigmasq(n_cols);
  for (int i = 0; i < n_cols; i++) {
    // scitbx::af::shared<double> result(n_refl);
    // for (int j=0; j < n_refl; j++){
    //  result[j] += jacobian_transpose.col(i) * var_cov_matrix.col(j);
    //  }

    for (scitbx::sparse::matrix<double>::row_iterator p =
           jacobian_transpose.col(i).begin();
         p != jacobian_transpose.col(i).end();
         ++p) {
      int k = p.index();
      sigmasq[i] +=
        *p * (jacobian_transpose.col(i) * var_cov_matrix.col(k));  // *p * result[k];
    }
  }
  return sigmasq;
}

scitbx::af::shared<scitbx::vec3<double> > rotate_vectors_about_axis(
  scitbx::af::shared<scitbx::vec3<double> > const rot_axis,
  scitbx::af::shared<scitbx::vec3<double> > const vectors,
  scitbx::af::shared<double> const angles) {
  // first normalise rotation axis
  double modulus = pow(
    (pow(rot_axis[0][0], 2) + pow(rot_axis[0][1], 2) + pow(rot_axis[0][2], 2)), 0.5);
  double ux = rot_axis[0][0] / modulus;
  double uy = rot_axis[0][1] / modulus;
  double uz = rot_axis[0][2] / modulus;
  int n_obs = angles.size();
  scitbx::af::shared<scitbx::vec3<double> > rotated_vectors(n_obs);

  for (int i = 0; i < n_obs; i++) {
    double cos_angle = std::cos(angles[i]);
    double sin_angle = std::sin(angles[i]);
    rotated_vectors[i] = scitbx::vec3<double>(
      (((cos_angle + ((pow(ux, 2)) * (1.0 - cos_angle))) * vectors[i][0])
       + (((ux * uy * (1.0 - cos_angle)) - (uz * sin_angle)) * vectors[i][1])
       + (((uz * ux * (1.0 - cos_angle)) + (uy * sin_angle)) * vectors[i][2])),
      ((((ux * uy * (1.0 - cos_angle)) + (uz * sin_angle)) * vectors[i][0])
       + ((cos_angle + ((pow(uy, 2)) * (1.0 - cos_angle))) * vectors[i][1])
       + (((uz * uy * (1.0 - cos_angle)) - (ux * sin_angle)) * vectors[i][2])),
      ((((ux * uz * (1.0 - cos_angle)) - (uy * sin_angle)) * vectors[i][0])
       + (((uy * uz * (1.0 - cos_angle)) + (ux * sin_angle)) * vectors[i][1])
       + ((cos_angle + ((pow(uz, 2)) * (1.0 - cos_angle))) * vectors[i][2])));
  }
  return rotated_vectors;
}

/**
 * Spherical harmonic table
 */

using scitbx::math::zernike::log_factorial_generator;
using scitbx::math::zernike::nss_spherical_harmonics;
using scitbx::sparse::matrix;
using scitbx::sparse::vector;

boost::python::tuple calculate_harmonic_tables_from_selections(
  scitbx::af::shared<std::size_t> s0_selection,
  scitbx::af::shared<std::size_t> s1_selection,
  boost::python::list coefficients_list) {
  int n_refl = s0_selection.size();
  int n_param = boost::python::len(coefficients_list);
  boost::python::list output_coefficients_list;
  matrix<double> coefficients_matrix(n_refl, n_param);
  // loop though each param first, then loop over selection
  for (int i = 0; i < n_param; ++i) {
    scitbx::af::shared<double> coefs =
      boost::python::extract<scitbx::af::shared<double> >(coefficients_list[i]);
    scitbx::af::shared<double> coef_for_output(n_refl);
    // loop though each reflection
    for (int j = 0; j < n_refl; ++j) {
      double val0 = coefs[s0_selection[j]];
      double val1 = coefs[s1_selection[j]];
      double value = (val0 + val1) / 2.0;
      coefficients_matrix(j, i) = value;
      coef_for_output[j] = value;
    }
    output_coefficients_list.append(coef_for_output);
  }
  return boost::python::make_tuple(output_coefficients_list, coefficients_matrix);
}

matrix<double> create_sph_harm_table(
  scitbx::af::shared<scitbx::vec2<double> > const s0_theta_phi,
  scitbx::af::shared<scitbx::vec2<double> > const s1_theta_phi,
  int lmax) {
  nss_spherical_harmonics<double> nsssphe(
    lmax, 50000, log_factorial_generator<double>((2 * lmax) + 1));
  int n_abs_param = (2 * lmax) + (pow(double(lmax), 2));
  int n_obs = s1_theta_phi.size();
  matrix<double> sph_harm_terms_(n_abs_param, n_obs);
  double sqrt2 = 1.414213562;
  int counter = 0;
  for (int l = 1; l < lmax + 1; l++) {
    for (int m = -1 * l; m < l + 1; m++) {
      if (m < 0) {
        double prefactor = sqrt2 * pow(-1.0, m) / 2.0;
        for (int i = 0; i < n_obs; i++) {
          sph_harm_terms_(counter, i) =
            prefactor
            * (nsssphe
                 .spherical_harmonic_direct(
                   l, -1 * m, s0_theta_phi[i][0], s0_theta_phi[i][1])
                 .imag()
               + nsssphe
                   .spherical_harmonic_direct(
                     l, -1 * m, s1_theta_phi[i][0], s1_theta_phi[i][1])
                   .imag());
        }
      } else if (m == 0) {
        for (int i = 0; i < n_obs; i++) {
          sph_harm_terms_(counter, i) =
            (0.5
             * (nsssphe
                  .spherical_harmonic_direct(
                    l, 0, s0_theta_phi[i][0], s0_theta_phi[i][1])
                  .real()
                + nsssphe
                    .spherical_harmonic_direct(
                      l, 0, s1_theta_phi[i][0], s1_theta_phi[i][1])
                    .real()));
        }
      } else {
        double prefactor = sqrt2 * pow(-1.0, m) / 2.0;
        for (int i = 0; i < n_obs; i++) {
          double val = prefactor
                       * (nsssphe
                            .spherical_harmonic_direct(
                              l, m, s0_theta_phi[i][0], s0_theta_phi[i][1])
                            .real()
                          + nsssphe
                              .spherical_harmonic_direct(
                                l, m, s1_theta_phi[i][0], s1_theta_phi[i][1])
                              .real());
          sph_harm_terms_(counter, i) = val;
        }
      }
      counter += 1;
    }
  }
  return sph_harm_terms_;
}

boost::python::list create_sph_harm_lookup_table(int lmax, int points_per_degree) {
  nss_spherical_harmonics<double> nsssphe(
    lmax, 50000, log_factorial_generator<double>((2 * lmax) + 1));
  boost::python::list coefficients_list;
  double sqrt2 = 1.414213562;
  int n_items = 360 * 180 * points_per_degree * points_per_degree;
  for (int l = 1; l < lmax + 1; l++) {
    for (int m = -1 * l; m < l + 1; m++) {
      scitbx::af::shared<double> coefficients(n_items);
      if (m < 0) {
        double prefactor = sqrt2 * pow(-1.0, m);
        for (int i = 0; i < n_items; i++) {
          double theta = (floor(i / (360.0 * points_per_degree)) * scitbx::constants::pi
                          / (180.0 * points_per_degree));
          double phi = ((i % (360 * points_per_degree)) * scitbx::constants::pi
                        / (180.0 * points_per_degree));
          coefficients[i] =
            prefactor
            * (nsssphe.spherical_harmonic_direct(l, -1 * m, theta, phi).imag());
        }
      } else if (m == 0) {
        for (int i = 0; i < n_items; i++) {
          double theta = (floor(i / (360.0 * points_per_degree)) * scitbx::constants::pi
                          / (180.0 * points_per_degree));
          double phi = ((i % (360 * points_per_degree)) * scitbx::constants::pi
                        / (180.0 * points_per_degree));
          coefficients[i] = nsssphe.spherical_harmonic_direct(l, 0, theta, phi).real();
        }
      } else {
        double prefactor = sqrt2 * pow(-1.0, m);
        for (int i = 0; i < n_items; i++) {
          double theta = (floor(i / (360.0 * points_per_degree)) * scitbx::constants::pi
                          / (180.0 * points_per_degree));
          double phi = ((i % (360 * points_per_degree)) * scitbx::constants::pi
                        / (180.0 * points_per_degree));
          coefficients[i] =
            prefactor * (nsssphe.spherical_harmonic_direct(l, m, theta, phi).real());
        }
      }
      coefficients_list.append(coefficients);
    }
  }
  return coefficients_list;
}
struct ResultsStruct {
  scitbx::af::shared<double> data_1;
  scitbx::af::shared<double> data_2;
  scitbx::af::shared<double> sigma_1;
  scitbx::af::shared<double> sigma_2;
  scitbx::af::shared<cctbx::miller::index<> > indices;

  ResultsStruct(scitbx::af::shared<double> data_1,
                scitbx::af::shared<double> data_2,
                scitbx::af::shared<double> sigma_1,
                scitbx::af::shared<double> sigma_2,
                scitbx::af::shared<cctbx::miller::index<> > indices)
      : data_1(data_1),
        data_2(data_2),
        sigma_1(sigma_1),
        sigma_2(sigma_2),
        indices(indices) {}

  scitbx::af::shared<double> get_data1() const {
    return data_1;
  }
  scitbx::af::shared<double> get_data2() const {
    return data_2;
  }
  scitbx::af::shared<double> get_sigma1() const {
    return sigma_1;
  }
  scitbx::af::shared<double> get_sigma2() const {
    return sigma_2;
  }
  scitbx::af::shared<cctbx::miller::index<> > get_indices() const {
    return indices;
  }
};

class weighted_split_unmerged {
public:
  scitbx::af::shared<double> data_1;
  scitbx::af::shared<double> data_2;
  scitbx::af::shared<double> sigma_1;
  scitbx::af::shared<double> sigma_2;
  scitbx::af::shared<cctbx::miller::index<> > indices;

  weighted_split_unmerged(
    scitbx::af::const_ref<cctbx::miller::index<> > const& unmerged_indices,
    scitbx::af::const_ref<double> const& unmerged_data,
    scitbx::af::const_ref<double> const& unmerged_sigmas,
    bool weighted = true,
    unsigned seed = 0) {
    if (unmerged_indices.size() == 0) return;
    if (seed != 0) gen.seed(seed);
    CCTBX_ASSERT(unmerged_sigmas.all_gt(0.0));
    std::size_t group_begin = 0;
    std::size_t group_end = 1;
    for (; group_end < unmerged_indices.size(); group_end++) {
      if (unmerged_indices[group_end] != unmerged_indices[group_begin]) {
        process_group(group_begin,
                      group_end,
                      unmerged_indices[group_begin],
                      unmerged_data,
                      unmerged_sigmas,
                      weighted);
        group_begin = group_end;
      }
    }
    process_group(group_begin,
                  group_end,
                  unmerged_indices[group_begin],
                  unmerged_data,
                  unmerged_sigmas,
                  weighted);
  }

  ResultsStruct data() {
    return ResultsStruct(data_1, data_2, sigma_1, sigma_2, indices);
  }

protected:
  void process_group(std::size_t group_begin,
                     std::size_t group_end,
                     cctbx::miller::index<> const& current_index,
                     scitbx::af::const_ref<double> const& unmerged_data,
                     scitbx::af::const_ref<double> const& unmerged_sigmas,
                     bool weighted) {
    const std::size_t n = group_end - group_begin;
    if (n < 2) {
      return;
    } else {
      // temp is a copy of the array of intensites of each observation
      std::vector<double> temp(n), temp_w(n);
      for (std::size_t i = 0; i < n; i++) {
        temp[i] = unmerged_data[group_begin + i];
        if (weighted) {
          temp_w[i] =
            1.0 / (unmerged_sigmas[group_begin + i] * unmerged_sigmas[group_begin + i]);
        } else {
          temp_w[i] = 1.0;
        }
      }
      std::size_t nsum = n / 2;
      // actually I (Kay) don't think it matters, and we
      // don't do it in picknofm, but it's like that in the Science paper:
      if (2 * nsum != n && gen.random_double() < 0.5) nsum += 1;
      std::vector<double> i_obs(2, 0.), sum_w(2, 0.), sum_wx2(2, 0.), n_obs(2, 0.),
        v2(2, 0);
      n_obs[0] = nsum;
      n_obs[1] = n - nsum;
      for (std::size_t i = 0; i < nsum; i++) {
        // choose a random index ind from 0 to n-i-1
        const std::size_t ind =
          i + std::min(n - i - 1, std::size_t(gen.random_double() * (n - i)));
        i_obs[0] += temp[ind] * temp_w[ind];
        sum_w[0] += temp_w[ind];
        temp[ind] = temp[i];
        temp_w[ind] = temp_w[i];
      }
      for (std::size_t i = nsum; i < n; i++) {
        i_obs[1] += temp[i] * temp_w[i];
        sum_w[1] += temp_w[i];
      }
      float mu_0 = i_obs[0] / sum_w[0];
      float mu_1 = i_obs[1] / sum_w[1];
      data_1.push_back(mu_0);
      data_2.push_back(mu_1);
      sigma_1.push_back(std::sqrt(1.0 / sum_w[0]));
      sigma_2.push_back(std::sqrt(1.0 / sum_w[1]));
      indices.push_back(current_index);
    }
  }

  scitbx::random::mersenne_twister gen;
};
}  // namespace dials_scaling

#endif  // DIALS_SCALING_SCALING_HELPER_H

/*
 * fill_gaps.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */


#ifndef DIALS_ALGORITHMS_BACKGROUND_GMODEL_FILL_GAPS_H
#define DIALS_ALGORITHMS_BACKGROUND_GMODEL_FILL_GAPS_H

#include <vector>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/panel.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/error.h>

namespace dials { namespace algorithms { namespace background {

  using dxtbx::model::Beam;
  using dxtbx::model::Panel;
  using scitbx::vec2;
  using scitbx::vec3;

  class FillGaps {
  public:

    FillGaps(
        const Beam &beam,
        const Panel &panel)
      : resolution_(
          af::c_grid<2>(
            panel.get_image_size()[1],
            panel.get_image_size()[0])) {
      vec3<double> s0 = beam.get_s0();
      for (std::size_t j = 0; j < resolution_.accessor()[0]; ++j) {
        for (std::size_t i = 0; i < resolution_.accessor()[1]; ++i) {
          vec2<double> px(j+0.5, i+0.5);
          resolution_(j,i) = panel.get_resolution_at_pixel(s0, px);
        }
      }
    }


    void operator()(
        af::ref< double, af::c_grid<2> > data,
        const af::const_ref< bool, af::c_grid<2> > &mask,
        double sigma,
        int kernel_size,
        std::size_t niter) const {

      // Compute an image of sigmas for the gaussian kernel which varies with
      // resolution.
      af::versa< double, af::c_grid<2> > sigma_image(data.accessor());
      for (std::size_t j = 0; j < data.accessor()[0]; ++j) {
        for (std::size_t i = 0; i < data.accessor()[1]; ++i) {
          double d0 = resolution_(j,i);
          double dsum = 0.0;
          double dcnt = 0.0;
          if (j > 0) {
            dsum += std::abs(resolution_(j-1,i) - d0);
            dcnt++;
          }
          if (i > 0) {
            dsum += std::abs(resolution_(j,i-1) - d0);
            dcnt++;
          }
          if (j < data.accessor()[0]-1) {
            dsum += std::abs(resolution_(j+1,i) - d0);
            dcnt++;
          }
          if (i < data.accessor()[1]-1) {
            dsum += std::abs(resolution_(j,i+1) - d0);
            dcnt++;
          }
          sigma_image(j,i) = sigma * dsum / dcnt;
        }
      }

      // Iteratively filter the image
      for (std::size_t iter = 0; iter < niter; ++iter) {
        std::cout << iter << std::endl;
        fill_image(data, mask, sigma_image.const_ref(), kernel_size);
      }
    }

  private:

    void fill_image(
        af::ref< double, af::c_grid<2> > data,
        const af::const_ref< bool, af::c_grid<2> > &mask,
        const af::const_ref< double, af::c_grid<2> > &sigma_image,
        int kernel_size) const {
      int height = (int)data.accessor()[0];
      int width = (int)data.accessor()[1];
      for (std::size_t j = 0; j < data.accessor()[0]; ++j) {
        for (std::size_t i = 0; i < data.accessor()[1]; ++i) {
          if (mask(j,i) == false) {
            int jc = (int)j;
            int ic = (int)i;
            int j0 = std::max(jc - kernel_size, 0);
            int j1 = std::min(jc + kernel_size, height);
            int i0 = std::max(ic - kernel_size, 0);
            int i1 = std::min(ic + kernel_size, width);
            double d0 = resolution_(j,i);
            double kernel_data = 0.0;
            double kernel_sum = 0.0;
            double sigma = sigma_image(j,i);
            for (int jj = j0; jj < j1; ++jj) {
              for (int ii = i0; ii < i1; ++ii) {
                if (jj != j && ii != i) {
                  double d = resolution_(jj,ii);
                  double kernel_value = std::exp(-(d-d0)*(d-d0)/(2.0*sigma*sigma));
                  kernel_data += data(jj,ii) * kernel_value;
                  kernel_sum += kernel_value;
                }
              }
            }
            DIALS_ASSERT(kernel_sum > 0);
            data(j,i) = kernel_data / kernel_sum;
          }
        }
      }
    }

    af::versa< double, af::c_grid<2> > resolution_;
  };

}}}

#endif // DIALS_ALGORITHMS_BACKGROUND_GMODEL_FILL_GAPS_H

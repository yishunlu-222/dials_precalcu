/*
 * ext.cc
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <dials/algorithms/background/gmodel/pixel_filter.h>
#include <dials/algorithms/background/gmodel/fill_gaps.h>

namespace dials { namespace algorithms { namespace background {
  namespace boost_python {

  using namespace boost::python;

  BOOST_PYTHON_MODULE(dials_algorithms_background_gmodel_ext)
  {
    class_<PixelFilterResult>("PixelFilterResult", no_init)
      .def("data", &PixelFilterResult::data)
      .def("mask", &PixelFilterResult::mask)
      ;

    class_<PixelFilter>("PixelFilter", no_init)
      .def(init<std::size_t,std::size_t>())
      .def("add", &PixelFilter::add<double>)
      .def("add", &PixelFilter::add<int>)
      .def("compute", &PixelFilter::compute, (
            arg("min_count")=0,
            arg("nsigma")=6))
      .def("num_images", &PixelFilter::num_images)
      ;

    class_<PolarTransformResult>("PolarTransformResult", no_init)
      .def("data", &PolarTransformResult::data)
      .def("mask", &PolarTransformResult::mask)
      ;

    class_<PolarTransform>("PolarTransform", no_init)
      .def(init<const Beam&, const Panel&>())
      .def("r", &PolarTransform::r)
      .def("a", &PolarTransform::a)
      .def("to_polar", &PolarTransform::to_polar)
      .def("to_cartesian", &PolarTransform::to_cartesian)
      .def("xy", &PolarTransform::xy)
      .def("xy2", &PolarTransform::xy2)
      ;

    class_<FillGaps>("FillGaps", no_init)
      .def(init<const Beam&, const Panel&>())
      .def("__call__", &FillGaps::operator())
      ;

    def("row_median", &row_median);

    def("fill_gaps", &fill_gaps);
  }

}}}} // namespace = dials::algorithms::background::boost_python

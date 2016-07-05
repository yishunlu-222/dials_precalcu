# LIBTBX_SET_DISPATCHER_NAME dials.background

from __future__ import division

import iotbx.phil
import json
import numpy as np
import sys

help_message = '''

Examples::

  dials.extract_powder_data datablock.json strong.pickle

'''

phil_scope = iotbx.phil.parse("""\
bins = 1000
  .type = int
""", process_includes=True)

def main():
  import sys
  run(sys.argv[1:])

def run(args):
#  with open('powder.json', 'r') as fh:
#    powder = json.load(fh)
#  print powder
#
#  means = []
#  variances = []
#  for p in xrange(5, len(powder) - 5):
#    window = powder[(p-5):(p+6)]
#    window = zip(*window)[1]
#    mean = np.mean(window)
#    var = np.var(window)
#    means.append(mean)
#    variances.append(var)
#    print powder[p][1], mean + (var ** 0.5)
#    if powder[p][1] > (mean + 2*(var ** 0.5)):
#      print "**", powder[p][0], (powder[p][1] - mean) / (var**0.5)
#
#  powder = powder[5:-5]
#  with open("powder.dat", 'w') as fh:
#    for j in zip(powder, means, variances):
#      fh.write('%f %f %f %f\n' % (j[0][0], j[0][1], j[1], j[2]))
#
#  import sys
#  sys.exit(1)

  from dials.util.options import OptionParser
  from dials.util.options import flatten_datablocks, flatten_experiments, flatten_reflections
  import libtbx.load_env

  usage = "%s [options] datablock.json reflections.pickle" %(
    libtbx.env.dispatcher_name)

  parser = OptionParser(
    usage=usage,
    phil=phil_scope,
    read_datablocks=True,
    read_reflections=True,
    read_experiments=True,
    epilog=help_message)

  params, options = parser.parse_args(show_diff_phil=True)
  datablocks = flatten_datablocks(params.input.datablock)
  experiments = flatten_experiments(params.input.experiments)
  reflections = flatten_reflections(params.input.reflections)

  if len(datablocks) == 0 or len(reflections) == 0:
    parser.print_help()
    exit()
  assert(len(datablocks) == 1)
  datablock = datablocks[0]

  imagesets = datablock.extract_imagesets()
  assert(len(imagesets) == 1)
  imageset = imagesets[0]

  assert(len(reflections) == 1)
  reflections = reflections[0]
  extract_powder_data(imageset, reflections, params)

def extract_powder_data(imageset, reflections, params):
  detector = imageset.get_detector()
  beam = imageset.get_beam()
  assert(len(detector) == 1)
  detector = detector[0]
  trusted = detector.get_trusted_range()

  from dials.array_family import flex
  from libtbx.phil import parse
  from scitbx import matrix
  import math

  n = matrix.col(detector.get_normal()).normalize()
  s0 = beam.get_s0()
  b = matrix.col(s0).normalize()
  wavelength = beam.get_wavelength()

  if math.fabs(b.dot(n)) < 0.95:
    from libtbx.utils import Sorry
    print 'Detector not perpendicular to beam'

  joint_data = None
  bad = None

  tt = []
  for n in reflections:
    two_theta = detector.get_two_theta_at_pixel(s0, n['xyzobs.px.value'][0:2]) 
    d_spacing = wavelength / (2 * math.asin(0.5 * two_theta))
    two_theta_deg = 180 * two_theta / 3.14159

#    tt.append(two_theta_deg)
    tt.append(d_spacing)

  n_bins = params.bins
  h0 = flex.histogram(flex.double(tt), n_slots=n_bins)

  powder = []
  with open("powder.dat", 'w') as fh:
    for x, c in zip(h0.slot_centers(), h0.slots()):
      print '%8.3f %3d' % (x, c)
      fh.write('%f %d\n' % (x, c))
      powder.append((x, c))
  with open('powder.json', 'w') as fh:
    json.dump(powder, fh, indent=2)


if __name__ == '__main__':
  main()

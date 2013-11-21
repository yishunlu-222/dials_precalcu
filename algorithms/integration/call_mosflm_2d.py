from __future__ import division
from dials.algorithms.integration.mosflm_2D_profile import \
 fit_profile_2d, calc_background_n_make_2d_profile

from dials.model.data import Reflection, ReflectionList
def mosflm_caller(rlist, xmax, ymax, n_div):
  print "performing profile fitting  ...."
  ncol = n_div
  nrow = n_div
  arr_rlist = []

  for col in range(ncol):
    b = []
    for row in range(nrow):
      b.append(ReflectionList())
    arr_rlist.append(b)

  ncnt = 0
  lst_pos = []
  for r in rlist:
    if r.is_valid():
      x, y = r.image_coord_px
      col = int(float(x) / float(xmax) * n_div)
      row = int(float(y) / float(ymax) * n_div)
      arr_rlist[row][col].append(r)
      ncnt += 1
      pos = [row, col, len(arr_rlist[row][col]) - 1]
      lst_pos.append(pos)

  for col in range(ncol):
    for row in range(nrow):
      profile, tr_hold = calc_background_n_make_2d_profile(arr_rlist[row][col])      
      arr_rlist[row][col] = fit_profile_2d(arr_rlist[row][col], profile, tr_hold)

  new_rlist = ReflectionList()
  for numpos in lst_pos:
    row = numpos[0]
    col = numpos[1]
    deep = numpos[2]
    new_rlist.append(arr_rlist[row][col][deep])
  print "profile fitting  .... done"
  return new_rlist

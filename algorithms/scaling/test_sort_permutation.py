from scitbx.array_family import flex

def test_sort_permutation():
  t = flex.int([1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 2, 2, 3, 2, 2, 2, 2,
    3, 3, 2, 1, 2, 1, 1])
  perm = flex.sort_permutation(t, reverse=True, stable=True)
  assert list(t.select(perm)) == [3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  assert list(perm) == [13, 17, 22, 23, 2, 8, 14, 15, 16, 18, 19, 20, 21, 24, 26,
    0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 25, 27, 28]

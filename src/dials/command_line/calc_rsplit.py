from __future__ import annotations

import copy
import logging
import math
import sys

from cctbx import miller
from cctbx.miller import binned_data

from dials.array_family import flex
from dials.util import tabulate
from dials.util.filter_reflections import filtered_arrays_from_experiments_reflections
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files

logger = logging.getLogger(__name__)


def r_split(this, other, assume_index_matching=False, use_binning=False):
    # Used in Boutet et al. (2012), which credit it to Owen et al
    # (2006).  See also R_mrgd_I in Diederichs & Karplus (1997)?
    # Barends cites Collaborative Computational Project Number 4. The
    # CCP4 suite: programs for protein crystallography. Acta
    # Crystallogr. Sect. D-Biol. Crystallogr. 50, 760-763 (1994) and
    # White, T. A. et al. CrystFEL: a software suite for snapshot
    # serial crystallography. J. Appl. Cryst. 45, 335-341 (2012).
    if not use_binning:
        assert other.indices().size() == this.indices().size()
        if this.data().size() == 0:
            return None

        if assume_index_matching:
            (o, c) = (this, other)
        else:
            (o, c) = this.common_sets(other=other, assert_no_singles=True)

        # The case where the denominator is less or equal to zero is
        # pathological and should never arise in practice.
        den = flex.sum(flex.abs(o.data() + c.data()))
        assert den > 0
        return math.sqrt(2) * flex.sum(flex.abs(o.data() - c.data())) / den

    assert this.binner is not None
    results = []
    for i_bin in this.binner().range_all():
        sel = this.binner().selection(i_bin)
        results.append(
            r_split(
                this.select(sel),
                other.select(sel),
                assume_index_matching=assume_index_matching,
                use_binning=False,
            )
        )
    return binned_data(binner=this.binner(), data=results, data_fmt="%7.4f")


from libtbx import phil

phil_scope = phil.parse(
    """
    n_bins = 10
       .type = int
"""
)


def run(args=None, phil: phil.scope = phil_scope):

    parser = ArgumentParser(
        read_experiments=True,
        read_reflections=True,
        phil=phil,
        check_format=False,
        epilog=__doc__,
    )
    params, _ = parser.parse_args(args=args, show_diff_phil=False)

    if not params.input.experiments or not params.input.reflections:
        parser.print_help()
        sys.exit()

    reflections, scaled_expts = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )

    scaled_table = flex.reflection_table.concat(reflections)
    scaled_table = scaled_table.select(
        scaled_table.get_flags(scaled_table.flags.scaled)
    )

    scaled_array = filtered_arrays_from_experiments_reflections(
        scaled_expts,
        [scaled_table],
        outlier_rejection_after_filter=False,
        partiality_threshold=0.4,
    )[0]

    seed = 0

    scaled_array = scaled_array.sort("packed_indices")

    split_datasets = miller.split_unmerged(
        unmerged_indices=scaled_array.indices(),
        unmerged_data=scaled_array.data(),
        unmerged_sigmas=scaled_array.sigmas(),
        seed=seed,
    )

    m1 = miller.array(
        miller_set=miller.set(scaled_array.crystal_symmetry(), split_datasets.indices),
        data=split_datasets.data_1,
    )
    m2 = miller.array(
        miller_set=miller.set(scaled_array.crystal_symmetry(), split_datasets.indices),
        data=split_datasets.data_2,
    )
    print("Calculating r-split")
    m1_copy = copy.deepcopy(m1)
    m2_copy = copy.deepcopy(m2)

    rsplit_overall = r_split(
        m1_copy, m2_copy, assume_index_matching=True, use_binning=False
    )
    print(f"Overall r-split: {rsplit_overall:.5f}")

    m1.setup_binner_counting_sorted(n_bins=params.n_bins)
    m2.setup_binner_counting_sorted(n_bins=params.n_bins)
    data = r_split(m1, m2, assume_index_matching=True, use_binning=True)

    header = ["Resolution range", "r-split"]
    rows = []

    for i_bin, rsplit in zip(list(m1.binner().range_all())[1:-1], data.data[1:-1]):
        d_max_bin, d_min_bin = m1.binner().bin_d_range(i_bin)
        rows.append([f"{d_max_bin:.3f} - {d_min_bin:.3f}", f"{rsplit:.5f}"])
    rows.append(["Overall", f"{rsplit_overall:.5f}"])
    print(tabulate(rows, header))


if __name__ == "__main__":
    run()

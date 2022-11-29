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


def r_split(this, other, assume_index_matching=False, use_binning=False, weighted=True):
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
        if weighted:
            assert len(o.sigmas())
            assert len(c.sigmas())
            joint_var = (o.sigmas() ** 2) + (c.sigmas() ** 2)
            assert joint_var > 0
            den = flex.sum((o.data() + c.data()) / joint_var)
            if den == 0:
                return -1
            return (
                math.sqrt(2.0)
                * flex.sum(flex.abs(o.data() - c.data()) / joint_var)
                / den
            )
        else:
            den = flex.sum(o.data() + c.data())
            if den == 0:
                return -1
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
                weighted=weighted,
            )
        )
    return binned_data(binner=this.binner(), data=results, data_fmt="%7.4f")


def weighted_cchalf(
    this, other, assume_index_matching=False, use_binning=False, weighted=True
):
    if not use_binning:
        assert other.indices().size() == this.indices().size()
        if this.data().size() == 0:
            return None, None

        if assume_index_matching:
            (o, c) = (this, other)
        else:
            (o, c) = this.common_sets(other=other, assert_no_singles=True)

        # The case where the denominator is less or equal to zero is
        # pathological and should never arise in practice.
        if weighted:
            assert len(o.sigmas())
            assert len(c.sigmas())
            n = len(o.data())
            v_o = o.sigmas() ** 2
            v_c = c.sigmas() ** 2
            var_w = v_o + v_c
            joint_w = 1.0 / var_w
            sumjw = flex.sum(joint_w)
            norm_jw = joint_w / sumjw
            xbar = flex.sum(o.data() * norm_jw)
            ybar = flex.sum(c.data() * norm_jw)
            sxy = flex.sum((o.data() - xbar) * (c.data() - ybar) * norm_jw)

            sx = flex.sum((o.data() - xbar) ** 2 * norm_jw)
            sy = flex.sum((c.data() - ybar) ** 2 * norm_jw)
            # what is neff? neff = 1/V2
            V2 = flex.sum(norm_jw**2)

            return (sxy / ((sx * sy) ** 0.5), 1.0 / V2)
        else:
            n = len(o.data())
            xbar = flex.sum(o.data()) / n
            ybar = flex.sum(c.data()) / n
            sxy = flex.sum((o.data() - xbar) * (c.data() - ybar))
            sx = flex.sum((o.data() - xbar) ** 2)
            sy = flex.sum((c.data() - ybar) ** 2)

            return (sxy / ((sx * sy) ** 0.5), n)
    assert this.binner is not None
    results = []
    n_eff = []
    for i_bin in this.binner().range_all():
        sel = this.binner().selection(i_bin)
        cchalf, neff = weighted_cchalf(
            this.select(sel),
            other.select(sel),
            assume_index_matching=assume_index_matching,
            use_binning=False,
            weighted=weighted,
        )
        results.append(cchalf)
        n_eff.append(neff)
    return binned_data(binner=this.binner(), data=results, data_fmt="%7.4f"), n_eff


def compute_cc_significance(r, n, p):
    # https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Testing_using_Student.27s_t-distribution
    if r == -1 or n <= 2:
        significance = False
        critical_value = 0
    else:
        from scitbx.math import distributions

        dist = distributions.students_t_distribution(n - 2)
        t = dist.quantile(1 - p)
        critical_value = t / math.sqrt(n - 2 + t**2)
        significance = r > critical_value
    return significance, critical_value


def compute_cc_significance_levels(cchalfs, neffs, cc_one_half_significance_level=0.01):
    significances = []
    critical_vals = []
    for cc, n in zip(cchalfs, neffs):
        s, c = compute_cc_significance(
            cc, int(math.ceil(n)), cc_one_half_significance_level
        )
        significances.append(s)
        critical_vals.append(c)
    return significances, critical_vals


from libtbx import phil

phil_scope = phil.parse(
    """
    n_bins = 10
       .type = int
    d_min= None
       .type = float
    d_max= None
       .type = float

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

    # scaled_table = flex.reflection_table.concat(reflections)
    scaled_table = reflections[0]
    scaled_table = scaled_table.select(
        scaled_table.get_flags(scaled_table.flags.scaled)
    )
    from dials.algorithms.scaling.combine_intensities import map_indices_to_asu

    scaled_table["miller_index"] = map_indices_to_asu(
        scaled_table["miller_index"], scaled_expts[0].crystal.get_space_group()
    )

    scaled_array = filtered_arrays_from_experiments_reflections(
        scaled_expts,
        [scaled_table],
        outlier_rejection_after_filter=False,
        partiality_threshold=0.4,
    )[0]
    if params.d_min or params.d_max:
        scaled_array = scaled_array.resolution_filter(
            d_min=params.d_min, d_max=params.d_max
        )

    seed = 0

    scaled_array = scaled_array.sort("packed_indices")

    from dials_scaling_ext import weighted_split_unmerged

    data = weighted_split_unmerged(
        unmerged_indices=scaled_array.indices(),
        unmerged_data=scaled_array.data(),
        unmerged_sigmas=scaled_array.sigmas(),
        seed=seed,
    ).data()
    I_1 = data.get_data1()
    I_2 = data.get_data2()
    sig_1 = data.get_sigma1()
    sig_2 = data.get_sigma2()
    indices = data.get_indices()
    # import matplotlib.pyplot as plt
    # plt.errorbar(list(I_1), list(I_2), xerr=list(sig_1), yerr=list(sig_2), fmt='none')
    # plt.show()
    m1 = miller.array(
        miller_set=miller.set(scaled_array.crystal_symmetry(), indices),
        data=I_1,
        sigmas=sig_1,
    )
    m2 = miller.array(
        miller_set=miller.set(scaled_array.crystal_symmetry(), indices),
        data=I_2,
        sigmas=sig_2,
    )
    print("Calculating weighted r-split")
    m1_copy = copy.deepcopy(m1)
    m2_copy = copy.deepcopy(m2)

    rsplit_overall = r_split(
        m1_copy, m2_copy, assume_index_matching=True, use_binning=False, weighted=True
    )
    cchalf_overall, neff = weighted_cchalf(
        m1_copy, m2_copy, assume_index_matching=True, use_binning=False, weighted=True
    )
    print(f"Overall weighted r-split: {rsplit_overall:.5f}")
    print(f"Overall weighted cc-half: {cchalf_overall:.5f}")

    m1.setup_binner(n_bins=params.n_bins)
    m2.setup_binner(n_bins=params.n_bins)
    data = r_split(m1, m2, assume_index_matching=True, use_binning=True)

    header = ["Resolution range", "weighted r-split"]
    rows = []

    for i_bin, rsplit in zip(list(m1.binner().range_all())[1:-1], data.data[1:-1]):
        d_max_bin, d_min_bin = m1.binner().bin_d_range(i_bin)
        rows.append([f"{d_max_bin:.3f} - {d_min_bin:.3f}", f"{rsplit:.5f}"])
    rows.append(["Overall", f"{rsplit_overall:.5f}"])
    print(tabulate(rows, header))
    data, neff = weighted_cchalf(m1, m2, assume_index_matching=True, use_binning=True)

    header = ["Resolution range", "weighted cc-half"]
    rows = []

    for i_bin, rsplit in zip(list(m1.binner().range_all())[1:-1], data.data[1:-1]):
        d_max_bin, d_min_bin = m1.binner().bin_d_range(i_bin)
        rows.append([f"{d_max_bin:.3f} - {d_min_bin:.3f}", f"{rsplit:.5f}"])
    rows.append(["Overall", f"{cchalf_overall:.5f}"])
    print(tabulate(rows, header))

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
        m1_copy, m2_copy, assume_index_matching=True, use_binning=False, weighted=False
    )
    print(f"Overall r-split: {rsplit_overall:.5f}")
    cchalf_overall, neff = weighted_cchalf(
        m1_copy, m2_copy, assume_index_matching=True, use_binning=False, weighted=False
    )
    print(f"Overall cc-half: {cchalf_overall:.5f}")

    m1.setup_binner(n_bins=params.n_bins)
    m2.setup_binner(n_bins=params.n_bins)
    data = r_split(m1, m2, assume_index_matching=True, use_binning=True, weighted=False)

    header = ["Resolution range", "r-split"]
    rows = []

    for i_bin, rsplit in zip(list(m1.binner().range_all())[1:-1], data.data[1:-1]):
        d_max_bin, d_min_bin = m1.binner().bin_d_range(i_bin)
        rows.append([f"{d_max_bin:.3f} - {d_min_bin:.3f}", f"{rsplit:.5f}"])
    rows.append(["Overall", f"{rsplit_overall:.5f}"])
    print(tabulate(rows, header))
    data, neff = weighted_cchalf(
        m1, m2, assume_index_matching=True, use_binning=True, weighted=False
    )

    header = ["Resolution range", "cc-half"]
    rows = []

    for i_bin, rsplit in zip(list(m1.binner().range_all())[1:-1], data.data[1:-1]):
        d_max_bin, d_min_bin = m1.binner().bin_d_range(i_bin)
        rows.append([f"{d_max_bin:.3f} - {d_min_bin:.3f}", f"{rsplit:.5f}"])
    rows.append(["Overall", f"{cchalf_overall:.5f}"])
    print(tabulate(rows, header))


if __name__ == "__main__":
    run()

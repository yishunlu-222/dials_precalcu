"""
Module of library functions, to perform core scaling operations on reflection
tables and experiment lists. Some functions, such as create_scaling_model and
merging statistics calculations are called from the main dials.scale script,
whereas others are provided as library functions for calling from custom
scripts. The functions defined here should ideally only require reflection
tables and ExperimentList objects (and sometimes phil_scope objects if
necessary), and return common dials objects such as reflection tables and
ExperimentLists.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pkg_resources

import iotbx.merging_statistics
from cctbx import crystal, miller, uctbx
from dxtbx.model import Experiment
from dxtbx.util import ersatz_uuid4
from libtbx import Auto, phil

from dials.algorithms.scaling.Ih_table import IhTable
from dials.algorithms.scaling.model.model import KBScalingModel, PhysicalScalingModel
from dials.algorithms.scaling.scaling_utilities import (
    DialsMergingStatisticsError,
    calculate_prescaling_correction,
)
from dials.array_family import flex
from dials.util import Sorry
from dials.util.options import ArgumentParser
from dials.util.reference import intensities_from_reference_file

logger = logging.getLogger("dials")


def set_image_ranges_in_scaling_models(experiments):
    """Set the batch range in scaling models if not already set."""
    for exp in experiments:
        if exp.scan:
            valid_image_ranges = exp.scan.get_valid_image_ranges(exp.identifier)
            if "valid_image_range" not in exp.scaling_model.configdict:
                # only set if not currently set i.e. set initial
                exp.scaling_model.set_valid_image_range(exp.scan.get_image_range())
            if exp.scaling_model.configdict["valid_image_range"] != [
                valid_image_ranges[0][0],
                valid_image_ranges[-1][1],
            ]:
                # first and last values in whole list of tuples
                exp.scaling_model.limit_image_range(
                    (valid_image_ranges[0][0], valid_image_ranges[-1][1])
                )
    return experiments


def choose_initial_scaling_intensities(reflection_table, intensity_choice="profile"):
    """Choose which intensities to initially use for scaling. The LP, QE and
    partiality corrections are also applied. Two new columns are
    added to the reflection table 'intensity' and 'variance', which have
    all corrections applied except an inverse scale factor."""
    if intensity_choice == "profile":
        intensity_choice = "prf"  # rename to allow string matching with refl table
    if "intensity.prf.value" not in reflection_table:
        intensity_choice = "sum"
    elif intensity_choice == "prf":
        if (
            reflection_table.get_flags(reflection_table.flags.integrated_prf).count(
                True
            )
            == 0
        ):
            logger.warning(
                "No profile fitted reflections in this dataset, using summation intensities"
            )
            intensity_choice = "sum"
    reflection_table = calculate_prescaling_correction(reflection_table)
    conv = reflection_table["prescaling_correction"]
    # if prf/sum, use those. If combine, use prf else sum for each refl.
    if intensity_choice == "prf":
        reflection_table["intensity"] = reflection_table["intensity.prf.value"] * conv
        reflection_table["variance"] = (
            reflection_table["intensity.prf.variance"] * conv * conv
        )
    else:
        # first fill in summation intensities.
        if "partiality" in reflection_table:
            inverse_partiality = flex.double(reflection_table.size(), 1.0)
            nonzero_partiality_sel = reflection_table["partiality"] > 0.0
            good_refl = reflection_table.select(reflection_table["partiality"] > 0.0)
            inverse_partiality.set_selected(
                nonzero_partiality_sel.iselection(), 1.0 / good_refl["partiality"]
            )
            reflection_table["intensity"] = (
                reflection_table["intensity.sum.value"] * conv * inverse_partiality
            )
            reflection_table["variance"] = reflection_table[
                "intensity.sum.variance"
            ] * flex.pow2(conv * inverse_partiality)
            if "partiality.inv.variance" in reflection_table:
                reflection_table["variance"] += (
                    reflection_table["intensity.sum.value"]
                    * conv
                    * reflection_table["partiality.inv.variance"]
                )
        else:
            reflection_table["intensity"] = (
                reflection_table["intensity.sum.value"] * conv
            )
            reflection_table["variance"] = (
                reflection_table["intensity.sum.variance"] * conv * conv
            )
        if intensity_choice == "combine":
            # now overwrite prf if we have it.
            sel = reflection_table.get_flags(reflection_table.flags.integrated_prf)
            isel = sel.iselection()
            Iprf = (reflection_table["intensity.prf.value"] * conv).select(sel)
            Vprf = (reflection_table["intensity.prf.variance"] * conv * conv).select(
                sel
            )
            reflection_table["intensity"].set_selected(isel, Iprf)
            reflection_table["variance"].set_selected(isel, Vprf)
    variance_mask = reflection_table["variance"] <= 0.0
    reflection_table.set_flags(
        variance_mask, reflection_table.flags.excluded_for_scaling
    )
    return reflection_table


def scale_against_target(
    reflection_table,
    experiment,
    target_reflection_table,
    target_experiment,
    params=None,
    model="KB",
):
    """Determine scale factors for a single dataset, by scaling against a target
    reflection table. Requires a single reflection table for the reflections to
    scale and the target dataset, and an ExperimentList for both datasets. The
    params option can also be specified, if None then the default scaling
    configuration is used. The scaling model can be specified individually.

    Returns the reflection table, with added columns 'inverse_scale_factor' and
    'inverse_scale_factor_variance'."""

    if not params:
        phil_scope = phil.parse(
            """
      include scope dials.algorithms.scaling.scaling_options.phil_scope
      include scope dials.algorithms.scaling.model.model.model_phil_scope
      include scope dials.algorithms.scaling.scaling_refiner.scaling_refinery_phil_scope
    """,
            process_includes=True,
        )
        parser = ArgumentParser(phil=phil_scope, check_format=False)
        params, _ = parser.parse_args(args=[], quick_parse=True)
        params.model = model

    from dials.algorithms.scaling.scaler_factory import TargetScalerFactory

    reflections = [reflection_table, target_reflection_table]
    experiment.append(target_experiment[0])
    experiments = create_scaling_model(params, experiment, reflections)
    experiments[-1].scaling_model.set_scaling_model_as_scaled()
    scaler = TargetScalerFactory.create(params, experiments, reflections)
    scaler.perform_scaling()
    scaler.expand_scales_to_all_reflections(calc_cov=True)
    return scaler.unscaled_scalers[0].reflection_table


def scale_single_dataset(reflection_table, experiment, params=None, model="physical"):
    """Determine scale factors for a single dataset. Requires a reflection table
    and an ExperimentList with a single experiment. A custom params option can be
    specified, if not the default scaling params option will be used, with default
    configuration options. The model can be individually specified.

    Returns the reflection table, with added columns 'inverse_scale_factor' and
    'inverse_scale_factor_variance'."""

    if not params:
        phil_scope = phil.parse(
            """
      include scope dials.algorithms.scaling.model.model.model_phil_scope
      include scope dials.algorithms.scaling.scaling_options.phil_scope
      include scope dials.algorithms.scaling.scaling_refiner.scaling_refinery_phil_scope
    """,
            process_includes=True,
        )
        parser = ArgumentParser(phil=phil_scope, check_format=False)
        params, _ = parser.parse_args(args=[], quick_parse=True)

    params.model = model

    from dials.algorithms.scaling.scaler_factory import SingleScalerFactory

    experiments = create_scaling_model(params, experiment, [reflection_table])
    scaler = SingleScalerFactory.create(params, experiments[0], reflection_table)
    from dials.algorithms.scaling.algorithm import scaling_algorithm

    scaler = scaling_algorithm(scaler)
    return scaler.reflection_table


def create_scaling_model(params, experiments, reflections):
    """Loop through the experiments, creating the scaling models."""
    autos = [None, Auto, "auto", "Auto"]
    use_auto_model = params.model in autos
    # Determine non-auto model to use outside the loop over datasets.
    if not use_auto_model:
        model_class = None
        for entry_point in pkg_resources.iter_entry_points("dxtbx.scaling_model_ext"):
            if entry_point.name == params.model:
                model_class = entry_point.load()
                break
        if not model_class:
            raise ValueError(f"Unable to create scaling model of type {params.model}")

    for expt, refl in zip(experiments, reflections):
        if not expt.scaling_model or params.overwrite_existing_models:
            # need to make a new model
            if use_auto_model:
                if not expt.scan:
                    model = KBScalingModel
                else:  # set model as physical unless scan < 1.0 degree
                    osc_range = expt.scan.get_oscillation_range()
                    abs_osc_range = abs(osc_range[1] - osc_range[0])
                    if abs_osc_range < 1.0:
                        model = KBScalingModel
                    else:
                        model = PhysicalScalingModel
            else:
                model = model_class
            expt.scaling_model = model.from_data(params, expt, refl)
        else:
            # allow for updating of an existing model.
            expt.scaling_model.update(params)
    return experiments


def create_Ih_table(
    experiments, reflections, selections=None, n_blocks=1, anomalous=False
):
    """Create an Ih table from a list of experiments and reflections. Optionally,
    a selection list can also be given, to select data from each reflection table.
    Allow an unequal number of experiments and reflections, as only need to
    extract one space group value (can optionally check all same if many)."""
    if selections:
        assert len(selections) == len(
            reflections
        ), """Must have an equal number of
    reflection tables and selections in the input lists."""
    space_group_0 = experiments[0].crystal.get_space_group()
    for experiment in experiments:
        assert (
            experiment.crystal.get_space_group() == space_group_0
        ), """The space
    groups of all experiments must be equal."""
    input_tables = []
    indices_lists = []
    for i, reflection in enumerate(reflections):
        if "inverse_scale_factor" not in reflection:
            reflection["inverse_scale_factor"] = flex.double(reflection.size(), 1.0)
        if selections:
            input_tables.append(reflection.select(selections[i]))
            indices_lists.append(selections[i].iselection())
        else:
            input_tables.append(reflection)
            indices_lists = None
    Ih_table = IhTable(
        input_tables,
        space_group_0,
        indices_lists,
        nblocks=n_blocks,
        anomalous=anomalous,
    )
    return Ih_table


def scaled_data_as_miller_array(
    reflection_table_list,
    experiments,
    best_unit_cell=None,
    anomalous_flag=False,
    wavelength=None,
):
    """Get a scaled miller array from an experiment and reflection table."""
    if len(reflection_table_list) > 1:
        joint_table = flex.reflection_table()
        for reflection_table in reflection_table_list:
            # better to just create many miller arrays and join them?
            refl_for_joint_table = flex.reflection_table()
            for col in [
                "miller_index",
                "intensity.scale.value",
                "inverse_scale_factor",
                "intensity.scale.variance",
            ]:
                refl_for_joint_table[col] = reflection_table[col]
            good_refl_sel = ~reflection_table.get_flags(
                reflection_table.flags.bad_for_scaling, all=False
            )
            refl_for_joint_table = refl_for_joint_table.select(good_refl_sel)
            joint_table.extend(refl_for_joint_table)
    else:
        reflection_table = reflection_table_list[0]
        good_refl_sel = ~reflection_table.get_flags(
            reflection_table.flags.bad_for_scaling, all=False
        )
        joint_table = reflection_table.select(good_refl_sel)
    # Filter out negative scale factors to avoid merging statistics errors.
    # These are not removed from the output data, as it is likely one would
    # want to do further analysis e.g. delta cc1/2 and rescaling, to exclude
    # certain data and get better scale factors for all reflections.
    pos_scales = joint_table["inverse_scale_factor"] > 0
    if pos_scales.count(False) > 0:
        logger.info(
            """There are %s reflections with non-positive scale factors which
will not be used for calculating merging statistics""",
            pos_scales.count(False),
        )
        joint_table = joint_table.select(pos_scales)

    if best_unit_cell is None:
        best_unit_cell = determine_best_unit_cell(experiments)
    miller_set = miller.set(
        crystal_symmetry=crystal.symmetry(
            unit_cell=best_unit_cell,
            space_group=experiments[0].crystal.get_space_group(),
            assert_is_compatible_unit_cell=False,
        ),
        indices=joint_table["miller_index"],
        anomalous_flag=anomalous_flag,
    )
    i_obs = miller.array(
        miller_set,
        data=joint_table["intensity.scale.value"] / joint_table["inverse_scale_factor"],
    )
    i_obs.set_observation_type_xray_intensity()
    i_obs.set_sigmas(
        flex.sqrt(joint_table["intensity.scale.variance"])
        / joint_table["inverse_scale_factor"]
    )
    if not wavelength:
        wavelength = np.mean([expt.beam.get_wavelength() for expt in experiments])
    i_obs.set_info(
        miller.array_info(
            source="DIALS",
            source_type="reflection_tables",
            wavelength=wavelength,
        )
    )
    return i_obs


def determine_best_unit_cell(experiments):
    """Set the median unit cell as the best cell, for consistent d-values across
    experiments."""
    uc_params = [flex.double() for i in range(6)]
    for exp in experiments:
        unit_cell = (
            exp.crystal.get_recalculated_unit_cell() or exp.crystal.get_unit_cell()
        )
        for i, p in enumerate(unit_cell.parameters()):
            uc_params[i].append(p)
    best_unit_cell = uctbx.unit_cell(parameters=[flex.median(p) for p in uc_params])
    if len(experiments) > 1:
        logger.info("Using median unit cell across experiments : %s", best_unit_cell)
    return best_unit_cell


class ExtendedDatasetStatistics(iotbx.merging_statistics.dataset_statistics):
    def __init__(self, *args, stills=False, weighted=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighted_r_split = None
        self.unweighted_r_split = None
        self.weighted_cc_half = None
        self.weighted_r_split_binned = None
        self.unweighted_r_split_binned = None
        self.weighted_cc_half_binned = None
        if not (stills or weighted):
            return
        i_obs = kwargs.get("i_obs")
        n_bins = kwargs.get("n_bins", 20)
        if not i_obs:
            return
        i_obs = i_obs.map_to_asu()
        i_obs = i_obs.sort("packed_indices")
        from dials_scaling_ext import weighted_split_unmerged

        seed = 0
        data = weighted_split_unmerged(
            unmerged_indices=i_obs.indices(),
            unmerged_data=i_obs.data(),
            unmerged_sigmas=i_obs.sigmas(),
            seed=seed,
        ).data()
        I_1 = data.get_data1()
        I_2 = data.get_data2()
        sig_1 = data.get_sigma1()
        sig_2 = data.get_sigma2()
        indices = data.get_indices()
        m1 = miller.array(
            miller_set=miller.set(i_obs.crystal_symmetry(), indices),
            data=I_1,
            sigmas=sig_1,
        )
        m2 = miller.array(
            miller_set=miller.set(i_obs.crystal_symmetry(), indices),
            data=I_2,
            sigmas=sig_2,
        )
        m1.setup_binner(n_bins=n_bins)
        m2.setup_binner(n_bins=n_bins)
        self.weighted_binner = m1.binner()
        if stills:
            # now use i_obs to calc rsplit
            from dials.command_line.calc_rsplit import r_split

            self.weighted_r_split = r_split(
                m1, m2, assume_index_matching=True, use_binning=False, weighted=True
            )
            self.unweighted_r_split = r_split(
                m1, m2, assume_index_matching=True, use_binning=False, weighted=False
            )
            self.weighted_r_split_binned = r_split(
                m1, m2, assume_index_matching=True, use_binning=True, weighted=True
            ).data[1:-1]
            self.unweighted_r_split_binned = r_split(
                m1, m2, assume_index_matching=True, use_binning=True, weighted=False
            ).data[1:-1]
        if weighted:
            from dials.command_line.calc_rsplit import weighted_cchalf

            self.weighted_cc_half = weighted_cchalf(
                m1, m2, assume_index_matching=True, use_binning=False, weighted=True
            )
            self.weighted_cc_half_binned = weighted_cchalf(
                m1, m2, assume_index_matching=True, use_binning=True, weighted=True
            ).data[1:-1]


def merging_stats_from_scaled_array(
    scaled_miller_array,
    n_bins=20,
    use_internal_variance=False,
    anomalous=True,
    extra_stats=False,
):
    """Calculate the normal and anomalous merging statistics."""

    if scaled_miller_array.is_unique_set_under_symmetry():
        raise DialsMergingStatisticsError(
            "Dataset contains no equivalent reflections, merging statistics "
            "cannot be calculated."
        )
    stills, weighted = False, False
    if extra_stats:
        stills, weighted = True, True
    try:
        result = ExtendedDatasetStatistics(
            i_obs=scaled_miller_array,
            n_bins=n_bins,
            anomalous=False,
            sigma_filtering=None,
            eliminate_sys_absent=False,
            use_internal_variance=use_internal_variance,
            cc_one_half_significance_level=0.01,
            stills=stills,
            weighted=weighted,
        )
    except (RuntimeError, Sorry) as e:
        raise DialsMergingStatisticsError(
            f"Error encountered during merging statistics calculation:\n{e}"
        )

    anom_result = None

    if anomalous:
        intensities_anom = scaled_miller_array.as_anomalous_array()
        intensities_anom = intensities_anom.map_to_asu().customized_copy(
            info=scaled_miller_array.info()
        )
        if intensities_anom.is_unique_set_under_symmetry():
            logger.warning(
                "Anomalous dataset contains no equivalent reflections, anomalous "
                "merging statistics cannot be calculated."
            )
        else:
            try:
                anom_result = ExtendedDatasetStatistics(
                    i_obs=intensities_anom,
                    n_bins=n_bins,
                    anomalous=True,
                    sigma_filtering=None,
                    cc_one_half_significance_level=0.01,
                    eliminate_sys_absent=False,
                    use_internal_variance=use_internal_variance,
                )
            except (RuntimeError, Sorry) as e:
                logger.warning(
                    "Error encountered during anomalous merging statistics "
                    "calculation:\n%s",
                    e,
                    exc_info=True,
                )
    return result, anom_result


def create_datastructures_for_reference_file(
    experiments, reference_file, anomalous=True, d_min=2.0
):
    # If the file is a model file, then d_min is used to determine the highest
    # resolution calculated intensities.
    wavelength = np.mean([expt.beam.get_wavelength() for expt in experiments])
    intensities = intensities_from_reference_file(reference_file, d_min, wavelength)
    if not anomalous:
        intensities = intensities.as_non_anomalous_array().merge_equivalents().array()

    table = flex.reflection_table()
    table["intensity"] = intensities.data()
    table["miller_index"] = intensities.indices()
    table["id"] = flex.int(table.size(), len(experiments))
    table["d"] = intensities.d_spacings().data()

    if intensities.sigmas():  # only the case for a data file
        table["variance"] = intensities.sigmas() ** 2
        logger.info(f"Extracted {table.size()} intensities from reference file")
        table = table.select(table["variance"] > 0.0)
        if table.size() == 0:
            raise ValueError(
                "No reflections with positive sigma remain after filtering"
            )

    table.set_flags(flex.bool(table.size(), False), table.flags.bad_for_scaling)
    table.set_flags(flex.bool(table.size(), True), table.flags.integrated)

    expt = Experiment()
    expt.crystal = deepcopy(experiments[0].crystal)
    params = Mock()
    params.KB.decay_correction.return_value = False
    expt.scaling_model = KBScalingModel.from_data(params, [], [])
    expt.scaling_model.set_scaling_model_as_scaled()  # Set as scaled to fix scale.
    expt.identifier = ersatz_uuid4()

    table.experiment_identifiers()[len(experiments)] = expt.identifier

    table.experiment_identifiers()[len(experiments)] = expt.identifier

    return expt, table


def create_datastructures_for_target_mtz(experiments, mtz_file, anomalous=True):
    """
    Read a merged mtz file and extract miller indices, intensities and variances.
    Deprecated, retained for backwards compability.
    """
    return create_datastructures_for_reference_file(experiments, mtz_file, anomalous)


def create_datastructures_for_structural_model(
    experiments, model_file, anomalous=True, d_min=2.0
):
    """Read a cif/pdb file, calculate intensities. Return an experiment and
    reflection table to be used for the structural model in scaling."""
    return create_datastructures_for_reference_file(
        experiments, model_file, anomalous, d_min
    )

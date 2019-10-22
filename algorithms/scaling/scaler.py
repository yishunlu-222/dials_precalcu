"""
This module defines classes which implement the stages of the scaling algorithm.

These 'scalers' act to initialise and connect various parts of the scaling
algorithm and datastructures such as the Ih_table etc, and
present a united interface to the main scaling algorithm for single, multi
and targeted scaling.

The SingleScaler is defined, for scaling of a single dataset, a MultiScaler is
defined for scaling multiple datasets simultaneously and a TargetScaler is
defined for targeted scaling.
"""
from __future__ import absolute_import, division, print_function

import copy
import logging
import time
from six.moves import cStringIO as StringIO
from dials.array_family import flex
from dials.algorithms.scaling.basis_functions import RefinerCalculator
from dials.algorithms.scaling.outlier_rejection import determine_outlier_index_arrays
from dials.algorithms.scaling.Ih_table import IhTable
from dials.algorithms.scaling.target_function import ScalingTarget, ScalingTargetFixedIH
from dials.algorithms.scaling.scaling_refiner import (
    scaling_refinery,
    error_model_refinery,
)
from dials.algorithms.scaling.error_model.error_model import get_error_model
from dials.algorithms.scaling.error_model.error_model_target import ErrorModelTarget
from dials.algorithms.scaling.parameter_handler import ScalingParameterManagerGenerator
from dials.algorithms.scaling.scaling_utilities import (
    log_memory_usage,
    DialsMergingStatisticsError,
)
from dials.algorithms.scaling.scaling_library import (
    scaled_data_as_miller_array,
    merging_stats_from_scaled_array,
)
from dials.algorithms.scaling.combine_intensities import (
    SingleDatasetIntensityCombiner,
    MultiDatasetIntensityCombiner,
)
from dials.algorithms.scaling.reflection_selection import (
    calculate_scaling_subset_connected,
    calculate_scaling_subset_ranges_with_E2,
    calculate_scaling_subset_ranges,
    select_connected_reflections_across_datasets,
)
from dials.util.observer import Subject
from libtbx.table_utils import simple_table
from scitbx import sparse

logger = logging.getLogger("dials")


class MultiScalerBase(Subject):

    """Base class for Scaler to handle multiple datasets."""

    def __init__(self, single_scalers):
        """Define the properties of a scaler."""
        super(MultiScalerBase, self).__init__(
            events=[
                "performed_scaling",
                "performed_error_analysis",
                "performed_outlier_rejection",
            ]
        )
        self.single_scalers = single_scalers
        self._params = single_scalers[0].params
        self._Ih_table = None
        self._global_Ih_table = None
        self._free_Ih_table = None
        self._work_free_stats = []
        self._removed_datasets = []
        self._error_model = None
        self._active_scalers = []

    @property
    def active_scalers(self):
        """A list of scalers that are currently being used in the algorithm."""
        return self._active_scalers

    @property
    def error_model(self):
        """The error model minimised for the combined dataset."""
        return self._error_model

    @property
    def removed_datasets(self):
        """The list of removed datasets."""
        return self._removed_datasets

    @property
    def work_free_stats(self):
        """Holder for work/free set statistics."""
        return self._work_free_stats

    @property
    def Ih_table(self):
        """The Ih_table datastructure for use in minimisation."""
        return self._Ih_table

    @property
    def global_Ih_table(self):
        """
        An Ih_table datastructure containing all suitable reflections.

        This includes reflections across all datasets being minimised, and there
        should only be one instance, maintained by the highest level scaler, e.g.
        a multiscaler in a multi-dataset case.
        """
        return self._global_Ih_table

    @property
    def params(self):
        """The params phil scope."""
        return self._params

    ### Interface for scaling refiner

    def update_for_minimisation(self, apm, block_id):
        """Update the scale factors and Ih for the next iteration of minimisation."""
        self._update_for_minimisation(apm, block_id, calc_Ih=True)

    def _update_for_minimisation(self, apm, block_id, calc_Ih=True):
        scales = flex.double([])
        derivs = []
        for apm_i in apm.apm_list:
            scales_i, derivs_i = RefinerCalculator.calculate_scales_and_derivatives(
                apm_i, block_id
            )
            scales.extend(scales_i)
            derivs.append(derivs_i)
        deriv_matrix = sparse.matrix(scales.size(), apm.n_active_params)
        start_row_no = 0
        for j, deriv in enumerate(derivs):
            deriv_matrix.assign_block(deriv, start_row_no, apm.apm_data[j]["start_idx"])
            start_row_no += deriv.n_rows
        self.Ih_table.set_inverse_scale_factors(scales, block_id)
        self.Ih_table.set_derivatives(deriv_matrix, block_id)
        self.Ih_table.update_weights(block_id)
        if calc_Ih:
            self.Ih_table.calc_Ih(block_id)

    def get_blocks_for_minimisation(self):
        """Return the blocks to iterate over during refinement."""
        return self.Ih_table.blocked_data_list

    ## Interface for algorithms using the scaler

    @Subject.notify_event(event="performed_outlier_rejection")
    def round_of_outlier_rejection(self):
        """Perform a round of outlier rejection across all datasets."""
        self._round_of_outlier_rejection(target=None)

    def _round_of_outlier_rejection(self, target=None):
        """
        Perform a round of outlier rejection across all datasets.

        After identifying outliers, set the outliers property in individual scalers.
        """
        if self.params.scaling_options.outlier_rejection:
            outlier_index_arrays = determine_outlier_index_arrays(
                self.global_Ih_table,
                self.params.scaling_options.outlier_rejection,
                self.params.scaling_options.outlier_zmax,
                target=target,
            )
            for outlier_indices, scaler in zip(
                outlier_index_arrays, self.active_scalers
            ):
                scaler.outliers = flex.bool(scaler.n_suitable_refl, False)
                scaler.outliers.set_selected(outlier_indices, True)
            if self._free_Ih_table:
                free_outlier_index_arrays = determine_outlier_index_arrays(
                    self._free_Ih_table,
                    self.params.scaling_options.outlier_rejection,
                    self.params.scaling_options.outlier_zmax,
                    target=target,
                )
                for outlier_indices, scaler in zip(
                    free_outlier_index_arrays, self.active_scalers
                ):
                    scaler.outliers.set_selected(outlier_indices, True)
        logger.debug("Finished outlier rejection.")
        log_memory_usage()

    def expand_scales_to_all_reflections(self, calc_cov=False):
        """
        Calculate scale factors for all suitable reflections in the datasets.

        After the scale factors are updated, the global_Ih_table is updated also.
        """
        if calc_cov:
            logger.info("Calculating error estimates of inverse scale factors. \n")
        for i, scaler in enumerate(self.active_scalers):
            scaler.expand_scales_to_all_reflections(calc_cov=calc_cov)
            # now update global Ih table
            self.global_Ih_table.update_data_in_blocks(
                scaler.reflection_table["inverse_scale_factor"].select(
                    scaler.suitable_refl_for_scaling_sel
                ),
                dataset_id=i,
                column="inverse_scale_factor",
            )
            if self._free_Ih_table:
                self._free_Ih_table.update_data_in_blocks(
                    scaler.reflection_table["inverse_scale_factor"].select(
                        scaler.suitable_refl_for_scaling_sel
                    ),
                    dataset_id=i,
                    column="inverse_scale_factor",
                )
        self.global_Ih_table.calc_Ih()
        if self._free_Ih_table:
            self._free_Ih_table.calc_Ih()
        logger.info(
            "Scale factors determined during minimisation have now been\n"
            "applied to all datasets.\n"
        )

    def combine_intensities(self):
        """Combine reflection intensities, either jointly or separately."""
        multicombiner = None
        if self.params.reflection_selection.combine.joint_analysis:
            logger.info(
                "Performing multi-dataset profile/summation intensity optimisation."
            )
            try:
                multicombiner = MultiDatasetIntensityCombiner(self)
            except DialsMergingStatisticsError as e:
                logger.info("Intensity combination failed with the error %s", e)
        for j, scaler in enumerate(self.active_scalers):
            if not multicombiner:  # i.e not joint_analysis or if this failed
                singlecombiner = None
                try:
                    singlecombiner = SingleDatasetIntensityCombiner(scaler)
                except DialsMergingStatisticsError as e:
                    logger.info(
                        "Intensity combination failed for dataset %s with the error %s",
                        j,
                        e,
                    )
            if multicombiner:
                I, var = multicombiner.calculate_suitable_combined_intensities(j)
                Imid = multicombiner.max_key
            elif singlecombiner:
                I, var = singlecombiner.calculate_suitable_combined_intensities()
                Imid = singlecombiner.max_key
            scaler.experiment.scaling_model.record_intensity_combination_Imid(Imid)

            isel = scaler.suitable_refl_for_scaling_sel.iselection()
            scaler.reflection_table["intensity"].set_selected(isel, I)
            scaler.reflection_table["variance"].set_selected(isel, var)
            self.global_Ih_table.update_data_in_blocks(I, j, "intensity")
            self.global_Ih_table.update_data_in_blocks(var, j, "variance")
            if self._free_Ih_table:
                self._free_Ih_table.update_data_in_blocks(I, j, column="intensity")
                self._free_Ih_table.update_data_in_blocks(var, j, column="variance")
        self.global_Ih_table.calc_Ih()
        if self._free_Ih_table:
            self._free_Ih_table.calc_Ih()

    def make_ready_for_scaling(self, outlier=True):
        """
        Prepare the datastructures for a round of scaling.

        Update the scaling selection, create a new Ih_table and update the model
        data ready for minimisation. Also check to see if any datasets should be
        removed.
        """
        datasets_to_remove = []
        for i, scaler in enumerate(self.active_scalers):
            if outlier:
                scaler.scaling_selection = scaler.scaling_subset_sel & ~scaler.outliers
            else:
                scaler.scaling_selection = copy.deepcopy(scaler.scaling_subset_sel)
            if scaler.scaling_selection.count(True) == 0:
                datasets_to_remove.append(i)
        if datasets_to_remove:
            self._remove_datasets(self.active_scalers, datasets_to_remove)
        self._create_Ih_table()
        self._update_model_data()

    @Subject.notify_event(event="performed_scaling")
    def perform_scaling(self, engine=None, max_iterations=None, tolerance=None):
        """Minimise the scaling model."""
        self._perform_scaling(
            target_type=ScalingTarget,
            engine=engine,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    def _perform_scaling(
        self, target_type, engine=None, max_iterations=None, tolerance=None
    ):
        pmg = ScalingParameterManagerGenerator(
            self.active_scalers, self.params.scaling_refinery.refinement_order
        )
        for apm in pmg.parameter_managers():
            if not engine:
                engine = self.params.scaling_refinery.engine
            if not max_iterations:
                max_iterations = self.params.scaling_refinery.max_iterations
            st = time.time()
            refinery = scaling_refinery(
                engine=engine,
                scaler=self,
                target=target_type(),
                prediction_parameterisation=apm,
                max_iterations=max_iterations,
            )
            if tolerance:
                refinery.set_tolerance(tolerance)
            try:
                refinery.run()
            except RuntimeError as e:
                logger.error(e, exc_info=True)
            logger.info("Time taken for refinement %s", (time.time() - st))
            refinery.print_step_table()
            self._update_after_minimisation(apm)
            logger.info("\n" + "=" * 80 + "\n")

    @Subject.notify_event(event="performed_error_analysis")
    def perform_error_optimisation(self, update_Ih=True):
        """Perform an optimisation of the sigma values."""
        Ih_table = self.global_Ih_table
        Ih_table.reset_error_model()
        Ih_table.calc_Ih()
        error_model = get_error_model(self.params.weighting.error_model.error_model)
        try:
            logger.info("Performing a round of error model refinement.")
            refinery = error_model_refinery(
                engine="SimpleLBFGS",
                target=ErrorModelTarget(
                    error_model(
                        Ih_table.blocked_data_list[0],
                        self.params.weighting.error_model.n_bins,
                        self.params.weighting.error_model.min_Ih,
                        self.params.reflection_selection.min_partiality,
                    )
                ),
                max_iterations=100,
            )
            refinery.run()
        except (RuntimeError, ValueError) as e:
            logger.error(e, exc_info=True)
        else:
            error_model = refinery.return_error_model()
            logger.info(error_model)
            logger.info(error_model.minimisation_summary())
            self._update_error_model(error_model, update_Ih=update_Ih)
        return error_model

    def clear_Ih_table(self):
        """Delete the data from the current Ih_table, to free memory."""
        self._Ih_table = []

    def fix_initial_parameter(self):
        """Fix the initial parameter in the first suitable scaler"""
        for scaler in self.active_scalers:
            fixed = scaler.experiment.scaling_model.fix_initial_parameter(self.params)
            if fixed:
                return fixed

    def prepare_reflection_tables_for_output(self):
        """Finish adjust reflection table data at the end of the algorithm."""
        # First adjust variances
        for scaler in self.active_scalers:
            if self.error_model:
                scaler.reflection_table["variance"] = self.error_model.update_variances(
                    scaler.reflection_table["variance"],
                    scaler.reflection_table["intensity"],
                )
            # now increase the errors slightly to take into account the uncertainty in the
            # inverse scale factors
            if (
                scaler.var_cov_matrix.non_zeroes > 0
            ):  # parameters errors have been determined
                fractional_error = (
                    scaler.reflection_table["inverse_scale_factor_variance"] ** 0.5
                    / scaler.reflection_table["inverse_scale_factor"]
                )
                variance_scaling = (
                    flex.double(scaler.reflection_table.size(), 1.0) + fractional_error
                )
                scaler.reflection_table["variance"] *= variance_scaling
        self._set_outliers()
        self._clean_reflection_tables()
        msg = """
The reflection table variances have been adjusted to account for the
uncertainty in the scaling model""" + (
            "s for all datasets" if len(self.active_scalers) > 1 else ""
        )
        logger.info(msg)
        if self._free_Ih_table:
            # calc merging stats and log.
            free_miller_array = scaled_data_as_miller_array(
                [self._get_free_set_reflections()],
                [self.active_scalers[0].experiment],
                anomalous_flag=False,
            )
            work_miller_array = scaled_data_as_miller_array(
                [self._get_work_set_reflections()],
                [self.active_scalers[0].experiment],
                anomalous_flag=False,
            )
            free, _ = merging_stats_from_scaled_array(
                free_miller_array,
                self.params.output.merging.nbins,
                self.params.output.use_internal_variance,
                anomalous=False,
            )
            s = StringIO()
            s.write("\nFree set statistics\n")
            free.show(out=s)
            work, _ = merging_stats_from_scaled_array(
                work_miller_array,
                self.params.output.merging.nbins,
                self.params.output.use_internal_variance,
                anomalous=False,
            )
            s.write("\nWork set statistics\n")
            work.show(out=s)
            logger.debug(s.getvalue())
            self._work_free_stats = [
                work.overall.r_meas,
                free.overall.r_meas,
                free.overall.r_meas - work.overall.r_meas,
                work.overall.cc_one_half,
                free.overall.cc_one_half,
                work.overall.cc_one_half - free.overall.cc_one_half,
            ]

    # Internal general scaler methods

    def _remove_datasets(self, scalers, n_list):
        """
        Delete a scaler from the dataset.

        Code in this module does not necessarily have access to all references of
        experiments and reflections, so log the position in the list so that they
        can be deleted later. Scaling algorithm code should only depends on the
        scalers.
        """
        initial_number = len(scalers)
        for n in n_list[::-1]:
            self._removed_datasets.append(scalers[n].experiment.identifier)
            del scalers[n]
        assert len(scalers) == initial_number - len(n_list)
        logger.info("Removed datasets: %s", n_list)

    def _set_outliers(self):
        """Set the scaling outliers in the individual reflection tables."""
        for scaler in self.active_scalers:
            suitable_isel = scaler.suitable_refl_for_scaling_sel.iselection()
            outlier_isel = suitable_isel.select(scaler.outliers)
            outliers_mask = flex.bool(
                scaler.suitable_refl_for_scaling_sel.size(), False
            )
            outliers_mask.set_selected(outlier_isel, True)
            scaler.reflection_table.set_flags(
                outliers_mask, scaler.reflection_table.flags.outlier_in_scaling
            )

    def _clean_reflection_tables(self):
        """Remove unneccesary columns added to reflection tables."""
        for scaler in self.active_scalers:
            scaler.clean_reflection_table()

    def _update_error_model(self, error_model, update_Ih=True):
        """Update the error model in Ih table."""
        self._error_model = error_model
        if update_Ih:
            self.global_Ih_table.update_error_model(error_model)
            if self._free_Ih_table:
                self._free_Ih_table.update_error_model(error_model)
        for scaler in self.active_scalers:
            scaler.experiment.scaling_model.set_error_model(error_model)

    def _update_after_minimisation(self, parameter_manager):
        if parameter_manager.apm_list[0].var_cov_matrix:
            for i, scaler in enumerate(self.active_scalers):
                scaler.update_var_cov(parameter_manager.apm_list[i])
                scaler.experiment.scaling_model.set_scaling_model_as_scaled()

    def _get_free_set_reflections(self):
        """Get all reflections in the free set if it exists."""
        if self._free_Ih_table:
            refls = flex.reflection_table()
            for scaler in self.active_scalers:
                refls.extend(scaler.get_free_set_reflections())
            return refls
        return None

    def _get_work_set_reflections(self):
        """Get all reflections in the free set if it exists."""
        if self._free_Ih_table:
            refls = flex.reflection_table()
            for scaler in self.active_scalers:
                refls.extend(scaler.get_work_set_reflections())
            return refls
        return None

    def _update_model_data(self):
        for i, scaler in enumerate(self.active_scalers):
            block_selections = self.Ih_table.get_block_selections_for_dataset(i)
            for component in scaler.components.values():
                component.update_reflection_data(block_selections=block_selections)

    def _create_global_Ih_table(self):
        tables = [s.get_valid_reflections() for s in self.active_scalers]
        free_set_percentage = 0.0
        if self.params.scaling_options.use_free_set:
            free_set_percentage = self.params.scaling_options.free_set_percentage
        space_group = self.active_scalers[0].experiment.crystal.get_space_group()
        self._global_Ih_table = IhTable(
            tables,
            space_group,
            nblocks=1,
            additional_cols=["partiality"],
            free_set_percentage=free_set_percentage,
            free_set_offset=self.params.scaling_options.free_set_offset,
        )
        if free_set_percentage:
            # need to set free_set_selection in individual scalers
            tables = []
            free_tables = []
            indices_list = []
            free_indices_list = []
            for i, scaler in enumerate(self.active_scalers):
                sel = (
                    self.global_Ih_table.Ih_table_blocks[-1].Ih_table["dataset_id"] == i
                )
                indiv_Ih_block = self.global_Ih_table.Ih_table_blocks[-1].select(sel)
                loc_indices = indiv_Ih_block.Ih_table["loc_indices"]
                scaler.free_set_selection = flex.bool(scaler.n_suitable_refl, False)
                scaler.free_set_selection.set_selected(loc_indices, True)
                tables.append(scaler.get_work_set_reflections())
                free_tables.append(scaler.get_free_set_reflections())
                free_indices_list.append(scaler.free_set_selection.iselection())
                indices_list.append((~scaler.free_set_selection).iselection())
            self._global_Ih_table = IhTable(
                tables, space_group, indices_list, nblocks=1
            )
            self._free_Ih_table = IhTable(
                free_tables, space_group, free_indices_list, nblocks=1
            )

    def _create_Ih_table(self):
        """Create a new Ih table from the reflection tables."""
        tables = [
            s.get_reflections_for_model_minimisation() for s in self.active_scalers
        ]
        indices_lists = [s.scaling_selection.iselection() for s in self.active_scalers]
        self._Ih_table = IhTable(
            tables,
            self.active_scalers[0].experiment.crystal.get_space_group(),
            indices_lists=indices_lists,
            nblocks=self.params.scaling_options.nproc,
        )
        if self.error_model:
            for i, scaler in enumerate(self.active_scalers):
                variance = scaler.reflection_table["variance"].select(
                    scaler.suitable_refl_for_scaling_sel
                )
                intensity = scaler.reflection_table["intensity"].select(
                    scaler.suitable_refl_for_scaling_sel
                )
                new_vars = self.error_model.update_variances(variance, intensity)
                self._Ih_table.update_data_in_blocks(new_vars, i, column="variance")

    def _select_all_reflections_for_scaling(self):
        for scaler in self.active_scalers:
            if self._free_Ih_table:
                scaler.scaling_selection = ~scaler.free_set_selection
            else:
                scaler.scaling_selection = flex.bool(scaler.n_suitable_refl, True)
            scaler.scaling_subset_sel = copy.deepcopy(scaler.scaling_selection)
            scaler.scaling_selection &= ~scaler.outliers

    def _select_intensity_ranges_for_scaling(self):
        for scaler in self.active_scalers:
            overall_scaling_selection = calculate_scaling_subset_ranges_with_E2(
                scaler.reflection_table, scaler.params
            )
            scaler.scaling_selection = overall_scaling_selection.select(
                scaler.suitable_refl_for_scaling_sel
            )
            if self._free_Ih_table:
                scaler.scaling_selection.set_selected(scaler.free_set_selection, False)
            scaler.scaling_subset_sel = copy.deepcopy(scaler.scaling_selection)
            scaler.scaling_selection &= ~scaler.outliers

    def _select_random_intensities_for_scaling(self):
        n = self.params.reflection_selection.n_random
        isel = flex.random_selection(self.global_Ih_table.size, n)
        sel = flex.bool(self.global_Ih_table.size, False)
        sel.set_selected(isel, True)
        sel_Ih = self.global_Ih_table.Ih_table_blocks[0].select(sel)
        for i, scaler in enumerate(self.active_scalers):
            indiv_block = sel_Ih.select(sel_Ih.Ih_table["dataset_id"] == i)
            loc_indices = indiv_block.Ih_table["loc_indices"]
            scaler.scaling_selection = flex.bool(scaler.n_suitable_refl, False)
            scaler.scaling_selection.set_selected(loc_indices, True)
            scaler.scaling_subset_sel = copy.deepcopy(scaler.scaling_selection)
            scaler.scaling_selection &= ~scaler.outliers

    def _select_quasi_random_indices_for_individual_datasets(self):
        n_sel_per_dataset = []
        for i, scaler in enumerate(self.active_scalers):
            scaler.scaling_selection = flex.bool(scaler.n_suitable_refl, False)
            block = self.global_Ih_table.Ih_table_blocks[0].select(
                self.global_Ih_table.Ih_table_blocks[0].Ih_table["dataset_id"] == i
            )
            loc_indices = block.Ih_table["loc_indices"]
            block.Ih_table["s1c"] = (
                scaler.reflection_table["s1c"]
                .select(scaler.suitable_refl_for_scaling_sel)
                .select(loc_indices)
            )
            suitable_table = scaler.get_valid_reflections()
            presel = calculate_scaling_subset_ranges(suitable_table, self.params)
            preselection = presel.select(block.Ih_table["loc_indices"])
            indiv_block = block.select(preselection)
            indices_for_indiv = calculate_scaling_subset_connected(
                indiv_block, scaler.experiment, self.params
            )
            scaler.scaling_selection.set_selected(indices_for_indiv, True)
            n_sel_per_dataset.append(indices_for_indiv.size())
        return n_sel_per_dataset

    def _select_reflections_for_scaling(self):
        if self.params.reflection_selection.method == "quasi_random":
            qr = self.params.reflection_selection.quasi_random
            indices, dataset_ids, _ = select_connected_reflections_across_datasets(
                self.global_Ih_table,
                qr.multi_dataset.min_per_dataset,
                qr.multi_dataset.min_multiplicity,
                qr.multi_dataset.Isigma_cutoff,
            )
            header = [
                "Dataset id",
                "reflections \nconnected to \nother datasets",
                "reflections \nhighly connected \nwithin dataset",
                "combined number \nof reflections",
            ]
            rows = []
            n_sel_per_dataset = (
                self._select_quasi_random_indices_for_individual_datasets()
            )
            for i, scaler in enumerate(self.active_scalers):
                sel = dataset_ids == i
                indices_for_dataset = indices.select(sel)
                scaler.scaling_selection.set_selected(indices_for_dataset, True)
                rows.append(
                    [
                        scaler.experiment.identifier,
                        str(indices_for_dataset.size()),
                        str(n_sel_per_dataset[i]),
                        str(scaler.scaling_selection.count(True)),
                    ]
                )
                scaler.scaling_subset_sel = copy.deepcopy(scaler.scaling_selection)
                scaler.scaling_selection &= ~scaler.outliers
            st = simple_table(rows, header)
            logger.info(
                "Summary of reflections chosen for minimisation from each dataset:"
            )
            logger.info(st.format())
        elif self.params.reflection_selection.method == "intensity_ranges":
            self._select_intensity_ranges_for_scaling()
        elif self.params.reflection_selection.method == "use_all" or (
            self.params.reflection_selection.method == "random"
            and (self.params.reflection_selection.n_random >= self.global_Ih_table.size)
        ):
            self._select_all_reflections_for_scaling()
        elif self.params.reflection_selection.method == "random":
            self._select_random_reflections_for_scaling()
        else:
            raise ValueError("Invalid choice for 'reflection_selection.method'.")


class SingleScaler(MultiScalerBase):

    """A specialisation of multiscaler for single dataset case."""

    id_ = "single"

    def __init__(self, single_scalers):
        super(SingleScaler, self).__init__(single_scalers)
        self._active_scalers = self.single_scalers
        self._create_global_Ih_table()
        # now select reflections from across the datasets
        self._select_reflections_for_scaling()
        self._create_Ih_table()
        # now add data to scale components from datasets
        self._update_model_data()

    def update_for_minimisation(self, apm, block_id):
        """Update the scale factors and Ih for the next minimisation iteration."""
        apm_i = apm.apm_list[0]
        scales_i, derivs_i = RefinerCalculator.calculate_scales_and_derivatives(
            apm_i, block_id
        )
        self.Ih_table.set_derivatives(derivs_i, block_id)
        self.Ih_table.set_inverse_scale_factors(scales_i, block_id)
        self.Ih_table.update_weights(block_id)
        self.Ih_table.calc_Ih(block_id)

    def _select_reflections_for_scaling(self):
        """Select a subset of reflections to use in minimisation."""
        scaler = self.active_scalers[0]
        if self.params.reflection_selection.method == "quasi_random":
            n_sel_per_dataset = (
                self._select_quasi_random_indices_for_individual_datasets()
            )
            logger.info(
                "%s reflections were selected for scale factor determination \n"
                + "out of %s suitable reflections. ",
                n_sel_per_dataset[0],
                scaler.n_suitable_refl,
            )
            scaler.scaling_subset_sel = copy.deepcopy(scaler.scaling_selection)
            scaler.scaling_selection &= ~scaler.outliers  # now apply outliers
        elif self.params.reflection_selection.method == "intensity_ranges":
            self._select_intensity_ranges_for_scaling()
        elif self.params.reflection_selection.method == "use_all" or (
            self.params.reflection_selection.method == "random"
            and (self.params.reflection_selection.n_random >= self.global_Ih_table.size)
        ):
            self._select_all_reflections_for_scaling()
        elif self.params.reflection_selection.method == "random":
            self._select_random_intensities_for_scaling()
        else:
            raise ValueError("Invalid choice for 'reflection_selection.method'.")


class MultiScaler(MultiScalerBase):
    """Scaler for multiple datasets where all datasets are being minimised."""

    id_ = "multi"

    def __init__(self, single_scalers):
        """
        Initialise a multiscaler from a list of single scalers.

        Create a global_Ih_table, an Ih_table to use for minimisation and update
        the data in the model components.
        """
        super(MultiScaler, self).__init__(single_scalers)
        self._active_scalers = self.single_scalers
        self._create_global_Ih_table()
        # now select reflections from across the datasets
        self._select_reflections_for_scaling()
        self._create_Ih_table()
        # now add data to scale components from datasets
        self._update_model_data()
        log_memory_usage()


class TargetScaler(MultiScalerBase):
    """A target scaler for scaling datasets against already scaled data."""

    id_ = "target"

    def __init__(self, scaled_scalers, unscaled_scalers):
        """
        Initialise a multiscaler from a list of single and unscaled scalers.

        First, set the active scalers (the unscaled scalers) and use these to
        create a global_Ih_table. Then, use the scaled_scalers to create a
        target_Ih_table. Create an Ih_table to use for minimisation and use
        the target_Ih_table to set the Ih_values. Finally, update the data in
        the model components.
        """
        logger.info("\nInitialising a TargetScaler instance. \n")
        super(TargetScaler, self).__init__(scaled_scalers)
        logger.info("Determining symmetry equivalent reflections across datasets.\n")
        self.unscaled_scalers = unscaled_scalers
        self._active_scalers = unscaled_scalers
        self._create_global_Ih_table()
        self._select_reflections_for_scaling()
        tables = [
            s.reflection_table.select(s.suitable_refl_for_scaling_sel).select(
                s.scaling_selection
            )
            for s in self.single_scalers
        ]
        self._target_Ih_table = IhTable(
            tables,
            self.active_scalers[0].experiment.crystal.get_space_group(),
            nblocks=1,
        )  # Keep in one table for matching below
        self._create_Ih_table()
        self._update_model_data()
        logger.info("Completed initialisation of TargetScaler. \n\n" + "=" * 80 + "\n")
        log_memory_usage()

    @property
    def target_Ih_table(self):
        """An Ih_table containing data for the target."""
        return self._target_Ih_table

    def _create_Ih_table(self):
        super(TargetScaler, self)._create_Ih_table()
        for block in self._Ih_table.blocked_data_list:
            # this step reduces the number of reflections in each block
            block.match_Ih_values_to_target(self._target_Ih_table)
        self.Ih_table.generate_block_selections()

    def round_of_outlier_rejection(self):
        """Perform a round of targeted outlier rejection."""
        self._round_of_outlier_rejection(target=self._target_Ih_table)

    def update_for_minimisation(self, apm, block_id):
        """Calcalate the new parameters but don't calculate a new Ih."""
        self._update_for_minimisation(apm, block_id, calc_Ih=False)

    @Subject.notify_event(event="performed_scaling")
    def perform_scaling(self, engine=None, max_iterations=None, tolerance=None):
        """Minimise the scaling model, using a fixed-Ih target."""
        self._perform_scaling(
            target_type=ScalingTargetFixedIH,
            engine=engine,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

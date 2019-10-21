from __future__ import absolute_import, division, print_function

import copy
import logging
from collections import OrderedDict

import six
from dials_scaling_ext import row_multiply
from dials_scaling_ext import calc_sigmasq as cpp_calc_sigmasq
from dials.array_family import flex
from dials.algorithms.scaling.outlier_rejection import reject_outliers

from dials.algorithms.scaling.scaling_utilities import log_memory_usage
from dials.algorithms.scaling.combine_intensities import SingleDatasetIntensityCombiner
from libtbx.table_utils import simple_table
from scitbx import sparse

logger = logging.getLogger("dials")


class SingleDatasetScaler(object):
    """Definition of a scaler for a single dataset."""

    id_ = "single"

    def __init__(self, params, experiment, reflection_table):
        """
        Initialise a single-dataset scaler.

        The reflection table needs the columns 'inverse_scale_factor', 'Esq',
        'intensity', 'variance', 'id', which are guaranteed if the scaler is
        created using the SingleScalerFactory.
        """
        assert all(
            i in reflection_table
            for i in ["inverse_scale_factor", "intensity", "variance", "id"]
        )
        # super(SingleDatasetScaler, self).__init__()
        self._params = params
        self._experiment = experiment
        n_model_params = sum(val.n_params for val in self.components.values())
        self._var_cov_matrix = sparse.matrix(n_model_params, n_model_params)
        self._initial_keys = list(reflection_table.keys())
        self._reflection_table = reflection_table
        # self._Ih_table = None  # stores data for reflections used for minimisation
        self.suitable_refl_for_scaling_sel = self._get_suitable_for_scaling_sel(
            self._reflection_table
        )
        self.n_suitable_refl = self.suitable_refl_for_scaling_sel.count(True)
        if self._experiment.scaling_model.is_scaled:
            outliers = self._reflection_table.get_flags(
                self._reflection_table.flags.outlier_in_scaling
            )
            self.outliers = outliers.select(self.suitable_refl_for_scaling_sel)
        else:
            self.outliers = flex.bool(self.n_suitable_refl, False)
        self.scaling_subset_sel = (
            None
        )  # A selection of len n_suitable_refl of scaling subset selection
        self.scaling_selection = None  # As above, but with outliers deselected also
        self.free_set_selection = flex.bool(self.n_suitable_refl, False)

        # configure model
        sel_reflections = self.get_valid_reflections()
        self.experiment.scaling_model.configure_components(
            sel_reflections, self.experiment, self.params
        )
        rows = [[key, str(val.n_params)] for key, val in six.iteritems(self.components)]
        st = simple_table(rows, ["correction", "n_parameters"])
        logger.info("The following corrections will be applied to this dataset: \n")
        logger.info(st.format())
        if "Imid" in self.experiment.scaling_model.configdict:
            self._combine_intensities_using_Imid(
                self.experiment.scaling_model.configdict["Imid"]
            )
        if not self._experiment.scaling_model.is_scaled:
            self.individual_outlier_rejection()
        self.scaling_selection = ~self.outliers
        logger.info(
            "Completed preprocessing and initialisation for this dataset.\n"
            "\n" + "=" * 80 + "\n"
        )
        log_memory_usage()

    @property
    def params(self):
        """The params phil scope."""
        return self._params

    @property
    def experiment(self):
        """The experiment object associated with the dataset."""
        return self._experiment

    @property
    def reflection_table(self):
        """The reflection table of the datatset."""
        return self._reflection_table

    @reflection_table.setter
    def reflection_table(self, new_table):
        """Set the reflection table of the datatset."""
        self._reflection_table = new_table

    def get_valid_reflections(self):
        """All reflections not bad for scaling or user excluded."""
        return self.reflection_table.select(self.suitable_refl_for_scaling_sel)

    def get_work_set_reflections(self):
        valid = self.get_valid_reflections()
        return valid.select(~self.free_set_selection)

    def get_free_set_reflections(self):
        """Get all reflections in the free set if it exists."""
        valid = self.get_valid_reflections()
        return valid.select(self.free_set_selection)

    def get_reflections_for_model_minimisation(self):
        """Get the reflections selected for scaling model minimisation."""
        valid = self.get_valid_reflections()
        return valid.select(self.scaling_selection)

    def fix_initial_parameter(self):
        fixed = self.experiment.scaling_model.fix_initial_parameter(self.params)
        return fixed

    @staticmethod
    def _get_suitable_for_scaling_sel(reflections):
        """Extract suitable reflections for scaling from the reflection table."""
        user_excl = reflections.get_flags(reflections.flags.user_excluded_in_scaling)
        excl_for_scale = reflections.get_flags(reflections.flags.excluded_for_scaling)
        suitable_refl_for_scaling_sel = ~(user_excl | excl_for_scale)
        return suitable_refl_for_scaling_sel

    @property
    def components(self):
        """Shortcut to the scaling model components."""
        return self.experiment.scaling_model.components

    @property
    def consecutive_refinement_order(self):
        """Link to consecutive refinement order for parameter manager."""
        return self.experiment.scaling_model.consecutive_refinement_order

    @property
    def var_cov_matrix(self):
        """The variance covariance matrix for the parameters."""
        return self._var_cov_matrix

    # KEEP

    def update_var_cov(self, apm):
        """
        Update the full parameter variance covariance matrix after a refinement.

        If all parameters have been refined, then the full var_cov matrix can be set.
        Else one must select subblocks for pairs of parameters and assign these into
        the full var_cov matrix, taking care to out these in the correct position.
        This is applicable if only some parameters have been refined in this cycle.
        """
        var_cov_list = apm.var_cov_matrix  # values are passed as a list from refinery
        if int(var_cov_list.size() ** 0.5) == self.var_cov_matrix.n_rows:
            self._var_cov_matrix.assign_block(
                var_cov_list.matrix_copy_block(
                    0, 0, apm.n_active_params, apm.n_active_params
                ),
                0,
                0,
            )
        else:  # need to set part of the var_cov matrix e.g. if only refined some params
            # first work out the order in self._var_cov_matrix
            cumul_pos_dict = {}
            n_cumul_params = 0
            for name, component in six.iteritems(self.components):
                cumul_pos_dict[name] = n_cumul_params
                n_cumul_params += component.n_params
            # now get a var_cov_matrix subblock for pairs of parameters
            for name in apm.components_list:
                for name2 in apm.components_list:
                    n_rows = apm.components[name]["n_params"]
                    n_cols = apm.components[name2]["n_params"]
                    start_row = apm.components[name]["start_idx"]
                    start_col = apm.components[name2]["start_idx"]
                    sub = var_cov_list.matrix_copy_block(
                        start_row, start_col, n_rows, n_cols
                    )
                    # now set this block into correct location in overall var_cov
                    self._var_cov_matrix.assign_block(
                        sub, cumul_pos_dict[name], cumul_pos_dict[name2]
                    )

    def _combine_intensities_using_Imid(self, Imid):
        logger.info(
            "Using previously determined optimal intensity choice: %s\n",
            OrderedDict(
                [
                    (Imid, str(round(Imid, 4))),
                    (0, "profile intensities"),
                    (1, "summation intensities"),
                ]
            )[Imid],
        )
        combiner = SingleDatasetIntensityCombiner(self, Imid)
        intensity, variance = combiner.calculate_suitable_combined_intensities()
        # update data in reflection table
        isel = self.suitable_refl_for_scaling_sel.iselection()
        self._reflection_table["intensity"].set_selected(isel, intensity)
        self._reflection_table["variance"].set_selected(isel, variance)
        self.experiment.scaling_model.record_intensity_combination_Imid(
            combiner.max_key
        )

    def expand_scales_to_all_reflections(self, calc_cov=False):
        """
        Calculate scale factors for all suitable reflections.

        Use the current model to calculate scale factors for all suitable
        reflections, and set these in the reflection table. If caller=None,
        the global_Ih_table is updated. If calc_cov, an error estimate on the
        inverse scales is calculated.
        """
        self._reflection_table["inverse_scale_factor_variance"] = flex.double(
            self.reflection_table.size(), 0.0
        )
        n_blocks = self.params.scaling_options.nproc
        n_start = 0
        all_scales = flex.double([])
        all_invsfvars = flex.double([])
        n_param_tot = sum(c.n_params for c in self.components.values())
        for i in range(1, n_blocks + 1):  # do calc in blocks for speed/memory
            n_end = int(i * self.n_suitable_refl / n_blocks)
            block_isel = flex.size_t(range(n_start, n_end))
            n_start = n_end
            scales = flex.double(block_isel.size(), 1.0)
            scales_list = []
            derivs_list = []
            jacobian = sparse.matrix(block_isel.size(), n_param_tot)
            for component in self.components.values():
                component.update_reflection_data(block_selections=[block_isel])
                comp_scales, d = component.calculate_scales_and_derivatives(block_id=0)
                scales_list.append(comp_scales)
                if calc_cov:
                    derivs_list.append(d)
                scales *= comp_scales
            all_scales.extend(scales)
            if calc_cov and self.var_cov_matrix.non_zeroes > 0:
                n_cumulative_param = 0
                for j, component in enumerate(self.components):
                    d_block = derivs_list[j]
                    n_param = self.components[component].n_params
                    for k, component_2 in enumerate(self.components):
                        if component_2 != component:
                            d_block = row_multiply(d_block, scales_list[k])
                    jacobian.assign_block(d_block, 0, n_cumulative_param)
                    n_cumulative_param += n_param
                all_invsfvars.extend(
                    cpp_calc_sigmasq(jacobian.transpose(), self._var_cov_matrix)
                )
        scaled_isel = self.suitable_refl_for_scaling_sel.iselection()
        self.reflection_table["inverse_scale_factor"].set_selected(
            scaled_isel, all_scales
        )
        if calc_cov and self.var_cov_matrix.non_zeroes > 0:
            self.reflection_table["inverse_scale_factor_variance"].set_selected(
                scaled_isel, all_invsfvars
            )

    # @Subject.notify_event(event="performed_outlier_rejection")
    def individual_outlier_rejection(self):
        """Perform outlier rejection"""
        sel_reflections = self.get_valid_reflections()
        table_with_outliers = reject_outliers(
            sel_reflections,
            self.experiment,
            method=self.params.scaling_options.outlier_rejection,
            zmax=self.params.scaling_options.outlier_zmax,
        )
        self.outliers = copy.deepcopy(
            table_with_outliers.get_flags(table_with_outliers.flags.outlier_in_scaling)
        )
        assert self.outliers.size() == self.n_suitable_refl

    def clean_reflection_table(self):
        """Remove additional added columns that are not required for output."""
        self._initial_keys.append("inverse_scale_factor")
        self._initial_keys.append("inverse_scale_factor_variance")
        self._initial_keys.append("Ih_values")
        self._initial_keys.append("intensity.scale.value")
        self._initial_keys.append("intensity.scale.variance")
        self.reflection_table["intensity.scale.value"] = self.reflection_table[
            "intensity"
        ]
        self.reflection_table["intensity.scale.variance"] = self.reflection_table[
            "variance"
        ]
        if "Esq" in self.reflection_table:
            del self.reflection_table["Esq"]
        for key in self.reflection_table.keys():
            if key not in self._initial_keys:
                del self._reflection_table[key]


class NullScaler(object):
    """A singlescaler to allow targeted scaling against calculated intensities."""

    id_ = "null"

    def __init__(self, params, experiment, reflection):
        """Set the required properties to use as a scaler for targeted scaling."""
        self._params = params
        self._experiment = experiment
        self._reflection_table = reflection
        self.n_suitable_refl = self._reflection_table.size()
        self._reflection_table["inverse_scale_factor"] = flex.double(
            self.n_suitable_refl, 1.0
        )
        if "variance" not in self._reflection_table:
            self._reflection_table["variance"] = flex.double(self.n_suitable_refl, 1.0)
        self._reflection_table.set_flags(
            flex.bool(self.n_suitable_refl, False),
            self._reflection_table.flags.excluded_for_scaling,
        )
        self.suitable_refl_for_scaling_sel = flex.bool(self.n_suitable_refl, True)
        self.outliers = flex.bool(self.n_suitable_refl, False)
        self.scaling_selection = flex.bool(self.n_suitable_refl, True)
        logger.info("Target dataset contains %s reflections", self.n_suitable_refl)
        logger.info(
            "Completed preprocessing and initialisation for this dataset."
            "\n\n" + "=" * 80 + "\n"
        )

    @property
    def params(self):
        """The params phil scope object."""
        return self._params

    @property
    def experiment(self):
        """Return the experiment object for the dataset"""
        return self._experiment

    @property
    def reflection_table(self):
        """Return the reflection_table object for the dataset"""
        return self._reflection_table

    @property
    def components(self):
        """Shortcut to scaling model components."""
        return self.experiment.scaling_model.components

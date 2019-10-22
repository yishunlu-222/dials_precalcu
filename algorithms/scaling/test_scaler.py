from __future__ import absolute_import, division, print_function
import pytest
from mock import MagicMock
from scitbx import sparse
from libtbx import phil
from dxtbx.model.experiment_list import ExperimentList
from dxtbx.model import Crystal, Scan, Beam, Goniometer, Detector, Experiment
from dials.array_family import flex
from dials.util.options import OptionParser
from dials.algorithms.scaling.scaling_utilities import calculate_prescaling_correction
from dials.algorithms.scaling.scaler import SingleScaler, MultiScaler, TargetScaler
from dials.algorithms.scaling.parameter_handler import ScalingParameterManagerGenerator
from dials.algorithms.scaling.single_dataset_scaler import SingleDatasetScaler
from dials.algorithms.scaling.model.model import KBScalingModel


def side_effect_update_var(variances, intensities):
    """Side effect to mock configure reflection table
    call during initialisation."""
    return flex.double(range(1, len(variances) + 1))


@pytest.fixture
def mock_errormodel():
    """A mock error model."""
    em = MagicMock()
    em.refined_parameters = [1.0, 0.1]
    em.update_variances.side_effect = side_effect_update_var
    return em


@pytest.fixture
def mock_errormodel2():
    """A mock error model."""
    em = MagicMock()
    em.refined_parameters = [1.0, 0.1]
    em.update_variances.side_effect = side_effect_update_var
    # return_value = flex.double(range(2, 9))
    return em


def generated_exp(n=1):
    """Generate an experiment list with two experiments."""
    experiments = ExperimentList()
    exp_dict = {
        "__id__": "crystal",
        "real_space_a": [1.0, 0.0, 0.0],
        "real_space_b": [0.0, 1.0, 0.0],
        "real_space_c": [0.0, 0.0, 2.0],
        "space_group_hall_symbol": " C 2y",
    }
    crystal = Crystal.from_dict(exp_dict)
    scan = Scan(image_range=[0, 90], oscillation=[0.0, 1.0])
    beam = Beam(s0=(0.0, 0.0, 1.01))
    goniometer = Goniometer((1.0, 0.0, 0.0))
    detector = Detector()
    experiments.append(
        Experiment(
            beam=beam,
            scan=scan,
            goniometer=goniometer,
            detector=detector,
            crystal=crystal,
        )
    )
    experiments[0].identifier = "0"
    if n > 1:
        for i in range(n - 1):
            experiments.append(
                Experiment(
                    beam=beam,
                    scan=scan,
                    goniometer=goniometer,
                    detector=detector,
                    crystal=crystal,
                )
            )
            experiments[i + 1].identifier = str(i + 1)
    return experiments


def generated_param():
    """Generate a param phil scope."""
    phil_scope = phil.parse(
        """
      include scope dials.command_line.scale.phil_scope
  """,
        process_includes=True,
    )
    optionparser = OptionParser(phil=phil_scope, check_format=False)
    parameters, _ = optionparser.parse_args(
        args=[], quick_parse=True, show_diff_phil=False
    )
    parameters.model = "KB"
    parameters.reflection_selection.method = "use_all"
    parameters.scaling_options.nproc = 2
    return parameters


def generated_refl(id_=0):
    """Create a reflection table suitable for splitting into blocks."""
    reflections = flex.reflection_table()
    reflections["intensity"] = flex.double([1.0, 2.0, 3.0, 4.0, 500.0, 6.0, 2.0, 2.0])
    reflections["variance"] = flex.double(8, 1.0)
    reflections["miller_index"] = flex.miller_index(
        [
            (1, 0, 0),
            (2, 0, 0),
            (0, 0, 1),
            (2, 2, 2),
            (1, 0, 0),
            (2, 0, 0),
            (1, 0, 0),
            (1, 0, 0),
        ]
    )
    reflections["d"] = flex.double([0.8, 2.1, 2.0, 1.4, 1.6, 2.5, 2.5, 2.5])
    reflections["partiality"] = flex.double(8, 1.0)
    reflections["Esq"] = flex.double(8, 1.0)
    reflections["inverse_scale_factor"] = flex.double(8, 1.0)
    reflections["xyzobs.px.value"] = flex.vec3_double(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 5.0),
            (0.0, 0.0, 8.0),
            (0.0, 0.0, 10.0),
            (0.0, 0.0, 12.0),
            (0.0, 0.0, 15.0),
            (0.0, 0.0, 15.0),
            (0.0, 0.0, 15.0),
        ]
    )
    reflections["s1"] = flex.vec3_double([(0.0, 0.1, 1.0)] * 8)
    reflections.set_flags(flex.bool(8, True), reflections.flags.integrated)
    reflections.set_flags(
        flex.bool([False] * 5 + [True] + [False] * 2),
        reflections.flags.excluded_for_scaling,
    )
    reflections["id"] = flex.int(8, id_)
    reflections.experiment_identifiers()[id_] = str(id_)
    return reflections


def generated_refl_for_comb():
    """Create a reflection table suitable for splitting into blocks."""
    reflections = flex.reflection_table()
    reflections["intensity"] = flex.double([1.0, 2.0, 3.0, 4.0, 500.0, 6.0, 2.0, 2.0])
    reflections["variance"] = flex.double(8, 1.0)
    reflections["intensity.prf.value"] = flex.double(
        [1.0, 3.0, 3.0, 4.0, 50.0, 6.0, 3.0, 2.0]
    )
    reflections["intensity.prf.variance"] = flex.double(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    )
    reflections["intensity.sum.value"] = flex.double(
        [1.0, 4.0, 3.0, 4.0, 500.0, 6.0, 6.0, 2.0]
    )
    reflections["intensity.sum.variance"] = flex.double(8, 1.0)
    reflections["miller_index"] = flex.miller_index(
        [
            (1, 0, 0),
            (2, 0, 0),
            (0, 0, 1),
            (2, 2, 2),
            (1, 0, 0),
            (2, 0, 0),
            (1, 0, 0),
            (1, 0, 0),
        ]
    )
    reflections["d"] = flex.double([0.8, 2.1, 2.0, 1.4, 1.6, 2.5, 2.5, 2.5])
    reflections["partiality"] = flex.double(8, 1.0)
    reflections["Esq"] = flex.double(8, 1.0)
    reflections["inverse_scale_factor"] = flex.double(8, 1.0)
    reflections["xyzobs.px.value"] = flex.vec3_double(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 5.0),
            (0.0, 0.0, 8.0),
            (0.0, 0.0, 10.0),
            (0.0, 0.0, 12.0),
            (0.0, 0.0, 15.0),
            (0.0, 0.0, 15.0),
            (0.0, 0.0, 15.0),
        ]
    )
    reflections["s1"] = flex.vec3_double([(0.0, 0.1, 1.0)] * 8)
    reflections.set_flags(flex.bool(8, True), reflections.flags.integrated)
    reflections.set_flags(
        flex.bool([False] * 5 + [True] + [False] * 2), reflections.flags.bad_for_scaling
    )
    reflections["id"] = flex.int(8, 0)
    reflections.experiment_identifiers()[0] = "0"
    reflections = calculate_prescaling_correction(reflections)
    return reflections


def mock_scaling_component():
    """Mock scaling component to allow creation of a scaling model."""
    component = MagicMock()
    component.n_params = 2
    component.inverse_scales = [flex.double([0.9, 1.1])]
    component.derivatives = [sparse.matrix(2, 2)]
    component.derivatives[0][0, 0] = 0.5
    component.derivatives[0][1, 0] = 0.4
    return component


@pytest.fixture
def mock_apm():
    """mock parameter manager for testing var_cov_matrix setting."""
    apm = MagicMock()
    apm.var_cov_matrix = flex.double([2.0])
    apm.var_cov_matrix.reshape(flex.grid(1, 1))
    apm.n_active_params = 1
    apm.n_obs = [2]
    apm.derivatives = [sparse.matrix(1, 1)]
    apm.components_list = ["scale"]
    apm.components = {
        "scale": {"object": mock_scaling_component(), "n_params": 1, "start_idx": 0}
    }
    apm.constant_g_values = [flex.double(2, 1.0)]
    return apm


# All scalers take a list of SingleDatasetScalers, so would be wise to use
# real ones to catch any behaviour changes there.


def generate_test_multiscaler():
    # need a list of single dataset scalers
    refl1 = generated_refl(id_=0)
    refl2 = generated_refl(id_=1)
    experiments = generated_exp(n=2)
    params = generated_param()
    experiments[0].scaling_model = KBScalingModel.from_data(
        params, experiments[0], refl1
    )
    experiments[1].scaling_model = KBScalingModel.from_data(
        params, experiments[1], refl2
    )

    single_scalers = [
        SingleDatasetScaler(params, experiments[0], refl1),
        SingleDatasetScaler(params, experiments[1], refl2),
    ]
    return MultiScaler(single_scalers)


def generate_test_targetscaler():
    # need a list of single dataset scalers
    refl1 = generated_refl(id_=0)
    refl2 = generated_refl(id_=1)
    experiments = generated_exp(n=2)
    params = generated_param()
    experiments[0].scaling_model = KBScalingModel.from_data(
        params, experiments[0], refl1
    )
    experiments[1].scaling_model = KBScalingModel.from_data(
        params, experiments[1], refl2
    )

    single_scalers = [
        SingleDatasetScaler(params, experiments[0], refl1),
        SingleDatasetScaler(params, experiments[1], refl2),
    ]
    return TargetScaler([single_scalers[0]], [single_scalers[1]])


def generate_test_singlescaler():
    # need a list of single dataset scalers
    refl1 = generated_refl(id_=0)
    experiments = generated_exp(n=1)
    params = generated_param()
    experiments[0].scaling_model = KBScalingModel.from_data(
        params, experiments[0], refl1
    )
    single_scalers = [SingleDatasetScaler(params, experiments[0], refl1)]
    return SingleScaler(single_scalers)


# Test scaler interface for scaling refiner


def test_update_for_minimisation():
    """Test the update_for_minimisation method of the scalers."""

    # Test for multiscalerbase, singlescaler and targetscaler.
    multiscaler = generate_test_multiscaler()
    assert len(multiscaler.get_blocks_for_minimisation()) == 2

    # To test, need a parameter manager
    apmg = ScalingParameterManagerGenerator(
        multiscaler.single_scalers, mode="concurrent"
    )
    apm = apmg.parameter_managers()[0]

    # update should update the scales, derivatives and calc Ih.
    # First check initial values are as expected.
    block_0 = multiscaler.Ih_table.blocked_data_list[0]
    block_1 = multiscaler.Ih_table.blocked_data_list[1]
    assert set(block_0.inverse_scale_factors) == set([1.0])
    assert set(block_1.inverse_scale_factors) == set([1.0])
    assert list(block_0.Ih_values) == pytest.approx(([3.0] + [5.0 / 3.0] * 3) * 2)
    assert list(block_1.Ih_values) == [2.0, 4.0, 2.0, 4.0]
    assert block_0.derivatives is None
    assert block_1.derivatives is None

    # Update block 0 and test that only block 0 is updated
    apm.set_param_vals(flex.double([2.0, 0.0, 1.0, 0.0]))
    multiscaler.update_for_minimisation(apm, 0)
    assert block_0.derivatives is not None
    assert list(block_0.inverse_scale_factors) == [2.0] * 4 + [1.0] * 4
    assert list(block_0.Ih_values) == ([1.8] + [1.0] * 3) * 2
    assert block_1.derivatives is None
    assert set(block_1.inverse_scale_factors) == set([1.0])
    assert list(block_1.Ih_values) == [2.0, 4.0, 2.0, 4.0]

    # Update block 1 and test that only block 1 is updated
    apm.set_param_vals(flex.double([2.0, 0.0, 4.0, 0.0]))
    multiscaler.update_for_minimisation(apm, 1)
    assert block_0.derivatives is not None
    assert list(block_0.inverse_scale_factors) == [2.0] * 4 + [1.0] * 4
    assert list(block_0.Ih_values) == ([1.8] + [1.0] * 3) * 2
    assert block_1.derivatives is not None
    assert list(block_1.inverse_scale_factors) == [2.0] * 2 + [4.0] * 2
    assert list(block_1.Ih_values) == [0.6, 1.2] * 2

    # Repeat for singlescaler but just test once block updating.
    singlescaler = generate_test_singlescaler()
    assert len(singlescaler.get_blocks_for_minimisation()) == 2
    block_0 = singlescaler.Ih_table.blocked_data_list[0]
    block_1 = singlescaler.Ih_table.blocked_data_list[1]
    assert list(block_0.Ih_values) == [3.0] + [5.0 / 3.0] * 3
    assert list(block_1.Ih_values) == [2.0, 4.0]

    apmg = ScalingParameterManagerGenerator(
        singlescaler.single_scalers, mode="concurrent"
    )
    apm = apmg.parameter_managers()[0]
    apm.set_param_vals(flex.double([2.0, 0.0]))
    singlescaler.update_for_minimisation(apm, 0)
    assert block_0.derivatives is not None
    assert list(block_0.inverse_scale_factors) == [2.0] * 4
    assert list(block_0.Ih_values) == ([1.5] + [5.0 / 6.0] * 3)
    assert block_1.derivatives is None
    assert set(block_1.inverse_scale_factors) == set([1.0])
    assert list(block_1.Ih_values) == [2.0, 4.0]

    # Repeat for targetscaler
    targetscaler = generate_test_targetscaler()
    assert len(targetscaler.get_blocks_for_minimisation()) == 2
    block_0 = targetscaler.Ih_table.blocked_data_list[0]
    block_1 = targetscaler.Ih_table.blocked_data_list[1]
    expected_target_Ih_vals = [3.0] + [5.0 / 3.0] * 3
    assert list(block_0.inverse_scale_factors) == [1.0] * 4
    assert list(block_0.Ih_values) == expected_target_Ih_vals
    assert list(block_1.Ih_values) == [2.0, 4.0]
    apmg = ScalingParameterManagerGenerator(
        targetscaler.active_scalers, mode="concurrent"
    )
    apm = apmg.parameter_managers()[0]
    apm.set_param_vals(flex.double([2.0, 0.0]))
    targetscaler.update_for_minimisation(apm, 0)
    assert block_0.derivatives is not None
    assert list(block_0.inverse_scale_factors) == [2.0] * 4
    # target Ih values should be unchanged
    assert list(block_0.Ih_values) == expected_target_Ih_vals
    assert block_1.derivatives is None
    assert set(block_1.inverse_scale_factors) == set([1.0])
    assert list(block_1.Ih_values) == [2.0, 4.0]


# Test scaler interface for algorithms using the scaler.
def test_round_of_outlier_rejection():
    """Test that a round of outlier rejection updates the outliers array in
    the individual scalers."""
    scaler = generate_test_multiscaler()
    initial_outliers = [False, False, False, False, True, False, False]
    assert list(scaler.single_scalers[0].outliers) == initial_outliers
    assert list(scaler.single_scalers[1].outliers) == initial_outliers
    updated_outliers = [True, False, False, False, True, False, False]
    # and use a low zmax to force the first one to be an outlier
    scaler.single_scalers[0].params.scaling_options.outlier_zmax = 0.7
    scaler.round_of_outlier_rejection()
    assert list(scaler.single_scalers[0].outliers) == updated_outliers
    assert list(scaler.single_scalers[1].outliers) == updated_outliers


def test_expand_scales_to_all_reflections():
    # Test standard multi/single and target
    pass


def test_combine_intensities():
    # Test standard multi/single and target
    pass


def test_make_ready_for_scaling():
    pass


def test_perform_scaling():
    pass


def test_perform_error_optimisation():
    pass


def test_clear_Ih_table():
    pass


def test_fix_initial_parameter():
    pass


def test_prepare_reflection_tables_for_output():
    pass


# Test relevant hidden methods.


'''def test_SingleScaler_initialisation():
    """Test that all attributes are correctly set upon initialisation"""
    p, e, r = (generated_param(), generated_exp(), generated_refl())
    exp = create_scaling_model(p, e, r)
    p.reflection_selection.method = "use_all"
    # test initialised correctly
    scaler = SingleScaler(p, exp[0], r)
    assert (
        list(scaler.suitable_refl_for_scaling_sel) == [True] * 5 + [False] + [True] * 2
    )
    # all 7 of the suitable should be within the scaling_subset
    assert list(scaler.scaling_subset_sel) == [True] * 7
    # one of these is not in the scaling selection due to being an outlier.
    assert list(scaler.scaling_selection) == [True] * 4 + [False] + [True] * 2
    assert list(scaler.outliers) == [False] * 4 + [True] + [False] * 2
    assert scaler.n_suitable_refl == 7

    # check for correct setup of global_Ih_table
    # block selection is order to extract out from suitable_reflections
    assert scaler.global_Ih_table.size == 7
    assert list(scaler.global_Ih_table.blocked_data_list[0].intensities) == [
        3.0,
        1.0,
        500.0,
        2.0,
        2.0,
        2.0,
        4.0,
    ]
    block_selection = scaler.global_Ih_table.blocked_data_list[0].block_selections[0]
    assert list(block_selection) == [2, 0, 4, 5, 6, 1, 3]

    # check for correct setup of Ih_table
    assert scaler.Ih_table.size == 6
    assert list(scaler.Ih_table.blocked_data_list[0].intensities) == [
        3.0,
        1.0,
        2.0,
        2.0,
        2.0,
        4.0,
    ]
    block_selection = scaler.Ih_table.blocked_data_list[0].block_selections[0]
    assert list(block_selection) == [2, 0, 5, 6, 1, 3]

    # check for correct data/d_values in components
    d_suitable = r["d"].select(scaler.suitable_refl_for_scaling_sel)
    decay = scaler.experiment.scaling_model.components["decay"]
    # first check 'data' contains all suitable reflections
    assert list(decay.data["d"]) == list(d_suitable)
    # Now check 'd_values' (which will be used for minim.) matches Ih_table data
    assert list(decay.d_values[0]) == list(d_suitable.select(block_selection))

    # test make ready for scaling method
    # set some new outliers and check for updated datastructures
    outlier_list = [False] * 3 + [True] * 2 + [False] * 2
    scaler.outliers = flex.bool(outlier_list)
    scaler.make_ready_for_scaling(outlier=True)
    assert scaler.Ih_table.size == 5
    assert list(scaler.Ih_table.blocked_data_list[0].intensities) == [
        3.0,
        1.0,
        2.0,
        2.0,
        2.0,
    ]
    block_selection = scaler.Ih_table.blocked_data_list[0].block_selections[0]
    assert list(block_selection) == [2, 0, 5, 6, 1]
    assert list(decay.d_values[0]) == list(d_suitable.select(block_selection))

    # test set outliers
    assert list(r.get_flags(r.flags.outlier_in_scaling)) == [False] * 8
    scaler._set_outliers()
    assert list(r.get_flags(r.flags.outlier_in_scaling)) == outlier_list + [False]


def test_multiscaler_initialisation():
    """Unit tests for the MultiScalerBase class."""
    p, e = (generated_param(), generated_exp(2))
    r1 = generated_refl(id_=0)
    r1["intensity.sum.value"] = r1["intensity"]
    r1["intensity.sum.variance"] = r1["variance"]
    r2 = generated_refl(id_=1)
    r2["intensity.sum.value"] = r2["intensity"]
    r2["intensity.sum.variance"] = r2["variance"]
    exp = create_scaling_model(p, e, [r1, r2])
    singlescaler1 = create_scaler(p, [exp[0]], [r1])
    singlescaler2 = create_scaler(p, [exp[1]], [r2])

    multiscaler = MultiScaler([singlescaler1, singlescaler2])

    # check initialisation
    assert len(multiscaler.active_scalers) == 2
    assert multiscaler.active_scalers[0] == singlescaler1
    assert multiscaler.active_scalers[1] == singlescaler2

    # check for correct setup of global Ih table
    assert multiscaler.global_Ih_table.size == 14
    assert (
        list(multiscaler.global_Ih_table.blocked_data_list[0].intensities)
        == [3.0, 1.0, 500.0, 2.0, 2.0, 2.0, 4.0] * 2
    )
    block_selections = multiscaler.global_Ih_table.blocked_data_list[0].block_selections
    assert list(block_selections[0]) == [2, 0, 4, 5, 6, 1, 3]
    assert list(block_selections[1]) == [2, 0, 4, 5, 6, 1, 3]

    # check for correct setup of Ih_table
    assert multiscaler.Ih_table.size == 12
    assert (
        list(multiscaler.Ih_table.blocked_data_list[0].intensities)
        == [3.0, 1.0, 2.0, 2.0, 2.0, 4.0] * 2
    )
    block_selections = multiscaler.Ih_table.blocked_data_list[0].block_selections
    assert list(block_selections[0]) == [2, 0, 5, 6, 1, 3]
    assert list(block_selections[1]) == [2, 0, 5, 6, 1, 3]

    # check for correct data/d_values in components
    for i, scaler in enumerate(multiscaler.active_scalers):
        d_suitable = scaler.reflection_table["d"].select(
            scaler.suitable_refl_for_scaling_sel
        )
        decay = scaler.experiment.scaling_model.components["decay"]
        # first check 'data' contains all suitable reflections
        assert list(decay.data["d"]) == list(d_suitable)
        # Now check 'd_values' (which will be used for minim.) matches Ih_table data
        assert list(decay.d_values[0]) == list(d_suitable.select(block_selections[i]))


def test_targetscaler_initialisation():
    """Unit tests for the MultiScalerBase class."""
    p, e = (generated_param(), generated_exp(2))
    r1 = generated_refl(id_=0)
    p.reflection_selection.method = "intensity_ranges"

    r1["intensity.sum.value"] = r1["intensity"]
    r1["intensity.sum.variance"] = r1["variance"]
    r2 = generated_refl(id_=1)
    r2["intensity.sum.value"] = r2["intensity"]
    r2["intensity.sum.variance"] = r2["variance"]
    exp = create_scaling_model(p, e, [r1, r2])
    singlescaler1 = SingleScaler(p, exp[0], r1, for_multi=True)
    singlescaler2 = SingleScaler(p, exp[1], r2, for_multi=True)

    # singlescaler2.experiments.scaling_model.set_scaling_model_as_scaled()

    targetscaler = TargetScaler(
        scaled_scalers=[singlescaler1], unscaled_scalers=[singlescaler2]
    )

    # check initialisation
    assert len(targetscaler.active_scalers) == 1
    assert len(targetscaler.single_scalers) == 1
    assert targetscaler.active_scalers[0] == singlescaler2
    assert targetscaler.single_scalers[0] == singlescaler1

    # check for correct setup of global Ih table
    assert targetscaler.global_Ih_table.size == 7  # only for active scalers
    assert list(targetscaler.global_Ih_table.blocked_data_list[0].intensities) == [
        3.0,
        1.0,
        500.0,
        2.0,
        2.0,
        2.0,
        4.0,
    ]
    block_selections = targetscaler.global_Ih_table.blocked_data_list[
        0
    ].block_selections
    assert list(block_selections[0]) == [2, 0, 4, 5, 6, 1, 3]

    # check for correct setup of Ih_table
    assert targetscaler.Ih_table.size == 6
    assert list(targetscaler.Ih_table.blocked_data_list[0].intensities) == [
        3.0,
        1.0,
        2.0,
        2.0,
        2.0,
        4.0,
    ]
    block_selections = targetscaler.Ih_table.blocked_data_list[0].block_selections
    assert list(block_selections[0]) == [2, 0, 5, 6, 1, 3]

    # check for correct setup of target Ih_Table
    assert targetscaler.target_Ih_table.size == 6
    assert list(targetscaler.target_Ih_table.blocked_data_list[0].intensities) == [
        3.0,
        1.0,
        2.0,
        2.0,
        2.0,
        4.0,
    ]
    block_selections = targetscaler.target_Ih_table.blocked_data_list[
        0
    ].block_selections
    assert list(block_selections[0]) == [
        2,
        0,
        4,
        5,
        1,
        3,
    ]  # different as taget_Ih_table
    # not created with indices lists.

    block_selections = targetscaler.Ih_table.blocked_data_list[0].block_selections
    # check for correct data/d_values in components
    for i, scaler in enumerate(targetscaler.active_scalers):
        d_suitable = scaler.reflection_table["d"].select(
            scaler.suitable_refl_for_scaling_sel
        )
        decay = scaler.experiment.scaling_model.components["decay"]
        # first check 'data' contains all suitable reflections
        assert list(decay.data["d"]) == list(d_suitable)
        # Now check 'd_values' (which will be used for minim.) matches Ih_table data
        assert list(decay.d_values[0]) == list(d_suitable.select(block_selections[i]))

    # but shouldn't have updated other
    assert (
        targetscaler.single_scalers[0]
        .experiment.scaling_model.components["decay"]
        .d_values
        == []
    )


def test_SingleScaler_expand_scales_to_all_reflections(mock_apm):
    p, e, r = (generated_param(), generated_exp(), generated_refl())
    exp = create_scaling_model(p, e, r)
    p.reflection_selection.method = "use_all"
    scaler = SingleScaler(p, exp[0], r)
    # test expand to all reflections method. First check scales are all 1, then
    # update a component to simulate a minimisation result, then check that
    # scales are set only in all suitable reflections (as it may not be possible
    # to calculate scales for unsuitable reflections!)
    # Must also update the scales in the global_Ih_table
    assert list(scaler.reflection_table["inverse_scale_factor"]) == [1.0] * 8
    scaler.experiment.scaling_model.components["scale"].parameters = flex.double([2.0])
    scaler.expand_scales_to_all_reflections(calc_cov=False)
    assert (
        list(scaler.reflection_table["inverse_scale_factor"])
        == [2.0] * 5 + [1.0] + [2.0] * 2
    )
    assert (
        list(scaler.global_Ih_table.blocked_data_list[0].inverse_scale_factors)
        == [2.0] * 7
    )

    assert list(scaler.reflection_table["inverse_scale_factor_variance"]) == [0.0] * 8
    # now try again
    apm = Mock()
    apm.n_active_params = 2
    var_list = [1.0, 0.1, 0.1, 0.5]
    apm.var_cov_matrix = flex.double(var_list)
    apm.var_cov_matrix.reshape(flex.grid(2, 2))
    scaler.update_var_cov(apm)
    assert scaler.var_cov_matrix[0, 0] == var_list[0]
    assert scaler.var_cov_matrix[0, 1] == var_list[1]
    assert scaler.var_cov_matrix[1, 0] == var_list[2]
    assert scaler.var_cov_matrix[1, 1] == var_list[3]
    assert scaler.var_cov_matrix.non_zeroes == 4
    scaler.expand_scales_to_all_reflections(calc_cov=True)
    assert list(
        scaler.reflection_table["inverse_scale_factor_variance"]
    ) == pytest.approx(
        [2.53320, 1.07106, 1.08125, 1.23219, 1.15442, 0.0, 1.0448, 1.0448], 1e-4
    )

    # Second case - when var_cov_matrix is only part of full matrix.
    p, e, r = (generated_param(), generated_exp(), generated_refl())
    exp = create_scaling_model(p, e, r)
    p.reflection_selection.method = "use_all"
    scaler = SingleScaler(p, exp[0], r)
    apm = mock_apm
    scaler.update_var_cov(apm)
    assert scaler.var_cov_matrix.non_zeroes == 1
    assert scaler.var_cov_matrix[0, 0] == 2.0
    assert scaler.var_cov_matrix.n_cols == 2
    assert scaler.var_cov_matrix.n_rows == 2
    assert scaler.var_cov_matrix.non_zeroes == 1


def generated_refl_2(exclude_refl=True):
    """Generate a reflection table."""
    # these miller_idx/d_values don't make physical sense, but I didn't want to
    # have to write the tests for lots of reflections.
    reflections = flex.reflection_table()
    reflections["intensity"] = flex.double([1.0, 10.0, 100.0, 1.0])
    reflections["variance"] = flex.double([1.0, 10.0, 100.0, 1.0])
    reflections["miller_index"] = flex.miller_index(
        [(1, 0, 0), (0, 0, 1), (2, 0, 0), (2, 2, 2)]
    )  # don't change
    reflections["d"] = flex.double([0.8, 2.0, 2.0, 0.0])  # don't change
    reflections["d"] = flex.double([0.8, 2.0, 2.1, 0.1])
    reflections["Esq"] = flex.double([1.0, 1.0, 1.0, 1.0])
    reflections["inverse_scale_factor"] = flex.double([1.0, 1.0, 1.0, 1.0])
    reflections["id"] = flex.int(4, 0)
    reflections["xyzobs.px.value"] = flex.vec3_double(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 10.0), (0.0, 0.0, 10.0)]
    )
    reflections["s1"] = flex.vec3_double(
        [(0.0, 0.1, 1.0), (0.0, 0.1, 1.0), (0.0, 0.1, 1.0), (0.0, 0.1, 1.0)]
    )
    if exclude_refl:
        integrated_list = flex.bool([True, True, False, False])
        bad_list = flex.bool([False, False, True, True])
    else:
        integrated_list = flex.bool(4, True)
        bad_list = flex.bool(4, False)
    reflections.set_flags(integrated_list, reflections.flags.integrated)
    reflections.set_flags(bad_list, reflections.flags.bad_for_scaling)
    return reflections


# @pytest.mark.xfail(reason='need to rework mcok error model')
def test_update_error_model(mock_errormodel, mock_errormodel2):
    """Test the update_error_model method"""
    p, e, r = (generated_param(), generated_exp(), generated_refl())
    exp = create_scaling_model(p, e, r)
    p.reflection_selection.method = "use_all"
    # test initialised correctly
    scaler = SingleScaler(p, exp[0], r)
    block = scaler.global_Ih_table.blocked_data_list[0]
    original_vars = block.variances
    # test update error model - should update weights in global Ih
    # as will be setting different things in Ih_table and reflection table, split
    # up the test to use two different error models.
    scaler._update_error_model(mock_errormodel)
    assert list(block.variances) == list(original_vars)
    newvars = flex.double(range(1, 8))
    assert list(block.block_selections[0]) == [2, 0, 4, 5, 6, 1, 3]
    assert list(block.weights) == list(1.0 / newvars)
    assert scaler.experiment.scaling_model.error_model is mock_errormodel

    # now test for updating of reflection table
    # do again with second errormodel
    scaler.global_Ih_table.reset_error_model()
    scaler._update_error_model(mock_errormodel2)
    assert list(block.variances) == list(original_vars)
    newvars = flex.double(range(1, 9))
    assert list(block.block_selections[0]) == [2, 0, 4, 5, 6, 1, 3]
    # [2, 3, 4, 5, 6, 7, 8] < set these in ^ these positions (taking into account
    # the one non-suitable refl at index 5)
    assert list(block.weights) == list(1.0 / newvars)[:-1]
    assert scaler.experiment.scaling_model.error_model is mock_errormodel2


def test_SingleScaler_combine_intensities():
    """test combine intensities method"""
    p, e, r = (generated_param(), generated_exp(), generated_refl_for_comb())
    exp = create_scaling_model(p, e, r)
    p.reflection_selection.method = "use_all"
    scaler = SingleScaler(p, exp[0], r)
    scaler.combine_intensities()

    # The input makes the profile intensities best - so check these are set in the
    # reflection table and global_Ih_table
    assert list(scaler.reflection_table["intensity"]) == list(r["intensity.prf.value"])
    assert list(scaler.reflection_table["variance"]) == list(
        r["intensity.prf.variance"]
    )
    block = scaler.global_Ih_table.blocked_data_list[0]
    block_sel = block.block_selections[0]
    suitable = scaler.suitable_refl_for_scaling_sel
    assert list(block.intensities) == list(
        scaler.reflection_table["intensity"].select(suitable).select(block_sel)
    )
    assert list(block.variances) == list(
        scaler.reflection_table["variance"].select(suitable).select(block_sel)
    )


def test_NullScaler():
    """Test for successful creation of NullScaler."""
    p, e, r = (generated_param(), generated_exp(), generated_refl())
    exp = create_scaling_model(p, e, r)
    _ = NullScaler(p, exp[0], r)
    # What exactly should be tested here?


def test_sf_variance_calculation():
    """Test the calculation of scale factor variances."""
    test_experiments = generated_exp()
    test_params = generated_param()
    assert len(test_experiments) == 1
    experiments = create_scaling_model(test_params, test_experiments, [None])
    components = experiments[0].scaling_model.components
    rt = flex.reflection_table()
    d1 = 1.0
    d2 = 2.0
    d3 = 3.0
    rt["d"] = flex.double([d1, d2, d3])
    rt["id"] = flex.int([0, 0, 0])
    experiments[0].scaling_model.configure_components(rt, experiments[0], test_params)
    components["scale"].update_reflection_data()
    _, d = components["scale"].calculate_scales_and_derivatives()
    assert list(d.col(0)) == [(0, 1.0), (1, 1.0), (2, 1.0)]
    components["decay"].update_reflection_data()
    s, d = components["decay"].calculate_scales_and_derivatives()
    assert list(d.col(0)) == [
        (0, 1.0 / (2.0 * d1 * d1)),
        (1, 1.0 / (2.0 * d2 * d2)),
        (2, 1.0 / (2.0 * d3 * d3)),
    ]
    var_cov = sparse.matrix(2, 2)
    a = 0.2
    b = 0.3
    c = 0.1
    var_cov[0, 0] = a
    var_cov[0, 1] = c
    var_cov[1, 0] = c
    var_cov[1, 1] = b
    variances = calc_sf_variances(components, var_cov)
    assert list(variances) == pytest.approx(
        [
            b / (4.0 * (d1 ** 4.0)) + c / (d1 ** 2.0) + a,
            b / (4.0 * (d2 ** 4.0)) + c / (d2 ** 2.0) + a,
            b / (4.0 * (d3 ** 4.0)) + c / (d3 ** 2.0) + a,
        ]
    )'''

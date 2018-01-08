'''
Define a Data_Manager object used for calculating scaling factors
'''
from __future__ import print_function
import copy
from dials.array_family import flex
from cctbx import miller, crystal
#from minimiser_functions import error_scale_LBFGSoptimiser
from dials.util.options import flatten_experiments, flatten_reflections
import numpy as np
import cPickle as pickle
from target_function import target_function, target_function_fixedIh, xds_target_function_log
import basis_functions as bf
from scaling_utilities import calc_s2d, sph_harm_table
from Wilson_outlier_test import calculate_wilson_outliers, calc_normE2
import scale_factor as SF
from reflection_weighting import Weighting
from target_Ih import SingleIhTable, JointIhTable
import minimiser_functions as mf
from collections import OrderedDict
from scitbx import sparse
from aimless_outlier_rejection import reject_outliers

import logging
logger = logging.getLogger('dials.scale')

class DataManagerUtilities(object):
  '''Define a class of utility methods for the Data Manager.'''

  @staticmethod
  def _reflection_table_setup(initial_keys, reflections):
    '''initial filter to select integrated reflections.
    input - initial_keys and reflection table
    output - reflection table'''
    reflections = reflections.select(reflections.get_flags(
      reflections.flags.integrated))
    if 'intensity.prf.variance' in initial_keys:
      reflections = reflections.select(reflections['intensity.prf.variance'] > 0)
    if 'intensity.sum.variance' in initial_keys:
      reflections = reflections.select(reflections['intensity.sum.variance'] > 0)
    if not 'inverse_scale_factor' in initial_keys:
      reflections['inverse_scale_factor'] = (flex.double([1.0] * len(reflections)))
    reflections['Ih_values'] = flex.double([0.0] * len(reflections))
    return reflections

  @classmethod
  def _extract_reflections_for_scaling(cls, reflection_table, params,
    error_model_params=None):
    '''select the reflections with non-zero weight and update scale weights
    object.'''
    n_refl = len(reflection_table)
    weights_for_scaling = cls._update_weights_for_scaling(reflection_table,
      params, error_model_params=error_model_params)
    sel = weights_for_scaling.weights > 0.0
    sel1 = reflection_table['Esq'] > params.reflection_selection.E2min
    sel2 = reflection_table['Esq'] < params.reflection_selection.E2max
    selection = sel & sel1 & sel2
    reflections_for_scaling = reflection_table.select(selection)
    weights_for_scaling.weights= weights_for_scaling.weights.select(selection)
    msg = ('{0} reflections were selected for scale factor determination {sep}'
      'out of {1} reflections. This was based on selection criteria of {sep}'
      'E2min = {2}, E2max = {3}, Isigma_min = {4}, dmin = {5}. {sep}').format(
      reflections_for_scaling.size(), n_refl, params.reflection_selection.E2min,
      params.reflection_selection.E2max, params.reflection_selection.Isigma_min, 
      params.reflection_selection.d_min, sep='\n')
    logger.info(msg)
    return reflections_for_scaling, weights_for_scaling, selection

  @staticmethod
  def _update_weights_for_scaling(reflection_table, params, weights_filter=True,
    error_model_params=None):
    '''set the weights of each reflection to be used in scaling'''
    weights_for_scaling = Weighting(reflection_table)
    logger.info('Updating the weights associated with the intensities. \n')
    if weights_filter:
      weights_for_scaling.apply_Isigma_cutoff(reflection_table,
        params.reflection_selection.Isigma_min)
      weights_for_scaling.apply_dmin_cutoff(reflection_table,
        params.reflection_selection.d_min)
    weights_for_scaling.remove_wilson_outliers(reflection_table)
    if error_model_params:
      weights_for_scaling.apply_aimless_error_model(reflection_table,
        error_model_params)
    return weights_for_scaling

  @staticmethod
  def _map_indices_to_asu(reflection_table, experiments, params):
    '''Create a miller_set object, map to the asu and create a sorted
       reflection table, sorted by asu miller index'''
    u_c = experiments.crystal.get_unit_cell().parameters()
    if params.scaling_options.force_space_group:
      sg_from_file = experiments.crystal.get_space_group().info()
      s_g_symbol = params.scaling_options.force_space_group
      crystal_symmetry = crystal.symmetry(unit_cell=u_c,
        space_group_symbol=s_g_symbol)
      msg = ('WARNING: Manually overriding space group from {0} to {1}. {sep}'
        'If the reflection indexing in these space groups is different, {sep}'
        'bad things may happen!!! {sep}').format(sg_from_file, s_g_symbol, sep='\n')
      logger.info(msg)
    else:
      s_g = experiments.crystal.get_space_group()
      crystal_symmetry = crystal.symmetry(unit_cell=u_c, space_group=s_g)
    miller_set = miller.set(crystal_symmetry=crystal_symmetry,
      indices=reflection_table['miller_index'], anomalous_flag=True)
    miller_set_in_asu = miller_set.map_to_asu()
    reflection_table["asu_miller_index"] = miller_set_in_asu.indices()
    permuted = (miller_set.map_to_asu()).sort_permutation(by_value='packed_indices')
    reflection_table = reflection_table.select(permuted)
    return reflection_table

  @classmethod
  def _select_optimal_intensities(cls, reflection_table, params):
    '''method to choose which intensities to use for scaling'''
    if (params.scaling_options.integration_method == 'sum' or
        params.scaling_options.integration_method == 'prf'):
      intstr = params.scaling_options.integration_method
      reflection_table['intensity'] = (reflection_table['intensity.'+intstr+'.value']
                                       * reflection_table['lp']
                                       / reflection_table['dqe'])
      reflection_table['variance'] = (reflection_table['intensity.'+intstr+'.variance']
                                      * (reflection_table['lp']**2)
                                      / (reflection_table['dqe']**2))
      logger.info(('{0} intensity values will be used for scaling. {sep}').format(
        'Profile fitted' if intstr == 'prf' else 'Summation integrated', sep='\n'))
    #perform a combined prf/sum in a similar fashion to aimless
    elif params.scaling_options.integration_method == 'combine':
      int_prf = (reflection_table['intensity.prf.value']
                 * reflection_table['lp'] / reflection_table['dqe'])
      int_sum = (reflection_table['intensity.sum.value']
                 * reflection_table['lp'] / reflection_table['dqe'])
      var_prf = (reflection_table['intensity.prf.variance']
                 * (reflection_table['lp']**2) / (reflection_table['dqe']**2))
      var_sum = (reflection_table['intensity.sum.variance']
                 * (reflection_table['lp']**2) / (reflection_table['dqe']**2))
      Imid = max(int_sum)/2.0
      weight = 1.0/(1.0 + ((int_prf/Imid)**3))
      reflection_table['intensity'] = ((weight * int_prf) + ((1.0 - weight) * int_sum))
      reflection_table['variance'] = ((weight * var_prf) + ((1.0 - weight) * var_sum))
      msg = ('Combined profile/summation intensity values will be used for {sep}'
      'scaling, with an Imid of {0}. {sep}').format(Imid, sep='\n')
      logger.info(msg)
    else:
      logger.info('Invalid integration_method choice, using default profile fitted intensities')
      params.scaling_options.integration_method = 'prf'
      cls._select_optimal_intensities(reflection_table, params)
    return reflection_table


class ScalingDataManager(DataManagerUtilities):
  '''Data Manager takes a params parsestring containing the parsed
     integrated.pickle and integrated_experiments.json files'''
  def __init__(self, reflections, experiments, params):
    super(ScalingDataManager, self).__init__()
    'General attributes relevant for all parameterisations'
    logger.info('\nInitialising a data manager instance. \n')
    self._experiments = experiments
    self._params = params
    self._initial_keys = [key for key in reflections.keys()]
    'choose intensities, map to asu, assign unique refl. index'
    reflection_table = self._reflection_table_setup(self._initial_keys, reflections)
    reflection_table = self._select_optimal_intensities(reflection_table, self.params)
    reflection_table = self._map_indices_to_asu(reflection_table,
      self.experiments, self.params)
    #add in a method that automatically updates the weights when the reflection table is changed?
    'assign initial weights (will be statistical weights at this point)'
    reflection_table = calc_normE2(reflection_table, self.experiments)
    reflection_table['wilson_outlier_flag'] = calculate_wilson_outliers(
      reflection_table)
    self._reflection_table = reflection_table
    #extra attributes that need to be filled in by subclasses.
    self.Ih_table = None
    self.g_parameterisation = None
    self.apm = None 

  @property
  def experiments(self):
    return self._experiments

  @property
  def reflection_table(self):
    return self._reflection_table

  @property
  def params(self):
    return self._params

  def update_for_minimisation(self, apm):
    '''update the scale factors and Ih for the next iteration of minimisation'''
    basis_fn = self.get_basis_function(apm)
    apm.active_derivatives = basis_fn[1]
    self.Ih_table.inverse_scale_factors = basis_fn[0]
    self.Ih_table.calc_Ih()

  def get_target_function(self, apm):
    '''call the target function'''
    return target_function(self, apm).return_targets()

  def get_basis_function(self, apm):
    '''call thebasis function'''
    return bf.basis_function(self, apm).return_basis()

  def expand_scales_to_all_reflections(self):
    '''method to be filled in by subclasses'''
    pass

  def clean_reflection_table(self):
    '''method to be filled in by subclasses'''
    pass

  '''define a few methods for saving the data'''
  def save_reflection_table(self, filename):
    ''' Save the reflections to file. '''
    self.reflection_table.as_pickle(filename)

  def save_data_manager(self, filename):
    ''' Save the data manager to file. '''
    data_file = open(filename, 'w')
    pickle.dump(self, data_file)
    data_file.close()


class AimlessDataManager(ScalingDataManager):
  '''Data Manager subclass for implementing XDS parameterisation'''
  def __init__(self, reflections, experiments, params):
    super(AimlessDataManager, self).__init__(reflections, experiments, params)
    'initialise g-value objects'
    (self.g_absorption, self.g_scale, self.g_decay) = (None, None, None)
    self.g_parameterisation = OrderedDict()
    '''bin reflections, determine outliers, extract reflections and weights for
    scaling and set normalised values.'''
    if self.params.scaling_options.reject_outliers:
      weights = self._update_weights_for_scaling(self._reflection_table,
        self.params).weights
      self.Ih_table = SingleIhTable(self.reflection_table, weights)
      n_outliers = self.round_of_outlier_rejection()
      msg = ('An initial outlier rejection has been performed, {0} outliers {sep}'
        'were found which have been removed from the dataset. {sep}'.format(
        n_outliers, sep='\n'))
      logger.info(msg)
    self.initialise_scale_factors()
    self.select_reflections_for_scaling()
    logger.info('Completed initialisation of aimless data manager. \n' + '*'*40 + '\n')

  def select_reflections_for_scaling(self):
    (refl_for_scaling, weights_for_scaling, selection) = (
      self._extract_reflections_for_scaling(self.reflection_table, self.params,
      error_model_params=self.params.scaling_options.error_model_params))
    self.Ih_table = SingleIhTable(refl_for_scaling, weights_for_scaling.weights)
    '''refactor the next two operations into extract_reflections?
    reset the normalised values within the scale_factor object to current'''
    self.g_scale.update_reflection_data(
      normalised_values=refl_for_scaling['normalised_rotation_angle'])
    #self.g_scale.normalised_values = (refl_for_scaling['normalised_rotation_angle'])
    if self.params.parameterisation.decay_term:
      self.g_decay.update_reflection_data(dvalues=refl_for_scaling['d'],
        normalised_values=refl_for_scaling['normalised_time_values'])
      #self.g_decay.normalised_values = (refl_for_scaling['normalised_time_values'])
      #self.g_decay.d_values = refl_for_scaling['d']
    if self.params.parameterisation.absorption_term:
      sph_harm_table_T = self.sph_harm_table.transpose()
      sel_sph_harm_table = sph_harm_table_T.select_columns(selection.iselection())
      self.g_absorption.update_reflection_data(sel_sph_harm_table.transpose())

  def update_error_model(self):
    error_model_params = mf.error_scale_LBFGSoptimiser(self.Ih_table,
      flex.double([1.0,0.01])).x
    self.params.scaling_options.error_model_params = error_model_params
    (reflections_for_scaling, weights_for_scaling, selection) = (
      self._extract_reflections_for_scaling(self.reflection_table, self.params,
      error_model_params=self.params.scaling_options.error_model_params))
    self.Ih_table = SingleIhTable(reflections_for_scaling,
      weights_for_scaling.weights)

  def initialise_scale_factors(self):
    '''initialise scale factors and add to self.active_parameters'''
    logger.info('Initialising scale factor objects. \n')
    self.initialise_scale_term(self.reflection_table)
    self.initialise_decay_term(self.reflection_table)
    self.initialise_absorption_scales(self.reflection_table,
      self.params.parameterisation.lmax)

  def initialise_decay_term(self, refl_table):
    '''calculate the relative, normalised rotation angle. Here this is called
    normalised time to allow a different rotation interval compare to the scale
    correction. A SmoothScaleFactor_1D object is then initialised'''
    if self.params.parameterisation.decay_term:
      if self.params.parameterisation.B_factor_interval:
        rot_int = self.params.parameterisation.B_factor_interval + 0.001
      else:
        rot_int = 20.0 + 0.001
      osc_range = self.experiments.scan.get_oscillation_range()
      if ((osc_range[1] - osc_range[0]) / rot_int) % 1 < 0.33:
        #if last bin less than 33% filled'
        n_phi_bins = int((osc_range[1] - osc_range[0]) / rot_int)
        'increase rotation interval slightly'
        rot_int = (osc_range[1] - osc_range[0])/float(n_phi_bins) + 0.001
      'extend by 0.001 to make sure all datapoints within min/max'
      one_osc_width = self.experiments.scan.get_oscillation()[1]
      refl_table['normalised_time_values'] = ((one_osc_width
        * refl_table['xyzobs.px.value'].parts()[2]) + 0.001) / rot_int
      '''define the highest and lowest gridpoints:
      go out two further than the max/min int values'''
      high_param_val = int((max(refl_table['normalised_time_values'])//1)+2)#was +2
      low_param_val = int((min(refl_table['normalised_time_values'])//1)-1)#was -1
      n_decay_param = high_param_val - low_param_val + 1
      self.g_decay = SF.SmoothBScaleFactor1D(0.0, n_decay_param)#, refl_table['d'])
      self.g_parameterisation['g_decay'] = self.g_decay
      msg = ('The decay term ScaleFactor object was successfully initialised. {sep}'
        'The B-factor parameter interval has been set to {0} degrees. {sep}'
        '{1} parameters will be used to parameterise the time-dependent decay. {sep}'
        ).format(rot_int, n_decay_param, sep='\n')
      logger.info(msg)

  def initialise_scale_term(self, refl_table):
    '''calculate the relative, normalised rotation angle.
    A SmoothScaleFactor_1D object is then initialised'''
    if self.params.parameterisation.rotation_interval:
      rot_int = self.params.parameterisation.rotation_interval + 0.001
    else:
      rot_int = 15.0 + 0.001
    osc_range = self.experiments.scan.get_oscillation_range()
    if ((osc_range[1] - osc_range[0])/ rot_int) % 1 < 0.33:
      #if last bin less than 33% filled
      n_phi_bins = int((osc_range[1] - osc_range[0])/ rot_int)
      'increase rotation interval slightly'
      rot_int = (osc_range[1] - osc_range[0])/float(n_phi_bins) + 0.001
    'extend by 0.001 to make sure all datapoints within min/max'
    one_osc_width = self.experiments.scan.get_oscillation()[1]
    refl_table['normalised_rotation_angle'] = ((one_osc_width
      * refl_table['xyzobs.px.value'].parts()[2]) + 0.001) / rot_int
    '''define the highest and lowest gridpoints:
    go out two further than the max/min int values'''
    high_param_val = int((max(refl_table['normalised_rotation_angle'])//1)+2)
    low_param_val = int((min(refl_table['normalised_rotation_angle'])//1)-1)
    n_scale_param = high_param_val - low_param_val + 1
    self.g_scale = SF.SmoothScaleFactor1D(1.0, n_scale_param)
    self.g_parameterisation['g_scale'] = self.g_scale
    msg = ('The scale term ScaleFactor object was successfully initialised. {sep}'
      'The scale term parameter interval has been set to {0} degrees. {sep}'
      '{1} parameters will be used to parameterise the time-dependent scale. {sep}'
        ).format(rot_int, n_scale_param, sep='\n')
    logger.info(msg)

  def initialise_absorption_scales(self, reflection_table, lmax):
    if self.params.parameterisation.absorption_term:
      reflection_table = calc_s2d(reflection_table, self.experiments)
      n_abs_params = 0
      for i in range(lmax):
        n_abs_params += (2*(i+1))+1
      self.sph_harm_table = sph_harm_table(reflection_table, lmax)
      self.g_absorption = SF.SHScaleFactor(0.0, n_abs_params)#, self.sph_harm_table)
      #self.g_absorption.update_reflection_data(self.sph_harm_table)
      self.g_parameterisation['g_absorption'] = self.g_absorption
      msg = ('The absorption term ScaleFactor object was successfully initialised. {sep}'
        'The absorption term will be parameterised by a set of spherical {sep}'
        'harmonics up to an lmax of {0} ({1} parameters). {sep}'
        ).format(lmax, n_abs_params, sep='\n')
      logger.info(msg)
    else:
      reflection_table['phi'] = (reflection_table['xyzobs.px.value'].parts()[2]
                                 * self.experiments.scan.get_oscillation()[1])
    

  def calc_absorption_constraint(self, apm):
    #this should only be called if g_absorption in active params
    idx = apm.active_parameterisation.index('g_absorption')
    start_idx = apm.cumulative_active_params[idx]
    end_idx = apm.cumulative_active_params[idx+1]
    weight = 1e5
    abs_params = apm.x[start_idx:end_idx]
    residual = (weight * (abs_params)**2)
    gradient = (2.0 * weight * abs_params)
    #need to make sure gradient is returned is same size as gradient calculated in target fn-
    #would only be triggered if refining absorption as same time as another scale factor.
    gradient_vector = flex.double([])
    for i, param in enumerate(apm.active_parameterisation):
      if param != 'g_absorption':
        gradient_vector.extend(flex.double([0.0]*apm.active_params_list[i]))
      elif param == 'g_absorption':
        gradient_vector.extend(gradient)
    return (residual, gradient_vector)

  def normalise_scales_and_B(self):
    if self.params.parameterisation.decay_term:
      maxB = max(flex.double(np.log(self.g_decay.inverse_scales))
                 * 2.0 * (self.g_decay.d_values**2))
      self.g_decay.parameters -= flex.double([maxB] * self.g_decay.n_params)
      self.g_decay.calculate_scales_and_derivatives()
    sel = (self.g_scale.normalised_values == min(self.g_scale.normalised_values))
    initial_scale = self.g_scale.inverse_scales.select(sel)[0]
    self.g_scale.parameters /= initial_scale
    self.g_scale.calculate_scales_and_derivatives()

  def round_of_outlier_rejection(self):
    sel = flex.bool([True]*len(self.reflection_table))
    _, indices_of_outliers = reject_outliers(self, max_deviation=6.0)
    sel.set_selected(flex.size_t(indices_of_outliers), flex.bool([False]*len(indices_of_outliers)))
    self._reflection_table = self._reflection_table.select(sel)
    new_weights = self._update_weights_for_scaling(self.reflection_table,
      self.params, weights_filter=False, error_model_params=None).weights
    self.Ih_table = SingleIhTable(self.reflection_table, new_weights)
    self._reflection_table = self._reflection_table.select(new_weights != 0)
    self._reflection_table['Ih_values'] = self.Ih_table.Ih_values
    return len(indices_of_outliers)

  def expand_scales_to_all_reflections(self):
    if not self.params.scaling_options.multi_mode:
      self.normalise_scales_and_B()
    "recalculate scales for reflections in sorted_reflection table"
    expanded_scale_factors = flex.double([1.0]*len(self.reflection_table))
    self.g_scale.update_reflection_data(
      normalised_values=self.reflection_table['normalised_rotation_angle'])
    #self.g_scale.normalised_values = self.reflection_table['normalised_rotation_angle']
    self.g_scale.calculate_scales()
    expanded_scale_factors *= self.g_scale.inverse_scales
    if self.params.parameterisation.decay_term:
      self.g_decay.update_reflection_data(dvalues=self.reflection_table['d'],
        normalised_values=self.reflection_table['normalised_time_values'])
      #self.g_decay.normalised_values = self.reflection_table['normalised_time_values']
      #self.g_decay.d_values = self.reflection_table['d']
      self.g_decay.calculate_scales()
      expanded_scale_factors *= self.g_decay.inverse_scales
    if self.params.parameterisation.absorption_term:
      self.g_absorption.update_reflection_data(self.sph_harm_table)
      self.g_absorption.calculate_scales_and_derivatives()
      expanded_scale_factors  *= self.g_absorption.inverse_scales
    self._reflection_table['inverse_scale_factor'] = expanded_scale_factors
    logger.info(('Scale factors determined during minimisation have now been applied {sep}'
      'to all reflections. {sep}').format(sep='\n'))
    weights = self._update_weights_for_scaling(self.reflection_table,
      self.params, weights_filter=False, error_model_params=None).weights
    self.Ih_table = SingleIhTable(self.reflection_table, weights)
    self._reflection_table = self._reflection_table.select(weights != 0.0)
    self._reflection_table['Ih_values'] = self.Ih_table.Ih_values
    if (self.params.scaling_options.reject_outliers and not 
        self.params.scaling_options.multi_mode):
      n_outliers = self.round_of_outlier_rejection()
      msg = ('A final outlier rejection has been performed, {0} outliers {sep}'
        'were found which have been removed from the dataset. {sep}'.format(
        n_outliers, sep='\n'))
      logger.info(msg)
    logger.info('A new best estimate for I_h for all reflections has now been calculated. \n')

  def clean_reflection_table(self):
    self._initial_keys.append('inverse_scale_factor')
    self._initial_keys.append('phi')
    for key in self.reflection_table.keys():
      if not key in self._initial_keys:
        del self.reflection_table[key]
    #keep phi column for now for comparing to aimless
    added_columns = ['s2', 's2d', 'decay_factor', 'angular_scale_factor',
      'normalised_rotation_angle', 'normalised_time_values', 'h_index',
      'wilson_outlier_flag', 'centric_flag', 'absorption_factor']
    for key in added_columns:
      del self.reflection_table[key]


class KB_Data_Manager(ScalingDataManager):
  def __init__(self, reflections, experiments, params):
    super(KB_Data_Manager, self).__init__(reflections, experiments, params)
    self.active_parameters = flex.double([])
    (reflections_for_scaling, weights_for_scaling, selection) = (
      self._extract_reflections_for_scaling(self.reflection_table, self.params))
    self.Ih_table = SingleIhTable(reflections_for_scaling, weights_for_scaling.weights)
    self.g_parameterisation = OrderedDict()
    if self.params.parameterisation.scale_term:
      self.g_scale = SF.KScaleFactor(1.0)
      self.g_scale.update_reflection_data(n_refl=self.Ih_table.size)
      self.g_parameterisation['g_scale'] = self.g_scale
    if self.params.parameterisation.decay_term:
      self.g_decay = SF.BScaleFactor(0.0)#, reflections_for_scaling['d'])
      self.g_decay.update_reflection_data(dvalues=reflections_for_scaling['d'])
      self.g_parameterisation['g_decay'] = self.g_decay
    logger.info('Completed initialisation of KB data manager. \n' + '*'*40 + '\n')

  def get_target_function(self, apm):
    '''override the target function method for fixed Ih'''
    return target_function_fixedIh(self, apm).return_targets()

  def update_for_minimisation(self, apm):
    '''update the scale factors and Ih for the next iteration of minimisation'''
    basis_fn = self.get_basis_function(apm)
    apm.active_derivatives = basis_fn[1]
    self.Ih_table.inverse_scale_factors = basis_fn[0]
    #note - we don't calculate Ih here as using a target instead

  def expand_scales_to_all_reflections(self):
    expanded_scale_factors = flex.double([1.0]*len(self.reflection_table))
    if self.params.parameterisation.scale_term:
      #self.g_scale.n_refl = len(self.reflection_table)
      self.g_scale.update_reflection_data(n_refl=len(self.reflection_table))
      self.g_scale.calculate_scales_and_derivatives()
      expanded_scale_factors *= self.g_scale.inverse_scales
    if self.params.parameterisation.decay_term:
      self.g_decay.update_reflection_data(dvalues=self.reflection_table['d'])
      #self.g_decay.d_values = self.reflection_table['d']
      self.g_decay.calculate_scales_and_derivatives()
      expanded_scale_factors *= self.g_decay.inverse_scales
      logger.info(('Scale factors determined during minimisation have now been applied {sep}'
        'to all reflections. Scale factors were determined to be K = {0:.4f}, {sep}'
        'B = {1:.4f}. {sep}').format(list(self.g_scale.parameters)[0], 
        list(self.g_decay.parameters)[0], sep='\n'))
    else:
      logger.info(('The scale factor determined during minimisation has now been applied {sep}'
        'to all reflections. The scale factor was determined to be K = {0:.4f}. {sep}'
        ).format(list(self.g_scale.inverse_scales)[0], sep='\n'))
    self._reflection_table['inverse_scale_factor'] = expanded_scale_factors


class active_parameter_manager(object):
  ''' object to manage the current active parameters during minimisation.
  Separated out to provide a consistent interface between the data manager and
  minimiser.
  Takes in a data manager, needed to access SF objects through g_parameterisation,
  and a param_name list indicating the active parameters.'''
  def __init__(self, Data_Manager, param_name):
    constant_g_values = []
    self.x = flex.double([])
    self.n_active_params = 0
    self.active_parameterisation = []
    self.active_params_list = []
    self.cumulative_active_params = [0]
    self.active_derivatives = None
    for p_type, scalefactor in Data_Manager.g_parameterisation.iteritems():
      if p_type in param_name:
        self.x.extend(scalefactor.parameters)
        self.n_active_params = len(self.x) #update n_active_params
        self.active_parameterisation.append(p_type)
        n_params = scalefactor.n_params#len(scalefactor.parameters)
        self.active_params_list.append(n_params)
        self.cumulative_active_params.append(self.cumulative_active_params[-1] + n_params)
      else:
        constant_g_values.append(scalefactor.inverse_scales)
    if len(constant_g_values) > 0.0:
      self.constant_g_values = flex.double(np.prod(np.array(constant_g_values), axis=0))
    else:
      self.constant_g_values = None
    if not Data_Manager.params.scaling_options.multi_mode:
      logger.info(('Set up parameter handler for following corrections: {0}\n').format(
        ''.join(i.lstrip('g_')+' ' for i in self.active_parameterisation)))
  
  def get_current_parameters(self):
    return self.x

class multi_active_parameter_manager(object):
  def __init__(self, Data_Manager, param_name):
    self.apm_list = []
    for DM in Data_Manager.data_managers:
      apm = active_parameter_manager(DM, param_name)
      self.apm_list.append(apm)
    self.active_parameterisation = self.apm_list[0].active_parameterisation
    self.x = flex.double([])
    self.n_active_params_list = []
    self.n_active_params = 0
    self.n_cumulative_params = [0]
    for apm in self.apm_list:
      self.x.extend(apm.x)
      self.n_active_params_list.append(len(apm.x))
      self.n_active_params += len(apm.x)
      self.n_cumulative_params.append(copy.deepcopy(self.n_active_params))
    logger.info(('Set up joint parameter handler for following corrections: {0}\n').format(
        ''.join(i.lstrip('g_')+' ' for i in self.active_parameterisation)))

  def get_current_parameters(self):
    return self.x

class XDS_Data_Manager(ScalingDataManager):
  '''Data Manager subclass for implementing XDS parameterisation'''
  def __init__(self, reflections, experiments, params):
    Data_Manager.__init__(self, reflections, experiments, params)
    #initialise g-value arrays
    (self.g_absorption, self.g_modulation, self.g_decay) = (None, None, None)
    self.g_parameterisation = {}
    self.initialise_scale_factors() #this creates ScaleFactor objects
    (reflections_for_scaling, weights_for_scaling, selection) = (
      self._extract_reflections_for_scaling(self.reflection_table, self.params))
    self.Ih_table = SingleIhTable(reflections_for_scaling, weights_for_scaling.weights)
    #update normalised values after extracting reflections for scaling
    if self.params.parameterisation.modulation:
      self.g_modulation.set_normalised_values(reflections_for_scaling['normalised_x_values'],
        reflections_for_scaling['normalised_y_values'])
    if self.params.parameterisation.decay:
      self.g_decay.set_normalised_values(reflections_for_scaling['normalised_res_values'],
        reflections_for_scaling['normalised_time_values'])
    if self.params.parameterisation.absorption:
      self.g_absorption.set_normalised_values(reflections_for_scaling['normalised_x_abs_values'],
        reflections_for_scaling['normalised_y_abs_values'],
        reflections_for_scaling['normalised_time_values'])
    logger.info('Completed initialisation of XDS data manager. \n' + '*'*40 + '\n')

  def initialise_scale_factors(self):
    logger.info('Initialising scale factor objects. \n')
    self.bin_reflections_decay()
    if self.params.parameterisation.absorption:
      self.bin_reflections_absorption()
    if self.params.parameterisation.modulation:
      self.bin_reflections_modulation()

  def get_target_function(self, parameters):
    '''call the xds target function method'''
    if self.params.scaling_options.minimisation_parameterisation == 'log':
      return xds_target_function_log(self, parameters).return_targets()
    else:
      return target_function(self, parameters).return_targets()

  def get_basis_function(self, parameters):
    '''call the xds basis function method'''
    if self.params.scaling_options.minimisation_parameterisation == 'log':
      return bf.xds_basis_function_log(self, parameters).return_basis()
    else:
      return bf.basis_function(self, parameters).return_basis()

  def bin_reflections_decay(self):
    '''bin reflections for decay correction'''
    ndbins = self.params.scaling_options.n_d_bins
    if self.params.parameterisation.rotation_interval:
      rotation_interval = self.params.parameterisation.rotation_interval
    else:
      rotation_interval = 15.0
    osc_range = self.experiments.scan.get_oscillation_range()
    if ((osc_range[1] - osc_range[0]) / rotation_interval) % 1 < 0.33:
      #if last bin less than 33% filled'
      n_phi_bins = int((osc_range[1] - osc_range[0])/ rotation_interval)
      'increase rotation interval slightly'
      rotation_interval = (osc_range[1] - osc_range[0])/float(n_phi_bins) + 0.001
    else:
      rotation_interval = 15.0 + 0.001
      n_phi_bins = int((osc_range[1] - osc_range[0]) / rotation_interval) + 1
    self.n_phi_bins = n_phi_bins
    nzbins = n_phi_bins
    '''Bin the data into resolution and time 'z' bins'''
    zmax = max(self.reflection_table['xyzobs.px.value'].parts()[2]) + 0.001
    zmin = min(self.reflection_table['xyzobs.px.value'].parts()[2]) - 0.001
    resmax = (1.0 / (min(self.reflection_table['d'])**2)) + 0.001
    resmin = (1.0 / (max(self.reflection_table['d'])**2)) - 0.001
    one_resbin_width = (resmax - resmin) / ndbins
    one_time_width = (zmax - zmin) / nzbins
    self._reflection_table['normalised_res_values'] = (
      ((1.0 / (self.reflection_table['d']**2)) - resmin) / one_resbin_width)
    #define the highest and lowest gridpoints: go out one further than the max/min int values
    highest_parameter_value = int((max(self.reflection_table['normalised_res_values'])//1)+2)
    lowest_parameter_value = int((min(self.reflection_table['normalised_res_values'])//1)-1)
    n_res_parameters = highest_parameter_value - lowest_parameter_value + 1
    self._reflection_table['normalised_time_values'] = (
      (self.reflection_table['xyzobs.px.value'].parts()[2] - zmin) / one_time_width)
    if self.params.parameterisation.decay:
      #put this here to allow calculation of normalised time values needed for abscor, will rearrange.
      #define the highest and lowest gridpoints: go out one further than the max/min int values
      highest_parameter_value = int((max(self.reflection_table['normalised_time_values'])//1)+2)
      lowest_parameter_value = int((min(self.reflection_table['normalised_time_values'])//1)-1)
      n_time_parameters = highest_parameter_value - lowest_parameter_value + 1
      if self.params.scaling_options.minimisation_parameterisation == 'log':
        self.g_decay = SF.SmoothScaleFactor_2D(0.0, n_res_parameters, n_time_parameters)
      else:
        self.g_decay = SF.SmoothScaleFactor_2D(1.0, n_res_parameters, n_time_parameters)
      self.g_decay.set_normalised_values(self.reflection_table['normalised_res_values'],
        self.reflection_table['normalised_time_values'])
      self.g_parameterisation['g_decay'] = self.g_decay
    msg = ('The decay term ScaleFactor object was successfully initialised. {sep}'
      'The rotational parameter interval has been set to {0} degrees, giving {sep}'
      '{1} phi bins, while the resolution has been binned into {2} bins. {sep}'
      'In total, this ScaleFactor object uses {3} parameters. {sep}'
      ).format(rotation_interval, n_phi_bins, ndbins, 
               n_time_parameters * n_res_parameters, sep='\n')
    logger.info(msg)

  def bin_reflections_modulation(self):
    '''bin reflections for modulation correction'''
    nxbins = nybins = params.parameterisation.n_detector_bins
    xvalues = self.reflection_table['xyzobs.px.value'].parts()[0]
    (xmax, xmin) = (max(xvalues) + 0.001, min(xvalues) - 0.001)
    yvalues = self.reflection_table['xyzobs.px.value'].parts()[1]
    (ymax, ymin) = (max(yvalues) + 0.001, min(yvalues) - 0.001)
    one_x_width = (xmax - xmin) / float(nxbins)
    one_y_width = (ymax - ymin) / float(nybins)
    self._reflection_table['normalised_x_values'] = ((xvalues - xmin) / one_x_width)
    self._reflection_table['normalised_y_values'] = ((yvalues - ymin) / one_y_width)
    #define the highest and lowest gridpoints: go out one further than the max/min int values
    highest_x_parameter_value = int((max(self.reflection_table['normalised_x_values'])//1)+2)
    lowest_x_parameter_value = int((min(self.reflection_table['normalised_x_values'])//1)-1)
    n_x_parameters = highest_x_parameter_value - lowest_x_parameter_value + 1
    highest_y_parameter_value = int((max(self.reflection_table['normalised_y_values'])//1)+2)
    lowest_y_parameter_value = int((min(self.reflection_table['normalised_y_values'])//1)-1)
    n_y_parameters = highest_y_parameter_value - lowest_y_parameter_value + 1
    if self.params.scaling_options.minimisation_parameterisation == 'log':
      self.g_modulation = SF.SmoothScaleFactor_2D(0.0, n_x_parameters, n_y_parameters)
    else:
      self.g_modulation = SF.SmoothScaleFactor_2D(1.0, n_x_parameters, n_y_parameters)
    self.g_modulation.set_normalised_values(self.reflection_table['normalised_x_values'],
      self.reflection_table['normalised_y_values'])
    self.g_parameterisation['g_modulation'] = self.g_modulation
    msg = ('The modulation term ScaleFactor object was successfully initialised. {sep}'
      'This parameterises a correction across the detector area, using a {sep}'
      'square grid of {0} parameters. {sep}').format(n_x_parameters**2, sep='\n')
    logger.info(msg)


  def bin_reflections_absorption(self):
    '''bin reflections for absorption correction'''
    nxbins = nybins = 4.0
    xvalues = self.reflection_table['xyzobs.px.value'].parts()[0]
    (xmax, xmin) = (max(xvalues) + 0.001, min(xvalues) - 0.001)
    yvalues = self.reflection_table['xyzobs.px.value'].parts()[1]
    (ymax, ymin) = (max(yvalues) + 0.001, min(yvalues) - 0.001)
    one_x_width = (xmax - xmin) / float(nxbins)
    one_y_width = (ymax - ymin) / float(nybins)
    self._reflection_table['normalised_x_abs_values'] = ((xvalues - xmin) / one_x_width)
    self._reflection_table['normalised_y_abs_values'] = ((yvalues - ymin) / one_y_width)
    #define the highest and lowest gridpoints: go out one further than the max/min int values
    highest_x_parameter_value = int((max(self.reflection_table['normalised_x_abs_values'])//1)+1)
    lowest_x_parameter_value = int((min(self.reflection_table['normalised_x_abs_values'])//1))
    n_x_parameters = highest_x_parameter_value - lowest_x_parameter_value + 1
    highest_y_parameter_value = int((max(self.reflection_table['normalised_y_abs_values'])//1)+1)
    lowest_y_parameter_value = int((min(self.reflection_table['normalised_y_abs_values'])//1))
    n_y_parameters = highest_y_parameter_value - lowest_y_parameter_value + 1
    #new code to change time binning to smooth parameters
    highest_parameter_value = int((max(self.reflection_table['normalised_time_values'])//1)+1)
    lowest_parameter_value = int((min(self.reflection_table['normalised_time_values'])//1)-0)
    n_time_parameters = highest_parameter_value - lowest_parameter_value + 1
    #n_time_bins = int((max(self.reflection_table['normalised_time_values'])//1)+1)
    if self.params.scaling_options.minimisation_parameterisation == 'log':
      self.g_absorption = SF.SmoothScaleFactor_GridAbsorption(
        0.0, n_x_parameters, n_y_parameters, n_time_parameters)
    else:
      self.g_absorption = SF.SmoothScaleFactor_GridAbsorption(
        1.0, n_x_parameters, n_y_parameters, n_time_parameters)
    self.g_absorption.set_normalised_values(self.reflection_table['normalised_x_abs_values'],
      self.reflection_table['normalised_y_abs_values'],
      self.reflection_table['normalised_time_values'])
    self.g_parameterisation['g_absorption'] = self.g_absorption
    msg = ('The absorption term ScaleFactor object was successfully initialised. {sep}'
      'This defines a {0} by {0} grid of absorption correction parameters {sep}'
      'at {1} time values, giving a total of {2} parameters. {sep}'
      ).format(n_x_parameters, n_time_parameters, 
               n_time_parameters * (n_x_parameters**2), sep='\n')
    logger.info(msg)


  def bin_reflections_absorption_radially(self):
    from math import pi
    n_rad_bins = 3.0
    n_ang_bins = 8.0
    image_size = self.experiments.detector.to_dict()['panels'][0]['image_size']
    xvalues = self.reflection_table['xyzobs.px.value'].parts()[0]
    yvalues = self.reflection_table['xyzobs.px.value'].parts()[1]
    x_center = image_size[0]/2.0
    y_center = image_size[1]/2.0
    xrelvalues = xvalues - x_center
    yrelvalues = yvalues - y_center
    radial_values = ((xrelvalues**2) + (yrelvalues**2))**0.5
    angular_values = flex.double(np.arctan2(yrelvalues, xrelvalues))
    normalised_radial_values = n_rad_bins * radial_values / (max(radial_values) * 1.001)
    normalised_angular_values = n_ang_bins * (angular_values + pi) / (2.0 * pi)
    #catchers to stop values being exactly on the boundary and causing errors later.
    sel = normalised_angular_values == 8.0
    normalised_angular_values.set_selected(sel, 7.9999)
    sel = normalised_angular_values == 0.0
    normalised_angular_values.set_selected(sel, 0.0001)
    self._reflection_table['normalised_angle_values'] = normalised_angular_values
    self._reflection_table['normalised_radial_values'] = normalised_radial_values
    n_ang_parameters = n_rad_bins + 1
    n_rad_parameters = n_rad_bins + 1
    #new code to change time binning to smooth parameters
    highest_parameter_value = int((max(self.reflection_table['normalised_time_values'])//1)+1)
    lowest_parameter_value = int((min(self.reflection_table['normalised_time_values'])//1)-0)
    n_time_parameters = highest_parameter_value - lowest_parameter_value + 1

  def expand_scales_to_all_reflections(self):
    #currently we have Ih, scales and weights for reflections_for_scaling
    #first calculate the scales for all reflections (Sorted reflections)
    scales_to_expand = []
    if self.params.parameterisation.modulation:
      self.g_modulation.set_normalised_values(self.reflection_table['normalised_x_values'],
        self.reflection_table['normalised_y_values'])
      scales_to_expand.append(self.g_modulation.calculate_smooth_scales())
    if self.params.parameterisation.decay:
      self.g_decay.set_normalised_values(self.reflection_table['normalised_res_values'],
        self.reflection_table['normalised_time_values'])
      scales_to_expand.append(self.g_decay.calculate_smooth_scales())
    if self.params.parameterisation.absorption:
      self.g_absorption.set_normalised_values(self.reflection_table['normalised_x_abs_values'],
        self.reflection_table['normalised_y_abs_values'],
        self.reflection_table['normalised_time_values'])
      scales_to_expand.append(self.g_absorption.calculate_smooth_scales())
    
    if self.params.scaling_options.minimisation_parameterisation == 'log':
      self.reflection_table['inverse_scale_factor'] = flex.double(
        np.exp(np.sum(np.array(scales_to_expand), axis=0)))
    else:
      self.reflection_table['inverse_scale_factor'] = flex.double(
        np.prod(np.array(scales_to_expand), axis=0))
    'remove reflections that were determined as outliers'
    logger.info(('Scale factors determined during minimisation have now been applied {sep}'
      'to all reflections. {sep}').format(sep='\n'))
    weights = self._update_weights_for_scaling(self.reflection_table,
      self.params, weights_filter=False).weights
    #sel = self.weights_for_scaling.weights != 0.0
    self._reflection_table = self._reflection_table.select(weights != 0.0)
    weights = self._update_weights_for_scaling(
      self.reflection_table, self.params, weights_filter=False).weights
    #now calculate Ih for all reflections.
    self.Ih_table = SingleIhTable(self.reflection_table, weights)
    logger.info('A new best estimate for I_h for all reflections has now been calculated. \n')

  def clean_reflection_table(self):
    #add keys for additional data that is to be exported
    self._initial_keys.append('inverse_scale_factor')
    for key in self.reflection_table.keys():
      if not key in self._initial_keys:
        del self.reflection_table[key]
    added_columns = ['l_bin_index', 'a_bin_index', 'xy_bin_index', 'h_index',
      'normalised_y_values', 'normalised_x_values', 'normalised_y_abs_values',
      'normalised_x_abs_values', 'normalised_time_values', 
      'normalised_res_values', 'wilson_outlier_flag', 'centric_flag']
    for key in added_columns:
      del self.reflection_table[key]


class MultiCrystalDataManager(ScalingDataManager):
  def __init__(self, reflections, experiments, params):
    self.data_managers = []
    self._params = params
    if self.params.scaling_method == 'xds':
      for reflection, experiment in zip(reflections, experiments):
        self.data_managers.append(XDS_Data_Manager(reflection, experiment, params))
    elif self.params.scaling_method == 'aimless':
      for reflection, experiment in zip(reflections, experiments):
        self.data_managers.append(AimlessDataManager(reflection, experiment, params))
    else:
      assert 0, "Incorrect scaling method passed to multicrystal datamanager (not 'xds' or 'aimless')"
    self.Ih_table = JointIhTable(self.data_managers)
    self.update_counter=0
    self.n_active_params = None
    self.n_active_params_dataset1 = None
    self.n_active_params_dataset2 = None
    logger.info('Completed initialisation of multicrystal data manager. \n' + '*'*40 + '\n')

  def calc_absorption_constraint(self, apm):
    'method only called in aimless scaling'
    R = flex.double([])
    G = flex.double([])
    for i, dm in enumerate(self.data_managers):
      R.extend(dm.calc_absorption_constraint(apm.apm_list[i])[0])
      G.extend(dm.calc_absorption_constraint(apm.apm_list[i])[1])
    return (R, G)

  def update_error_model(self):
    for dm in self.data_managers:
      dm.update_error_model()
    self.Ih_table = JointIhTable(self.data_managers)

  def zip_together_derivatives(self, apm, basis_fn_deriv_list):#derivs1, derivs2):
    n_refl_total = self.Ih_table.size
    n_param_total = apm.n_cumulative_params[-1]
    active_derivatives = sparse.matrix(n_refl_total, n_param_total) 
    for i, derivs in enumerate(basis_fn_deriv_list):
      derivs_T = derivs.transpose()
      expanded = derivs_T * self.Ih_table.h_index_expand_list[i]
      active_derivatives.assign_block(expanded.transpose(), 0, apm.n_cumulative_params[i])
    return active_derivatives

  def update_for_minimisation(self, apm):
    '''update the scale factors and Ih for the next iteration of minimisation,
    update the x values from the amp to the individual apms, as this is where 
    basis functions, target functions etc get access to the parameters.'''
    for i, val in enumerate(apm.n_active_params_list):
      apm.apm_list[i].x = apm.x[apm.n_cumulative_params[i]:apm.n_cumulative_params[i+1]]
    basis_fn_deriv_list = []
    for i, dm in enumerate(self.data_managers):
      basis_fn = dm.get_basis_function(apm.apm_list[i])
      dm.Ih_table.inverse_scale_factors = basis_fn[0]
      basis_fn_deriv_list.append(basis_fn[1])
    apm.active_derivatives = self.zip_together_derivatives(apm, basis_fn_deriv_list)
    self.Ih_table.calc_Ih()

  def expand_scales_to_all_reflections(self):
    for dm in self.data_managers:
      dm.expand_scales_to_all_reflections()

  def join_multiple_datasets(self):
    self.JointIhTable = JointIhTable(self.data_managers)
    joined_reflections = flex.reflection_table()
    for dm in self.data_managers:
      joined_reflections.extend(dm.reflection_table)
    u_c = self.data_managers[0].experiments.crystal.get_unit_cell().parameters()
    s_g = self.data_managers[0].experiments.crystal.get_space_group()
    crystal_symmetry = crystal.symmetry(unit_cell=u_c, space_group=s_g)
    miller_set = miller.set(crystal_symmetry=crystal_symmetry,
                            indices=joined_reflections['asu_miller_index'], anomalous_flag=True)
    permuted = miller_set.sort_permutation(by_value='packed_indices')
    self._reflection_table = joined_reflections.select(permuted)
    #self.weights_for_scaling = Weighting(self.reflection_table)
    weights = Weighting(self.reflection_table).weights
    self.Ih_table = SingleIhTable(self.reflection_table, weights)
    self._reflection_table['Ih_values'] = self.Ih_table.Ih_values

    if self.params.scaling_options.reject_outliers:
      sel = flex.bool([True]*len(self.reflection_table))
      _, indices_of_outliers = reject_outliers(self, max_deviation=6.0)
      msg = ('Combined outlier rejection has been performed across all datasets, {sep}'
        '{0} outliers were found which have been removed from the dataset. {sep}'.format(
        len(indices_of_outliers), sep='\n'))
      logger.info(msg)
      sel.set_selected(flex.size_t(indices_of_outliers), flex.bool([False]*len(indices_of_outliers)))
      self._reflection_table = self._reflection_table.select(sel)
      weights = self._update_weights_for_scaling(self.reflection_table,
        self.params, weights_filter=False, error_model_params=None).weights
      self.Ih_table = SingleIhTable(self.reflection_table, weights)
      self._reflection_table['Ih_values'] = self.Ih_table.Ih_values
      msg = ('A new best estimate for I_h for all reflections across all datasets {sep}'
        'has now been calculated. {sep}').format(sep='\n')
      logger.info(msg)

class TargetedDataManager(ScalingDataManager):
  def __init__(self, reflections1, experiments1, reflections_scaled, params):
    #first assume that the Ih_values of reflections_scaled are the best estimates
    osc_range = experiments1.scan.get_oscillation_range()
    self._experiments = experiments1
    self._params = params
    if osc_range[1]-osc_range[0]<1000.0: 
      #usually would have it osc_range <10, big here just for testing on LCY dataset
      'first make a simple KB data manager'
      self.dm1 = KB_Data_Manager(reflections1, experiments1, params)
      'now extract Ih values from reflections_scaled'
      self.target_dm = ScalingDataManager(reflections_scaled, experiments1, params)
      (target_reflections, target_weights, target_selection) = (
      self._extract_reflections_for_scaling(self.target_dm.reflection_table, self.params))
      target_Ih_table = SingleIhTable(target_reflections, target_weights.weights)
      'find common reflections in the two datasets'
      for i, miller_idx in enumerate(self.dm1.Ih_table.asu_miller_index):
        sel = target_Ih_table.asu_miller_index == miller_idx
        Ih_values = target_Ih_table.Ih_values.select(sel)
        if Ih_values:
          self.dm1.Ih_table.Ih_values[i] = Ih_values[0]
          #select [0] above as all Ih values are the same for asu_miller_idx
        else:
          self.dm1.Ih_table.Ih_values[i] = 0.0
          #should already be zero but set again just in case
      'select only those reflections matched in the target dataset'
      #print(list(self.dm1.Ih_table.Ih_values))
      sel = self.dm1.Ih_table.Ih_values != 0.0
      self.dm1.Ih_table = self.dm1.Ih_table.select(sel)
      #print(list(self.dm1.Ih_table.Ih_values))
      #exit()
      'update the data in the SF objects'
      if self.params.parameterisation.decay_term:
        self.dm1.g_decay.update_reflection_data(
          dvalues=self.dm1.g_decay.d_values.select(sel))
        #d_values = self.dm1.g_decay.d_values.select(sel)
      if self.params.parameterisation.scale_term:
        self.dm1.g_scale.update_reflection_data(n_refl=self.dm1.Ih_table.size)
        #self.dm1.g_scale.n_refl = self.dm1.Ih_table.size
      self.g_parameterisation = self.dm1.g_parameterisation
    else:
      #do full aimless/xds scaling of one dataset against the other?
      #self.target_refl_table = reflections_scaled
      assert 0, "no methods specified yet for scaling a large dataset against another"

  def get_target_function(self, apm):
    '''call the target function method'''
    return self.dm1.get_target_function(apm)

  def get_basis_function(self, apm):
    '''call the KB basis function method'''
    return self.dm1.get_basis_function(apm)

  def update_for_minimisation(self, apm):
    '''update the scale factors and Ih for the next iteration of minimisation'''
    return self.dm1.update_for_minimisation(apm)

  def expand_scales_to_all_reflections(self):
    return self.dm1.expand_scales_to_all_reflections()

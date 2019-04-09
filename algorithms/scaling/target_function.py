"""
This file defines targets for scaling.

These are initialised with a scaler and an active parameter manager,
and have implementations of residual/gradient calculations for
scaling.
"""
from __future__ import division
from __future__ import absolute_import, print_function
from copy import copy
from dials.array_family import flex
from dials.algorithms.scaling.scaling_restraints import MultiScalingRestraints
from dials_scaling_ext import row_multiply, calc_dIh_by_dpi, calc_jacobian


class ScalingTarget(object):
    """
    A class to be used by a Scaling Refinery to calculate gradients,
    residuals etc required by the Refinery for minimisation.
    '"""

    _grad_names = ["dI_dp"]
    rmsd_names = ["RMSD_I"]
    rmsd_units = ["a.u"]

    def __init__(self, curvatures=False):
        self._restr = None
        self.curvatures = curvatures
        self.rmsd_names = ["RMSD_I"]
        self.rmsd_units = ["a.u"]
        # Quantities to cache each step
        self._rmsds = None
        self.restraints_calculator = MultiScalingRestraints()
        self.param_restraints = True  # If one tests for restraints and None is
        # returned, then this is set to False and restraints calculations are not
        # attempted for the remainder of the minimisation with this target function.

    def rmsds(self, Ih_table, apm):
        """Calculate RMSDs for the matches. Also calculate R-factors."""
        R = flex.double([])
        n = 0
        if Ih_table.free_Ih_table:
            work_blocks = Ih_table.blocked_data_list[:-1]
            free_block = Ih_table.blocked_data_list[-1]
            self.rmsd_names = ["RMSD_I", "RMSD_I (no restraints)", "Free RMSD_I"]
            self.rmsd_units = ["a.u", "a.u", "a.u"]
        else:
            work_blocks = Ih_table.blocked_data_list
            self.rmsd_names = ["RMSD_I"]
            self.rmsd_units = ["a.u"]
        for block in work_blocks:
            R.extend((self.calculate_residuals(block) ** 2) * block.weights)
            n += block.size
        unrestr_R = copy(R)
        if self.param_restraints:
            restraints = self.restraints_calculator.calculate_restraints(apm)
            if restraints:
                R.extend(restraints[0])
            else:
                self.param_restraints = False
        self._rmsds = [(flex.sum((R)) / n) ** 0.5]
        if Ih_table.free_Ih_table:
            self._rmsds.append((flex.sum((unrestr_R)) / n) ** 0.5)
            Rmsdfree = (self.calculate_residuals(free_block) ** 2) * free_block.weights
            self._rmsds.append((flex.sum((Rmsdfree)) / free_block.size) ** 0.5)
        return self._rmsds

    @staticmethod
    def achieved():
        """Method required by refinement engine."""
        return False  # implement a method here?

    @staticmethod
    def calculate_residuals(Ih_table):
        """Return the residual vector."""
        R = Ih_table.intensities - (Ih_table.inverse_scale_factors * Ih_table.Ih_values)
        return R

    @staticmethod
    def calculate_gradients(Ih_table):
        """Return a gradient vector on length len(self.apm.x)."""
        prefactor = (
            -2.0
            * Ih_table.weights
            * (
                Ih_table.intensities
                - (Ih_table.Ih_values * Ih_table.inverse_scale_factors)
            )
        )
        dIhbydpi = dIh_by_dpi(Ih_table)
        term_1 = (prefactor * Ih_table.Ih_values) * Ih_table.derivatives
        term_2 = (
            prefactor * Ih_table.inverse_scale_factors * Ih_table.h_index_matrix
        ) * dIhbydpi
        gradient = term_1 + term_2
        return gradient

    @staticmethod
    def calculate_jacobian(Ih_table):
        """Calculate the jacobian matrix, size Ih_table.size by len(self.apm.x)."""
        gsq = Ih_table.inverse_scale_factors ** 2 * Ih_table.weights
        sumgsq = gsq * Ih_table.h_index_matrix
        dIh = (
            (
                Ih_table.intensities
                - (Ih_table.Ih_values * 2.0 * Ih_table.inverse_scale_factors)
            )
        ) * Ih_table.weights
        jacobian = calc_jacobian(
            Ih_table.derivatives.transpose(),
            Ih_table.h_index_matrix,
            Ih_table.Ih_values,
            Ih_table.inverse_scale_factors,
            dIh,
            sumgsq,
        )
        return jacobian

    def calculate_curvatures(self, _):
        """Return the second derivative of the target function."""
        assert 0, "method not yet implemented."

    # The following methods are for adaptlbfgs.
    def compute_functional_gradients(self, Ih_table):
        """Return the functional and gradients."""
        resids = self.calculate_residuals(Ih_table)
        gradients = self.calculate_gradients(Ih_table)
        weights = Ih_table.weights
        functional = flex.sum(resids ** 2 * weights)
        return functional, gradients

    def compute_functional_gradients_and_curvatures(self, Ih_table):
        """Return the functional, gradients and curvatures."""
        resids = self.calculate_residuals(Ih_table)
        gradients = self.calculate_gradients(Ih_table)
        curvatures = self.calculate_curvatures(Ih_table)
        weights = Ih_table.weights
        functional = flex.sum(resids ** 2 * weights)
        return functional, gradients, curvatures

    def compute_restraints_functional_gradients_and_curvatures(self, apm):
        """Return the restrains for functional, gradients and curvatures."""
        restraints = None
        if self.param_restraints:
            restr = self.restraints_calculator.calculate_restraints(apm)
            if restr:
                resid_restr = flex.sum(restr[0])  # add to total functional here
                grad_restr = restr[1]
                restraints = [resid_restr, grad_restr, None]
            else:
                self.param_restraints = False
        return restraints  # list of restraints to add to resid, grads and curvs

    # The following methods are for adaptlstbx (GN/ LM algorithms)
    def compute_residuals(self, Ih_table):
        """Return the residuals array and weights."""
        residuals = self.calculate_residuals(Ih_table)
        weights = Ih_table.weights
        return residuals, weights

    def compute_residuals_and_gradients(self, Ih_table):
        """Return the residuals array, jacobian matrix and weights."""
        residuals = self.calculate_residuals(Ih_table)
        jacobian = self.calculate_jacobian(Ih_table)
        weights = Ih_table.weights
        return residuals, jacobian, weights

    def compute_restraints_residuals_and_gradients(self, apm):
        """Return the restraints for the residuals and jacobian."""
        if self.param_restraints:
            restr = self.restraints_calculator.calculate_jacobian_restraints(apm)
            if not restr:
                self.param_restraints = False
            return restr
        return None

class ScalingVarianceTarget(object):

    _grad_names = ["dI_dp"]
    rmsd_names = ["RMSD_I"]
    rmsd_units = ["a.u"]

    def __init__(self, curvatures=False):
        self._restr = None
        self.curvatures = curvatures
        self.rmsd_names = ["RMSD_I"]
        self.rmsd_units = ["a.u"]
        # Quantities to cache each step
        self._rmsds = None
        self.restraints_calculator = None
        self.param_restraints = False

    @staticmethod
    def calculate_residuals(Ih_table):
        """Return the residual vector."""
        R = Ih_table.intensities - (Ih_table.inverse_scale_factors * Ih_table.Ih_values)
        return R

    def calculate_gradients(self, Ih_table):
        dIhbydpi = dIh_by_dpi(Ih_table)# already summed by group
        r = self.calculate_residuals(Ih_table)
        n = Ih_table.size
        weights = Ih_table.weights
        first_term = (-2.0 * Ih_table.weights * r * Ih_table.Ih_values * Ih_table.derivatives) / n
        second_term = ((-2.0 * Ih_table.weights * r * Ih_table.inverse_scale_factors * Ih_table.h_index_matrix) * dIhbydpi) / n
        sqrtweights = Ih_table.weights ** 0.5
        third_term = 2.0 * flex.sum(sqrtweights * r) * (sqrtweights * Ih_table.Ih_values * Ih_table.derivatives) / (n ** 2)
        fourth_term = 2.0 * flex.sum(sqrtweights * r) * ((sqrtweights * Ih_table.inverse_scale_factors * Ih_table.h_index_matrix) * dIhbydpi) / (n ** 2)
        dvdpi = first_term + second_term + third_term + fourth_term
        var = ((flex.sum(r ** 2 * weights)) / n) - ((flex.sum(r * (weights ** 0.5)))**2 / (n ** 2))
        gradients = -2.0 * (1.0 - var) * var * dvdpi
        return gradients

    def compute_functional_gradients(self, Ih_table):
        """Return the functional and gradients."""
        resids = self.calculate_residuals(Ih_table)
        gradients = self.calculate_gradients(Ih_table)
        weights = Ih_table.weights
        n = resids.size()
        var = ((flex.sum(resids ** 2 * weights)) / n) - ((flex.sum(resids * (weights ** 0.5)))**2 / (n ** 2))
        functional = (1.0 - var)**2
        return functional, gradients

    def rmsds(self, Ih_table, apm):
        """Calculate RMSDs for the matches. Also calculate R-factors."""
        R = []
        #n = 0
        work_blocks = Ih_table.blocked_data_list
        self.rmsd_names = ["RMSD_I"]
        self.rmsd_units = ["a.u"]
        for block in work_blocks:
            r = self.calculate_residuals(block)
            weights = block.weights
            n = weights.size()
            var = ((flex.sum(r ** 2 * weights)) / n) - ((flex.sum(r * (weights ** 0.5)))**2 / (n ** 2))
            R.append((1.0 - var)**2)
        print(R)
        self._rmsds = [(sum(R) / len(R)) ** 0.5]
        return self._rmsds

    def compute_restraints_functional_gradients_and_curvatures(self, apm):
        return None

    @staticmethod
    def achieved():
        """Method required by refinement engine."""
        return False  # implement a method here?


class ScalingTargetFixedIH(ScalingTarget):
    """An implementation of scaling target for when the scaling is to be
    done against a fixed reference Ih set (i.e scaler is a TargetScaler)
    """

    @staticmethod
    def calculate_gradients(Ih_table):
        rhl = Ih_table.intensities - (
            Ih_table.Ih_values * Ih_table.inverse_scale_factors
        )
        G = -2.0 * rhl * Ih_table.weights * Ih_table.Ih_values * Ih_table.derivatives
        return G

    @staticmethod
    def calculate_jacobian(Ih_table):
        """Calculate the jacobian matrix, size Ih_table.size by len(self.apm.x)."""
        jacobian = row_multiply(Ih_table.derivatives, -1.0 * Ih_table.Ih_values)
        return jacobian

def dIh_by_dpi(Ih_table):
    gsq = Ih_table.inverse_scale_factors ** 2 * Ih_table.weights
    sumgsq = gsq * Ih_table.h_index_matrix

    dIh = (
        Ih_table.intensities
        - (Ih_table.Ih_values * 2.0 * Ih_table.inverse_scale_factors)
    ) * Ih_table.weights
    dIh_by_dpi = calc_dIh_by_dpi(
        dIh, sumgsq, Ih_table.h_index_matrix, Ih_table.derivatives.transpose()
    )
    return dIh_by_dpi

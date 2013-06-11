#!/usr/bin/env python
#
# dials.algorithms.integration.integrator.py
#
#  Copyright (C) 2013 Diamond Light Source
#
#  Author: James Parkhurst
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.
from __future__ import division


class Integrator(object):
    ''' The integrator base class. '''

    def __init__(self, compute_spots, compute_background, compute_intensity):
        ''' Initialise the integrator base class.

        Params:
            compute_spots The spot extractor strategy
            compute_background The background strategy
            compute_intensity The intensity strategy

        '''
        # Initialise the reflection extractor
        self.compute_spots = compute_spots
        self.compute_background = compute_background
        self.compute_intensity = compute_intensity

    def __call__(self, sweep, crystal, reflections=None):
        ''' Call to integrate.

        Params:
            sweep The sweep to process
            crystal The crystal to process
            reflections The reflection list

        Returns:
            A reflection list

        '''
        # Extract the reflections from the sweep
        if reflections == None:
            reflections = self.compute_spots(sweep, crystal)

        # Calculate the background
        reflections = self.compute_background(sweep, crystal, reflections)

        # Calculate the intensity and return
        return self.compute_intensity(sweep, crystal, reflections)


class IntegratorFactory(object):
    ''' Factory class to create integrators '''

    @staticmethod
    def from_parameters(params):
        ''' Given a set of parametets, construct the integrator

        Params:
            params The input parameters

        Returns:
            The integrator instance

        '''
        # Configure the algorithms to extract reflections, compute the
        # background intensity and integrate the reflection intensity
        compute_spots = IntegratorFactory.configure_extractor(params)
        compute_background = IntegratorFactory.configure_background(params)
        compute_intensity = IntegratorFactory.configure_intensity(params)

        # Return the integrator with the given strategies
        return Integrator(compute_spots=compute_spots,
                          compute_background=compute_background,
                          compute_intensity=compute_intensity)

    @staticmethod
    def configure_extractor(params):
        ''' Given a set of parameters, configure the reflection extractor

        Params:
            params The input parameters

        Returns:
            The extractor instance

        '''
        from dials.algorithms.integration import ReflectionExtractor
        return ReflectionExtractor(params.integration.shoebox.n_sigma)

    @staticmethod
    def configure_background(params):
        ''' Given a set of parameters, configure the background calculator

        Params:
            params The input parameters

        Returns:
            The background calculator instance

        '''
        from dials.algorithms.background import NullSubtractor
        from dials.algorithms.background import XdsSubtractor
        from dials.algorithms.background import FableSubtractor

        # Configure the NULL subtractor
        if params.integration.background.algorithm == 'none':
            algorithm = NullSubtractor()

        # Configure the XDS subtractor
        elif params.integration.background.algorithm == 'xds':
            algorithm = XdsSubtractor(
                min_data=params.integration.background.min_pixels,
                n_sigma=params.integration.background.n_sigma)

        # Configure the Fable subtractor
        elif params.integration.background.algorithm == 'fable':
            algorithm = FableSubtractor(
                min_data=params.integration.background.min_pixels,
                n_sigma=params.integration.background.n_sigma)

        # Configure the flat subtractor
        elif params.integration.background.algorithm == 'flat':
            raise RuntimeError('Not implemented yet')

        # Configure the curved subtractor
        elif params.integration.background.algorithm == 'curved':
            raise RuntimeError('Not implemented yet')

        # Configure the esmerelda subtractor
        elif params.integration.background.algorithm == 'esmerelda':
            raise RuntimeError('Not implemented yet')

        # Unknown subtractor
        else:
            raise RuntimeError('Unknown background algorithm')

        # Return the algorithm
        return algorithm

    @staticmethod
    def configure_intensity(params):
        ''' Given a set of parameters, configure the intensity calculator

        Params:
            params The input parameters

        Returns:
            The intensity calculator instance

        '''
        from dials.algorithms.integration import Summation3d

        # Configure the 2D summation algorithm
        if params.integration.algorithm == 'sum2d':
            raise RuntimeError('Not implemented yet')

        # Configure the 3D summation algorithm
        elif params.integration.algorithm == 'sum3d':
            algorithm = Summation3d()

        # Configure the reciprocal space summation algorithm
        elif params.integration.algorithm == 'sum_rs':
            raise RuntimeError('Not implemented yet')

        # Configure the 2D profile fitting algorithm
        elif params.integration.algorithm == 'fit_2d':
            raise RuntimeError('Not implemented yet')

        # Configure the 3D profile fitting algorithm
        elif params.integration.algorithm == 'fit_3d':
            raise RuntimeError('Not implemented yet')

        # Configure the reciprocal space profile fitting algorithm
        elif params.integration.algorithm == 'fit_rs':
            raise RuntimeError('Not implemented yet')

        # Unknown algorithm
        else:
            raise RuntimeError('Unknown integration algorithm')

        # Return the algorithm
        return algorithm

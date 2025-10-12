import numpy as np
import math
from typing import List

from CEE225_Dynamics.sdof_numerical_methods import SDOFHarmonicVibration, AverageAccelerationMethod, SolutionPoint
from CEE225_Dynamics.sdof_numerical_methods.plotly_generator import plot_pseudo_acceleration_vs_time

class ElasticResponseSpectrumBuilder:

    def __init__(self, sdof_system: SDOFHarmonicVibration, period_range:tuple = (0.02, 10), periods_count:int = 200,
                 normalization_value:float = 386):

        self.base_sdof_system = sdof_system
        self.period_range = period_range
        self.periods_count = periods_count
        self.normalization_value = normalization_value
        self.spectrum_cloud_point = self.build_spectrum()

    def build_spectrum(self) -> List[SolutionPoint]:
        periods_list = np.logspace(
            np.log10(self.period_range[0]),
            np.log10(self.period_range[1]),
            self.periods_count
        )
        solution_set = []
        for period in periods_list:
            wn_i = (2 * math.pi)/period
            sdof_properties = dict(
                time_step=min(period/20,0.1),
                time_stop=min(period*1000, 4),
                elastic_constant=wn_i**2*self.base_sdof_system.m,
                damping_ratio=self.base_sdof_system.zeta,
                natural_frequency=wn_i,
                initial_displacement=0,
                initial_velocity=0,
                forcing_function=self.base_sdof_system.p,
            )
            exact_solution = SDOFHarmonicVibration(**sdof_properties)
            average_acceleration_method = AverageAccelerationMethod(**sdof_properties,
                                                                    exact_solution=exact_solution.eom)
            maximum_displacement = average_acceleration_method.get_maximum_displacement()
            solution_set.append(SolutionPoint(
                time=period,
                displacement=maximum_displacement,
                velocity=maximum_displacement,
                acceleration=(wn_i**2)*maximum_displacement/self.normalization_value, # Normalized in g
            ))
        return solution_set

    def plot(self, **kwargs):
        plot_pseudo_acceleration_vs_time(
            "Elastic Response Spectrum",
            {"Pseudo-Aceleration": self.spectrum_cloud_point},
            **kwargs
        )

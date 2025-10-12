import math
from functools import lru_cache

from CEE225_Dynamics.response_spectrum_builder.response_spectrum_builder import ElasticResponseSpectrumBuilder
from CEE225_Dynamics.sdof_numerical_methods import SDOFHarmonicVibration

wn = math.pi * 2  # Initial Natural frequency [rad/s] - used to obtain mass
zeta = 0.05  # Damping ratio
k = 5  # Elastic constant [kips/in]
m = k/(wn**2)  # [kips s²/in]

p0 = 8*m*12  # [kips]
w = math.pi / 0.4
t_change = 1.2



@lru_cache(512)
def forcing_function(t: float):
    if t < 0:
        raise ValueError("Error: the analyzed time should always be t>0")
    elif t <= t_change:
        return p0 * math.sin(t * w)
    return 0

sdof_system = SDOFHarmonicVibration(
    forcing_function=forcing_function,
    elastic_constant=k,
    damping_ratio=zeta,
    natural_frequency=wn,
    initial_displacement=0,
    initial_velocity=0,
)

if __name__ == "__main__":
    builder = ElasticResponseSpectrumBuilder(
        sdof_system,
        normalization_value = 8*12,
        period_range=(0.02,20)
    )
    builder.plot(
        use_log_x=True,
        use_log_y=True
    )

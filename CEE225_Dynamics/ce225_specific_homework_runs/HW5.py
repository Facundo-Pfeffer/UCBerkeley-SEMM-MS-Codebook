from CEE225_Dynamics.sdof_numerical_methods import SDOFHarmonicVibration, CentralDifferenceMethod, AverageAccelerationMethod
from CEE225_Dynamics.sdof_numerical_methods.plotly_generator import plot_displacement_vs_time

import math
from functools import lru_cache

wn = math.pi*2 # Natural frequency [rad/s]
zeta = 0.05 # Damping ratio
k = 5  # Elastic constant [kips/in]

p0= 8
w = math.pi/0.4
t_change = 1.2

@lru_cache(512)
def forcing_function(t: float):
    if t < 0:
        raise ValueError("Error: the analyzed time should always be t>0")
    elif t<=t_change:
        return p0*math.sin(t*w)
    return 0

if __name__ == "__main__":
    problem_a_params = dict(
        time_step=0.1,
        time_stop=4,
        elastic_constant=k,
        damping_ratio=zeta,
        natural_frequency=wn,
        initial_displacement=0,
        initial_velocity=0,
        forcing_function=forcing_function,
    )
    exact_solution = SDOFHarmonicVibration(**problem_a_params)
    central_difference_method = CentralDifferenceMethod(**problem_a_params, exact_solution=exact_solution.eom)
    average_acceleration_method = AverageAccelerationMethod(**problem_a_params, exact_solution=exact_solution.eom)

    plot_displacement_vs_time(
        title="Dynamic Response for ζ=0.05: Displacement vs Time",
        solutions={f"Central Difference (E:{central_difference_method.accum_abs_error:.2f}in)": central_difference_method.solution_set,
                   f"Average Acceleration Method (E:{average_acceleration_method.accum_abs_error:.2f}in)": average_acceleration_method.solution_set,
                   "Exact Solution": exact_solution.get_cloud_points(central_difference_method.dt, central_difference_method.time_stop),
                   "Static Solution": exact_solution.get_static_solution_cloud_points(central_difference_method.dt, central_difference_method.time_stop)})
    print(central_difference_method.solution_set)

    problem_b_params = problem_a_params.copy()
    for z in [0.01, 0.1, 0.25]:
        params = problem_b_params.copy()
        params["damping_ratio"] = z
        exact_solution = SDOFHarmonicVibration(**params)
        central_difference_method = CentralDifferenceMethod(**params, exact_solution=exact_solution.eom)
        average_acceleration_method = AverageAccelerationMethod(**params, exact_solution=exact_solution.eom)
        plot_displacement_vs_time(
            title=f"Dynamic Response for ζ={z}: Displacement vs Time",
            solutions={
                f"Central Difference (E:{central_difference_method.accum_abs_error:.2f}in)": central_difference_method.solution_set,
                f"Average Acceleration Method (E:{average_acceleration_method.accum_abs_error:.2f}in)": average_acceleration_method.solution_set,
                "Exact Solution": exact_solution.get_cloud_points(central_difference_method.dt,
                                                                  central_difference_method.time_stop)},
            filename=f'Dynamic_Response_zeta_{z}.html')

    problem_c_params = problem_a_params.copy()
    problem_c_central_solutions = {}
    problem_c_average_acceleration_solutions = {}
    for delta_t in [0.35, 0.20, 0.05]:
        params = problem_c_params.copy()
        params["time_step"] = delta_t
        exact_solution = SDOFHarmonicVibration(**params)

        central_difference_method = CentralDifferenceMethod(**params, exact_solution=exact_solution.eom)
        problem_c_central_solutions[f"Δt={delta_t}"] = central_difference_method.solution_set
        average_acceleration_method = AverageAccelerationMethod(**params, exact_solution=exact_solution.eom)
        problem_c_average_acceleration_solutions[f"Δt={delta_t})"] = average_acceleration_method.solution_set
        exact_solution = SDOFHarmonicVibration(**params)

    problem_c_central_solutions["Exact Solution"] = exact_solution.get_cloud_points(0.05, 4)
    problem_c_average_acceleration_solutions["Exact Solution"] = exact_solution.get_cloud_points(0.05, 4)

    plot_displacement_vs_time(
        title=f"UNSTABLE Comparison of Central Difference Method for different Δt values at ζ={0.05}",
        solutions=problem_c_central_solutions,
        filename='Unstable_Central_Difference_Comparison_Delta_t.html'
    )
    first_key, first_value = next(iter(problem_c_central_solutions.items()))
    problem_c_central_solutions.pop(first_key)

    plot_displacement_vs_time(
        title=f"STABLE Comparison of Central Difference Method for different Δt values at ζ={0.05}",
        solutions=problem_c_central_solutions,
        filename='Stable_Central_Difference_Comparison_Delta_t.html'
    )

    plot_displacement_vs_time(
        title=f"Comparison of Average Acceleration Method for different Δt values at ζ={0.05}",
        solutions=problem_c_average_acceleration_solutions,
        filename='Average_Acceleration_Comparison_Delta_t.html'
    )

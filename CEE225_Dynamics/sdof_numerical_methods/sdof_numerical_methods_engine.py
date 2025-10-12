import numpy as np
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


class SolutionPoint:
    """Single solution point to be plotted."""
    def __init__(self, time, displacement, velocity=None, acceleration=None, metadata:dict = None):
        metadata = metadata or {}
        self.t = time
        self.u = displacement
        self.v = velocity
        self.a = acceleration
        self.metadata = metadata


class SDOFSystem:
    """System of SDOFs."""
    def __init__(self, elastic_constant, damping_ratio, natural_frequency, **kwargs):

        self.k, self.zeta, self.wn, self.m, self.c, self.Tn  = self.populate_sdof_constants(
            elastic_constant, damping_ratio, natural_frequency)

    @staticmethod
    def populate_sdof_constants(elastic_constant, damping_ratio, natural_frequency):
        k = elastic_constant
        zeta = damping_ratio
        wn = natural_frequency
        m = k / (wn ** 2)
        c = 2 * math.sqrt(k * m) * zeta
        Tn = 2 * math.pi / wn
        return k, zeta, wn, m, c, Tn


class SDOFHarmonicVibration(SDOFSystem):
    """Harmonic vibration of a single degree of freedom until t_change, then free vibration."""

    def __init__(self, forcing_function,
                 initial_displacement=None, initial_velocity=None, initial_acceleration=None, **kwargs):

        super().__init__(**kwargs)
        self.p = forcing_function
        self.u0, self.u1, self.u2 = self.populate_initial_conditions(initial_displacement, initial_velocity,
                                                                 initial_acceleration)
        self.eom = self.get_eom()

    def populate_initial_conditions(self, initial_displacement, initial_velocity, initial_acceleration):
        if len([x for x in (initial_displacement, initial_velocity, initial_acceleration) if x is None]) > 1:
            raise ValueError("Error: at least 2 initial conditions need to be provided")
        elif initial_acceleration is None:
            initial_acceleration = (self.p(0)-self.c*initial_velocity-self.k*initial_displacement)/self.m
        elif initial_velocity is None:
            initial_velocity = (self.p(0)-self.m*initial_acceleration-self.k*initial_displacement)/self.c
        else:
            initial_displacement = (self.p(0)-self.c*initial_velocity-self.m*initial_acceleration)/self.k
        return initial_displacement, initial_velocity, initial_acceleration


    def get_eom(self):

        r = w/self.wn
        C = p0/self.k * (1-r**2)/((1-r**2)**2 + (2*self.zeta*r)**2)
        D = p0/self.k * (-2*self.zeta*r)/((1-r**2)**2 + (2*self.zeta*r)**2)

        w_d = self.wn*math.sqrt(1-self.zeta**2)
        A = self.u0 - D
        B = self.u1 - (C*w-A*self.wn*self.zeta)/w_d
        func_1 = lambda t: math.exp(-self.zeta*self.wn*t)*(A*math.cos(w_d*t) + B*math.sin(w_d*t)) + C*math.sin(w*t) + D*math.cos(w*t)
        vel_func = lambda t: math.exp(-self.zeta*self.wn*t)*(-self.zeta*self.wn*(A*math.cos(w_d*t) + B*math.sin(w_d*t))
                                                             -A*w_d*math.sin(w_d*t)+B*w_d*math.cos(w_d*t)) + w*C*math.cos(w*t) - w*D*math.sin(w*t)
        A2 = func_1(t_change)
        B2 = (vel_func(t_change)+A2*self.zeta*self.wn)/w_d
        func_2 = lambda t: math.exp(-self.zeta*self.wn*(t-t_change))*(A2*math.cos(w_d*(t-t_change)) + B2*math.sin(w_d*(t-t_change)))
        def final_func(t):
            if t<=t_change:
                return func_1(t)
            return func_2(t)
        return final_func

    def get_cloud_points(self, dt, time_stop):
        times_range = np.arange(0, dt * int(time_stop / dt) + dt, dt)
        solution_set = []
        for t in times_range:
            solution_set.append(SolutionPoint(time=t, displacement=self.eom(t), metadata={
                "Method": "Analytical Solution",
            }))
        return solution_set


    def get_static_solution_cloud_points(self, dt, time_stop):
        times_range = np.arange(0, dt * int(time_stop / dt) + dt, dt)
        solution_set = []
        for t in times_range:
            solution_set.append(SolutionPoint(time=t, displacement=self.p(t) / self.k, metadata={
                "Method": "Static Solution",
            }))
        return solution_set



class AbsSDOFNumericMethod:
    def __init__(self, time_step, time_stop,
                 elastic_constant, damping_ratio, natural_frequency,
                 initial_displacement=None, initial_velocity=None, initial_acceleration=None,
                 forcing_function=forcing_function,
                 exact_solution=None):
        self.dt = time_step
        self.time_stop = time_stop
        self.p = forcing_function
        self.exact_solution = exact_solution

        self.k, self.zeta, self.wn, self.m, self.c = self.populate_sdof_constants(
            elastic_constant, damping_ratio, natural_frequency)

        self.u0, self.u1, self.u2 = self.populate_initial_conditions(initial_displacement, initial_displacement, initial_displacement)
        self.point_metadata = {}
        self.accum_abs_error = 0



    @staticmethod
    def populate_sdof_constants(elastic_constant, damping_ratio, natural_frequency):
        k = elastic_constant
        zeta = damping_ratio
        wn = natural_frequency
        m = k / (wn ** 2)
        c = 2 * math.sqrt(k * m) * zeta
        return k, zeta, wn, m, c

    def populate_initial_conditions(self, initial_displacement, initial_velocity, initial_acceleration):
        if len([x for x in (initial_displacement, initial_velocity, initial_acceleration) if x is None]) > 1:
            raise ValueError("Error: at least 2 initial conditions need to be provided")
        elif initial_acceleration is None:
            initial_acceleration = (self.p(0) - self.c * initial_velocity - self.k * initial_displacement) / self.m
        elif initial_velocity is None:
            initial_velocity = (self.p(0) - self.m * initial_acceleration - self.k * initial_displacement) / self.c
        else:
            initial_displacement = (self.p(0) - self.c * initial_velocity - self.m * initial_acceleration) / self.k
        return initial_displacement, initial_velocity, initial_acceleration

    def populate_point_metadata(self, t, disp, metadata=None):
        metadata = self.point_metadata.copy() if metadata is None else metadata
        if t==0:
            return metadata
        exact_disp = self.exact_solution(t)
        abs_error = abs(disp - exact_disp)
        net_error = disp - exact_disp
        metadata["NetError"] = net_error
        metadata["AbsError"] = abs_error
        metadata["PercentageError"] = net_error / exact_disp
        self.accum_abs_error += abs_error
        return metadata


class CentralDifferenceMethod(AbsSDOFNumericMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.k_hat, self.a, self.b = self.populate_method_constants()

        self.point_metadata = { "Method": "Central Difference Method", "Constants": {
                        "k_hat": self.k_hat,
                        "a": self.a,
                        "b": self.b
                    }
                }

        self.solution_set = self.generate_point_cloud()


    def populate_method_constants(self):
        k_hat = self.m/(self.dt**2) + self.c/(2*self.dt)
        a = self.m/(self.dt**2) - self.c/(2*self.dt)
        b = self.k - 2*self.m/(self.dt**2)
        return k_hat, a, b

    def generate_point_cloud(self):
        times_range = np.arange(0, self.dt * int(self.time_stop / self.dt) + self.dt, self.dt)
        solution_set = [SolutionPoint(time=0, displacement=self.u0, metadata=self.point_metadata)]
        u_1_first_step = self.u0 - self.u1 * self.dt + self.u2 * self.dt ** 2 / 2
        for i, t in enumerate(times_range):
            if i == 0:
                ui_1 = u_1_first_step
                ui = self.u0
            else:
                ui_1 = solution_set[i-1].u
                ui = solution_set[i].u

            p_hat = self.p(t) - self.a * ui_1 - self.b * ui

            disp = p_hat/self.k_hat
            solution_set.append(SolutionPoint(
                time=t + self.dt,
                displacement=disp,
                metadata=self.populate_point_metadata(t, disp))
            )
        return solution_set



class AverageAccelerationMethod(AbsSDOFNumericMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.a1, self.a2, self.a3, self.k_hat = self.populate_method_constants()

        self.point_metadata = { "Method": "AverageAccelerationMethod", "Constants": {
                        "a1": self.a1,
                        "a2": self.a2,
                        "a3": self.a3
                    }
                }

        self.solution_set = self.generate_point_cloud()


    def populate_sdof_constants(self, elastic_constant, damping_ratio, natural_frequency):
        k = elastic_constant
        zeta = damping_ratio
        wn = natural_frequency
        m = k/(wn**2)
        c = 2*math.sqrt(k*m)*zeta
        return k, zeta, wn, m, c

    def populate_method_constants(self):
        a1 = 4/(self.dt**2)*self.m+2/self.dt*self.c
        a2 = 4/self.dt*self.m+self.c
        a3=self.m
        k_hat = a1+k
        return a1, a2, a3, k_hat

    def populate_initial_conditions(self, initial_displacement, initial_velocity, initial_acceleration):
        if len([x for x in (initial_displacement, initial_velocity, initial_acceleration) if x is None]) > 1:
            raise ValueError("Error: at least 2 initial conditions need to be provided")
        elif initial_acceleration is None:
            initial_acceleration = (self.p(0)-self.c*initial_velocity-self.k*initial_displacement)/self.m
        elif initial_velocity is None:
            initial_velocity = (self.p(0)-self.m*initial_acceleration-self.k*initial_displacement)/self.c
        else:
            initial_displacement = (self.p(0)-self.c*initial_velocity-self.m*initial_acceleration)/self.k
        return initial_displacement, initial_velocity, initial_acceleration


    def generate_point_cloud(self):
        times_range = np.arange(0, self.dt * int(self.time_stop / self.dt) + self.dt, self.dt)
        solution_set = [SolutionPoint(time=0, displacement=self.u0, velocity=self.u1, acceleration=self.u2, metadata=self.point_metadata)]
        for i, t in enumerate(times_range):
            if i == 0:
                ui = self.u0
                u1i = self.u1
                u2i = self.u2
            else:
                ui = solution_set[i].u
                u1i = solution_set[i].v
                u2i = solution_set[i].a

            p_hat_next_step = self.p(t+self.dt) + self.a1*ui + self.a2*u1i + self.a3*u2i

            metadata = self.point_metadata.copy()
            metadata["p_hat_next_step"] = p_hat_next_step

            ui_next_step = p_hat_next_step/self.k_hat
            u1i_next_step = (2/self.dt)*(ui_next_step-ui)-u1i
            u2i_next_step = (4/self.dt**2)*(ui_next_step-ui)-4*u1i/self.dt-u2i
            solution_set.append(SolutionPoint(
                time=t + self.dt,
                displacement=ui_next_step,
                velocity=u1i_next_step,
                acceleration=u2i_next_step,
                metadata=self.populate_point_metadata(t, ui_next_step, metadata))
            )
        return solution_set

    def get_maximum_displacement(self)->float:
        """Gets the maximum displacement of the system among the cloud points in inches."""
        return max([abs(point.u) for point in self.solution_set])

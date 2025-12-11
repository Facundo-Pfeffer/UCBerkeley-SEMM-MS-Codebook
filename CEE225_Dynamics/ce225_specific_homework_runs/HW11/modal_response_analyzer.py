"""
Modal Response Analysis for MDOF Systems Subjected to Ground Motion.

This module implements modal analysis for computing the response of a
multi-degree-of-freedom structure to earthquake ground motion.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from newmark_sdof import newmark_sdof
from data_loader import DataLoader


class ModalResponseAnalyzer:
    """Analyzes modal response of MDOF structure to ground motion."""
    
    def __init__(self, mass_matrix, mode_shapes, natural_freqs, damping_ratios,
                 floor_heights=None):
        """
        Parameters:
        -----------
        mass_matrix : array-like
            Mass matrix m (n_floors x n_floors) [kg]
        mode_shapes : array-like
            Mass-orthonormal mode shape matrix Φ (n_floors x n_modes)
            Each column is a mode shape vector
        natural_freqs : array-like
            Natural frequencies ωn [rad/s] for each mode
        damping_ratios : array-like
            Damping ratios ζ [-] for each mode
        floor_heights : array-like, optional
            Floor heights from base [m]. If None, assumes [1, 2, 3, ...]
        """
        self.m = np.asarray(mass_matrix, dtype=float)
        self.Phi = np.asarray(mode_shapes, dtype=float)
        self.omega_n = np.asarray(natural_freqs, dtype=float)
        self.zeta = np.asarray(damping_ratios, dtype=float)
        
        if self.Phi.ndim == 1:
            self.Phi = self.Phi.reshape(-1, 1)
        
        self.n_floors, self.n_modes = self.Phi.shape
        
        if len(self.omega_n) != self.n_modes:
            raise ValueError(f"natural_freqs length ({len(self.omega_n)}) must match number of modes ({self.n_modes})")
        if len(self.zeta) != self.n_modes:
            raise ValueError(f"damping_ratios length ({len(self.zeta)}) must match number of modes ({self.n_modes})")
        
        if floor_heights is None:
            self.floor_heights = np.array([i + 1 for i in range(self.n_floors)], dtype=float)
        else:
            self.floor_heights = np.asarray(floor_heights, dtype=float)
        
        # Influence vector (uniform ground motion)
        self.iota = np.ones(self.n_floors)
        
        # Compute modal properties
        self._compute_modal_properties()
    
    def _compute_modal_properties(self):
        """Compute participation factors and modal static responses."""
        # Participation numerator: L_n = φ_n^T m ι
        self.L = np.array([self.Phi[:, n].T @ self.m @ self.iota for n in range(self.n_modes)])
        
        # Modal mass: M_n = φ_n^T m φ_n (should be 1 for mass-orthonormal)
        self.M_modal = np.array([self.Phi[:, n].T @ self.m @ self.Phi[:, n] for n in range(self.n_modes)])
        
        # Participation factor: Γ_n = L_n / M_n
        self.Gamma = self.L / self.M_modal
        
        # Effective modal mass: M_n* = Γ_n^2 M_n
        self.M_eff = self.Gamma**2 * self.M_modal
        
        # Effective modal height
        self.h_star = np.zeros(self.n_modes)
        for n in range(self.n_modes):
            numerator = sum(self.m[j, j] * self.Phi[j, n] * self.floor_heights[j] 
                          for j in range(self.n_floors))
            denominator = sum(self.m[j, j] * self.Phi[j, n] for j in range(self.n_floors))
            if abs(denominator) > 1e-12:
                self.h_star[n] = numerator / denominator
            else:
                self.h_star[n] = 0.0
        
        # Modal static story shears (for each story and mode)
        # Story r is between floors r and r+1 (r=0 is base, r=n_floors-1 is top)
        self.V_static = np.zeros((self.n_floors, self.n_modes))
        for n in range(self.n_modes):
            # Modal lateral force pattern: s_n = Γ_n m φ_n
            s_n = self.Gamma[n] * self.m @ self.Phi[:, n]
            # Story shear = sum of forces above that story
            for r in range(self.n_floors):
                self.V_static[r, n] = np.sum(s_n[r:])
        
        # Modal static base moment
        self.Mb_static = np.zeros(self.n_modes)
        for n in range(self.n_modes):
            s_n = self.Gamma[n] * self.m @ self.Phi[:, n]
            # Base moment = sum of (force * height)
            self.Mb_static[n] = sum(s_n[j] * self.floor_heights[j] for j in range(self.n_floors))
    
    def solve_modal_equations(self, ug_ddot, time, dt=None):
        """
        Solve modal equations for D_n(t) using Newmark's method.
        
        Parameters:
        -----------
        ug_ddot : array-like
            Ground acceleration üg(t) [m/s²]
        time : array-like
            Time vector [s]
        dt : float, optional
            Time step [s]. If None, computed from time array.
        
        Returns:
        --------
        D : np.ndarray
            Modal displacement D_n(t) [m] for each mode (n_modes x n_steps)
        D_dot : np.ndarray
            Modal velocity Ḋ_n(t) [m/s] for each mode
        D_ddot : np.ndarray
            Modal acceleration D̈_n(t) [m/s²] for each mode
        """
        ug_ddot = np.asarray(ug_ddot, dtype=float).flatten()
        time = np.asarray(time, dtype=float).flatten()
        
        if len(ug_ddot) != len(time):
            raise ValueError(f"ug_ddot length ({len(ug_ddot)}) must match time length ({len(time)})")
        
        if dt is None:
            dt = time[1] - time[0] if len(time) > 1 else 0.01
        
        n_steps = len(time)
        D = np.zeros((self.n_modes, n_steps))
        D_dot = np.zeros((self.n_modes, n_steps))
        D_ddot = np.zeros((self.n_modes, n_steps))
        
        # Solve for each mode
        for n in range(self.n_modes):
            # Modal equation: D̈_n + 2ζ_n ω_n Ḋ_n + ω_n² D_n = -üg
            # Equivalent SDOF: m_eff * D̈ + c_eff * Ḋ + k_eff * D = p_eff
            # where m_eff = 1, c_eff = 2ζ_n ω_n, k_eff = ω_n², p_eff = -üg
            
            m_eff = 1.0
            c_eff = 2.0 * self.zeta[n] * self.omega_n[n]
            k_eff = self.omega_n[n]**2
            p_eff = -ug_ddot  # Effective force
            
            # Initial conditions: D(0) = 0, Ḋ(0) = 0
            results = newmark_sdof(m_eff, k_eff, c_eff, p_eff, dt, 0.0, 0.0, 'constant')
            
            D[n, :] = results[:, 0]
            D_dot[n, :] = results[:, 1]
            D_ddot[n, :] = results[:, 2]
        
        return D, D_dot, D_ddot
    
    def compute_floor_responses(self, D, D_dot, D_ddot, ug_ddot=None):
        """
        Compute floor displacements, velocities, and accelerations.
        
        Parameters:
        -----------
        D : np.ndarray
            Modal displacement D_n(t) [m] (n_modes x n_steps)
        D_dot : np.ndarray
            Modal velocity Ḋ_n(t) [m/s] (n_modes x n_steps)
        D_ddot : np.ndarray
            Modal acceleration D̈_n(t) [m/s²] (n_modes x n_steps)
        ug_ddot : np.ndarray, optional
            Ground acceleration üg(t) [m/s²] (n_steps,). If provided,
            total floor accelerations = relative + ground.
        
        Returns:
        --------
        u : np.ndarray
            Floor displacements u_j(t) [m] (n_floors x n_steps)
        u_dot : np.ndarray
            Floor velocities [m/s] (n_floors x n_steps)
        u_ddot : np.ndarray
            Floor accelerations [m/s²] (n_floors x n_steps)
            If ug_ddot provided, returns total acceleration (relative + ground)
        """
        n_steps = D.shape[1]
        u = np.zeros((self.n_floors, n_steps))
        u_dot = np.zeros((self.n_floors, n_steps))
        u_ddot = np.zeros((self.n_floors, n_steps))
        
        # u_j = Σ Γ_n φ_jn D_n
        for j in range(self.n_floors):
            for n in range(self.n_modes):
                q_n = self.Gamma[n] * D[n, :]  # Modal coordinate
                u[j, :] += self.Phi[j, n] * q_n
                u_dot[j, :] += self.Phi[j, n] * self.Gamma[n] * D_dot[n, :]
                u_ddot[j, :] += self.Phi[j, n] * self.Gamma[n] * D_ddot[n, :]
        
        # Add ground acceleration to get total floor accelerations
        if ug_ddot is not None:
            for j in range(self.n_floors):
                u_ddot[j, :] += ug_ddot
        
        return u, u_dot, u_ddot
    
    def compute_base_shear(self, D):
        """
        Compute base shear V_b(t) = Σ V_{b,n}^st * A_n(t).
        
        Parameters:
        -----------
        D : np.ndarray
            Modal displacement D_n(t) [m] (n_modes x n_steps)
        
        Returns:
        --------
        V_base : np.ndarray
            Base shear [N] (n_steps,)
        """
        n_steps = D.shape[1]
        V_base = np.zeros(n_steps)
        
        # A_n = ω_n² D_n (pseudo-acceleration)
        # Base shear = V_{b,0}^st (story 0 is base)
        V_b_static = self.V_static[0, :]  # Base story shear for each mode
        
        for n in range(self.n_modes):
            A_n = self.omega_n[n]**2 * D[n, :]  # Pseudo-acceleration
            V_base += V_b_static[n] * A_n
        
        return V_base
    
    def compute_base_moment(self, D):
        """
        Compute base overturning moment M_b(t) = Σ M_{b,n}^st * A_n(t).
        
        Parameters:
        -----------
        D : np.ndarray
            Modal displacement D_n(t) [m] (n_modes x n_steps)
        
        Returns:
        --------
        M_base : np.ndarray
            Base moment [N·m] (n_steps,)
        """
        n_steps = D.shape[1]
        M_base = np.zeros(n_steps)
        
        # A_n = ω_n² D_n (pseudo-acceleration)
        for n in range(self.n_modes):
            A_n = self.omega_n[n]**2 * D[n, :]
            M_base += self.Mb_static[n] * A_n
        
        return M_base
    
    def compute_modal_coordinates(self, D):
        """
        Compute modal coordinates q_n(t) = Γ_n D_n(t).
        
        Parameters:
        -----------
        D : np.ndarray
            Modal displacement D_n(t) [m] (n_modes x n_steps)
        
        Returns:
        --------
        q : np.ndarray
            Modal coordinates q_n(t) [m] (n_modes x n_steps)
        """
        q = np.zeros_like(D)
        for n in range(self.n_modes):
            q[n, :] = self.Gamma[n] * D[n, :]
        return q


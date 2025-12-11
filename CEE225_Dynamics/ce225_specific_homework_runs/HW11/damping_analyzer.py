"""Damping analysis using logarithmic decrement method."""

import numpy as np
from scipy import signal


class DampingAnalyzer:
    """Computes damping ratios using logarithmic decrement from free vibration decay."""
    
    def __init__(self, natural_freq, min_cycles=3, peak_prominence_ratio=0.1):
        """
        Parameters:
        -----------
        natural_freq : float
            Natural frequency in Hz.
        min_cycles : int
            Minimum number of cycles to average over.
        peak_prominence_ratio : float
            Minimum peak prominence as ratio of max amplitude (0.1 = 10%).
        """
        self.natural_freq = natural_freq
        self.min_cycles = min_cycles
        self.peak_prominence_ratio = peak_prominence_ratio
        self.period = 1.0 / natural_freq

    def analyze_damping(self, time, acc_data, mode_shape=None):
        """
        Analyze damping from acceleration data.

        Parameters:
        -----------
        time : array-like
            Time vector.
        acc_data : list of array-like
            Acceleration data for each floor.
        mode_shape : array-like, optional
            Mode shape vector. If provided, computes modal acceleration.

        Returns:
        --------
        dict
            Dictionary with damping estimates for each floor and combined statistics.
        """
        time = np.asarray(time, dtype=float)
        num_floors = len(acc_data)
        acc_data = [np.asarray(acc, dtype=float) for acc in acc_data]

        # Compute modal accelerations for all floors
        modal_acc_data = []
        for floor_idx in range(num_floors):
            if mode_shape is not None:
                modal_acc = acc_data[floor_idx] * mode_shape[floor_idx]
            else:
                modal_acc = acc_data[floor_idx]
            modal_acc_data.append(modal_acc)

        # Detect decay start once using combined signal (RMS of all modal accelerations)
        combined_signal = np.sqrt(np.mean([acc**2 for acc in modal_acc_data], axis=0))
        global_decay_start_idx = self._detect_decay_start(time, combined_signal)

        results = {}
        floor_dampings = []

        for floor_idx in range(num_floors):
            modal_acc = modal_acc_data[floor_idx]

            floor_result = self._analyze_single_floor(
                time, modal_acc, floor_idx, decay_start_idx=global_decay_start_idx
            )
            results[f'floor_{floor_idx + 1}'] = floor_result

            if floor_result['damping_ratio'] is not None:
                floor_dampings.append(floor_result['damping_ratio'])

        if floor_dampings:
            results['mean_damping'] = np.mean(floor_dampings)
            results['std_damping'] = np.std(floor_dampings)
            results['cv_damping'] = results['std_damping'] / (results['mean_damping'] + 1e-10)
        else:
            results['mean_damping'] = None
            results['std_damping'] = None
            results['cv_damping'] = None

        results['all_floor_dampings'] = floor_dampings
        results['num_floors_used'] = len(floor_dampings)
        results['natural_freq'] = self.natural_freq

        return results

    def _analyze_single_floor(self, time, acc, floor_idx, decay_start_idx=None):
        """Analyze damping for a single floor."""
        if decay_start_idx is None:
            decay_start_idx = self._detect_decay_start(time, acc)

        if decay_start_idx is None:
            return {
                'damping_ratio': None,
                'log_decrement': None,
                'decay_start_time': None,
                'num_peaks': 0,
                'peaks': None,
                'peak_times': None,
                'envelope': None,
                'error': 'Could not detect decay start'
            }

        decay_time_full = time[decay_start_idx:]
        decay_acc_full = acc[decay_start_idx:]

        # Trim noisy end portion
        decay_end_idx = self._detect_decay_end(decay_time_full, decay_acc_full)
        if decay_end_idx is not None:
            decay_time = decay_time_full[:decay_end_idx]
            decay_acc = decay_acc_full[:decay_end_idx]
        else:
            decay_time = decay_time_full
            decay_acc = decay_acc_full

        peaks, peak_times = self._extract_peaks(decay_time, decay_acc)

        if len(peaks) < self.min_cycles + 1:
            return {
                'damping_ratio': None,
                'log_decrement': None,
                'decay_start_time': time[decay_start_idx],
                'num_peaks': len(peaks),
                'peaks': peaks,
                'peak_times': peak_times,
                'envelope': None,
                'error': f'Insufficient peaks: {len(peaks)} < {self.min_cycles + 1}'
            }

        log_decrement, damping_ratio = self._compute_log_decrement(peaks, peak_times)

        envelope = self._compute_envelope(decay_time, decay_acc)

        return {
            'damping_ratio': damping_ratio,
            'log_decrement': log_decrement,
            'decay_start_time': time[decay_start_idx],
            'decay_start_idx': decay_start_idx,
            'num_peaks': len(peaks),
            'peaks': peaks,
            'peak_times': peak_times,
            'envelope': envelope,
            'decay_time': decay_time,
            'decay_acc': decay_acc,
            'natural_freq': self.natural_freq,
            'error': None
        }

    def _detect_decay_start(self, time, acc):
        """
        Robust decay detection that adapts to signal duration.
        """
        acc_abs = np.abs(acc)
        if len(time) < 3:
            return None

        dt = time[1] - time[0]
        total_duration = time[-1] - time[0]

        max_idx = np.argmax(acc_abs)
        max_amp = acc_abs[max_idx]

        window_size = int(0.5 * self.period / dt)
        window_size = max(10, min(window_size, len(acc) // 20))

        moving_avg = np.convolve(acc_abs, np.ones(window_size) / window_size, mode='same')
        moving_avg_diff = np.diff(moving_avg)

        envelope = self._compute_envelope(time, acc)
        envelope_diff = np.diff(envelope)

        search_start_ratio = 0.3 if total_duration < 40 else 0.4
        search_start_idx = int(len(time) * search_start_ratio)
        search_start_idx = max(search_start_idx, max_idx + int(2 * self.period / dt))

        search_end_ratio = 0.85 if total_duration < 40 else 0.90
        search_end_idx = int(len(time) * search_end_ratio)

        best_decay_idx = None
        best_score = -np.inf

        for i in range(search_start_idx, min(search_end_idx, len(moving_avg_diff) - window_size)):
            window_diff = moving_avg_diff[i:i + window_size]
            envelope_window_diff = envelope_diff[i:min(i + window_size, len(envelope_diff))]

            score = 0.0

            negative_ratio = np.sum(window_diff < 0) / len(window_diff)
            if negative_ratio > 0.7:
                score += negative_ratio * 2.0

            if len(envelope_window_diff) > 0:
                envelope_negative_ratio = np.sum(envelope_window_diff < 0) / len(envelope_window_diff)
                if envelope_negative_ratio > 0.7:
                    score += envelope_negative_ratio * 1.5

            if np.all(window_diff < 0):
                score += 3.0

            amp_ratio = moving_avg[i] / max_amp if max_amp > 0 else 1.0
            if 0.1 < amp_ratio < 0.9:
                score += 1.0

            if score > best_score:
                best_score = score
                best_decay_idx = i

        if best_decay_idx is not None and best_score > 2.0:
            return best_decay_idx

        if max_idx < len(time) * 0.75:
            fallback_start = int(max_idx + 3 * self.period / dt)
            if fallback_start < len(time) - int(5 * self.period / dt):
                return min(fallback_start, len(time) - 1)

        return None

    def _detect_decay_end(self, time, acc):
        """
        Detect when the decay portion becomes too noisy or clearly deviates
        from exponential decay.

        Strategy:
        1. Compute envelope.
        2. Fit exponential in log-space on an early, high-amplitude window.
        3. Scan forward in time with a sliding window; locate the earliest time
           where the local log-slope becomes much less negative than the fitted
           slope (or positive), or where the envelope is close to the noise floor.
        """
        n = len(time)
        if n < 10:
            return None

        dt = time[1] - time[0]
        envelope = self._compute_envelope(time, acc)
        A0 = envelope[0] if n > 0 else np.max(np.abs(acc))

        # Minimum required samples (at least 5 periods)
        min_samples = int(5 * self.period / dt)
        min_samples = max(10, min_samples)
        if n <= min_samples:
            return None

        # Early window for fitting: high amplitude and limited length
        amp_cutoff = 0.4  # 40% of initial amplitude
        high_mask = envelope > amp_cutoff * A0
        if np.any(high_mask):
            last_high = np.where(high_mask)[0].max()
        else:
            last_high = min_samples

        early_end_idx = max(min_samples, min(last_high + 1, int(0.25 * n)))
        early_end_idx = min(early_end_idx, n - 1)

        t_early = time[:early_end_idx]
        e_early = envelope[:early_end_idx]

        log_e = np.log(e_early + 1e-10)
        valid = np.isfinite(log_e) & (e_early > 1e-10)
        if np.sum(valid) < 5:
            # fallback: simple amplitude threshold
            noise_level = 0.01 * A0
            for i in range(n):
                if envelope[i] < noise_level:
                    return i
            return None

        t_fit = t_early[valid]
        log_fit = log_e[valid]
        t0 = t_fit[0]
        slope_ref, intercept_ref = np.polyfit(t_fit - t0, log_fit, 1)
        # slope_ref should be negative

        # Sliding window length (~2 periods)
        win = int(2 * self.period / dt)
        win = max(5, min(win, n // 5))

        # Noise floor for a very conservative cutoff
        noise_floor = 0.01 * A0

        # Scan forward from the end of the fitting window
        start_idx = early_end_idx
        end_idx = n - win

        for i in range(start_idx, end_idx):
            j = i + win
            t_w = time[i:j]
            e_w = envelope[i:j]

            # If already clearly below noise floor, cut here
            if np.all(e_w < noise_floor):
                return i

            log_w = np.log(e_w + 1e-10)
            if not np.all(np.isfinite(log_w)):
                continue

            t_loc = t_w - t_w[0]
            s_loc, _ = np.polyfit(t_loc, log_w, 1)

            # Conditions:
            # 1) local slope becomes positive (envelope growing)
            if s_loc > 0.0:
                return i

            # 2) local decay rate much slower (less negative) than reference
            # slope_ref < 0, so 0.3 * slope_ref is closer to zero.
            if s_loc > 0.3 * slope_ref:
                return i

        return None

    def _extract_peaks(self, time, acc):
        """Extract positive peaks only for logarithmic decrement analysis."""
        acc_abs = np.abs(acc)
        max_amp = np.max(acc_abs)

        decay_duration = time[-1] - time[0]
        min_cycles_available = decay_duration * self.natural_freq

        if min_cycles_available < 5:
            prominence_ratio = max(0.05, self.peak_prominence_ratio * 0.5)
        elif min_cycles_available < 10:
            prominence_ratio = max(0.07, self.peak_prominence_ratio * 0.7)
        else:
            prominence_ratio = self.peak_prominence_ratio

        min_prominence = max_amp * prominence_ratio

        min_distance = int(0.7 * self.period / (time[1] - time[0]))
        min_distance = max(5, min_distance)

        peaks_pos, _ = signal.find_peaks(
            acc,
            prominence=min_prominence,
            distance=min_distance
        )

        if len(peaks_pos) < 2:
            return np.array([]), np.array([])

        peak_values = acc[peaks_pos]
        peak_times = time[peaks_pos]

        return peak_values, peak_times

    def _compute_log_decrement(self, peaks, peak_times):
        """
        Compute logarithmic decrement by averaging over multiple cycles.
        δ = (1/n) * ln(x₀/xₙ) for n cycles.
        ζ ≈ δ / (2π) for small damping.
        """
        if len(peaks) < 2:
            return None, None

        peak_amplitudes = peaks

        n_cycles = min(self.min_cycles, len(peak_amplitudes) - 1)
        if n_cycles < 1:
            n_cycles = 1

        log_decrements = []

        for i in range(len(peak_amplitudes) - n_cycles):
            x0 = peak_amplitudes[i]
            xn = peak_amplitudes[i + n_cycles]

            if x0 > 1e-12 and xn > 1e-12:
                delta = (1.0 / n_cycles) * np.log(x0 / xn)
                if 0.0 < delta < 2.0:
                    log_decrements.append(delta)

        if not log_decrements:
            return None, None

        mean_delta = np.mean(log_decrements)
        damping_ratio = mean_delta / np.sqrt(4 * np.pi**2 + mean_delta**2)

        return mean_delta, damping_ratio

    def _compute_envelope(self, time, acc):
        """Compute envelope using Hilbert transform."""
        from scipy.signal import hilbert

        analytic_signal = hilbert(acc)
        envelope = np.abs(analytic_signal)
        return envelope

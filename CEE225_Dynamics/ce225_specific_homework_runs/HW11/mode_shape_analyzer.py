"""Mode shape computation from acceleration data."""

import numpy as np


class ModeShapeAnalyzer:
    """Computes mode shapes using RMS ratios and correlation."""
    
    def __init__(self, num_floors, reference_floor=None, filter_thresholds=None):
        """
        Parameters:
        -----------
        num_floors : int
            Number of floors
        reference_floor : int, optional
            Reference floor (1-indexed). If None, uses top floor.
        filter_thresholds : dict, optional
            Keys: 'ref_amp_ratio' (0.05), 'glob_amp_ratio' (0.02), 
            'cos_theta_min' (0.95), 'corr_threshold' (0.05)
        """
        self.num_floors = num_floors
        self.reference_floor = reference_floor if reference_floor is not None else num_floors
        
        if self.reference_floor < 1 or self.reference_floor > num_floors:
            raise ValueError(f"reference_floor must be between 1 and {num_floors}")
        
        self.ref_idx = self.reference_floor - 1
        
        default_thresholds = {
            'ref_amp_ratio': 0.05,
            'glob_amp_ratio': 0.02,
            'cos_theta_min': 0.95,
            'corr_threshold': 0.05
        }
        
        if filter_thresholds:
            default_thresholds.update(filter_thresholds)
        
        self.filter_thresholds = default_thresholds
    
    def compute_mode_shape_statistics(self, time, *acc_data, use_filter=True):
        """Compute mode shape from acceleration data. Returns (mode_shape, statistics)."""
        if len(acc_data) != self.num_floors:
            raise ValueError(f"Expected {self.num_floors} acceleration arrays, got {len(acc_data)}")
        
        time = np.asarray(time, dtype=float)
        acc_data = [np.asarray(acc, dtype=float) for acc in acc_data]
        rms_values = np.array([np.sqrt(np.mean(acc ** 2)) for acc in acc_data])
        
        if rms_values[self.ref_idx] < 1e-12:
            raise ValueError(f"RMS of reference floor {self.reference_floor} is essentially zero")
        
        rms_ratios = rms_values / rms_values[self.ref_idx]
        signs = self._compute_signs(acc_data, self.ref_idx)
        phi_raw = signs * rms_ratios
        
        max_abs = np.max(np.abs(phi_raw))
        if max_abs < 1e-12:
            mode_shape_normalized = phi_raw.copy()
        else:
            mode_shape_normalized = phi_raw / max_abs
        
        ref_idx_max = int(np.argmax(np.abs(mode_shape_normalized)))
        if mode_shape_normalized[ref_idx_max] < 0.0:
            mode_shape_normalized *= -1.0
        
        raw_shapes, raw_times, raw_ref_amp, raw_max_inst = self._compute_instantaneous_shapes(
            time, acc_data, mode_shape_normalized
        )
        stats_raw = self._compute_statistics(raw_shapes)
        
        if use_filter and len(raw_shapes) > 0:
            filtered_shapes, filtered_times, filter_effective = self._apply_filters(
                raw_shapes, raw_times, raw_ref_amp, raw_max_inst, mode_shape_normalized
            )
        else:
            filtered_shapes = raw_shapes
            filtered_times = raw_times
            filter_effective = False
        
        if use_filter:
            used_shapes = filtered_shapes
            used_times = filtered_times
        else:
            used_shapes = raw_shapes
            used_times = raw_times
            filter_effective = False
        
        stats_used = self._compute_statistics(used_shapes)
        statistics = {
            "mode_shape_reference": mode_shape_normalized,
            "rms_values": rms_values,
            "rms_ratios": signs * rms_ratios,
            "mean_mode_shape_raw": stats_raw['mean'],
            "std_mode_shape_raw": stats_raw['std'],
            "coefficient_of_variation_raw": stats_raw['cv'],
            "confidence_95_lower_raw": stats_raw['ci_lower'],
            "confidence_95_upper_raw": stats_raw['ci_upper'],
            "all_mode_shapes_raw": raw_shapes,
            "time_raw": raw_times,
            "n_points_raw": len(raw_shapes),
            "mean_mode_shape": stats_used['mean'],
            "std_mode_shape": stats_used['std'],
            "coefficient_of_variation": stats_used['cv'],
            "confidence_95_lower": stats_used['ci_lower'],
            "confidence_95_upper": stats_used['ci_upper'],
            "all_mode_shapes_used": used_shapes,
            "time_used": used_times,
            "n_points_used": len(used_shapes),
            "use_filter": bool(use_filter),
            "filter_effective": bool(filter_effective),
            "all_mode_shapes": used_shapes,
            "time": used_times,
        }
        
        return mode_shape_normalized, statistics
    
    def _compute_signs(self, acc_data, ref_idx):
        """Compute signs from correlation with reference floor."""
        signs = np.ones(self.num_floors)
        ref_acc = acc_data[ref_idx]
        
        for i in range(self.num_floors):
            if i == ref_idx:
                continue
            
            num = np.mean(acc_data[i] * ref_acc)
            den = np.sqrt(np.mean(acc_data[i]**2) * np.mean(ref_acc**2)) + 1e-12
            c = num / den
            
            if abs(c) > self.filter_thresholds['corr_threshold']:
                signs[i] = np.sign(c)
            else:
                signs[i] = 1.0
        
        return signs
    
    def _compute_instantaneous_shapes(self, time, acc_data, mode_shape_ref):
        """Compute instantaneous normalized shapes from acceleration."""
        n_samples = len(time)
        raw_shapes = []
        raw_times = []
        raw_ref_amp = []
        raw_max_inst = []
        
        for i in range(n_samples):
            inst_acc = np.array([acc[i] for acc in acc_data], dtype=float)
            max_inst = np.max(np.abs(inst_acc))
            
            if max_inst < 1e-12:
                continue
            
            inst_norm = inst_acc / max_inst
            if np.dot(inst_norm, mode_shape_ref) < 0.0:
                inst_norm = -inst_norm
            
            raw_shapes.append(inst_norm)
            raw_times.append(time[i])
            raw_ref_amp.append(abs(acc_data[self.ref_idx][i]))
            raw_max_inst.append(max_inst)
        
        if not raw_shapes:
            raw_shapes = np.tile(mode_shape_ref, (1, 1))
            raw_times = np.array([time[0] if len(time) > 0 else 0.0])
            raw_ref_amp = np.array([abs(acc_data[self.ref_idx][0]) if len(acc_data[self.ref_idx]) > 0 else 0.0])
            raw_max_inst = np.array([1.0])
        else:
            raw_shapes = np.array(raw_shapes)
            raw_times = np.array(raw_times)
            raw_ref_amp = np.array(raw_ref_amp)
            raw_max_inst = np.array(raw_max_inst)
        
        return raw_shapes, raw_times, raw_ref_amp, raw_max_inst
    
    def _apply_filters(self, raw_shapes, raw_times, raw_ref_amp, raw_max_inst, mode_shape_ref):
        """Apply amplitude and shape consistency filters."""
        ref_max = np.max(raw_ref_amp)
        glob_max = np.max(raw_max_inst)
        
        ref_amp_thresh = self.filter_thresholds['ref_amp_ratio'] * ref_max
        glob_amp_thresh = self.filter_thresholds['glob_amp_ratio'] * glob_max
        cos_theta_min = self.filter_thresholds['cos_theta_min']
        
        shapes_new = []
        times_new = []
        
        for j in range(len(raw_shapes)):
            if raw_ref_amp[j] < ref_amp_thresh:
                continue
            if raw_max_inst[j] < glob_amp_thresh:
                continue
            
            v = raw_shapes[j]
            num = float(np.dot(v, mode_shape_ref))
            den = (np.linalg.norm(v) * np.linalg.norm(mode_shape_ref) + 1e-12)  # Regularization
            cos_theta = num / den
            
            if cos_theta < cos_theta_min:
                continue
            
            shapes_new.append(v)
            times_new.append(raw_times[j])
        
        if len(shapes_new) > 0:
            filtered_shapes = np.array(shapes_new)
            filtered_times = np.array(times_new)
            filter_effective = True
        else:
            filtered_shapes = raw_shapes
            filtered_times = raw_times
            filter_effective = False
        
        return filtered_shapes, filtered_times, filter_effective
    
    def _compute_statistics(self, shapes):
        """Compute statistics over instantaneous shapes."""
        if len(shapes) == 0:
            n = self.num_floors
            return {
                'mean': np.zeros(n),
                'std': np.zeros(n),
                'cv': np.zeros(n),
                'ci_lower': np.zeros(n),
                'ci_upper': np.zeros(n)
            }
        
        mean = np.mean(shapes, axis=0)
        std = np.std(shapes, axis=0)
        cv = std / (np.abs(mean) + 1e-10)  # Regularization to avoid division by zero
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std
        
        return {
            'mean': mean,
            'std': std,
            'cv': cv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }


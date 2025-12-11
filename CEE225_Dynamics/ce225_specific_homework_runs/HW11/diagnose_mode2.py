#!/usr/bin/env python3
"""Diagnostic script for Mode 2 damping analysis."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import DataLoader
from damping_analyzer import DampingAnalyzer
from mode_shape_analyzer import ModeShapeAnalyzer
import numpy as np

print("=" * 70)
print("Mode 2 Damping Analysis Diagnostic")
print("=" * 70)

# Load data
dl = DataLoader()
data = dl.load_mode_data(2)
time = data[0]
acc_data = data[1:]

print(f"\nData loaded:")
print(f"  Time range: {time[0]:.3f} to {time[-1]:.3f} seconds")
print(f"  Number of samples: {len(time)}")
print(f"  Number of floors: {len(acc_data)}")
print(f"  Sampling rate: {1.0/(time[1]-time[0]):.1f} Hz")

# Compute mode shape
analyzer = ModeShapeAnalyzer(3, reference_floor=3)
mode_shape, stats = analyzer.compute_mode_shape_statistics(time, *acc_data, use_filter=True)
print(f"\nMode shape: {mode_shape}")

# Analyze damping
natural_freq = 7.20
damping_analyzer = DampingAnalyzer(natural_freq=natural_freq)
results = damping_analyzer.analyze_damping(time, acc_data, mode_shape=mode_shape)

print(f"\nOverall Results:")
print(f"  Mean damping: {results.get('mean_damping')}")
print(f"  Std damping: {results.get('std_damping')}")
print(f"  Floors used: {results.get('num_floors_used')}")
print(f"  Natural frequency: {natural_freq} Hz")
print(f"  Period: {1.0/natural_freq:.4f} s")

print(f"\nPer-Floor Analysis:")
for i in range(3):
    floor_res = results.get(f'floor_{i+1}', {})
    print(f"\n  Floor {i+1}:")
    print(f"    Damping ratio: {floor_res.get('damping_ratio')}")
    print(f"    Log decrement: {floor_res.get('log_decrement')}")
    print(f"    Decay start time: {floor_res.get('decay_start_time')}")
    print(f"    Decay start index: {floor_res.get('decay_start_idx')}")
    
    if floor_res.get('decay_start_idx') is not None:
        decay_idx = floor_res.get('decay_start_idx')
        total_time = time[-1] - time[0]
        decay_time_remaining = time[-1] - time[decay_idx]
        print(f"    Time remaining after decay start: {decay_time_remaining:.3f} s ({100*decay_time_remaining/total_time:.1f}% of total)")
        print(f"    Expected cycles in decay: {decay_time_remaining * natural_freq:.1f}")
    
    print(f"    Number of peaks: {floor_res.get('num_peaks')}")
    print(f"    Error: {floor_res.get('error')}")
    
    if floor_res.get('decay_time') is not None:
        decay_time = floor_res.get('decay_time')
        print(f"    Decay time range: {decay_time[0]:.3f} to {decay_time[-1]:.3f} s")
        print(f"    Decay duration: {decay_time[-1] - decay_time[0]:.3f} s")
        print(f"    Decay samples: {len(decay_time)}")
    
    if floor_res.get('peaks') is not None:
        peaks = floor_res.get('peaks')
        peak_times = floor_res.get('peak_times')
        if len(peaks) > 0:
            print(f"    Peak values range: {np.min(np.abs(peaks)):.6f} to {np.max(np.abs(peaks)):.6f}")
            print(f"    First peak: {peaks[0]:.6f} at {peak_times[0]:.3f} s")
            print(f"    Last peak: {peaks[-1]:.6f} at {peak_times[-1]:.3f} s")
            if len(peaks) > 1:
                periods = np.diff(peak_times)
                print(f"    Average period between peaks: {np.mean(periods):.4f} s (expected: {1.0/natural_freq:.4f} s)")
                print(f"    Period std: {np.std(periods):.4f} s")
    
    # Check modal acceleration
    modal_acc = acc_data[i] * mode_shape[i]
    acc_max = np.max(np.abs(modal_acc))
    print(f"    Modal acceleration max: {acc_max:.6f}")
    
    if floor_res.get('decay_start_idx') is not None:
        decay_idx = floor_res.get('decay_start_idx')
        acc_before = np.max(np.abs(modal_acc[:decay_idx]))
        acc_after = np.max(np.abs(modal_acc[decay_idx:]))
        print(f"    Max acc before decay: {acc_before:.6f}")
        print(f"    Max acc after decay: {acc_after:.6f}")
        print(f"    Ratio (after/before): {acc_after/acc_before if acc_before > 0 else 0:.3f}")

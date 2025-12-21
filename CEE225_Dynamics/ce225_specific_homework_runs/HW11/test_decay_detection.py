#!/usr/bin/env python3
"""Test script to diagnose decay detection and exponential fit issues."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import DataLoader
from damping_analyzer import DampingAnalyzer
from mode_shape_analyzer import ModeShapeAnalyzer
import numpy as np

def test_mode(mode_num, natural_freq):
    """Test decay detection for a specific mode."""
    print(f"\n{'='*70}")
    print(f"Testing Mode {mode_num} (fn = {natural_freq} Hz)")
    print(f"{'='*70}")
    
    # Load data
    dl = DataLoader()
    data = dl.load_mode_data(mode_num)
    time = data[0]
    acc_data = data[1:]
    
    print(f"\nSignal characteristics:")
    print(f"  Duration: {time[-1] - time[0]:.2f} seconds")
    print(f"  Samples: {len(time)}")
    print(f"  Sampling rate: {1.0/(time[1]-time[0]):.1f} Hz")
    
    # Compute mode shape
    analyzer = ModeShapeAnalyzer(3, reference_floor=3)
    mode_shape, stats = analyzer.compute_mode_shape_statistics(time, *acc_data, use_filter=True)
    
    # Analyze damping
    damping_analyzer = DampingAnalyzer(natural_freq=natural_freq)
    results = damping_analyzer.analyze_damping(time, acc_data, mode_shape=mode_shape)
    
    print(f"\nOverall results:")
    print(f"  Mean damping: {results.get('mean_damping'):.4f}" if results.get('mean_damping') else "  Mean damping: None")
    print(f"  Floors used: {results.get('num_floors_used')}")
    
    print(f"\nPer-floor details:")
    for i in range(3):
        floor_res = results.get(f'floor_{i+1}', {})
        print(f"\n  Floor {i+1}:")
        print(f"    Damping: {floor_res.get('damping_ratio'):.4f}" if floor_res.get('damping_ratio') else "    Damping: None")
        print(f"    Decay start: {floor_res.get('decay_start_time'):.2f} s" if floor_res.get('decay_start_time') else "    Decay start: None")
        print(f"    Peaks found: {floor_res.get('num_peaks', 0)}")
        print(f"    Error: {floor_res.get('error', 'None')}")
        
        if floor_res.get('decay_time') is not None:
            decay_time = floor_res.get('decay_time')
            decay_duration = decay_time[-1] - decay_time[0]
            print(f"    Decay duration: {decay_duration:.2f} s")
            print(f"    Decay cycles: {decay_duration * natural_freq:.1f}")
            
            if floor_res.get('envelope') is not None:
                envelope = floor_res.get('envelope')
                A0 = envelope[0]
                A_end = envelope[-1]
                print(f"    Envelope start: {A0:.6f}")
                print(f"    Envelope end: {A_end:.6f}")
                print(f"    Envelope ratio (end/start): {A_end/A0:.4f}" if A0 > 0 else "    Envelope ratio: N/A")
                
                # Check exponential fit parameters
                damping_ratio = floor_res.get('damping_ratio')
                if damping_ratio:
                    omega_n = 2 * np.pi * natural_freq
                    t0 = decay_time[0]
                    t_end = decay_time[-1]
                    expected_A_end = A0 * np.exp(-damping_ratio * omega_n * (t_end - t0))
                    print(f"    Expected A_end (from fit): {expected_A_end:.6f}")
                    print(f"    Actual A_end: {A_end:.6f}")
                    print(f"    Fit error: {abs(expected_A_end - A_end)/A0*100:.2f}%")
                    
                    # Decay rate
                    decay_rate = damping_ratio * omega_n
                    print(f"    Decay rate (ζ*ωn): {decay_rate:.4f} rad/s")
                    print(f"    Time constant (1/(ζ*ωn)): {1.0/decay_rate:.2f} s" if decay_rate > 0 else "    Time constant: N/A")

if __name__ == '__main__':
    # Test all modes
    mode_freqs = {1: 3.50, 2: 7.20, 3: 10.50}
    
    for mode_num, freq in mode_freqs.items():
        test_mode(mode_num, freq)












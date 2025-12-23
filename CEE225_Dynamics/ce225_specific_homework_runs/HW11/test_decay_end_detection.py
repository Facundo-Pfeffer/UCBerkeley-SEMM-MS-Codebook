#!/usr/bin/env python3
"""Test script to diagnose decay end detection."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import DataLoader
from damping_analyzer import DampingAnalyzer
from mode_shape_analyzer import ModeShapeAnalyzer
import numpy as np
import matplotlib.pyplot as plt

def test_mode2_decay_end():
    """Test decay end detection for Mode 2."""
    print("=" * 70)
    print("Testing Decay End Detection for Mode 2")
    print("=" * 70)
    
    # Load data
    dl = DataLoader()
    data = dl.load_mode_data(2)
    time = data[0]
    acc_data = data[1:]
    
    print(f"\nSignal characteristics:")
    print(f"  Duration: {time[-1] - time[0]:.2f} seconds")
    print(f"  Samples: {len(time)}")
    
    # Compute mode shape
    analyzer = ModeShapeAnalyzer(3, reference_floor=3)
    mode_shape, stats = analyzer.compute_mode_shape_statistics(time, *acc_data, use_filter=True)
    
    # Analyze damping
    damping_analyzer = DampingAnalyzer(natural_freq=7.20)
    results = damping_analyzer.analyze_damping(time, acc_data, mode_shape=mode_shape)
    
    print(f"\nDecay detection results:")
    for i in range(3):
        floor_res = results.get(f'floor_{i+1}', {})
        print(f"\n  Floor {i+1}:")
        print(f"    Decay start: {floor_res.get('decay_start_time'):.2f} s" if floor_res.get('decay_start_time') else "    Decay start: None")
        
        if floor_res.get('decay_time') is not None:
            decay_time = floor_res.get('decay_time')
            print(f"    Decay time range: {decay_time[0]:.2f} to {decay_time[-1]:.2f} s")
            print(f"    Decay duration: {decay_time[-1] - decay_time[0]:.2f} s")
            print(f"    Original signal end: {time[-1]:.2f} s")
            print(f"    Trimmed by: {time[-1] - decay_time[-1]:.2f} s")
            
            if floor_res.get('envelope') is not None:
                envelope = floor_res.get('envelope')
                print(f"    Envelope start: {envelope[0]:.6f}")
                print(f"    Envelope end: {envelope[-1]:.6f}")
                print(f"    Envelope ratio (end/start): {envelope[-1]/envelope[0]:.4f}" if envelope[0] > 0 else "    N/A")
        
        print(f"    Peaks found: {floor_res.get('num_peaks', 0)}")
        print(f"    Damping: {floor_res.get('damping_ratio'):.4f}" if floor_res.get('damping_ratio') else "    Damping: None")
    
    # Test the decay end detection directly
    print(f"\n\nDirect decay end detection test:")
    print("-" * 70)
    
    # Get modal acceleration for floor 1
    modal_acc = acc_data[0] * mode_shape[0]
    
    # Detect decay start
    decay_start_idx = damping_analyzer._detect_decay_start(time, modal_acc)
    if decay_start_idx is None:
        print("Could not detect decay start!")
        return
    
    print(f"Decay start detected at: {time[decay_start_idx]:.2f} s (index {decay_start_idx})")
    
    # Get full decay portion
    decay_time_full = time[decay_start_idx:]
    decay_acc_full = modal_acc[decay_start_idx:]
    
    print(f"Full decay portion: {decay_time_full[0]:.2f} to {decay_time_full[-1]:.2f} s")
    print(f"Full decay duration: {decay_time_full[-1] - decay_time_full[0]:.2f} s")
    
    # Test end detection
    decay_end_idx = damping_analyzer._detect_decay_end(decay_time_full, decay_acc_full)
    
    if decay_end_idx is not None:
        print(f"\nDecay end detected at: {decay_time_full[decay_end_idx]:.2f} s (index {decay_end_idx})")
        print(f"Trimmed decay portion: {decay_time_full[0]:.2f} to {decay_time_full[decay_end_idx]:.2f} s")
        print(f"Trimmed duration: {decay_time_full[decay_end_idx] - decay_time_full[0]:.2f} s")
        print(f"Data trimmed from end: {decay_time_full[-1] - decay_time_full[decay_end_idx]:.2f} s")
    else:
        print("\nNo decay end detected - using full decay portion")
        print("This might be the problem!")

if __name__ == '__main__':
    test_mode2_decay_end()















#!/usr/bin/env python3
"""
HW11 Main Runner - Mode Shape Analysis for MDOF Building
========================================================
Main script for Homework 11: analyzes mode shapes from acceleration data.

This script orchestrates the complete analysis workflow:
- Loads acceleration data from CSV files
- Computes mode shapes using RMS ratios and correlation
- Generates interactive visualizations
- Updates HTML files with results

Author: Facundo L. Pfeffer
Course: CEE225 - Structural Dynamics
University of California, Berkeley
"""

from pathlib import Path
import numpy as np
from data_loader import DataLoader
from mode_shape_analyzer import ModeShapeAnalyzer
from mode_shape_plotter import ModeShapePlotter
from damping_analyzer import DampingAnalyzer
from damping_plotter import DampingPlotter
from html_updater import update_mode_shape_matrix_in_html
from problem3_modal_response import run_problem3_analysis
from problem4_response_spectrum import run_problem4


def main():
    """Main analysis function."""
    print("=" * 70)
    print("Mode Shape Analysis - MDOF Building")
    print("=" * 70)

    # Configuration
    mode_freqs = {1: 2.00, 2: 7.20, 3: 13.75}  # Natural frequencies in Hz
    mode_numbers = [1, 2, 3]

    # Mass matrix (from your image: diagonal with 1180, 1180, 910)
    M = np.array([
        [1180, 0, 0],
        [0, 1180, 0],
        [0, 0, 910]
    ])
    # Floor heights consistent with overturning moment lever arms (m)
    floor_heights = np.array([2.0828, 4.1656, 6.2484])

    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / 'highlighted_htmls'
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")

    # Initialize components
    data_loader = DataLoader()
    num_floors = data_loader.get_num_floors(mode_number=1)
    print(f"Detected {num_floors} floors from data")

    # Initialize analyzer with mass matrix for post-processing
    analyzer = ModeShapeAnalyzer(
        num_floors=num_floors,
        reference_floor=num_floors
    )
    plotter = ModeShapePlotter(num_floors=num_floors)
    damping_plotter = DampingPlotter(num_floors=num_floors)

    # Storage for results
    all_mode_shapes = []
    all_statistics_used = []
    all_statistics_raw = []
    all_damping_results = []

    # Analyze each mode
    for mode_num in mode_numbers:
        print(f"\nAnalyzing Mode {mode_num}...")

        # Load data
        data = data_loader.load_mode_data(mode_num)
        time = data[0]
        acc_data = data[1:]
        
        print(f"  Loaded {len(time)} time steps")
        print(f"  Time range: {time[0]:.3f} to {time[-1]:.3f} seconds")

        # Compute mode shapes (filtered)
        mode_shape, statistics_used = analyzer.compute_mode_shape_statistics(
            time, *acc_data, use_filter=True
        )

        # Compute mode shapes (raw, for sensitivity analysis)
        _, statistics_raw = analyzer.compute_mode_shape_statistics(
            time, *acc_data, use_filter=False
        )

        # Report results
        f_nat = mode_freqs.get(mode_num, None)
        if f_nat is not None:
            print(f"  Mode {mode_num} natural frequency (given): {f_nat:.2f} Hz")

        print(f"  Reference mode shape (from RMS ratios): {mode_shape}")
        print(f"  RMS ratios to floor {num_floors}: {statistics_used['rms_ratios']}")
        print(f"  Mean from raw instantaneous shapes: {statistics_used['mean_mode_shape_raw']}")
        print(f"  Mean from used (filtered) shapes:   {statistics_used['mean_mode_shape']}")
        print(f"  CoV (raw):  {statistics_used['coefficient_of_variation_raw']}")
        print(f"  CoV (used): {statistics_used['coefficient_of_variation']}")
        print(f"  Raw samples: {statistics_used['n_points_raw']}, "
              f"used after filter: {statistics_used['n_points_used']}")

        all_mode_shapes.append(mode_shape)
        all_statistics_used.append(statistics_used)
        all_statistics_raw.append(statistics_raw)
        
        # Damping analysis
        damping_analyzer = DampingAnalyzer(natural_freq=mode_freqs[mode_num])
        damping_results = damping_analyzer.analyze_damping(time, acc_data, mode_shape=mode_shape)
        all_damping_results.append(damping_results)
        
        if damping_results['mean_damping'] is not None:
            print(f"  Damping ratio (mean): {damping_results['mean_damping']:.4f}")
            if damping_results['std_damping'] is not None:
                print(f"  Damping ratio (std):  {damping_results['std_damping']:.4f}")
            print(f"  Floors used: {damping_results['num_floors_used']}")
        else:
            print(f"  Damping analysis: Failed to compute")

    # Post-process: assemble mode shape matrix (columns are mode shapes)
    print("\n" + "-" * 70)
    print("Post-processing: Assemble mode shape matrix (no mass-normalization)")
    print("-" * 70)
    Phi_original = np.column_stack(all_mode_shapes)
    print(f"\nMode shape matrix Φ (from RMS ratios):")
    print(Phi_original)
    all_mode_shapes_mass_normalized = None
    has_mass_normalized = False

    # Generate plots (filtered results)
    print("\nGenerating filtered visualizations...")
    for mode_num in mode_numbers:
        mode_idx = mode_num - 1
        output_path = output_dir / f'mode_{mode_num}_analysis.html'
        plotter.plot_mode_shape(
            mode_num,
            all_mode_shapes[mode_idx],
            all_statistics_used[mode_idx],
            output_path,
            mode_shape_mass_normalized=None
        )

    # Generate plots (raw results for sensitivity page)
    print("\nGenerating raw (unfiltered) visualizations...")
    for mode_num in mode_numbers:
        mode_idx = mode_num - 1
        output_path_raw = output_dir / f'mode_{mode_num}_analysis_raw.html'
        plotter.plot_mode_shape(
            mode_num,
            all_mode_shapes[mode_idx],
            all_statistics_raw[mode_idx],
            output_path_raw,
        )

    # Combined visualization
    print("\nGenerating combined visualization...")
    combined_output = output_dir / 'all_mode_shapes.html'
    plotter.plot_combined_mode_shapes(
        all_mode_shapes,
        all_statistics_used,
        combined_output,
        all_mode_shapes_mass_normalized=all_mode_shapes_mass_normalized if has_mass_normalized else None
    )

    # Damping analysis plots
    print("\nGenerating damping visualizations...")
    for mode_num in mode_numbers:
        mode_idx = mode_num - 1
        
        data = data_loader.load_mode_data(mode_num)
        time_damping = data[0]
        acc_data_damping = data[1:]
        
        damping_output = output_dir / f'mode_{mode_num}_damping.html'
        damping_plotter.plot_damping_analysis(
            mode_num,
            time_damping,
            acc_data_damping,
            all_damping_results[mode_idx],
            all_mode_shapes[mode_idx],
            damping_output
        )

    # Update HTML with mode shape matrix
    print("\nUpdating mode shape matrix in HTML...")
    update_mode_shape_matrix_in_html(
        all_mode_shapes, 
        output_dir,
        all_mode_shapes_mass_normalized=None,
        mass_matrix=None
    )

    # Common inputs for Problems 3 and 4 (mass-normalize with M)
    def mass_normalize(Phi, M):
        Phi_n = Phi.copy().astype(float)
        for k in range(Phi.shape[1]):
            mk = float(Phi[:, k].T @ M @ Phi[:, k])
            if mk > 1e-12:
                Phi_n[:, k] /= np.sqrt(mk)
        return Phi_n

    Phi_used = mass_normalize(np.column_stack(all_mode_shapes), M)
    omega_used = 2 * np.pi * np.array([mode_freqs[n] for n in mode_numbers])
    zeta_used = np.array([
        all_damping_results[i]['mean_damping']
        if all_damping_results[i]['mean_damping'] is not None
        else [0.0113, 0.0157, 0.0093][i]
        for i in range(len(mode_numbers))
    ])
    # floor_heights already defined on line 47: [2.0828, 4.1656, 6.2484] meters

    # Step 3: Modal Response Analysis
    print("\n" + "=" * 70)
    print("Problem #3: Modal Response Analysis")
    print("=" * 70)
    try:
        run_problem3_analysis(
            mass_matrix=M,
            mode_shapes=Phi_used,
            natural_freqs=omega_used,
            damping_ratios=zeta_used,
            floor_heights=floor_heights,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"\nWarning: Step 3 analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 4: Response Spectrum Analysis
    print("\n" + "=" * 70)
    print("Step 4: Response Spectrum Analysis")
    print("=" * 70)
    try:
        run_problem4(
            mass_matrix=M,
            mode_shapes=Phi_used,
            natural_freqs=omega_used,
            damping_ratios=zeta_used,
            floor_heights=floor_heights,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"\nWarning: Step 4 analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")
    for mode_num in mode_numbers:
        print(f"  - mode_{mode_num}_analysis.html")
        print(f"  - mode_{mode_num}_analysis_raw.html")
        print(f"  - mode_{mode_num}_damping.html")
    print("  - all_mode_shapes.html")
    print("  - problem3_modal_displacements.html")
    print("  - problem3_floor_displacements.html")
    print("  - problem3_floor_accelerations.html")
    print("  - problem3_base_shear.html")
    print("  - problem3_base_moment.html")
    print("  - step3_modal_response.html")
    print("  - problem4_spectrum.html")
    print("  - step4_response_spectrum.html")


if __name__ == '__main__':
    main()

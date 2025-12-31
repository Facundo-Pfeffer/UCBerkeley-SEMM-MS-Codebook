#!/usr/bin/env python3
"""
Test script for mass normalization and orthogonalization of mode shapes.

This script demonstrates:
1. Computing mode shapes from acceleration data
2. Normalizing them so that φᵀ M φ = 1 for each mode
3. Orthogonalizing them so that Φᵀ M Φ = I (diagonal)
"""

import numpy as np
from pathlib import Path
from data_loader import DataLoader
from mode_shape_analyzer import ModeShapeAnalyzer


def main():
    """Test mass normalization and orthogonalization."""
    print("=" * 70)
    print("Testing Mass Normalization and Orthogonalization")
    print("=" * 70)
    
    # Define mass matrix (from your image: diagonal with 1180, 1180, 910)
    # Assuming 3 floors based on the mass matrix dimensions
    M = np.array([
        [1180, 0, 0],
        [0, 1180, 0],
        [0, 0, 910]
    ])
    print(f"\nMass matrix M:\n{M}")
    
    # Configuration
    mode_numbers = [1, 2, 3]
    num_floors = 3
    
    # Initialize components
    data_loader = DataLoader()
    analyzer = ModeShapeAnalyzer(
        num_floors=num_floors,
        reference_floor=num_floors,
        mass_matrix=M
    )
    
    # Storage for mode shapes
    all_mode_shapes = []
    
    # Compute mode shapes for each mode
    print("\n" + "-" * 70)
    print("Step 1: Computing mode shapes from acceleration data")
    print("-" * 70)
    for mode_num in mode_numbers:
        print(f"\nMode {mode_num}:")
        
        # Load data
        data = data_loader.load_mode_data(mode_num)
        time = data[0]
        acc_data = data[1:]
        
        # Compute mode shape
        mode_shape, statistics = analyzer.compute_mode_shape_statistics(
            time, *acc_data, use_filter=True
        )
        
        print(f"  Original mode shape: {mode_shape}")
        all_mode_shapes.append(mode_shape)
    
    # Convert to matrix (columns are mode shapes)
    Phi_original = np.column_stack(all_mode_shapes)
    print(f"\nOriginal mode shape matrix Φ (columns are modes):")
    print(Phi_original)
    
    # Check original modal mass matrix
    print("\n" + "-" * 70)
    print("Step 2: Checking original Φᵀ M Φ")
    print("-" * 70)
    M_modal_original = analyzer.compute_modal_mass_matrix(Phi_original)
    print(f"\nOriginal Φᵀ M Φ:\n{M_modal_original}")
    print(f"\nDiagonal elements: {np.diag(M_modal_original)}")
    off_diag = M_modal_original - np.diag(np.diag(M_modal_original))
    print(f"Max off-diagonal: {np.max(np.abs(off_diag)):.6e}")
    
    # Normalize each mode individually
    print("\n" + "-" * 70)
    print("Step 3: Mass-normalizing each mode (φᵢᵀ M φᵢ = 1)")
    print("-" * 70)
    Phi_normalized = np.zeros_like(Phi_original)
    for i, mode_shape in enumerate(all_mode_shapes):
        normalized = analyzer.normalize_mode_shape_mass(mode_shape)
        Phi_normalized[:, i] = normalized
        print(f"\nMode {i+1}:")
        print(f"  Original: {mode_shape}")
        print(f"  Normalized: {normalized}")
        # Verify normalization
        modal_mass = normalized.T @ M @ normalized
        print(f"  Verification: φᵀ M φ = {modal_mass:.6f} (should be 1.0)")
    
    # Check normalized modal mass matrix
    print("\n" + "-" * 70)
    print("Step 4: Checking normalized Φᵀ M Φ")
    print("-" * 70)
    M_modal_normalized = analyzer.compute_modal_mass_matrix(Phi_normalized)
    print(f"\nNormalized Φᵀ M Φ:\n{M_modal_normalized}")
    print(f"\nDiagonal elements: {np.diag(M_modal_normalized)}")
    off_diag = M_modal_normalized - np.diag(np.diag(M_modal_normalized))
    print(f"Max off-diagonal: {np.max(np.abs(off_diag)):.6e}")
    
    # Orthogonalize modes
    print("\n" + "-" * 70)
    print("Step 5: Orthogonalizing modes (making Φᵀ M Φ diagonal)")
    print("-" * 70)
    Phi_orthogonalized = analyzer.orthogonalize_mode_shapes_mass(Phi_original)
    print(f"\nOrthogonalized mode shape matrix:")
    print(Phi_orthogonalized)
    
    # Check orthogonalized modal mass matrix
    print("\n" + "-" * 70)
    print("Step 6: Verifying orthogonalized Φᵀ M Φ ≈ I")
    print("-" * 70)
    is_orthonormal, M_modal_ortho, max_off_diag, max_diag_dev = \
        analyzer.verify_mass_orthonormality(Phi_orthogonalized)
    
    print(f"\nOrthogonalized Φᵀ M Φ:\n{M_modal_ortho}")
    print(f"\nMax off-diagonal: {max_off_diag:.6e}")
    print(f"Max diagonal deviation from 1.0: {max_diag_dev:.6e}")
    print(f"Mass-orthonormal: {'✓ YES' if is_orthonormal else '✗ NO'}")
    
    # Compare changes
    print("\n" + "-" * 70)
    print("Step 7: Comparison")
    print("-" * 70)
    print("\nOriginal vs Orthogonalized mode shapes:")
    for i in range(num_floors):
        print(f"\nMode {i+1}:")
        print(f"  Original:      {Phi_original[:, i]}")
        print(f"  Orthogonalized: {Phi_orthogonalized[:, i]}")
        change = Phi_orthogonalized[:, i] - Phi_original[:, i]
        print(f"  Change:         {change}")
        relative_change = np.linalg.norm(change) / np.linalg.norm(Phi_original[:, i])
        print(f"  Relative change: {relative_change*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()














# Mass Normalization and Orthogonalization of Mode Shapes

## Overview

This document explains the implementation of mass-normalization and mass-orthogonalization for mode shapes, ensuring that **Φᵀ M Φ** is as diagonal as possible (ideally equal to the identity matrix **I**).

## Theory

### Mass-Orthonormal Mode Shapes

In structural dynamics, mode shapes are typically normalized so that they satisfy:

```
Φᵀ M Φ = I
```

where:
- **Φ** is the mode shape matrix (columns are individual mode shapes)
- **M** is the mass matrix
- **I** is the identity matrix

This condition means:
1. **Mass normalization**: Each mode shape φᵢ satisfies φᵢᵀ M φᵢ = 1
2. **Mass orthogonality**: Different mode shapes are orthogonal: φᵢᵀ M φⱼ = 0 for i ≠ j

### Why This Matters

When mode shapes are mass-orthonormal:
- Modal equations decouple completely
- Modal masses are all equal to 1
- Modal analysis becomes simpler and more numerically stable
- The transformation to modal coordinates is well-conditioned

## Algorithm Changes

### Original Algorithm

The original algorithm:
1. Computes mode shapes from acceleration data using RMS ratios
2. Normalizes by maximum amplitude
3. Processes each mode independently

**Result**: Mode shapes are not necessarily mass-orthonormal.

### Modified Algorithm

The modified algorithm adds two optional steps:

#### Step 1: Mass Normalization (Individual Modes)

For each mode shape φᵢ:
```
M_r = φᵢᵀ M φᵢ  (modal mass)
φ_normalized = φᵢ / √(M_r)
```

This ensures φ_normalizedᵀ M φ_normalized = 1.

#### Step 2: Mass Orthogonalization (Multiple Modes)

Using **mass-weighted Gram-Schmidt orthogonalization**:

1. Start with the first mode: φ₁_normalized
2. For each subsequent mode φᵢ:
   - Subtract its projection onto all previous modes:
     ```
     v = φᵢ
     for j = 1 to i-1:
         proj = vᵀ M φⱼ
         v = v - proj × φⱼ
     ```
   - Normalize: φᵢ = v / √(vᵀ M v)

**Result**: All mode shapes are mass-orthonormal: Φᵀ M Φ = I

## Implementation

### New Methods in `ModeShapeAnalyzer`

1. **`normalize_mode_shape_mass(mode_shape)`**
   - Normalizes a single mode shape so φᵀ M φ = 1
   - Returns the normalized mode shape

2. **`orthogonalize_mode_shapes_mass(mode_shapes)`**
   - Takes a matrix of mode shapes (columns)
   - Returns mass-orthonormal mode shapes
   - Uses modified Gram-Schmidt with mass-weighted inner product

3. **`compute_modal_mass_matrix(mode_shapes)`**
   - Computes Φᵀ M Φ for verification
   - Returns the modal mass matrix

4. **`verify_mass_orthonormality(mode_shapes, tolerance=1e-3)`**
   - Checks if Φᵀ M Φ ≈ I
   - Returns verification results and diagnostics

### Usage

```python
# Initialize analyzer with mass matrix
M = np.array([[1180, 0, 0],
              [0, 1180, 0],
              [0, 0, 910]])

analyzer = ModeShapeAnalyzer(
    num_floors=3,
    reference_floor=3,
    mass_matrix=M
)

# Compute mode shapes (as before)
mode_shape_1, stats_1 = analyzer.compute_mode_shape_statistics(time, *acc_data_1)
mode_shape_2, stats_2 = analyzer.compute_mode_shape_statistics(time, *acc_data_2)
mode_shape_3, stats_3 = analyzer.compute_mode_shape_statistics(time, *acc_data_3)

# Collect into matrix
Phi = np.column_stack([mode_shape_1, mode_shape_2, mode_shape_3])

# Option 1: Normalize each mode individually
Phi_normalized = np.column_stack([
    analyzer.normalize_mode_shape_mass(mode_shape_1),
    analyzer.normalize_mode_shape_mass(mode_shape_2),
    analyzer.normalize_mode_shape_mass(mode_shape_3)
])

# Option 2: Orthogonalize all modes together (recommended)
Phi_orthogonalized = analyzer.orthogonalize_mode_shapes_mass(Phi)

# Verify
is_orthonormal, M_modal, max_off_diag, max_diag_dev = \
    analyzer.verify_mass_orthonormality(Phi_orthogonalized)
```

## Impact on Results

### What Changes

1. **Mode shape magnitudes**: Will be scaled differently
2. **Modal mass matrix**: Will be diagonal (identity) instead of arbitrary
3. **Numerical stability**: Improved for subsequent modal analysis

### What Stays the Same

1. **Mode shape shapes**: The relative shapes (ratios between floors) are preserved
2. **Physical interpretation**: Still represents the same vibration modes
3. **Computation method**: Still uses RMS ratios and correlation from acceleration data

## Testing

Run the test script to see the effect:

```bash
python test_mass_normalization.py
```

This will:
1. Compute original mode shapes
2. Show original Φᵀ M Φ (likely not diagonal)
3. Normalize each mode
4. Show normalized Φᵀ M Φ (diagonal, but may have off-diagonal terms)
5. Orthogonalize all modes
6. Show final Φᵀ M Φ (should be identity matrix)
7. Compare original vs orthogonalized mode shapes

## Notes

- **Mass matrix must be provided**: The analyzer needs the mass matrix to perform normalization
- **Orthogonalization may change mode shapes**: If modes are not naturally orthogonal (due to measurement noise), orthogonalization will modify them slightly
- **Order matters**: Modes are processed in the order provided. If numerical issues occur, you may need to reorder modes
- **Tolerance**: The verification uses a default tolerance of 1e-3. Adjust if needed for your application




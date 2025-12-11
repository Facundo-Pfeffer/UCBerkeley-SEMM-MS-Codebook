# Robust Decay Detection - Implementation Summary

## Changes Made

### 1. Adaptive Decay Detection (`_detect_decay_start`)

**Previous Method:**
- Fixed search start at 50% of signal
- Fixed fallback at 80% of signal
- Simple monotonic decrease check

**New Robust Method:**
- **Adaptive search start**: 
  - Short signals (<40s): starts at 30% of signal
  - Long signals (≥40s): starts at 40% of signal
  - Ensures at least 2 periods after maximum amplitude
  
- **Multi-criteria scoring**:
  - Monotonic decrease in moving average (weight: 2.0)
  - Monotonic decrease in envelope (weight: 1.5)
  - Perfect monotonic window (bonus: +3.0)
  - Amplitude ratio check (0.1 < ratio < 0.9, weight: 1.0)
  
- **Adaptive fallback**:
  - Uses 75% threshold (instead of 80%)
  - Adds 3 periods after max (instead of 2)
  - Ensures at least 5 periods remain for analysis

### 2. Adaptive Peak Extraction (`_extract_peaks`)

**Previous Method:**
- Fixed prominence threshold: 10% of max in decay portion

**New Method:**
- **Adaptive prominence based on available cycles**:
  - < 5 cycles available: 5% prominence (50% of default)
  - 5-10 cycles available: 7% prominence (70% of default)
  - ≥ 10 cycles available: 10% prominence (default)
  
- **Rationale**: Short decay windows have lower amplitudes, so lower prominence threshold helps detect more peaks

## Why Exponential Fits Appear Horizontal/Shifted

### Position Determination:
- **A0**: Initial amplitude = `envelope[0]` at decay start
- **t0**: Decay start time = `decay_time[0]`
- **Fit formula**: `A(t) = A0 * exp(-ζ * ωn * (t - t0))`

### Why They Look Horizontal:
1. **Low damping**: If ζ is very small (e.g., 0.01-0.02), the decay rate `ζ * ωn` is small, making decay very slow
2. **Short time window**: Limited decay duration means little visible change
3. **Log scale**: On logarithmic y-axis, slow exponential decay can appear nearly flat
4. **Visual effect**: The fit is mathematically correct, but visually appears horizontal due to the combination of low damping and short window

### Example Calculation:
For Mode 1 with ζ = 0.02, fn = 3.5 Hz:
- ωn = 2π * 3.5 = 22.0 rad/s
- Decay rate = 0.02 * 22.0 = 0.44 rad/s
- Over 15 seconds: exp(-0.44 * 15) = exp(-6.6) ≈ 0.0014
- This is a 99.86% reduction, but if starting from 0.15, it goes to 0.0002
- On log scale from 0.15 to 0.0002, this should show clear decay

**If fits appear horizontal, it likely means:**
- The damping ratio is very small (< 0.01)
- OR the decay window is very short (< 5 seconds)
- OR there's an issue with the natural frequency used

### 3. Unified Decay Start Detection

**Previous Method:**
- Each floor detected decay start independently
- Different floors could have different decay start times
- Caused envelopes to appear "distant in time" on plots

**New Method:**
- **Single decay detection**: Uses RMS of all modal accelerations to detect decay start once
- **Shared decay start**: All floors use the same decay start time
- **Physical consistency**: All floors are part of the same structure in the same mode, so decay should start simultaneously

**Why This Matters:**
- In Mode 3, red and blue envelopes appeared distant in time because:
  - Each floor's modal acceleration has different amplitude characteristics (different mode shape components)
  - The scoring algorithm found different "best" decay start times for each floor
  - This caused envelopes to start at different times (e.g., 17s vs 20s)
- With unified detection, all envelopes start at the same time, making comparisons meaningful

## Testing Recommendations

Before finalizing, test on all three modes:

1. **Mode 1 (60s, 3.5 Hz)**:
   - Should detect decay earlier (around 30-40s instead of 43s)
   - Should have more peaks detected
   - Exponential fit should show clear decay

2. **Mode 2 (32s, 7.20 Hz)**:
   - Should detect decay earlier (around 20-25s instead of 30.5s)
   - Should detect more peaks with adaptive prominence
   - Should have sufficient data for analysis

3. **Mode 3 (32s, 10.50 Hz)**:
   - Similar improvements as Mode 2
   - Higher frequency means more cycles in same time window

## Expected Improvements

- **Earlier decay detection**: 10-15 seconds earlier for short signals
- **More peaks detected**: 2-3x more peaks for short decay windows
- **Better analysis success**: All floors should have valid damping estimates
- **More consistent results**: Less variation between floors


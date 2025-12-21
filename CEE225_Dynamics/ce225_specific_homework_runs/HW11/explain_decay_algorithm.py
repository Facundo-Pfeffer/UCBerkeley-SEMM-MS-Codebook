#!/usr/bin/env python3
"""Explanation of decay detection algorithm and signal duration effects."""

print("=" * 70)
print("Decay Detection Algorithm Explanation")
print("=" * 70)

print("\n1. DECAY DETECTION ALGORITHM (_detect_decay_start):")
print("-" * 70)
print("""
Step 1: Compute moving average envelope
  - Window size = 0.5 * period (e.g., for 7.20 Hz: period = 0.139s, window ≈ 0.07s)
  - Creates a smoothed amplitude envelope of |acceleration|

Step 2: Compute derivative (difference) of moving average
  - moving_avg_diff[i] = moving_avg[i+1] - moving_avg[i]
  - Negative values indicate decreasing amplitude

Step 3: Search for decay start (from midpoint)
  - Starts searching at: decay_candidate_start = len(time) // 2
    * Mode 1 (60s): starts at 30s
    * Mode 2/3 (32s): starts at 16s
  - Looks for a window where ALL differences are negative:
    * if np.all(window_diff < 0): return i
    * This means the envelope is monotonically decreasing

Step 4: Fallback if no monotonic decrease found
  - Finds max amplitude index: max_idx = np.argmax(|acc|)
  - Checks if max_idx < len(time) * 0.8 (80% of signal)
    * Mode 1 (60s): max_idx must be < 48s
    * Mode 2/3 (32s): max_idx must be < 25.6s
  - If true: decay_start = max_idx + 2*period
    * Assumes decay starts 2 periods after peak
    * For Mode 2 (7.20 Hz): +0.278s after peak
""")

print("\n2. PEAK EXTRACTION ALGORITHM (_extract_peaks):")
print("-" * 70)
print("""
Step 1: Compute prominence threshold
  - min_prominence = max(|acc|) * 0.1 (10% of max in DECAY PORTION)
  - This is RELATIVE to the decay portion's maximum, not full signal

Step 2: Compute minimum distance between peaks
  - min_distance = 0.7 * period / dt (70% of period in samples)
  - Prevents detecting multiple peaks in same cycle

Step 3: Find peaks
  - Finds positive peaks: signal.find_peaks(acc, prominence, distance)
  - Finds negative peaks: signal.find_peaks(-acc, prominence, distance)
  - Combines and sorts by time

Step 4: Validation
  - Requires at least 2 peaks to compute log decrement
  - Requires min_cycles + 1 peaks for analysis (default: 4 peaks)
""")

print("\n3. SIGNAL DURATION EFFECTS:")
print("-" * 70)
print("""
Mode 1 (60 seconds):
  - Decay search starts at: 30s (50% of signal)
  - Has 30 seconds to find decay
  - Fallback check: max_idx < 48s (80% of signal)
  - If forced vibration is long, decay might start at 30-40s
  - Remaining decay data: 20-30 seconds (plenty of cycles)

Mode 2/3 (32 seconds):
  - Decay search starts at: 16s (50% of signal)
  - Has only 16 seconds to find decay
  - Fallback check: max_idx < 25.6s (80% of signal)
  - If forced vibration is long, decay might start at 16-25s
  - Remaining decay data: 7-16 seconds (fewer cycles)

PROBLEM IDENTIFIED:
  - The algorithm uses FIXED RATIOS (50%, 80%) that don't scale with signal duration
  - For shorter signals (32s), if forced vibration lasts until ~20s, decay detection
    might start very late (e.g., 30.5s for Mode 2), leaving only ~2s of decay data
  - The fallback (max_idx + 2*period) assumes decay starts shortly after peak,
    but this may not be true if forced vibration continues after the peak
""")

print("\n4. WHY MODE 2 SHOWS LIMITED DATA:")
print("-" * 70)
print("""
Scenario for Mode 2 (32s signal, decay detected at 30.5s):
  
  a) Decay detection:
     - Forced vibration might continue until ~28-30s
     - Algorithm searches from 16s but doesn't find monotonic decrease
     - Falls back to max_idx + 2*period
     - If max_idx is early (e.g., at 5s), decay_start = 5 + 0.278 = 5.3s (too early)
     - If max_idx is late (e.g., at 28s), decay_start = 28 + 0.278 = 28.3s (reasonable)
     - BUT if forced vibration continues, the actual decay might start later
  
  b) Peak extraction in decay portion:
     - Only ~2.5 seconds of decay data (30.5s to 33s)
     - At 7.20 Hz: ~18 cycles available
     - BUT: prominence threshold = 10% of max in decay portion
     - If decay amplitude is already low, 10% might be too high
     - Result: Few peaks detected, insufficient for analysis
  
  c) Minimum cycles requirement:
     - Default: min_cycles = 3
     - Needs at least 4 peaks (min_cycles + 1)
     - If only 2-3 peaks found, analysis fails
""")

print("\n5. RECOMMENDED FIXES:")
print("-" * 70)
print("""
1. Make decay search start adaptive:
   - Instead of fixed 50%, use earlier start (e.g., 30% or 40%)
   - For shorter signals, start even earlier

2. Improve fallback logic:
   - Don't assume decay starts 2 periods after peak
   - Use envelope analysis to find actual transition
   - Consider signal duration when setting fallback

3. Adaptive prominence threshold:
   - For short decay portions, reduce prominence threshold
   - Or use absolute threshold based on noise level

4. Account for signal duration:
   - Adjust search parameters based on total signal length
   - For 32s signals, be more aggressive in finding decay start
""")












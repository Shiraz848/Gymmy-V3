"""
CalibrationLogic.py
Pure logic module for ROM (Range of Motion) assessment calculations.
No ZED/Tkinter/hardware dependencies - only mathematical processing.

NOTE: This is for Patient ROM Assessment, NOT for the T-Pose system calibration.

UPDATED: Now returns both peaks (up position) and valleys (down position)
         with avg ± std for personalized thresholds.
"""

import numpy as np
from typing import List, Dict, Tuple


def find_local_peaks(data: List[float], min_distance: int = 5) -> Tuple[List[float], List[int]]:
    """
    Find local maxima (peaks) in the data.
    Used to identify the UP position of each repetition.
    
    Args:
        data: List of angle values over time
        min_distance: Minimum number of samples between peaks
    
    Returns:
        Tuple of (peak_values, peak_indices)
    """
    if len(data) < 3:
        return list(data), list(range(len(data)))
    
    peaks = []
    indices = []
    arr = np.array(data)
    
    i = 1
    while i < len(arr) - 1:
        # Check if current point is a local maximum
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            # Also check it's significantly higher than surrounding values
            window_start = max(0, i - min_distance)
            window_end = min(len(arr), i + min_distance + 1)
            window = arr[window_start:window_end]
            
            if arr[i] >= np.max(window) * 0.95:  # Within 5% of window max
                peaks.append(float(arr[i]))
                indices.append(i)
                i += min_distance  # Skip ahead to avoid nearby peaks
                continue
        i += 1
    
    return peaks, indices


def find_local_valleys(data: List[float], min_distance: int = 5) -> Tuple[List[float], List[int]]:
    """
    Find local minima (valleys) in the data.
    Used to identify the DOWN position of each repetition.
    
    Args:
        data: List of angle values over time
        min_distance: Minimum number of samples between valleys
    
    Returns:
        Tuple of (valley_values, valley_indices)
    """
    if len(data) < 3:
        return list(data), list(range(len(data)))
    
    valleys = []
    indices = []
    arr = np.array(data)
    
    i = 1
    while i < len(arr) - 1:
        # Check if current point is a local minimum
        if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
            # Also check it's significantly lower than surrounding values
            window_start = max(0, i - min_distance)
            window_end = min(len(arr), i + min_distance + 1)
            window = arr[window_start:window_end]
            
            if arr[i] <= np.min(window) * 1.05:  # Within 5% of window min
                valleys.append(float(arr[i]))
                indices.append(i)
                i += min_distance  # Skip ahead to avoid nearby valleys
                continue
        i += 1
    
    return valleys, indices


def filter_outliers(values: List[float], threshold: float = 0.15, is_peaks: bool = True) -> List[float]:
    """
    Filter outliers from peak/valley values.
    
    For peaks: If highest is > threshold% larger than 2nd highest, discard it.
    For valleys: If lowest is > threshold% smaller than 2nd lowest, discard it.
    
    Args:
        values: List of values (sorted appropriately)
        threshold: Percentage threshold for outlier detection (default 15%)
        is_peaks: True if filtering peaks (max values), False for valleys (min values)
    
    Returns:
        Filtered list of values
    """
    if len(values) < 2:
        return values
    
    filtered = values.copy()
    
    if is_peaks:
        # Sort descending for peaks
        filtered = sorted(filtered, reverse=True)
        if len(filtered) >= 2:
            highest = filtered[0]
            second_highest = filtered[1]
            if highest > second_highest * (1 + threshold):
                filtered = filtered[1:]  # Remove the outlier
    else:
        # Sort ascending for valleys
        filtered = sorted(filtered)
        if len(filtered) >= 2:
            lowest = filtered[0]
            second_lowest = filtered[1]
            if lowest < second_lowest * (1 - threshold):
                filtered = filtered[1:]  # Remove the outlier
    
    return filtered


def calculate_rom_thresholds(angle_history: List[float], 
                             num_to_use: int = 5,
                             default_up: float = 140.0,
                             default_down: float = 40.0) -> Dict:
    """
    Calculate ROM thresholds for both UP and DOWN positions.
    
    For each position (up/down):
    - avg: Average of best N values
    - std: Standard deviation of those values
    - ub: Upper bound = avg + std (above this = too much)
    - lb: Lower bound = avg - std (below this = not enough)
    
    Args:
        angle_history: List of angle values recorded during ROM assessment
        num_to_use: Number of best peaks/valleys to use (default 5)
        default_up: Default value for UP position if insufficient data
        default_down: Default value for DOWN position if insufficient data
    
    Returns:
        Dict with structure:
        {
            'up': {'avg': float, 'std': float, 'ub': float, 'lb': float},
            'down': {'avg': float, 'std': float, 'ub': float, 'lb': float},
            'peaks_used': int,
            'valleys_used': int
        }
    """
    result = {
        'up': {'avg': default_up, 'std': 0, 'ub': default_up, 'lb': default_up},
        'down': {'avg': default_down, 'std': 0, 'ub': default_down, 'lb': default_down},
        'peaks_used': 0,
        'valleys_used': 0
    }
    
    # Handle empty or insufficient data
    if not angle_history or len(angle_history) < 10:
        print(f"[ROM] Insufficient data ({len(angle_history) if angle_history else 0} samples). Using defaults.")
        return result
    
    # Remove None and NaN values
    clean_data = [x for x in angle_history if x is not None and not np.isnan(x)]
    
    if len(clean_data) < 10:
        print(f"[ROM] Insufficient valid data after cleaning ({len(clean_data)} samples). Using defaults.")
        return result
    
    # ==================== FIND PEAKS (UP POSITION) ====================
    peaks, peak_indices = find_local_peaks(clean_data)
    
    if len(peaks) < 2:
        # Fallback: use top values directly
        peaks = sorted(clean_data, reverse=True)[:num_to_use * 2]
    
    # Filter outliers and take best N
    peaks_filtered = filter_outliers(peaks, is_peaks=True)
    top_peaks = sorted(peaks_filtered, reverse=True)[:num_to_use]
    
    if top_peaks:
        up_avg = float(np.mean(top_peaks))
        up_std = float(np.std(top_peaks)) if len(top_peaks) > 1 else 0
        
        # Simple fixed margin: average ± 10 degrees
        # This provides consistent error tolerance for all patients
        FIXED_MARGIN = 10.0  # degrees above/below average
        
        result['up'] = {
            'avg': up_avg,
            'std': up_std,
            'ub': up_avg + FIXED_MARGIN,  # Upper bound = avg + 10°
            'lb': up_avg - FIXED_MARGIN   # Lower bound = avg - 10°
        }
        result['peaks_used'] = len(top_peaks)
        
        print(f"[ROM] UP position (peaks):")
        print(f"      Values used: {[f'{p:.1f}' for p in top_peaks]}")
        print(f"      avg = {up_avg:.2f}°, std = {up_std:.2f}°")
        print(f"      Valid range: [{result['up']['lb']:.2f}°, {result['up']['ub']:.2f}°] (avg ± {FIXED_MARGIN}°)")
    
    # ==================== FIND VALLEYS (DOWN POSITION) ====================
    valleys, valley_indices = find_local_valleys(clean_data)
    
    if len(valleys) < 2:
        # Fallback: use bottom values directly
        valleys = sorted(clean_data)[:num_to_use * 2]
    
    # Filter outliers and take best N (lowest values)
    valleys_filtered = filter_outliers(valleys, is_peaks=False)
    bottom_valleys = sorted(valleys_filtered)[:num_to_use]
    
    if bottom_valleys:
        down_avg = float(np.mean(bottom_valleys))
        down_std = float(np.std(bottom_valleys)) if len(bottom_valleys) > 1 else 0
        
        # Simple fixed margin: average ± 10 degrees
        # This provides consistent error tolerance for all patients
        FIXED_MARGIN = 10.0  # degrees above/below average
        
        result['down'] = {
            'avg': down_avg,
            'std': down_std,
            'ub': down_avg + FIXED_MARGIN,  # Upper bound = avg + 10°
            'lb': down_avg - FIXED_MARGIN   # Lower bound = avg - 10°
        }
        result['valleys_used'] = len(bottom_valleys)
        
        # Ensure lower bound doesn't go below 0
        if result['down']['lb'] < 0:
            result['down']['lb'] = 0.0
        
        print(f"[ROM] DOWN position (valleys):")
        print(f"      Values used: {[f'{v:.1f}' for v in bottom_valleys]}")
        print(f"      avg = {down_avg:.2f}°, std = {down_std:.2f}°")
        print(f"      Valid range: [{result['down']['lb']:.2f}°, {result['down']['ub']:.2f}°] (avg ± {FIXED_MARGIN}°)")
    
    # ==================== OVERLAP CHECK (DISABLED) ====================
    # NOTE: Automatic overlap adjustment has been disabled.
    # For patients with limited ROM, overlapping ranges may be valid.
    # The flag-based state machine in Camera.py handles transitions correctly.
    # 
    # If you want to re-enable overlap adjustment, uncomment the section below:
    # -------------------------------------------------------------------
    # MIN_GAP = 10.0  # Minimum degrees between DOWN upper bound and UP lower bound
    # 
    # if 'up' in result and 'down' in result:
    #     up_lb = result['up']['lb']
    #     down_ub = result['down']['ub']
    #     
    #     if down_ub >= up_lb - MIN_GAP:
    #         # Ranges overlap or are too close - need to create gap
    #         midpoint = (result['up']['avg'] + result['down']['avg']) / 2
    #         
    #         # Adjust bounds to create a gap at the midpoint
    #         new_down_ub = midpoint - (MIN_GAP / 2)
    #         new_up_lb = midpoint + (MIN_GAP / 2)
    #         
    #         print(f"[ROM] ⚠️  Ranges overlapped! Adjusting to create {MIN_GAP}° gap:")
    #         print(f"      DOWN ub: {down_ub:.1f}° -> {new_down_ub:.1f}°")
    #         print(f"      UP lb: {up_lb:.1f}° -> {new_up_lb:.1f}°")
    #         
    #         result['down']['ub'] = new_down_ub
    #         result['up']['lb'] = new_up_lb
    #         
    #         # Also ensure lb doesn't go below 0
    #         if result['down']['lb'] < 0:
    #             result['down']['lb'] = 0.0
    # -------------------------------------------------------------------
    
    # Check for overlap and just warn (don't adjust)
    if 'up' in result and 'down' in result:
        up_lb = result['up']['lb']
        down_ub = result['down']['ub']
        if down_ub >= up_lb:
            print(f"[ROM] ℹ️  Note: Ranges have some overlap (DOWN_UB={down_ub:.1f}° >= UP_LB={up_lb:.1f}°)")
            print(f"          This is allowed for limited ROM patients.")
    
    return result


# Backward compatibility wrapper
def calculate_max_rom(angle_history: List[float], 
                      num_peaks_to_use: int = 5,
                      default_value: float = 90.0) -> Dict:
    """
    Backward compatible wrapper that returns the new dict format.
    
    Returns:
        Dict with 'up' and 'down' positions, each containing avg, std, ub, lb
    """
    return calculate_rom_thresholds(
        angle_history=angle_history,
        num_to_use=num_peaks_to_use,
        default_up=default_value,
        default_down=default_value * 0.3  # Rough estimate for down position
    )


# ==================== TESTING ====================
if __name__ == "__main__":
    import random
    
    print("="*60)
    print("Testing CalibrationLogic with simulated exercise data")
    print("="*60)
    
    # Simulate ROM assessment data (10 reps)
    # Up position: around 140-150 degrees
    # Down position: around 30-40 degrees
    test_data = []
    for rep in range(10):
        # Go up (from ~35 to ~145)
        for i in range(20):
            test_data.append(35 + i * 5.5 + random.uniform(-3, 3))
        
        # Peak (up position)
        peak = 145 + random.uniform(-8, 8)
        if rep == 3:  # Simulate one outlier
            peak = 175
        test_data.append(peak)
        
        # Go down (from ~145 to ~35)
        for i in range(20):
            test_data.append(145 - i * 5.5 + random.uniform(-3, 3))
        
        # Valley (down position)
        valley = 35 + random.uniform(-5, 5)
        if rep == 7:  # Simulate one outlier
            valley = 10
        test_data.append(valley)
    
    print(f"\nSimulated {len(test_data)} data points over 10 repetitions")
    print(f"Expected UP: ~145° ± 8°")
    print(f"Expected DOWN: ~35° ± 5°")
    print()
    
    result = calculate_rom_thresholds(test_data)
    
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"\nUP position:")
    print(f"  Average: {result['up']['avg']:.2f}°")
    print(f"  Std Dev: {result['up']['std']:.2f}°")
    print(f"  Valid range: [{result['up']['lb']:.2f}°, {result['up']['ub']:.2f}°]")
    print(f"  Peaks used: {result['peaks_used']}")
    
    print(f"\nDOWN position:")
    print(f"  Average: {result['down']['avg']:.2f}°")
    print(f"  Std Dev: {result['down']['std']:.2f}°")
    print(f"  Valid range: [{result['down']['lb']:.2f}°, {result['down']['ub']:.2f}°]")
    print(f"  Valleys used: {result['valleys_used']}")
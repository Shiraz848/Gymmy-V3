"""
ExerciseConfig.py
Comprehensive exercise configuration for ROM assessment and training.

This file maps ALL exercises to their angle measurements.
Each exercise has its own thresholds - NOT shared by "joint type".

Structure:
- EXERCISE_CONFIG: Maps each exercise to its angle indices and defaults
- Helper functions to get patient's selected exercises from Excel
"""

import pandas as pd
import Settings as s

# =============================================================================
# EXERCISE CONFIGURATION
# =============================================================================
# Each exercise maps to:
#   - angles: List of dicts with angle_index, side (right/left), and default thresholds
#   - The thresholds will be personalized PER EXERCISE, not per joint type
#
# angle_index corresponds to the position in s.last_entry_angles:
#   0 = first angle (usually right primary)
#   1 = second angle (usually left primary)
#   2 = third angle (often secondary measurement)
#   3 = fourth angle
# =============================================================================

EXERCISE_CONFIG = {
    # ============ BALL EXERCISES ============
    "ball_bend_elbows": {
        "category": "ball",
        "description": "כיפוף מרפקים עם כדור",
        "angles": [
            {"index": 0, "side": "right", "default_up": 140, "default_down": 40},
            {"index": 1, "side": "left", "default_up": 140, "default_down": 40},
        ],
        "video_file": "ball_bend_elbows.mp4",
    },
    
    "ball_raise_arms_above_head": {
        "category": "ball",
        "description": "הרמת ידיים מעל הראש עם כדור",
        "angles": [
            {"index": 0, "side": "right", "default_up": 160, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 160, "default_down": 20},
        ],
        "video_file": "ball_raise_arms_above_head.mp4",
    },
    
    "ball_switch": {
        "category": "ball",
        "description": "החלפות עם כדור",
        "angles": [
            {"index": 0, "side": "right", "default_up": 90, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 90, "default_down": 20},
            {"index": 2, "side": "right", "default_up": 130, "default_down": 70},
            {"index": 3, "side": "left", "default_up": 130, "default_down": 70},
        ],
        "video_file": "ball_switch.mp4",
    },
    
    "ball_open_arms_and_forward": {
        "category": "ball",
        "description": "פתיחת ידיים וקדימה עם כדור",
        "angles": [
            {"index": 0, "side": "right", "default_up": 140, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 140, "default_down": 20},
        ],
        "video_file": "ball_open_arms_and_forward.mp4",
    },
    
    # NOTE: ball_open_arms_above_head is commented out in Gymmy.py, so we disable it here
    # to prevent AttributeError during ROM assessment.
    # Uncomment this and the Gymmy method when the exercise is ready.
    # "ball_open_arms_above_head": {
    #     "category": "ball",
    #     "description": "פתיחת ידיים מעל הראש עם כדור",
    #     "angles": [
    #         {"index": 0, "side": "right", "default_up": 160, "default_down": 30},
    #         {"index": 1, "side": "left", "default_up": 160, "default_down": 30},
    #     ],
    #     "video_file": "ball_open_arms_above_head.mp4",
    # },
    
    # ============ BAND EXERCISES ============
    "band_open_arms": {
        "category": "band",
        "description": "פתיחת ידיים עם גומיה",
        "angles": [
            {"index": 0, "side": "right", "default_up": 120, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 120, "default_down": 20},
        ],
        "video_file": "band_open_arms.mp4",
    },
    
    "band_open_arms_and_up": {
        "category": "band",
        "description": "פתיחת ידיים והרמה עם גומיה",
        "angles": [
            {"index": 0, "side": "right", "default_up": 140, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 140, "default_down": 20},
        ],
        "video_file": "band_open_arms_and_up.mp4",
    },
    
    "band_up_and_lean": {
        "category": "band",
        "description": "הרמה והטיה עם גומיה",
        "angles": [
            {"index": 0, "side": "right", "default_up": 160, "default_down": 90},
            {"index": 1, "side": "left", "default_up": 160, "default_down": 90},
            {"index": 2, "side": "right", "default_up": 140, "default_down": 100},
            {"index": 3, "side": "left", "default_up": 140, "default_down": 100},
        ],
        "video_file": "band_up_and_lean.mp4",
    },
    
    "band_straighten_left_arm_elbows_bend_to_sides": {
        "category": "band",
        "description": "יישור יד שמאל וכיפוף מרפקים לצדדים",
        "angles": [
            {"index": 0, "side": "right", "default_up": 140, "default_down": 40},
            {"index": 1, "side": "left", "default_up": 140, "default_down": 40},
        ],
        "video_file": "band_straighten_left_arm_elbows_bend_to_sides.mp4",
    },
    
    "band_straighten_right_arm_elbows_bend_to_sides": {
        "category": "band",
        "description": "יישור יד ימין וכיפוף מרפקים לצדדים",
        "angles": [
            {"index": 0, "side": "right", "default_up": 140, "default_down": 40},
            {"index": 1, "side": "left", "default_up": 140, "default_down": 40},
        ],
        "video_file": "band_straighten_right_arm_elbows_bend_to_sides.mp4",
    },
    
    # ============ STICK EXERCISES ============
    "stick_bend_elbows": {
        "category": "stick",
        "description": "כיפוף מרפקים עם מקל",
        "angles": [
            {"index": 0, "side": "right", "default_up": 140, "default_down": 30},
            {"index": 1, "side": "left", "default_up": 140, "default_down": 30},
        ],
        "video_file": "stick_bend_elbows.mp4",
    },
    
    "stick_bend_elbows_and_up": {
        "category": "stick",
        "description": "כיפוף מרפקים והרמה עם מקל",
        "angles": [
            {"index": 0, "side": "right", "default_up": 160, "default_down": 40},
            {"index": 1, "side": "left", "default_up": 160, "default_down": 40},
        ],
        "video_file": "stick_bend_elbows_and_up.mp4",
    },
    
    "stick_raise_arms_above_head": {
        "category": "stick",
        "description": "הרמת ידיים מעל הראש עם מקל",
        "angles": [
            {"index": 0, "side": "right", "default_up": 170, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 170, "default_down": 20},
        ],
        "video_file": "stick_raise_arms_above_head.mp4",
    },
    
    "stick_switch": {
        "category": "stick",
        "description": "החלפות עם מקל",
        "angles": [
            {"index": 0, "side": "right", "default_up": 90, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 90, "default_down": 20},
            {"index": 2, "side": "right", "default_up": 130, "default_down": 70},
            {"index": 3, "side": "left", "default_up": 130, "default_down": 70},
        ],
        "video_file": "stick_switch.mp4",
    },
    
    "stick_up_and_lean": {
        "category": "stick",
        "description": "הרמה והטיה עם מקל",
        "angles": [
            {"index": 0, "side": "right", "default_up": 160, "default_down": 90},
            {"index": 1, "side": "left", "default_up": 160, "default_down": 90},
            {"index": 2, "side": "right", "default_up": 140, "default_down": 100},
            {"index": 3, "side": "left", "default_up": 140, "default_down": 100},
        ],
        "video_file": "stick_up_and_lean.mp4",
    },
    
    # ============ WEIGHTS EXERCISES ============
    "weights_open_arms_and_forward": {
        "category": "weights",
        "description": "פתיחת ידיים וקדימה עם משקולות",
        "angles": [
            {"index": 0, "side": "right", "default_up": 130, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 130, "default_down": 20},
        ],
        "video_file": "weights_open_arms_and_forward.mp4",
    },
    
    "weights_abduction": {
        "category": "weights",
        "description": "הרחקה עם משקולות",
        "angles": [
            {"index": 0, "side": "right", "default_up": 110, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 110, "default_down": 20},
        ],
        "video_file": "weights_abduction.mp4",
    },
    
    # ============ NO-TOOL EXERCISES ============
    "notool_hands_behind_and_lean": {
        "category": "notool",
        "description": "ידיים מאחור והטיה",
        "angles": [
            {"index": 0, "side": "right", "default_up": 160, "default_down": 90},
            {"index": 1, "side": "left", "default_up": 160, "default_down": 90},
            {"index": 2, "side": "right", "default_up": 140, "default_down": 100},
            {"index": 3, "side": "left", "default_up": 140, "default_down": 100},
        ],
        "video_file": "notool_hands_behind_and_lean.mp4",
    },
    
    "notool_right_hand_up_and_bend": {
        "category": "notool",
        "description": "יד ימין למעלה וכיפוף",
        "angles": [
            {"index": 0, "side": "right", "default_up": 160, "default_down": 90},
            {"index": 1, "side": "left", "default_up": 160, "default_down": 90},
        ],
        "video_file": "notool_right_hand_up_and_bend.mp4",
    },
    
    "notool_left_hand_up_and_bend": {
        "category": "notool",
        "description": "יד שמאל למעלה וכיפוף",
        "angles": [
            {"index": 0, "side": "right", "default_up": 160, "default_down": 90},
            {"index": 1, "side": "left", "default_up": 160, "default_down": 90},
        ],
        "video_file": "notool_left_hand_up_and_bend.mp4",
    },
    
    "notool_raising_hands_diagonally": {
        "category": "notool",
        "description": "הרמת ידיים באלכסון",
        "angles": [
            {"index": 0, "side": "right", "default_up": 150, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 150, "default_down": 20},
        ],
        "video_file": "notool_raising_hands_diagonally.mp4",
    },
    
    "notool_right_bend_left_up_from_side": {
        "category": "notool",
        "description": "כיפוף ימין והרמה שמאל מהצד",
        "angles": [
            {"index": 0, "side": "right", "default_up": 90, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 140, "default_down": 20},
        ],
        "video_file": "notool_right_bend_left_up_from_side.mp4",
    },
    
    "notool_left_bend_right_up_from_side": {
        "category": "notool",
        "description": "כיפוף שמאל והרמה ימין מהצד",
        "angles": [
            {"index": 0, "side": "right", "default_up": 140, "default_down": 20},
            {"index": 1, "side": "left", "default_up": 90, "default_down": 20},
        ],
        "video_file": "notool_left_bend_right_up_from_side.mp4",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_exercises():
    """Return list of all exercise names."""
    return list(EXERCISE_CONFIG.keys())


def get_exercises_by_category(category):
    """Return list of exercises in a specific category (ball, band, stick, weights, notool)."""
    return [ex for ex, config in EXERCISE_CONFIG.items() if config["category"] == category]


def get_exercise_config(exercise_name):
    """Get full configuration for an exercise."""
    return EXERCISE_CONFIG.get(exercise_name, None)


def get_patient_selected_exercises(patient_id=None):
    """
    Get list of exercises that the physiotherapist selected for a patient.
    Reads from Patients.xlsx, sheet 'patients_exercises'.
    
    Args:
        patient_id: Patient ID. If None, uses s.chosen_patient_ID
    
    Returns:
        List of exercise names that are enabled (True) for this patient
    """
    if patient_id is None:
        patient_id = s.chosen_patient_ID
    
    try:
        excel_file_path = "Patients.xlsx"
        df = pd.read_excel(excel_file_path, sheet_name="patients_exercises")
        
        # Convert first column (ID) to string
        df.iloc[:, 0] = df.iloc[:, 0].astype(str)
        patient_id_str = str(patient_id).strip()
        
        # Find patient's row
        patient_row = df[df.iloc[:, 0] == patient_id_str]
        
        if patient_row.empty:
            print(f"[ExerciseConfig] Patient {patient_id} not found in patients_exercises")
            return []
        
        # Get all exercises where value is True
        selected_exercises = []
        for exercise_name in EXERCISE_CONFIG.keys():
            if exercise_name in df.columns:
                value = patient_row.iloc[0][exercise_name]
                if value == True or value == 1 or str(value).lower() == 'true':
                    selected_exercises.append(exercise_name)
        
        print(f"[ExerciseConfig] Patient {patient_id} selected exercises: {selected_exercises}")
        return selected_exercises
        
    except Exception as e:
        print(f"[ExerciseConfig] Error reading patient exercises: {e}")
        return []


def get_rom_threshold_key(exercise_name, side, position, bound):
    """
    Generate a ROM threshold key for a specific exercise.
    
    Args:
        exercise_name: e.g., "ball_bend_elbows"
        side: "right" or "left"
        position: "up" or "down"
        bound: "avg", "std", "ub", "lb"
    
    Returns:
        Key like "ball_bend_elbows_right_up_avg"
    """
    return f"{exercise_name}_{side}_{position}_{bound}"


def get_default_threshold(exercise_name, side, position):
    """
    Get default threshold for an exercise.
    
    Args:
        exercise_name: e.g., "ball_bend_elbows"
        side: "right" or "left"
        position: "up" or "down"
    
    Returns:
        Default angle value
    """
    config = EXERCISE_CONFIG.get(exercise_name)
    if not config:
        return 90  # fallback
    
    for angle_config in config["angles"]:
        if angle_config["side"] == side:
            if position == "up":
                return angle_config["default_up"]
            else:
                return angle_config["default_down"]
    
    return 90  # fallback


def get_exercise_threshold(exercise_name, side, position, bound="avg"):
    """
    Get personalized threshold for an exercise from patient_rom_limits.
    Falls back to default if not found.
    
    Args:
        exercise_name: e.g., "ball_bend_elbows"
        side: "right" or "left"
        position: "up" or "down"
        bound: "avg", "ub", "lb" (default: "avg")
    
    Returns:
        Threshold angle value
    """
    if not hasattr(s, 'patient_rom_limits') or not s.patient_rom_limits:
        return get_default_threshold(exercise_name, side, position)
    
    key = get_rom_threshold_key(exercise_name, side, position, bound)
    
    if key in s.patient_rom_limits:
        value = s.patient_rom_limits[key]
        if value is not None and value > 0:
            return value
    
    return get_default_threshold(exercise_name, side, position)


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ExerciseConfig Test")
    print("=" * 60)
    
    print(f"\nTotal exercises: {len(EXERCISE_CONFIG)}")
    
    for category in ["ball", "band", "stick", "weights", "notool"]:
        exercises = get_exercises_by_category(category)
        print(f"\n{category.upper()} exercises ({len(exercises)}):")
        for ex in exercises:
            config = get_exercise_config(ex)
            print(f"  - {ex}: {len(config['angles'])} angles")


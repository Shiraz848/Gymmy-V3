"""
Exercise_ROM_Mapping.py
Maps all exercises to their relevant ROM joint measurements.
Used to determine which patient ROM limits apply to each exercise.
"""

# Maps each exercise to the ROM joints it uses
EXERCISE_TO_ROM_JOINTS = {
    # ============ BALL EXERCISES ============
    "ball_bend_elbows": {
        "primary_movement": "elbow_flexion",
        "joints": ["elbow_flexion_right", "elbow_flexion_left"],
        "is_rom_test": True
    },
    "ball_raise_arms_above_head": {
        "primary_movement": "shoulder_flexion",
        "joints": ["shoulder_flexion_right", "shoulder_flexion_left"],
        "is_rom_test": True
    },
    "ball_switch": {
        "primary_movement": "torso_rotation",
        "joints": ["torso_rotation_right", "torso_rotation_left"],
        "is_rom_test": False
    },
    "ball_open_arms_and_forward": {
        "primary_movement": "shoulder_flexion",
        "joints": ["shoulder_flexion_right", "shoulder_flexion_left"],
        "is_rom_test": False
    },
    "ball_open_arms_above_head": {
        "primary_movement": "shoulder_abduction",
        "joints": ["shoulder_abduction_right", "shoulder_abduction_left"],
        "is_rom_test": False
    },
    
    # ============ BAND EXERCISES ============
    "band_open_arms": {
        "primary_movement": "shoulder_abduction",
        "joints": ["shoulder_abduction_right", "shoulder_abduction_left"],
        "is_rom_test": True
    },
    "band_open_arms_and_up": {
        "primary_movement": "shoulder_abduction",
        "joints": ["shoulder_abduction_right", "shoulder_abduction_left"],
        "is_rom_test": False
    },
    "band_up_and_lean": {
        "primary_movement": "lateral_lean",
        "joints": ["lateral_lean_right", "lateral_lean_left"],
        "is_rom_test": False
    },
    "band_straighten_left_arm_elbows_bend_to_sides": {
        "primary_movement": "elbow_flexion",
        "joints": ["elbow_flexion_right", "elbow_flexion_left"],
        "is_rom_test": False
    },
    "band_straighten_right_arm_elbows_bend_to_sides": {
        "primary_movement": "elbow_flexion",
        "joints": ["elbow_flexion_right", "elbow_flexion_left"],
        "is_rom_test": False
    },
    
    # ============ STICK EXERCISES ============
    "stick_bend_elbows": {
        "primary_movement": "elbow_flexion",
        "joints": ["elbow_flexion_right", "elbow_flexion_left"],
        "is_rom_test": False
    },
    "stick_bend_elbows_and_up": {
        "primary_movement": "shoulder_flexion",
        "joints": ["shoulder_flexion_right", "shoulder_flexion_left"],
        "is_rom_test": False
    },
    "stick_raise_arms_above_head": {
        "primary_movement": "shoulder_flexion",
        "joints": ["shoulder_flexion_right", "shoulder_flexion_left"],
        "is_rom_test": False
    },
    "stick_switch": {
        "primary_movement": "torso_rotation",
        "joints": ["torso_rotation_right", "torso_rotation_left"],
        "is_rom_test": True  # ROM TEST EXERCISE!
    },
    "stick_up_and_lean": {
        "primary_movement": "lateral_lean",
        "joints": ["lateral_lean_right", "lateral_lean_left"],
        "is_rom_test": False
    },
    
    # ============ WEIGHTS EXERCISES ============
    "weights_open_arms_and_forward": {
        "primary_movement": "shoulder_flexion",
        "joints": ["shoulder_flexion_right", "shoulder_flexion_left"],
        "is_rom_test": False
    },
    "weights_abduction": {
        "primary_movement": "shoulder_abduction",
        "joints": ["shoulder_abduction_right", "shoulder_abduction_left"],
        "is_rom_test": False
    },
    
    # ============ NO-TOOL EXERCISES ============
    "notool_hands_behind_and_lean": {
        "primary_movement": "lateral_lean",
        "joints": ["lateral_lean_right", "lateral_lean_left"],
        "is_rom_test": True  # ROM TEST EXERCISE!
    },
    "notool_right_hand_up_and_bend": {
        "primary_movement": "lateral_lean",
        "joints": ["lateral_lean_right", "lateral_lean_left"],
        "is_rom_test": False
    },
    "notool_left_hand_up_and_bend": {
        "primary_movement": "lateral_lean",
        "joints": ["lateral_lean_right", "lateral_lean_left"],
        "is_rom_test": False
    },
    "notool_raising_hands_diagonally": {
        "primary_movement": "shoulder_flexion",
        "joints": ["shoulder_flexion_right", "shoulder_flexion_left"],
        "is_rom_test": False
    },
    "notool_right_bend_left_up_from_side": {
        "primary_movement": "shoulder_abduction",
        "joints": ["shoulder_abduction_right", "shoulder_abduction_left"],
        "is_rom_test": False
    },
    "notool_left_bend_right_up_from_side": {
        "primary_movement": "shoulder_abduction",
        "joints": ["shoulder_abduction_right", "shoulder_abduction_left"],
        "is_rom_test": False
    }
}


def get_rom_joints_for_exercise(exercise_name: str) -> list:
    """Get the ROM joint names relevant for a given exercise."""
    if exercise_name in EXERCISE_TO_ROM_JOINTS:
        return EXERCISE_TO_ROM_JOINTS[exercise_name]["joints"]
    return []


def get_primary_movement(exercise_name: str) -> str:
    """Get the primary movement type for an exercise."""
    if exercise_name in EXERCISE_TO_ROM_JOINTS:
        return EXERCISE_TO_ROM_JOINTS[exercise_name]["primary_movement"]
    return "unknown"


def is_rom_test_exercise(exercise_name: str) -> bool:
    """Check if an exercise is used for ROM testing."""
    if exercise_name in EXERCISE_TO_ROM_JOINTS:
        return EXERCISE_TO_ROM_JOINTS[exercise_name]["is_rom_test"]
    return False
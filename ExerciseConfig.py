"""
ExerciseConfig.py
Comprehensive exercise configuration for ROM assessment and training.

This file maps ALL exercises to their angle measurements.
Each exercise has its own thresholds - NOT shared by "joint type".

Structure:
- EXERCISE_CONFIG: Maps each exercise to its angle indices and defaults
- Helper functions to get patient's selected exercises from Excel

IMPORTANT: The default values (default_up_lb, default_up_ub, etc.) match the 
hardcoded values in Camera.py exercise functions. When ROM calibration is done,
personalized values will override these.
"""

import pandas as pd
import Settings as s

# =============================================================================
# EXERCISE CONFIGURATION
# =============================================================================
# Each exercise maps to:
#   - angles: List of dicts with angle_index, side, and default thresholds
#   - The thresholds will be personalized PER EXERCISE through ROM calibration
#
# angle_index corresponds to the position in s.last_entry_angles:
#   0 = first angle (usually right primary)
#   1 = second angle (usually left primary)
#   2 = third angle (secondary measurement - right)
#   3 = fourth angle (secondary measurement - left)
#   4 = fifth angle (third measurement - right) [for exercise_three_angles_3d]
#   5 = sixth angle (third measurement - left) [for exercise_three_angles_3d]
#
# For exercises using exercise_two_angles_3d_one_side, thresholds differ per side.
# =============================================================================

EXERCISE_CONFIG = {
    # ============ BALL EXERCISES ============
    
    # EX1: ball_bend_elbows - exercise_two_angles_3d
    # Primary: shoulder-elbow-wrist (elbow flexion)
    # Secondary: elbow-shoulder-hip (shoulder position)
    "ball_bend_elbows": {
        "category": "ball",
        "description": "כיפוף מרפקים עם כדור",
        "angles": [
            # Primary angle: elbow flexion (shoulder-elbow-wrist)
            # Camera.py: up_lb=10, up_ub=65, down_lb=95, down_ub=180
            {"index": 0, "side": "right", "default_up_lb": 10, "default_up_ub": 65, "default_down_lb": 95, "default_down_ub": 180},
            {"index": 1, "side": "left", "default_up_lb": 10, "default_up_ub": 65, "default_down_lb": 95, "default_down_ub": 180},
            # Secondary angle: shoulder position (elbow-shoulder-hip)
            # Camera.py: up_lb2=0, up_ub2=85, down_lb2=0, down_ub2=85
            {"index": 2, "side": "right", "default_up_lb": 0, "default_up_ub": 85, "default_down_lb": 0, "default_down_ub": 85, "angle_name": "shoulder"},
            {"index": 3, "side": "left", "default_up_lb": 0, "default_up_ub": 85, "default_down_lb": 0, "default_down_ub": 85, "angle_name": "shoulder"},
        ],
        "video_file": "ball_bend_elbows.mp4",
    },
    
    # EX2: ball_raise_arms_above_head - exercise_two_angles_3d
    # Primary: hip-shoulder-elbow (shoulder flexion)
    # Secondary: shoulder-elbow-wrist (elbow stays straight)
    "ball_raise_arms_above_head": {
        "category": "ball",
        "description": "הרמת ידיים מעל הראש עם כדור",
        "angles": [
            # Primary angle: shoulder flexion (hip-shoulder-elbow)
            # Camera.py: up_lb=100, up_ub=180, down_lb=0, down_ub=70
            {"index": 0, "side": "right", "default_up_lb": 100, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 70},
            {"index": 1, "side": "left", "default_up_lb": 100, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 70},
            # Secondary angle: elbow extension (shoulder-elbow-wrist)
            # Camera.py: up_lb2=120, up_ub2=180, down_lb2=120, down_ub2=180
            {"index": 2, "side": "right", "default_up_lb": 120, "default_up_ub": 180, "default_down_lb": 120, "default_down_ub": 180, "angle_name": "elbow"},
            {"index": 3, "side": "left", "default_up_lb": 120, "default_up_ub": 180, "default_down_lb": 120, "default_down_ub": 180, "angle_name": "elbow"},
        ],
        "video_file": "ball_raise_arms_above_head.mp4",
    },
    
    # EX3: ball_switch - exercise_two_angles_3d_with_axis_check
    # Primary: shoulder-elbow-wrist (elbow angle)
    # Secondary: wrist-hip-hip (trunk rotation)
    "ball_switch": {
        "category": "ball",
        "description": "החלפות עם כדור",
        "angles": [
            # Primary angle: elbow (shoulder-elbow-wrist)
            # Camera.py: up_lb=0, up_ub=180, down_lb=135, down_ub=180
            {"index": 0, "side": "right", "default_up_lb": 0, "default_up_ub": 180, "default_down_lb": 135, "default_down_ub": 180},
            {"index": 1, "side": "left", "default_up_lb": 0, "default_up_ub": 180, "default_down_lb": 135, "default_down_ub": 180},
            # Secondary angle: trunk rotation (wrist-hip-hip)
            # Camera.py: up_lb2=100, up_ub2=160, down_lb2=40, down_ub2=70
            {"index": 2, "side": "right", "default_up_lb": 100, "default_up_ub": 160, "default_down_lb": 40, "default_down_ub": 70, "angle_name": "trunk"},
            {"index": 3, "side": "left", "default_up_lb": 100, "default_up_ub": 160, "default_down_lb": 40, "default_down_ub": 70, "angle_name": "trunk"},
        ],
        "video_file": "ball_switch.mp4",
    },
    
    # EX4: ball_open_arms_and_forward - exercise_three_angles_3d
    # Primary: hip-shoulder-elbow (shoulder flexion)
    # Secondary: shoulder-elbow-wrist (elbow extension)
    # Tertiary: wrist-shoulder-wrist (arm spread)
    "ball_open_arms_and_forward": {
        "category": "ball",
        "description": "פתיחת ידיים וקדימה עם כדור",
        "angles": [
            # Primary angle: shoulder flexion (hip-shoulder-elbow)
            # Camera.py: up_lb=60, up_ub=120, down_lb=20, down_ub=110
            {"index": 0, "side": "right", "default_up_lb": 60, "default_up_ub": 120, "default_down_lb": 20, "default_down_ub": 110},
            {"index": 1, "side": "left", "default_up_lb": 60, "default_up_ub": 120, "default_down_lb": 20, "default_down_ub": 110},
            # Secondary angle: elbow extension (shoulder-elbow-wrist)
            # Camera.py: up_lb2=150, up_ub2=180, down_lb2=0, down_ub2=180
            {"index": 2, "side": "right", "default_up_lb": 150, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 180, "angle_name": "elbow"},
            {"index": 3, "side": "left", "default_up_lb": 150, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 180, "angle_name": "elbow"},
            # Tertiary angle: arm spread (wrist-shoulder-wrist)
            # Camera.py: up_lb3=140, up_ub3=180, down_lb3=0, down_ub3=105
            {"index": 4, "side": "right", "default_up_lb": 140, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 105, "angle_name": "spread"},
            {"index": 5, "side": "left", "default_up_lb": 140, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 105, "angle_name": "spread"},
        ],
        "video_file": "ball_open_arms_and_forward.mp4",
    },
    
    # EX5: ball_open_arms_above_head - exercise_two_angles_3d (COMMENTED OUT - not implemented in Gymmy.py)
    # "ball_open_arms_above_head": {...}
    
    # ============ BAND EXERCISES ============
    
    # EX6: band_open_arms - exercise_three_angles_3d
    # Primary: hip-shoulder-wrist (arm elevation)
    # Secondary: shoulder-elbow-wrist (elbow extension)
    # Tertiary: wrist-shoulder-wrist (arm spread)
    "band_open_arms": {
        "category": "band",
        "description": "פתיחת ידיים עם גומיה",
        "angles": [
            # Primary angle: arm elevation (hip-shoulder-wrist)
            # Camera.py: up_lb=65, up_ub=120, down_lb=40, down_ub=120
            {"index": 0, "side": "right", "default_up_lb": 65, "default_up_ub": 120, "default_down_lb": 40, "default_down_ub": 120},
            {"index": 1, "side": "left", "default_up_lb": 65, "default_up_ub": 120, "default_down_lb": 40, "default_down_ub": 120},
            # Secondary angle: elbow extension (shoulder-elbow-wrist)
            # Camera.py: up_lb2=135, up_ub2=180, down_lb2=0, down_ub2=180
            {"index": 2, "side": "right", "default_up_lb": 135, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 180, "angle_name": "elbow"},
            {"index": 3, "side": "left", "default_up_lb": 135, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 180, "angle_name": "elbow"},
            # Tertiary angle: arm spread (wrist-shoulder-wrist)
            # Camera.py: up_lb3=135, up_ub3=180, down_lb3=0, down_ub3=120
            {"index": 4, "side": "right", "default_up_lb": 135, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 120, "angle_name": "spread"},
            {"index": 5, "side": "left", "default_up_lb": 135, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 120, "angle_name": "spread"},
        ],
        "video_file": "band_open_arms.mp4",
    },
    
    # EX7: band_open_arms_and_up - exercise_three_angles_3d
    # Primary: shoulder-elbow-wrist (elbow extension)
    # Secondary: elbow-shoulder-hip (shoulder elevation)
    # Tertiary: wrist-shoulder-wrist (arm spread)
    "band_open_arms_and_up": {
        "category": "band",
        "description": "פתיחת ידיים והרמה עם גומיה",
        "angles": [
            # Primary angle: elbow extension (shoulder-elbow-wrist)
            # Camera.py: up_lb=135, up_ub=180, down_lb=0, down_ub=180
            {"index": 0, "side": "right", "default_up_lb": 135, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 180},
            {"index": 1, "side": "left", "default_up_lb": 135, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 180},
            # Secondary angle: shoulder elevation (elbow-shoulder-hip)
            # Camera.py: up_lb2=120, up_ub2=180, down_lb2=0, down_ub2=105
            {"index": 2, "side": "right", "default_up_lb": 120, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 105, "angle_name": "shoulder"},
            {"index": 3, "side": "left", "default_up_lb": 120, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 105, "angle_name": "shoulder"},
            # Tertiary angle: arm spread (wrist-shoulder-wrist)
            # Camera.py: up_lb3=70, up_ub3=170, down_lb3=20, down_ub3=130
            {"index": 4, "side": "right", "default_up_lb": 70, "default_up_ub": 170, "default_down_lb": 20, "default_down_ub": 130, "angle_name": "spread"},
            {"index": 5, "side": "left", "default_up_lb": 70, "default_up_ub": 170, "default_down_lb": 20, "default_down_ub": 130, "angle_name": "spread"},
        ],
        "video_file": "band_open_arms_and_up.mp4",
    },
    
    # EX8: band_up_and_lean - exercise_two_angles_3d_with_axis_check
    # Primary: shoulder-elbow-wrist (elbow extension)
    # Secondary: elbow-hip-hip (trunk lean)
    "band_up_and_lean": {
        "category": "band",
        "description": "הרמה והטיה עם גומיה",
        "angles": [
            # Primary angle: elbow extension (shoulder-elbow-wrist)
            # Camera.py: up_lb=110, up_ub=180, down_lb=90, down_ub=180
            {"index": 0, "side": "right", "default_up_lb": 110, "default_up_ub": 180, "default_down_lb": 90, "default_down_ub": 180},
            {"index": 1, "side": "left", "default_up_lb": 110, "default_up_ub": 180, "default_down_lb": 90, "default_down_ub": 180},
            # Secondary angle: trunk lean (elbow-hip-hip)
            # Camera.py: up_lb2=120, up_ub2=170, down_lb2=50, down_ub2=115
            {"index": 2, "side": "right", "default_up_lb": 120, "default_up_ub": 170, "default_down_lb": 50, "default_down_ub": 115, "angle_name": "trunk"},
            {"index": 3, "side": "left", "default_up_lb": 120, "default_up_ub": 170, "default_down_lb": 50, "default_down_ub": 115, "angle_name": "trunk"},
        ],
        "video_file": "band_up_and_lean.mp4",
    },
    
    # EX9: band_straighten_left_arm_elbows_bend_to_sides - exercise_two_angles_3d_one_side
    # Primary: shoulder-elbow-wrist (elbow extension) - DIFFERENT thresholds per side
    # Secondary: elbow-shoulder-hip (shoulder position)
    "band_straighten_left_arm_elbows_bend_to_sides": {
        "category": "band",
        "description": "יישור יד שמאל וכיפוף מרפקים לצדדים",
        "one_side_thresholds": True,  # Flag indicating different thresholds per side
        "angles": [
            # Primary angle RIGHT: shoulder-elbow-wrist
            # Camera.py: up_lb_right=0, up_ub_right=75, down_lb_right=0, down_ub_right=75
            {"index": 0, "side": "right", "default_up_lb": 0, "default_up_ub": 75, "default_down_lb": 0, "default_down_ub": 75},
            # Primary angle LEFT: shoulder-elbow-wrist
            # Camera.py: up_lb_left=135, up_ub_left=180, down_lb_left=0, down_ub_left=75
            {"index": 1, "side": "left", "default_up_lb": 135, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 75},
            # Secondary angle RIGHT: elbow-shoulder-hip
            # Camera.py: up_lb_right2=60, up_ub_right2=130, down_lb_right2=60, down_ub_right2=130
            {"index": 2, "side": "right", "default_up_lb": 60, "default_up_ub": 130, "default_down_lb": 60, "default_down_ub": 130, "angle_name": "shoulder"},
            # Secondary angle LEFT: elbow-shoulder-hip
            # Camera.py: up_lb_left2=60, up_ub_left2=130, down_lb_left2=60, down_ub_left2=130
            {"index": 3, "side": "left", "default_up_lb": 60, "default_up_ub": 130, "default_down_lb": 60, "default_down_ub": 130, "angle_name": "shoulder"},
        ],
        "video_file": "band_straighten_left_arm_elbows_bend_to_sides.mp4",
    },
    
    # EX10: band_straighten_right_arm_elbows_bend_to_sides - exercise_two_angles_3d_one_side
    # Primary: shoulder-elbow-wrist (elbow extension) - DIFFERENT thresholds per side
    # Secondary: elbow-shoulder-hip (shoulder position)
    "band_straighten_right_arm_elbows_bend_to_sides": {
        "category": "band",
        "description": "יישור יד ימין וכיפוף מרפקים לצדדים",
        "one_side_thresholds": True,
        "angles": [
            # Primary angle RIGHT: shoulder-elbow-wrist
            # Camera.py: up_lb_right=135, up_ub_right=180, down_lb_right=0, down_ub_right=75
            {"index": 0, "side": "right", "default_up_lb": 135, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 75},
            # Primary angle LEFT: shoulder-elbow-wrist
            # Camera.py: up_lb_left=0, up_ub_left=75, down_lb_left=0, down_ub_left=75
            {"index": 1, "side": "left", "default_up_lb": 0, "default_up_ub": 75, "default_down_lb": 0, "default_down_ub": 75},
            # Secondary angle RIGHT: elbow-shoulder-hip
            # Camera.py: up_lb_right2=60, up_ub_right2=120, down_lb_right2=60, down_ub_right2=120
            {"index": 2, "side": "right", "default_up_lb": 60, "default_up_ub": 120, "default_down_lb": 60, "default_down_ub": 120, "angle_name": "shoulder"},
            # Secondary angle LEFT: elbow-shoulder-hip
            # Camera.py: up_lb_left2=60, up_ub_left2=120, down_lb_left2=60, down_ub_left2=120
            {"index": 3, "side": "left", "default_up_lb": 60, "default_up_ub": 120, "default_down_lb": 60, "default_down_ub": 120, "angle_name": "shoulder"},
        ],
        "video_file": "band_straighten_right_arm_elbows_bend_to_sides.mp4",
    },
    
    # ============ STICK EXERCISES ============
    
    # EX11: stick_bend_elbows - exercise_two_angles_3d
    # Primary: shoulder-elbow-wrist (elbow flexion)
    # Secondary: elbow-shoulder-hip (shoulder position)
    "stick_bend_elbows": {
        "category": "stick",
        "description": "כיפוף מרפקים עם מקל",
        "angles": [
            # Primary angle: elbow flexion (shoulder-elbow-wrist)
            # Camera.py: up_lb=10, up_ub=70, down_lb=95, down_ub=180
            {"index": 0, "side": "right", "default_up_lb": 10, "default_up_ub": 70, "default_down_lb": 95, "default_down_ub": 180},
            {"index": 1, "side": "left", "default_up_lb": 10, "default_up_ub": 70, "default_down_lb": 95, "default_down_ub": 180},
            # Secondary angle: shoulder position (elbow-shoulder-hip)
            # Camera.py: up_lb2=0, up_ub2=70, down_lb2=0, down_ub2=70
            {"index": 2, "side": "right", "default_up_lb": 0, "default_up_ub": 70, "default_down_lb": 0, "default_down_ub": 70, "angle_name": "shoulder"},
            {"index": 3, "side": "left", "default_up_lb": 0, "default_up_ub": 70, "default_down_lb": 0, "default_down_ub": 70, "angle_name": "shoulder"},
        ],
        "video_file": "stick_bend_elbows.mp4",
    },
    
    # EX12: stick_bend_elbows_and_up - exercise_two_angles_3d
    # Primary: hip-shoulder-elbow (shoulder flexion)
    # Secondary: shoulder-elbow-wrist (elbow extension)
    "stick_bend_elbows_and_up": {
        "category": "stick",
        "description": "כיפוף מרפקים והרמה עם מקל",
        "angles": [
            # Primary angle: shoulder flexion (hip-shoulder-elbow)
            # Camera.py: up_lb=110, up_ub=180, down_lb=0, down_ub=70
            {"index": 0, "side": "right", "default_up_lb": 110, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 70},
            {"index": 1, "side": "left", "default_up_lb": 110, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 70},
            # Secondary angle: elbow extension (shoulder-elbow-wrist)
            # Camera.py: up_lb2=120, up_ub2=180, down_lb2=0, down_ub2=75
            {"index": 2, "side": "right", "default_up_lb": 120, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 75, "angle_name": "elbow"},
            {"index": 3, "side": "left", "default_up_lb": 120, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 75, "angle_name": "elbow"},
        ],
        "video_file": "stick_bend_elbows_and_up.mp4",
    },
    
    # EX13: stick_raise_arms_above_head - exercise_two_angles_3d
    # Primary: hip-shoulder-elbow (shoulder flexion)
    # Secondary: wrist-elbow-shoulder (elbow extension)
    "stick_raise_arms_above_head": {
        "category": "stick",
        "description": "הרמת ידיים מעל הראש עם מקל",
        "angles": [
            # Primary angle: shoulder flexion (hip-shoulder-elbow)
            # Camera.py: up_lb=105, up_ub=180, down_lb=10, down_ub=70
            {"index": 0, "side": "right", "default_up_lb": 105, "default_up_ub": 180, "default_down_lb": 10, "default_down_ub": 70},
            {"index": 1, "side": "left", "default_up_lb": 105, "default_up_ub": 180, "default_down_lb": 10, "default_down_ub": 70},
            # Secondary angle: elbow extension (wrist-elbow-shoulder)
            # Camera.py: up_lb2=125, up_ub2=180, down_lb2=125, down_ub2=180
            {"index": 2, "side": "right", "default_up_lb": 125, "default_up_ub": 180, "default_down_lb": 125, "default_down_ub": 180, "angle_name": "elbow"},
            {"index": 3, "side": "left", "default_up_lb": 125, "default_up_ub": 180, "default_down_lb": 125, "default_down_ub": 180, "angle_name": "elbow"},
        ],
        "video_file": "stick_raise_arms_above_head.mp4",
    },
    
    # EX14: stick_switch - exercise_two_angles_3d_with_axis_check
    # Primary: shoulder-elbow-wrist (elbow angle)
    # Secondary: wrist-hip-hip (trunk rotation)
    "stick_switch": {
        "category": "stick",
        "description": "החלפות עם מקל",
        "angles": [
            # Primary angle: elbow (shoulder-elbow-wrist)
            # Camera.py: up_lb=0, up_ub=180, down_lb=135, down_ub=180
            {"index": 0, "side": "right", "default_up_lb": 0, "default_up_ub": 180, "default_down_lb": 135, "default_down_ub": 180},
            {"index": 1, "side": "left", "default_up_lb": 0, "default_up_ub": 180, "default_down_lb": 135, "default_down_ub": 180},
            # Secondary angle: trunk rotation (wrist-hip-hip)
            # Camera.py: up_lb2=85, up_ub2=160, down_lb2=10, down_ub2=70
            {"index": 2, "side": "right", "default_up_lb": 85, "default_up_ub": 160, "default_down_lb": 10, "default_down_ub": 70, "angle_name": "trunk"},
            {"index": 3, "side": "left", "default_up_lb": 85, "default_up_ub": 160, "default_down_lb": 10, "default_down_ub": 70, "angle_name": "trunk"},
        ],
        "video_file": "stick_switch.mp4",
    },
    
    # EX15: stick_up_and_lean - exercise_two_angles_3d_with_axis_check
    # Primary: shoulder-elbow-wrist (elbow extension)
    # Secondary: elbow-hip-hip (trunk lean)
    "stick_up_and_lean": {
        "category": "stick",
        "description": "הרמה והטיה עם מקל",
        "angles": [
            # Primary angle: elbow extension (shoulder-elbow-wrist)
            # Camera.py: up_lb=110, up_ub=180, down_lb=90, down_ub=180
            {"index": 0, "side": "right", "default_up_lb": 110, "default_up_ub": 180, "default_down_lb": 90, "default_down_ub": 180},
            {"index": 1, "side": "left", "default_up_lb": 110, "default_up_ub": 180, "default_down_lb": 90, "default_down_ub": 180},
            # Secondary angle: trunk lean (elbow-hip-hip)
            # Camera.py: up_lb2=115, up_ub2=170, down_lb2=50, down_ub2=110
            {"index": 2, "side": "right", "default_up_lb": 115, "default_up_ub": 170, "default_down_lb": 50, "default_down_ub": 110, "angle_name": "trunk"},
            {"index": 3, "side": "left", "default_up_lb": 115, "default_up_ub": 170, "default_down_lb": 50, "default_down_ub": 110, "angle_name": "trunk"},
        ],
        "video_file": "stick_up_and_lean.mp4",
    },
    
    # ============ WEIGHTS EXERCISES ============
    
    # EX18: weights_open_arms_and_forward - exercise_three_angles_3d
    # Primary: hip-shoulder-elbow (shoulder flexion)
    # Secondary: shoulder-elbow-wrist (elbow extension)
    # Tertiary: wrist-shoulder-wrist (arm spread)
    "weights_open_arms_and_forward": {
        "category": "weights",
        "description": "פתיחת ידיים וקדימה עם משקולות",
        "angles": [
            # Primary angle: shoulder flexion (hip-shoulder-elbow)
            # Camera.py: up_lb=60, up_ub=120, down_lb=20, down_ub=110
            {"index": 0, "side": "right", "default_up_lb": 60, "default_up_ub": 120, "default_down_lb": 20, "default_down_ub": 110},
            {"index": 1, "side": "left", "default_up_lb": 60, "default_up_ub": 120, "default_down_lb": 20, "default_down_ub": 110},
            # Secondary angle: elbow extension (shoulder-elbow-wrist)
            # Camera.py: up_lb2=150, up_ub2=180, down_lb2=0, down_ub2=180
            {"index": 2, "side": "right", "default_up_lb": 150, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 180, "angle_name": "elbow"},
            {"index": 3, "side": "left", "default_up_lb": 150, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 180, "angle_name": "elbow"},
            # Tertiary angle: arm spread (wrist-shoulder-wrist)
            # Camera.py: up_lb3=140, up_ub3=180, down_lb3=0, down_ub3=105
            {"index": 4, "side": "right", "default_up_lb": 140, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 105, "angle_name": "spread"},
            {"index": 5, "side": "left", "default_up_lb": 140, "default_up_ub": 180, "default_down_lb": 0, "default_down_ub": 105, "angle_name": "spread"},
        ],
        "video_file": "weights_open_arms_and_forward.mp4",
    },
    
    # EX19: weights_abduction - exercise_two_angles_3d
    # Primary: hip-shoulder-elbow (shoulder abduction)
    # Secondary: shoulder-elbow-wrist (elbow stays straight)
    "weights_abduction": {
        "category": "weights",
        "description": "הרחקה עם משקולות",
        "angles": [
            # Primary angle: shoulder abduction (hip-shoulder-elbow)
            # Camera.py: up_lb=80, up_ub=120, down_lb=0, down_ub=55
            {"index": 0, "side": "right", "default_up_lb": 80, "default_up_ub": 120, "default_down_lb": 0, "default_down_ub": 55},
            {"index": 1, "side": "left", "default_up_lb": 80, "default_up_ub": 120, "default_down_lb": 0, "default_down_ub": 55},
            # Secondary angle: elbow extension (shoulder-elbow-wrist)
            # Camera.py: up_lb2=140, up_ub2=180, down_lb2=140, down_ub2=180
            {"index": 2, "side": "right", "default_up_lb": 140, "default_up_ub": 180, "default_down_lb": 140, "default_down_ub": 180, "angle_name": "elbow"},
            {"index": 3, "side": "left", "default_up_lb": 140, "default_up_ub": 180, "default_down_lb": 140, "default_down_ub": 180, "angle_name": "elbow"},
        ],
        "video_file": "weights_abduction.mp4",
    },
    
    # ============ NO-TOOL EXERCISES ============
    
    # EX20: notool_hands_behind_and_lean - exercise_two_angles_3d_with_axis_check
    # Primary: shoulder-elbow-wrist (arms behind)
    # Secondary: elbow-hip-hip (trunk lean)
    "notool_hands_behind_and_lean": {
        "category": "notool",
        "description": "ידיים מאחור והטיה",
        "angles": [
            # Primary angle: arms behind (shoulder-elbow-wrist)
            # Camera.py: up_lb=10, up_ub=70, down_lb=10, down_ub=70
            {"index": 0, "side": "right", "default_up_lb": 10, "default_up_ub": 70, "default_down_lb": 10, "default_down_ub": 70},
            {"index": 1, "side": "left", "default_up_lb": 10, "default_up_ub": 70, "default_down_lb": 10, "default_down_ub": 70},
            # Secondary angle: trunk lean (elbow-hip-hip)
            # Camera.py: up_lb2=115, up_ub2=170, down_lb2=80, down_ub2=115
            {"index": 2, "side": "right", "default_up_lb": 115, "default_up_ub": 170, "default_down_lb": 80, "default_down_ub": 115, "angle_name": "trunk"},
            {"index": 3, "side": "left", "default_up_lb": 115, "default_up_ub": 170, "default_down_lb": 80, "default_down_ub": 115, "angle_name": "trunk"},
        ],
        "video_file": "notool_hands_behind_and_lean.mp4",
    },
    
    # EX21: notool_right_hand_up_and_bend - uses hand_up_and_bend_angles (different function)
    "notool_right_hand_up_and_bend": {
        "category": "notool",
        "description": "יד ימין למעלה וכיפוף",
        "angles": [
            # This exercise uses a different function (hand_up_and_bend_angles)
            # Camera.py: wrist-hip-hip, up_lb=20, up_ub=100, down_lb=90, down_ub=180, side="right"
            {"index": 0, "side": "right", "default_up_lb": 20, "default_up_ub": 100, "default_down_lb": 90, "default_down_ub": 180},
            {"index": 1, "side": "left", "default_up_lb": 20, "default_up_ub": 100, "default_down_lb": 90, "default_down_ub": 180},
        ],
        "video_file": "notool_right_hand_up_and_bend.mp4",
    },
    
    # EX22: notool_left_hand_up_and_bend - uses hand_up_and_bend_angles (different function)
    "notool_left_hand_up_and_bend": {
        "category": "notool",
        "description": "יד שמאל למעלה וכיפוף",
        "angles": [
            # This exercise uses a different function (hand_up_and_bend_angles)
            # Camera.py: wrist-hip-hip, up_lb=20, up_ub=100, down_lb=90, down_ub=180, side="left"
            {"index": 0, "side": "right", "default_up_lb": 20, "default_up_ub": 100, "default_down_lb": 90, "default_down_ub": 180},
            {"index": 1, "side": "left", "default_up_lb": 20, "default_up_ub": 100, "default_down_lb": 90, "default_down_ub": 180},
        ],
        "video_file": "notool_left_hand_up_and_bend.mp4",
    },
    
    # EX23: notool_raising_hands_diagonally - exercise_two_angles_3d_with_axis_check
    # Primary: wrist-shoulder-hip (arm elevation diagonal)
    # Secondary: shoulder-elbow-wrist (elbow extension)
    "notool_raising_hands_diagonally": {
        "category": "notool",
        "description": "הרמת ידיים באלכסון",
        "angles": [
            # Primary angle: arm elevation diagonal (wrist-shoulder-hip)
            # Camera.py: up_lb=80, up_ub=135, down_lb=105, down_ub=150
            {"index": 0, "side": "right", "default_up_lb": 80, "default_up_ub": 135, "default_down_lb": 105, "default_down_ub": 150},
            {"index": 1, "side": "left", "default_up_lb": 80, "default_up_ub": 135, "default_down_lb": 105, "default_down_ub": 150},
            # Secondary angle: elbow extension (shoulder-elbow-wrist)
            # Camera.py: up_lb2=0, up_ub2=180, down_lb2=120, down_ub2=180
            {"index": 2, "side": "right", "default_up_lb": 0, "default_up_ub": 180, "default_down_lb": 120, "default_down_ub": 180, "angle_name": "elbow"},
            {"index": 3, "side": "left", "default_up_lb": 0, "default_up_ub": 180, "default_down_lb": 120, "default_down_ub": 180, "angle_name": "elbow"},
        ],
        "video_file": "notool_raising_hands_diagonally.mp4",
    },
    
    # EX24: notool_right_bend_left_up_from_side - exercise_two_angles_3d_one_side
    # Primary: wrist-elbow-shoulder (elbow angle) - DIFFERENT thresholds per side
    # Secondary: hip-shoulder-elbow (shoulder elevation)
    "notool_right_bend_left_up_from_side": {
        "category": "notool",
        "description": "כיפוף ימין והרמה שמאל מהצד",
        "one_side_thresholds": True,
        "angles": [
            # Primary angle RIGHT: wrist-elbow-shoulder
            # Camera.py: up_lb_right=95, up_ub_right=170, down_lb_right=0, down_ub_right=50
            {"index": 0, "side": "right", "default_up_lb": 95, "default_up_ub": 170, "default_down_lb": 0, "default_down_ub": 50},
            # Primary angle LEFT: wrist-elbow-shoulder
            # Camera.py: up_lb_left=140, up_ub_left=180, down_lb_left=140, down_ub_left=180
            {"index": 1, "side": "left", "default_up_lb": 140, "default_up_ub": 180, "default_down_lb": 140, "default_down_ub": 180},
            # Secondary angle RIGHT: hip-shoulder-elbow
            # Camera.py: up_lb_right2=0, up_ub_right2=60, down_lb_right2=0, down_ub_right2=60
            {"index": 2, "side": "right", "default_up_lb": 0, "default_up_ub": 60, "default_down_lb": 0, "default_down_ub": 60, "angle_name": "shoulder"},
            # Secondary angle LEFT: hip-shoulder-elbow
            # Camera.py: up_lb_left2=70, up_ub_left2=120, down_lb_left2=0, down_ub_left2=50
            {"index": 3, "side": "left", "default_up_lb": 70, "default_up_ub": 120, "default_down_lb": 0, "default_down_ub": 50, "angle_name": "shoulder"},
        ],
        "video_file": "notool_right_bend_left_up_from_side.mp4",
    },
    
    # EX25: notool_left_bend_right_up_from_side - exercise_two_angles_3d_one_side
    # Primary: wrist-elbow-shoulder (elbow angle) - DIFFERENT thresholds per side
    # Secondary: hip-shoulder-elbow (shoulder elevation)
    "notool_left_bend_right_up_from_side": {
        "category": "notool",
        "description": "כיפוף שמאל והרמה ימין מהצד",
        "one_side_thresholds": True,
        "angles": [
            # Primary angle RIGHT: wrist-elbow-shoulder
            # Camera.py: up_lb_right=140, up_ub_right=180, down_lb_right=140, down_ub_right=180
            {"index": 0, "side": "right", "default_up_lb": 140, "default_up_ub": 180, "default_down_lb": 140, "default_down_ub": 180},
            # Primary angle LEFT: wrist-elbow-shoulder
            # Camera.py: up_lb_left=95, up_ub_left=170, down_lb_left=0, down_ub_left=40
            {"index": 1, "side": "left", "default_up_lb": 95, "default_up_ub": 170, "default_down_lb": 0, "default_down_ub": 40},
            # Secondary angle RIGHT: hip-shoulder-elbow
            # Camera.py: up_lb_right2=80, up_ub_right2=120, down_lb_right2=0, down_ub_right2=50
            {"index": 2, "side": "right", "default_up_lb": 80, "default_up_ub": 120, "default_down_lb": 0, "default_down_ub": 50, "angle_name": "shoulder"},
            # Secondary angle LEFT: hip-shoulder-elbow
            # Camera.py: up_lb_left2=0, up_ub_left2=60, down_lb_left2=0, down_ub_left2=60
            {"index": 3, "side": "left", "default_up_lb": 0, "default_up_ub": 60, "default_down_lb": 0, "default_down_ub": 60, "angle_name": "shoulder"},
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


def get_default_thresholds(exercise_name, angle_index):
    """
    Get default thresholds for a specific angle of an exercise.
    
    Args:
        exercise_name: e.g., "ball_bend_elbows"
        angle_index: 0, 1, 2, 3, etc.
    
    Returns:
        Dict with up_lb, up_ub, down_lb, down_ub or None if not found
    """
    config = EXERCISE_CONFIG.get(exercise_name)
    if not config:
        return None
    
    for angle_config in config["angles"]:
        if angle_config["index"] == angle_index:
            return {
                'up_lb': angle_config.get("default_up_lb", 90),
                'up_ub': angle_config.get("default_up_ub", 180),
                'down_lb': angle_config.get("default_down_lb", 0),
                'down_ub': angle_config.get("default_down_ub", 90),
            }
    
    return None


def get_exercise_threshold(exercise_name, side, position, bound="lb"):
    """
    Get personalized threshold for an exercise from patient_rom_limits.
    Falls back to default if not found.
    
    Args:
        exercise_name: e.g., "ball_bend_elbows"
        side: "right" or "left"
        position: "up" or "down"
        bound: "lb" or "ub" (default: "lb")
    
    Returns:
        Threshold angle value
    """
    # Try to get from patient ROM limits
    if hasattr(s, 'patient_rom_limits') and s.patient_rom_limits:
        key = f"{exercise_name}_{side}_{position}_{bound}"
        if key in s.patient_rom_limits:
            value = s.patient_rom_limits[key]
            if value is not None and value > 0:
                return value
    
    # Fall back to config defaults
    config = EXERCISE_CONFIG.get(exercise_name)
    if not config:
        return 90  # fallback
    
    # Find the angle for this side (index 0 for right, 1 for left for primary)
    angle_index = 0 if side == "right" else 1
    
    for angle_config in config["angles"]:
        if angle_config["index"] == angle_index:
            key = f"default_{position}_{bound}"
            return angle_config.get(key, 90)
    
    return 90  # fallback


def has_secondary_angles(exercise_name):
    """Check if an exercise has secondary angles (index 2, 3)."""
    config = EXERCISE_CONFIG.get(exercise_name)
    if not config:
        return False
    
    for angle_config in config["angles"]:
        if angle_config["index"] >= 2:
            return True
    return False


def has_tertiary_angles(exercise_name):
    """Check if an exercise has tertiary angles (index 4, 5) - for exercise_three_angles_3d."""
    config = EXERCISE_CONFIG.get(exercise_name)
    if not config:
        return False
    
    for angle_config in config["angles"]:
        if angle_config["index"] >= 4:
            return True
    return False


def get_angle_name(exercise_name, angle_index):
    """Get the angle_name for a specific angle (e.g., 'elbow', 'shoulder', 'trunk')."""
    config = EXERCISE_CONFIG.get(exercise_name)
    if not config:
        return None
    
    for angle_config in config["angles"]:
        if angle_config["index"] == angle_index:
            return angle_config.get("angle_name", None)
    return None


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
            num_angles = len(config['angles'])
            has_sec = has_secondary_angles(ex)
            has_ter = has_tertiary_angles(ex)
            one_side = config.get('one_side_thresholds', False)
            
            flags = []
            if has_sec:
                flags.append("2nd")
            if has_ter:
                flags.append("3rd")
            if one_side:
                flags.append("one-side")
            
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  - {ex}: {num_angles} angles{flag_str}")

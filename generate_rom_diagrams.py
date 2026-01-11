"""
generate_rom_diagrams.py
Generate clean flowcharts for ROM module documentation.
Style: Black and white, similar to draw.io
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon, Rectangle, Arc
import numpy as np
import os

# Create output folder
OUTPUT_FOLDER = "Report_Diagrams"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def draw_oval(ax, x, y, width, height, text, fontsize=10):
    """Draw an oval (for start/end nodes)."""
    oval = Ellipse((x, y), width, height, facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(oval)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)


def draw_rect(ax, x, y, width, height, text, fontsize=10):
    """Draw a rectangle (for process nodes)."""
    rect = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor='white', edgecolor='black', linewidth=1.5
    )
    ax.add_patch(rect)
    
    # Handle multi-line text
    lines = text.split('\n')
    line_height = 0.25
    for i, line in enumerate(lines):
        y_pos = y + (len(lines)/2 - i - 0.5) * line_height
        ax.text(x, y_pos, line, ha='center', va='center', fontsize=fontsize)


def draw_diamond(ax, x, y, size, text, fontsize=9):
    """Draw a diamond (for decision nodes)."""
    diamond = Polygon([
        (x, y + size/2),
        (x + size/2, y),
        (x, y - size/2),
        (x - size/2, y),
    ], facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(diamond)
    
    lines = text.split('\n')
    line_height = 0.2
    for i, line in enumerate(lines):
        y_pos = y + (len(lines)/2 - i - 0.5) * line_height
        ax.text(x, y_pos, line, ha='center', va='center', fontsize=fontsize)


def draw_cylinder(ax, x, y, width, height, text, fontsize=9):
    """Draw a cylinder (for database/file nodes)."""
    # Main body
    rect = Rectangle((x - width/2, y - height/2), width, height * 0.85,
                     facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Top ellipse
    top_ellipse = Ellipse((x, y + height/2 * 0.85), width, height * 0.3,
                          facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(top_ellipse)
    
    # Bottom arc (half ellipse)
    bottom_arc = Arc((x, y - height/2), width, height * 0.3, 
                     angle=0, theta1=180, theta2=360, linewidth=1.5)
    ax.add_patch(bottom_arc)
    
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)


def draw_arrow(ax, start, end):
    """Draw a simple arrow."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))


def draw_arrow_with_label(ax, start, end, label, label_offset=(0.15, 0)):
    """Draw an arrow with a label."""
    draw_arrow(ax, start, end)
    mid_x = (start[0] + end[0]) / 2 + label_offset[0]
    mid_y = (start[1] + end[1]) / 2 + label_offset[1]
    ax.text(mid_x, mid_y, label, fontsize=9, ha='center', va='center')


# =============================================================================
# DIAGRAM 1: ROM Assessment Process Flow
# =============================================================================
def create_rom_process_flow():
    """Create the main ROM assessment process flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Nodes
    draw_oval(ax, 4, 11, 2, 0.7, 'Start')
    draw_arrow(ax, (4, 10.65), (4, 10.1))
    
    draw_rect(ax, 4, 9.5, 3.5, 0.8, 'Select Patient')
    draw_arrow(ax, (4, 9.1), (4, 8.5))
    
    draw_rect(ax, 4, 8, 3.5, 0.8, 'Select Exercises')
    draw_arrow(ax, (4, 7.6), (4, 7))
    
    draw_rect(ax, 4, 6.5, 3.5, 0.8, 'Start ROM Assessment')
    draw_arrow(ax, (4, 6.1), (4, 5.5))
    
    draw_rect(ax, 4, 5, 3.8, 0.8, 'Record Angle Data\n(No Feedback)')
    draw_arrow(ax, (4, 4.6), (4, 4))
    
    draw_rect(ax, 4, 3.5, 3.8, 0.8, 'Calculate Personal\nThresholds')
    draw_arrow(ax, (4, 3.1), (4, 2.5))
    
    draw_cylinder(ax, 4, 2, 3, 0.8, 'Save to\nPatients.xlsx')
    draw_arrow(ax, (4, 1.6), (4, 1))
    
    draw_oval(ax, 4, 0.6, 2, 0.6, 'End')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, '1_ROM_Process_Flow.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {filepath}")


# =============================================================================
# DIAGRAM 2: Threshold Calculation Algorithm
# =============================================================================
def create_threshold_algorithm():
    """Create the threshold calculation algorithm flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input
    draw_cylinder(ax, 6, 9, 4, 0.8, 'Angle History Data')
    draw_arrow(ax, (6, 8.6), (6, 8.1))
    
    # Split
    draw_rect(ax, 6, 7.7, 3.5, 0.6, 'Analyze Movement Pattern')
    draw_arrow(ax, (4.25, 7.7), (3.5, 7.1))
    draw_arrow(ax, (7.75, 7.7), (8.5, 7.1))
    
    # LEFT: Peaks
    ax.text(3, 7.3, 'UP Position', ha='center', fontsize=10, fontweight='bold')
    draw_rect(ax, 3, 6.5, 2.8, 0.6, 'Find Local Peaks')
    draw_arrow(ax, (3, 6.2), (3, 5.7))
    
    draw_rect(ax, 3, 5.3, 2.8, 0.6, 'Filter Outliers')
    draw_arrow(ax, (3, 5), (3, 4.5))
    
    draw_rect(ax, 3, 4.1, 2.8, 0.6, 'Take Top 5 Values')
    draw_arrow(ax, (3, 3.8), (3, 3.3))
    
    draw_rect(ax, 3, 2.9, 2.8, 0.6, 'Calculate avg')
    draw_arrow(ax, (3, 2.6), (3, 2.1))
    
    draw_rect(ax, 3, 1.7, 3, 0.6, 'up_ub = avg + 10°\nup_lb = avg - 10°', fontsize=9)
    
    # RIGHT: Valleys
    ax.text(9, 7.3, 'DOWN Position', ha='center', fontsize=10, fontweight='bold')
    draw_rect(ax, 9, 6.5, 2.8, 0.6, 'Find Local Valleys')
    draw_arrow(ax, (9, 6.2), (9, 5.7))
    
    draw_rect(ax, 9, 5.3, 2.8, 0.6, 'Filter Outliers')
    draw_arrow(ax, (9, 5), (9, 4.5))
    
    draw_rect(ax, 9, 4.1, 3, 0.6, 'Take Bottom 5 Values')
    draw_arrow(ax, (9, 3.8), (9, 3.3))
    
    draw_rect(ax, 9, 2.9, 2.8, 0.6, 'Calculate avg')
    draw_arrow(ax, (9, 2.6), (9, 2.1))
    
    draw_rect(ax, 9, 1.7, 3.2, 0.6, 'down_ub = avg + 10°\ndown_lb = avg - 10°', fontsize=9)
    
    # Merge
    draw_arrow(ax, (3, 1.4), (4.5, 0.8))
    draw_arrow(ax, (9, 1.4), (7.5, 0.8))
    draw_cylinder(ax, 6, 0.5, 3.5, 0.6, 'Save ROM Thresholds')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, '2_Threshold_Algorithm.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {filepath}")


# =============================================================================
# DIAGRAM 3: Training Session Flow (with ROM integration)
# =============================================================================
def create_training_flow():
    """Create training session flow showing ROM integration point."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Start
    draw_oval(ax, 5, 13.3, 2.2, 0.6, 'Training starts')
    draw_arrow(ax, (5, 13), (5, 12.5))
    
    # Opening
    draw_rect(ax, 5, 12, 3.5, 0.7, 'Training opening\nbriefing')
    draw_arrow(ax, (5, 11.65), (5, 11.15))
    
    # Calibration
    draw_rect(ax, 5, 10.7, 3, 0.6, 'Calibration')
    draw_arrow(ax, (5, 10.4), (5, 9.9))
    
    # NEW: Load ROM thresholds
    draw_cylinder(ax, 5, 9.4, 3.5, 0.7, 'Load Patient ROM\nThresholds')
    draw_arrow(ax, (5, 9.05), (5, 8.55))
    
    # Equipment
    draw_rect(ax, 5, 8.1, 3.5, 0.6, 'Equipment request\nscreen')
    draw_arrow(ax, (5, 7.8), (5, 7.3))
    
    # Wait for hand
    draw_oval(ax, 5, 6.9, 3, 0.6, 'Wait for right\nhand up')
    draw_arrow(ax, (5, 6.6), (5, 6.1))
    
    # Exercise explanation
    draw_rect(ax, 5, 5.6, 3.8, 0.7, 'Exercise explanation\nand demonstration')
    draw_arrow(ax, (5, 5.25), (5, 4.75))
    
    # Exercise process
    draw_rect(ax, 5, 4.3, 3.8, 0.7, 'Exercise process\n(uses ROM thresholds)')
    draw_arrow(ax, (5, 3.95), (5, 3.45))
    
    # Decision: more exercises?
    draw_diamond(ax, 5, 2.9, 1.2, 'More\nexercises?')
    
    # Yes - loop back
    draw_arrow_with_label(ax, (5.6, 2.9), (7.5, 2.9), 'Yes', (0, 0.2))
    ax.annotate('', xy=(7.5, 5.6), xytext=(7.5, 2.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(6.9, 5.6), xytext=(7.5, 5.6),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    
    # No - continue
    draw_arrow_with_label(ax, (5, 2.3), (5, 1.8), 'No', (0.3, 0))
    
    # Effort rating
    draw_rect(ax, 5, 1.4, 3, 0.5, 'Effort rating screen')
    draw_arrow(ax, (5, 1.15), (5, 0.75))
    
    # End
    draw_oval(ax, 5, 0.45, 2.5, 0.5, 'Back to Entry\nScreen')
    
    # Annotation for new component
    ax.annotate('NEW:\nROM Module', xy=(6.8, 9.4), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#ffffcc', edgecolor='black'))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, '3_Training_Flow_with_ROM.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {filepath}")


# =============================================================================
# DIAGRAM 4: Exercise Process with Personalized Thresholds
# =============================================================================
def create_exercise_process():
    """Create exercise process flowchart with threshold checking."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Start
    draw_oval(ax, 5, 11.3, 2.5, 0.6, 'Exercise starts')
    draw_arrow(ax, (5, 11), (5, 10.5))
    
    # Load thresholds
    draw_cylinder(ax, 5, 10, 3.5, 0.7, 'Load exercise\nthresholds')
    draw_arrow(ax, (5, 9.65), (5, 9.15))
    
    # For each frame
    draw_rect(ax, 5, 8.7, 3, 0.6, 'For each frame:')
    draw_arrow(ax, (5, 8.4), (5, 7.9))
    
    # Get angle
    draw_rect(ax, 5, 7.5, 3.5, 0.6, 'Get current angle')
    draw_arrow(ax, (5, 7.2), (5, 6.6))
    
    # Decision: angle >= up_lb?
    draw_diamond(ax, 5, 6, 1.3, 'angle >=\nup_lb?')
    
    # Yes - mark up
    draw_arrow_with_label(ax, (5.65, 6), (7.5, 6), 'Yes', (0, 0.2))
    draw_rect(ax, 8.5, 6, 1.8, 0.5, 'flag = True', fontsize=9)
    
    # No - check down
    draw_arrow_with_label(ax, (5, 5.35), (5, 4.85), 'No', (0.25, 0))
    
    # Decision: angle <= down_ub?
    draw_diamond(ax, 5, 4.2, 1.4, 'angle <=\ndown_ub?')
    
    # Yes - check flag
    draw_arrow_with_label(ax, (5.7, 4.2), (7.2, 4.2), 'Yes', (0, 0.2))
    draw_diamond(ax, 8, 4.2, 1, 'flag?')
    
    # Count rep
    draw_arrow_with_label(ax, (8.5, 4.2), (9.3, 4.2), 'Yes', (0, 0.15))
    ax.text(9.3, 4.2, 'rep++\nflag=False', fontsize=8, ha='left', va='center')
    
    # No - continue
    draw_arrow_with_label(ax, (5, 3.5), (5, 3), 'No', (0.25, 0))
    
    # Give feedback
    draw_rect(ax, 5, 2.5, 3, 0.6, 'Give feedback')
    draw_arrow(ax, (5, 2.2), (5, 1.7))
    
    # Done?
    draw_diamond(ax, 5, 1.2, 1, 'Done?')
    
    # Loop back
    draw_arrow_with_label(ax, (4.5, 1.2), (2.5, 1.2), 'No', (0, 0.2))
    ax.annotate('', xy=(2.5, 7.5), xytext=(2.5, 1.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(3.25, 7.5), xytext=(2.5, 7.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    
    # End
    draw_arrow_with_label(ax, (5, 0.7), (5, 0.3), 'Yes', (0.25, 0))
    draw_oval(ax, 5, 0, 2, 0.5, 'End')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, '4_Exercise_Process.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {filepath}")


# =============================================================================
# DIAGRAM 5: ROM vs Regular Training Comparison
# =============================================================================
def create_comparison():
    """Create comparison between ROM and Regular training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    
    # LEFT: ROM Assessment
    ax1 = axes[0]
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('ROM Assessment Training', fontsize=14, fontweight='bold', pad=20)
    
    draw_oval(ax1, 3.5, 9.3, 2, 0.5, 'Start')
    draw_arrow(ax1, (3.5, 9.05), (3.5, 8.6))
    
    draw_rect(ax1, 3.5, 8.2, 3, 0.6, 'Select exercises')
    draw_arrow(ax1, (3.5, 7.9), (3.5, 7.4))
    
    draw_rect(ax1, 3.5, 7, 3.5, 0.6, 'Perform 10 repetitions\n(FREE movement)')
    draw_arrow(ax1, (3.5, 6.7), (3.5, 6.2))
    
    draw_rect(ax1, 3.5, 5.8, 3, 0.6, 'Record angles')
    draw_arrow(ax1, (3.5, 5.5), (3.5, 5))
    
    draw_rect(ax1, 3.5, 4.6, 3.2, 0.6, 'Find peaks/valleys')
    draw_arrow(ax1, (3.5, 4.3), (3.5, 3.8))
    
    draw_rect(ax1, 3.5, 3.4, 3.5, 0.6, 'Calculate thresholds\n(avg +/- 10 deg)')
    draw_arrow(ax1, (3.5, 3.1), (3.5, 2.6))
    
    draw_cylinder(ax1, 3.5, 2.2, 3, 0.6, 'Save to Excel')
    draw_arrow(ax1, (3.5, 1.9), (3.5, 1.4))
    
    draw_oval(ax1, 3.5, 1.1, 2, 0.5, 'End')
    
    # Key characteristics
    ax1.text(3.5, 0.3, 'Key: No feedback given', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='#e8e8e8'))
    
    # RIGHT: Regular Training
    ax2 = axes[1]
    ax2.set_xlim(0, 7)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Regular Training', fontsize=14, fontweight='bold', pad=20)
    
    draw_oval(ax2, 3.5, 9.3, 2, 0.5, 'Start')
    draw_arrow(ax2, (3.5, 9.05), (3.5, 8.6))
    
    draw_cylinder(ax2, 3.5, 8.2, 3.5, 0.6, 'Load ROM thresholds')
    draw_arrow(ax2, (3.5, 7.9), (3.5, 7.4))
    
    draw_rect(ax2, 3.5, 7, 3.5, 0.6, 'Perform exercise\n(GUIDED movement)')
    draw_arrow(ax2, (3.5, 6.7), (3.5, 6.2))
    
    draw_rect(ax2, 3.5, 5.8, 3, 0.6, 'Compare to\nthresholds')
    draw_arrow(ax2, (3.5, 5.5), (3.5, 5))
    
    draw_diamond(ax2, 3.5, 4.5, 0.8, 'In range?')
    
    # Yes
    draw_arrow_with_label(ax2, (3.9, 4.5), (5.5, 4.5), 'Yes', (0, 0.15))
    draw_rect(ax2, 6.2, 4.5, 1.3, 0.5, 'Count\nrep', fontsize=9)
    
    # No
    draw_arrow_with_label(ax2, (3.5, 4.1), (3.5, 3.6), 'No', (0.25, 0))
    draw_rect(ax2, 3.5, 3.2, 3, 0.6, 'Give corrective\nfeedback')
    draw_arrow(ax2, (3.5, 2.9), (3.5, 2.4))
    
    draw_diamond(ax2, 3.5, 1.9, 0.8, 'Done?')
    
    draw_arrow_with_label(ax2, (3.5, 1.5), (3.5, 1.1), 'Yes', (0.25, 0))
    draw_oval(ax2, 3.5, 0.8, 2, 0.5, 'End')
    
    # Loop
    draw_arrow_with_label(ax2, (3.1, 1.9), (1.5, 1.9), 'No', (0, 0.15))
    ax2.annotate('', xy=(1.5, 5.8), xytext=(1.5, 1.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax2.annotate('', xy=(2, 5.8), xytext=(1.5, 5.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    
    # Key characteristics
    ax2.text(3.5, 0.1, 'Key: Feedback based on\npersonal thresholds', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='#e8e8e8'))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, '5_ROM_vs_Regular_Training.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {filepath}")


# =============================================================================
# DIAGRAM 6: Data Flow Diagram
# =============================================================================
def create_data_flow():
    """Create data flow diagram showing ROM data storage."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'ROM Data Flow', ha='center', fontsize=14, fontweight='bold')
    
    # ROM Training
    draw_rect(ax, 2, 6, 2.5, 0.8, 'ROM Assessment\nTraining')
    
    # Arrow to processing
    draw_arrow(ax, (3.25, 6), (4.5, 6))
    ax.text(3.87, 6.2, 'angle data', fontsize=8, ha='center')
    
    # CalibrationLogic
    draw_rect(ax, 6, 6, 2.8, 0.8, 'CalibrationLogic.py\n(calculate thresholds)')
    
    # Arrow to Excel
    draw_arrow(ax, (7.4, 6), (8.7, 6))
    ax.text(8.05, 6.2, 'thresholds', fontsize=8, ha='center')
    
    # Excel storage
    draw_cylinder(ax, 10, 6, 2.2, 0.9, 'Patients.xlsx\nPatient_ROM sheet')
    
    # Arrow down to Settings
    draw_arrow(ax, (10, 5.55), (10, 4.5))
    
    # Settings
    draw_rect(ax, 10, 4, 2.5, 0.8, 's.patient_rom_limits\n(runtime)')
    
    # Arrow to Camera
    draw_arrow(ax, (8.75, 4), (7.4, 4))
    
    # Camera
    draw_rect(ax, 6, 4, 2.5, 0.8, 'Camera.py\nget_dynamic_threshold()')
    
    # Arrow to training
    draw_arrow(ax, (4.75, 4), (3.3, 4))
    ax.text(4, 4.2, 'thresholds', fontsize=8, ha='center')
    
    # Regular training
    draw_rect(ax, 2, 4, 2.4, 0.8, 'Regular Training\n(exercise functions)')
    
    # ExerciseConfig
    draw_cylinder(ax, 2, 2, 2.5, 0.8, 'ExerciseConfig.py\n(default thresholds)')
    
    # Arrow to Camera (fallback)
    draw_arrow(ax, (3.25, 2), (5, 3.6))
    ax.text(3.8, 2.6, 'fallback', fontsize=8, ha='center')
    
    # Legend
    ax.text(9, 1.5, 'Legend:', fontsize=10, fontweight='bold')
    draw_rect(ax, 9.3, 1, 1.2, 0.4, 'Process', fontsize=8)
    draw_cylinder(ax, 10.8, 1, 1.2, 0.4, 'Data', fontsize=8)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, '6_Data_Flow.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {filepath}")


# =============================================================================
# DIAGRAM 7: Sample Angle Data Visualization
# =============================================================================
def create_sample_data_viz():
    """Create visualization of angle data with peaks and valleys."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Generate sample data (simulating 4 repetitions)
    np.random.seed(42)
    t = np.linspace(0, 15, 300)
    
    # Create wave pattern
    base = 90 + 50 * np.sin(t * 1.2)
    noise = np.random.normal(0, 2, len(t))
    angles = base + noise
    
    # Manually identify peaks and valleys for clean visualization
    peaks_idx = [33, 85, 137, 189, 241]
    valleys_idx = [7, 59, 111, 163, 215, 267]
    
    # Plot
    ax.plot(t, angles, 'k-', linewidth=1.5, label='Angle Data')
    ax.scatter(t[peaks_idx], angles[peaks_idx], c='black', s=80, zorder=5, 
               marker='^', label='Peaks (UP position)')
    ax.scatter(t[valleys_idx], angles[valleys_idx], c='black', s=80, zorder=5,
               marker='v', label='Valleys (DOWN position)')
    
    # Calculate thresholds
    up_avg = np.mean(angles[peaks_idx])
    down_avg = np.mean(angles[valleys_idx])
    
    # Threshold zones
    ax.axhline(y=up_avg, color='black', linestyle='--', linewidth=1.5, label=f'UP avg = {up_avg:.0f}°')
    ax.fill_between(t, up_avg - 10, up_avg + 10, color='gray', alpha=0.2, label='UP threshold zone (±10°)')
    
    ax.axhline(y=down_avg, color='black', linestyle=':', linewidth=1.5, label=f'DOWN avg = {down_avg:.0f}°')
    ax.fill_between(t, down_avg - 10, down_avg + 10, color='gray', alpha=0.2, label='DOWN threshold zone (±10°)')
    
    # Labels
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_title('ROM Assessment: Detecting Peaks and Valleys', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Annotations
    ax.annotate(f'UP avg: {up_avg:.0f}°', xy=(14, up_avg), fontsize=10, va='center')
    ax.annotate(f'DOWN avg: {down_avg:.0f}°', xy=(14, down_avg), fontsize=10, va='center')
    ax.annotate(f'ROM Range: {up_avg - down_avg:.0f}°', xy=(7.5, 90), fontsize=11,
                ha='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, '7_Sample_Data_Analysis.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {filepath}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("Generating ROM Module Diagrams (draw.io style)")
    print("="*60)
    
    create_rom_process_flow()
    create_threshold_algorithm()
    create_training_flow()
    create_exercise_process()
    create_comparison()
    create_data_flow()
    create_sample_data_viz()
    
    print("="*60)
    print(f"All diagrams saved to: {OUTPUT_FOLDER}/")
    print("="*60)
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_FOLDER)):
        if f.endswith('.png'):
            print(f"  - {f}")

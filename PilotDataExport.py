"""
PilotDataExport.py
Export pilot study data to a clean, analysis-ready Excel format.

PILOT STUDY: Validating ROM Module
==================================
Goal: Prove that the ROM module works correctly:
1. ROM thresholds are measured reasonably
2. Personalized thresholds are applied in regular training
3. Repetitions are counted correctly

PILOT PARTICIPANTS: IDs 10-19 only
All other patient IDs are excluded from pilot analysis.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

# ==================== PILOT CONFIGURATION ====================
PILOT_PARTICIPANT_IDS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# No exercise limitation - each participant can do any exercises selected by physiotherapist

def is_pilot_participant(patient_id) -> bool:
    """Check if a patient ID belongs to the pilot study."""
    try:
        pid = int(patient_id)
        return pid in PILOT_PARTICIPANT_IDS
    except (ValueError, TypeError):
        return False


def export_pilot_summary(output_folder: str = "Pilot_Data"):
    """
    Export all pilot data to a clean Excel file for analysis.
    
    Creates:
    - Pilot_Summary_[date].xlsx with multiple sheets:
      - Participants - Basic participant info
      - ROM_Measurements - All ROM measurements (long format)
      - Training_Sessions - All training session results
    """
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("="*60)
    print(f"[Pilot Export] Starting data export...")
    print(f"[Pilot Export] Output folder: {output_folder}")
    print("="*60)
    
    # Output Excel file
    excel_file = os.path.join(output_folder, f"Pilot_Summary_{timestamp}.xlsx")
    
    # Create Excel writer
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        # Flag to track if we wrote any sheets
        sheets_written = False
        
        # ==================== 1. PARTICIPANTS SUMMARY ====================
        try:
            df_patients = pd.read_excel("Patients.xlsx", sheet_name="patients_details")
            
            # Select relevant columns
            cols_to_keep = ['ID', 'gender', 'number of repetitions in each exercise', 'rate']
            available_cols = [c for c in cols_to_keep if c in df_patients.columns]
            df_participants = df_patients[available_cols].copy()
            df_participants.rename(columns={
                'ID': 'participant_id',
                'number of repetitions in each exercise': 'reps_per_exercise'
            }, inplace=True)
            
            # Filter only pilot participants (IDs 10-19)
            df_participants = df_participants[df_participants['participant_id'].apply(is_pilot_participant)]
            
            if len(df_participants) > 0:
                df_participants.to_excel(writer, sheet_name='Participants', index=False)
                sheets_written = True
                print(f"[Pilot Export] OK - Sheet 'Participants': {len(df_participants)} pilot participants")
            else:
                print(f"[Pilot Export] WARNING - No pilot participants found (IDs 10-19)")
            
        except Exception as e:
            print(f"[Pilot Export] ERROR - Error exporting participants: {e}")
    
        # ==================== 2. ROM MEASUREMENTS (LONG FORMAT) ====================
        try:
            df_rom = pd.read_excel("Patients.xlsx", sheet_name="Patient_ROM")
            
            # Convert wide format to long format for easier analysis
            rom_records = []
            
            for idx, row in df_rom.iterrows():
                patient_id = row.get('patient_id')
                if pd.isna(patient_id):
                    continue
                
                # Filter only pilot participants (IDs 10-19)
                if not is_pilot_participant(patient_id):
                    continue
                    
                # Parse each column
                for col in df_rom.columns:
                    if col in ['patient_id', 'rom_last_updated']:
                        continue
                        
                    value = row.get(col)
                    if pd.isna(value):
                        continue
                    
                    # Parse column name: exercise_side_position_metric
                    # e.g., "band_open_arms_right_up_avg"
                    parts = col.rsplit('_', 3)
                    if len(parts) >= 4:
                        exercise = '_'.join(parts[:-3])
                        side = parts[-3]
                        position = parts[-2]
                        metric = parts[-1]
                        
                        rom_records.append({
                            'participant_id': int(patient_id),
                            'exercise': exercise,
                            'side': side,
                            'position': position,
                            'metric': metric,
                            'value': round(value, 2),
                            'last_updated': row.get('rom_last_updated', '')
                        })
            
            if rom_records:
                df_rom_long = pd.DataFrame(rom_records)
                df_rom_long.to_excel(writer, sheet_name='ROM_Measurements', index=False)
                sheets_written = True
                print(f"[Pilot Export] OK - Sheet 'ROM_Measurements': {len(df_rom_long)} measurements")
            else:
                print("[Pilot Export] WARNING - No ROM data for pilot participants (IDs 10-19)")
                
        except Exception as e:
            print(f"[Pilot Export] ERROR - Error exporting ROM data: {e}")
    
        # ==================== 3. TRAINING SESSIONS ====================
        try:
            training_records = []
            
            # Scan all patient folders for training data
            patients_folder = "Patients"
            if os.path.exists(patients_folder):
                for patient_id in os.listdir(patients_folder):
                    # Filter only pilot participants (IDs 10-19)
                    if not is_pilot_participant(patient_id):
                        continue
                    
                    patient_path = os.path.join(patients_folder, patient_id)
                    trainings_path = os.path.join(patient_path, "Trainings")
                    
                    if os.path.isdir(trainings_path):
                        for session in os.listdir(trainings_path):
                            session_path = os.path.join(trainings_path, session)
                            if os.path.isdir(session_path):
                                # Each subfolder is an exercise
                                for exercise in os.listdir(session_path):
                                    exercise_path = os.path.join(session_path, exercise)
                                    if os.path.isdir(exercise_path):
                                        training_records.append({
                                            'participant_id': int(patient_id),
                                            'session_date': session,
                                            'exercise': exercise,
                                            'session_type': 'regular'
                                        })
                    
                    # Also check ROM assessments
                    rom_path = os.path.join(patient_path, "ROM_Assessments")
                    if os.path.isdir(rom_path):
                        for session in os.listdir(rom_path):
                            session_path = os.path.join(rom_path, session)
                            if os.path.isdir(session_path):
                                for exercise in os.listdir(session_path):
                                    exercise_path = os.path.join(session_path, exercise)
                                    if os.path.isdir(exercise_path):
                                        training_records.append({
                                            'participant_id': int(patient_id),
                                            'session_date': session,
                                            'exercise': exercise,
                                            'session_type': 'ROM'
                                        })
            
            if training_records:
                df_training = pd.DataFrame(training_records)
                df_training.to_excel(writer, sheet_name='Training_Sessions', index=False)
                sheets_written = True
                print(f"[Pilot Export] OK - Sheet 'Training_Sessions': {len(df_training)} sessions")
            else:
                print("[Pilot Export] WARNING - No training data for pilot participants (IDs 10-19)")
                
        except Exception as e:
            print(f"[Pilot Export] ERROR - Error exporting training data: {e}")
    
        # ==================== 4. ANALYSIS SUMMARY ====================
        try:
            analysis_data = []
            
            # If we have ROM data, create analysis summary
            if rom_records:
                df_rom_analysis = pd.DataFrame(rom_records)
                
                # Group by exercise and calculate statistics
                for exercise in df_rom_analysis['exercise'].unique():
                    ex_data = df_rom_analysis[df_rom_analysis['exercise'] == exercise]
                    
                    # Get UP avg values
                    up_avgs = ex_data[(ex_data['position'] == 'up') & (ex_data['metric'] == 'avg')]['value']
                    down_avgs = ex_data[(ex_data['position'] == 'down') & (ex_data['metric'] == 'avg')]['value']
                    
                    if len(up_avgs) > 0 and len(down_avgs) > 0:
                        analysis_data.append({
                            'exercise': exercise,
                            'num_participants': len(up_avgs) // 2,  # divided by 2 for left/right
                            'up_avg_mean': round(up_avgs.mean(), 1),
                            'up_avg_std': round(up_avgs.std(), 1) if len(up_avgs) > 1 else 0,
                            'down_avg_mean': round(down_avgs.mean(), 1),
                            'down_avg_std': round(down_avgs.std(), 1) if len(down_avgs) > 1 else 0,
                            'rom_range': round(up_avgs.mean() - down_avgs.mean(), 1),
                            'status': '✓ Valid' if 40 < (up_avgs.mean() - down_avgs.mean()) < 120 else '⚠ Check'
                        })
                
                if analysis_data:
                    df_analysis = pd.DataFrame(analysis_data)
                    df_analysis.to_excel(writer, sheet_name='Analysis_Summary', index=False)
                    print(f"[Pilot Export] OK - Sheet 'Analysis_Summary': {len(analysis_data)} exercises analyzed")
                    
        except Exception as e:
            print(f"[Pilot Export] WARNING - Could not create analysis summary: {e}")
        
        # If no sheets were written, create a placeholder sheet
        if not sheets_written:
            placeholder_df = pd.DataFrame({
                'Status': ['No pilot participants registered yet (IDs 10-19)'],
                'Instructions': ['Register participants and run ROM + Regular training first']
            })
            placeholder_df.to_excel(writer, sheet_name='Info', index=False)
            print("[Pilot Export] INFO - Created placeholder sheet (no pilot data yet)")
    
    print("="*60)
    print(f"[Pilot Export] OK - Export complete!")
    print(f"[Pilot Export] Excel file: {excel_file}")
    print("="*60)
    
    return excel_file


def get_pilot_status():
    """
    Get a quick status of pilot participants.
    """
    print("\n" + "="*60)
    print("PILOT STATUS - Participants 10-19")
    print("="*60)
    
    patients_folder = "Patients"
    for pid in PILOT_PARTICIPANT_IDS:
        patient_path = os.path.join(patients_folder, str(pid))
        if os.path.exists(patient_path):
            # Count ROM and regular trainings
            rom_count = 0
            training_count = 0
            
            rom_path = os.path.join(patient_path, "ROM_Assessments")
            if os.path.isdir(rom_path):
                rom_count = len([d for d in os.listdir(rom_path) if os.path.isdir(os.path.join(rom_path, d))])
            
            training_path = os.path.join(patient_path, "Trainings")
            if os.path.isdir(training_path):
                training_count = len([d for d in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, d))])
            
            print(f"  [OK] Participant {pid}: {rom_count} ROM, {training_count} Training sessions")
        else:
            print(f"  [  ] Participant {pid}: Not registered yet")
    
    print("="*60 + "\n")


def analyze_pilot_results():
    """
    Generate a comprehensive analysis report for the pilot study.
    Run this AFTER all 10 participants have completed both ROM and regular training.
    """
    print("="*70)
    print("PILOT STUDY ANALYSIS - ROM Module Validation")
    print("="*70)
    
    excel_file = export_pilot_summary()
    if not excel_file:
        return
    
    # Load the data
    try:
        df_rom = pd.read_excel(excel_file, sheet_name='ROM_Measurements')
        df_training = pd.read_excel(excel_file, sheet_name='Training_Sessions')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"\nPILOT ANALYSIS RESULTS")
    print("-"*50)
    
    # ========== ANALYSIS 1: ROM Thresholds Validity ==========
    print(f"\nCRITERION 1: Are ROM thresholds reasonable?")
    print("-"*50)
    
    if len(df_rom) > 0:
        # Get UP avg values
        up_values = df_rom[(df_rom['position'] == 'up') & (df_rom['metric'] == 'avg')]['value']
        down_values = df_rom[(df_rom['position'] == 'down') & (df_rom['metric'] == 'avg')]['value']
        
        if len(up_values) > 0 and len(down_values) > 0:
            print(f"UP position (avg):   Mean={up_values.mean():.1f}°, Std={up_values.std():.1f}°")
            print(f"DOWN position (avg): Mean={down_values.mean():.1f}°, Std={down_values.std():.1f}°")
            rom_range = up_values.mean() - down_values.mean()
            print(f"Average ROM range:   {rom_range:.1f}°")
            
            # Expected range for healthy adults
            if 30 < rom_range < 120:
                print("[PASSED] ROM values within expected range (30-120 degrees)")
            else:
                print("[REVIEW] ROM values outside expected range")
        else:
            print("No UP/DOWN measurements found")
    else:
        print("No ROM data available")
    
    # ========== ANALYSIS 2: Variance Between Participants ==========
    print(f"\nCRITERION 2: Personalization is meaningful?")
    print("-"*50)
    
    if len(df_rom) > 0:
        # Check variance between participants for avg values
        avg_values = df_rom[df_rom['metric'] == 'avg']
        if len(avg_values) > 0:
            participant_means = avg_values.groupby('participant_id')['value'].mean()
            variance = participant_means.var()
            print(f"Variance between participants: {variance:.2f}")
            print(f"Number of participants: {len(participant_means)}")
            
            if len(participant_means) > 1 and variance > 10:
                print("[PASSED] Different participants get different thresholds")
            elif len(participant_means) <= 1:
                print("[INFO] Need more participants to evaluate")
            else:
                print("[REVIEW] Low variance (may need more diverse participants)")
    
    # ========== ANALYSIS 3: Training Sessions Completed ==========
    print(f"\nCRITERION 3: Both sessions completed?")
    print("-"*50)
    
    if len(df_training) > 0:
        # Count ROM vs regular sessions per participant
        rom_sessions = df_training[df_training['session_type'] == 'ROM']
        regular_sessions = df_training[df_training['session_type'] == 'regular']
        
        rom_participants = rom_sessions['participant_id'].nunique()
        regular_participants = regular_sessions['participant_id'].nunique()
        
        print(f"Participants with ROM assessment: {rom_participants}")
        print(f"Participants with regular training: {regular_participants}")
        
        if rom_participants >= 1 and regular_participants >= 1:
            print("[PASSED] Both session types completed")
        else:
            print("[INFO] Need both ROM and regular sessions")
    else:
        print("No training data available")
    
    # ========== SUMMARY ==========
    print(f"\n{'='*70}")
    print("SUMMARY FOR REPORT")
    print("="*70)
    print("""
Key findings to include in your report:

1. ROM MEASUREMENT ACCURACY
   - The system successfully measured range of motion for X exercises
   - Mean UP position: XX.X° (SD: X.X°)
   - Mean DOWN position: XX.X° (SD: X.X°)
   - Average ROM range: XX.X°

2. PERSONALIZATION
   - Each participant received individualized thresholds
   - Variance between participants: X.X° (indicates personalization works)

3. SYSTEM FUNCTIONALITY
   - ROM assessment: Completed for X participants
   - Regular training: Completed for X participants
   - Thresholds successfully applied in regular training

CONCLUSION: The ROM module [WORKS/NEEDS REVIEW] as designed.
""")
    print("="*70)


def print_pilot_protocol():
    """Print the recommended pilot study protocol."""
    print("""
============================================================================
                         PILOT STUDY PROTOCOL
                        ROM Module Validation
============================================================================

  GOAL: Validate that the ROM module works correctly
  PARTICIPANTS: 10 healthy adults (IDs: 10, 11, 12, ... 19)
  DURATION: 10-15 minutes per participant

============================================================================

  STEP-BY-STEP FOR EACH PARTICIPANT:
  -----------------------------------

  1. REGISTER PARTICIPANT
     - Open the system (python main.py)
     - Add new patient with ID: 10, 11, 12...
     - Select exercises (any exercises the participant can do)

  2. ROM ASSESSMENT
     - Click "ROM Training" button
     - Participant performs 10 reps per exercise
     - Move freely - NO feedback given
     - System records range of motion

  3. REGULAR TRAINING
     - Click "Start Training" button
     - Participant performs exercises with robot
     - Feedback IS given (based on ROM thresholds)
     - System counts repetitions

  4. DOCUMENT RESULTS
     - Note any issues observed
     - Check that repetitions were counted

============================================================================

  AFTER ALL 10 PARTICIPANTS:
  --------------------------

  Run in terminal:
      python PilotDataExport.py analyze

  This generates the analysis report for your thesis!

============================================================================
""")


# ==================== MAIN ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # Run full analysis
        analyze_pilot_results()
    elif len(sys.argv) > 1 and sys.argv[1] == "protocol":
        # Print protocol
        print_pilot_protocol()
    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        # Just show status
        get_pilot_status()
    else:
        # Default: show status and export
        print("\n" + "="*60)
        print("PILOT DATA EXPORT TOOL")
        print("Pilot Participants: IDs 10-19")
        print("="*60)
        print("\nUsage:")
        print("  python PilotDataExport.py           - Export data")
        print("  python PilotDataExport.py protocol  - Show pilot protocol")
        print("  python PilotDataExport.py analyze   - Run full analysis")
        print("  python PilotDataExport.py status    - Check participant status")
        print()
        
        # Show pilot status
        get_pilot_status()
        
        # Export current data
        print("Exporting pilot data to Excel...")
        export_pilot_summary()
        
        print("\nOpen 'Pilot_Data' folder for your Excel file!")


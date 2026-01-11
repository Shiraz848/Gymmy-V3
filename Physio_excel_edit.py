"""
Physiotherapist Management Tool
Manage physiotherapist IDs in Physiotherapists.xlsx
"""

import pandas as pd
import os

EXCEL_FILE = "Physiotherapists.xlsx"
SHEET_NAME = "details"


def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def show_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("🏥 PHYSIOTHERAPIST MANAGEMENT SYSTEM")
    print("="*60)
    print("\n1. 📋 View All Physiotherapists")
    print("2. ➕ Add New Physiotherapist")
    print("3. ❌ Delete Physiotherapist")
    print("4. 🚪 Exit")
    print("\n" + "="*60)


def view_physiotherapists():
    """Display all physiotherapists in the Excel file"""
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
        
        print("\n" + "="*60)
        print("📋 CURRENT PHYSIOTHERAPISTS:")
        print("="*60)
        
        if df.empty:
            print("\n❌ No physiotherapists found in the system.")
        else:
            print(f"\n{'#':<5} {'ID':<15} {'First Name':<20} {'Last Name':<20}")
            print("-"*60)
            for idx, row in df.iterrows():
                physio_id = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else "N/A"
                first_name = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else "N/A"
                last_name = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else "N/A"
                print(f"{idx+1:<5} {physio_id:<15} {first_name:<20} {last_name:<20}")
            
            print(f"\n✅ Total: {len(df)} physiotherapist(s)")
        
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File '{EXCEL_FILE}' not found!")
        print("   Make sure you're in the correct directory.")
    except Exception as e:
        print(f"\n❌ ERROR: Could not read file: {e}")


def add_physiotherapist():
    """Add a new physiotherapist to the Excel file"""
    try:
        print("\n" + "="*60)
        print("➕ ADD NEW PHYSIOTHERAPIST")
        print("="*60)
        
        # Get user input
        print("\n📝 Enter details (press Enter to skip optional fields):")
        physio_id = input("\n🆔 ID Number (required): ").strip()
        
        if not physio_id:
            print("\n❌ ERROR: ID is required!")
            return
        
        # Check if ID already exists
        df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
        df['ID_str'] = df.iloc[:, 0].astype(str).str.strip()
        
        if physio_id in df['ID_str'].values:
            print(f"\n❌ ERROR: ID '{physio_id}' already exists in the system!")
            return
        
        first_name = input("👤 First Name (optional): ").strip()
        last_name = input("👤 Last Name (optional): ").strip()
        
        # Create new row
        new_row = pd.DataFrame({
            'ID': [physio_id],
            'first name': [first_name if first_name else ''],
            'last name': [last_name if last_name else '']
        })
        
        # Append to existing dataframe
        df_original = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
        df_updated = pd.concat([df_original, new_row], ignore_index=True)
        
        # Save to Excel
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='w') as writer:
            df_updated.to_excel(writer, sheet_name=SHEET_NAME, index=False)
        
        print("\n" + "="*60)
        print("✅ SUCCESS! Physiotherapist added:")
        print(f"   🆔 ID: {physio_id}")
        print(f"   👤 Name: {first_name} {last_name}")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File '{EXCEL_FILE}' not found!")
    except Exception as e:
        print(f"\n❌ ERROR: Could not add physiotherapist: {e}")


def delete_physiotherapist():
    """Delete a physiotherapist from the Excel file"""
    try:
        # First, show all physiotherapists
        view_physiotherapists()
        
        print("\n" + "="*60)
        print("❌ DELETE PHYSIOTHERAPIST")
        print("="*60)
        
        # Get ID to delete
        physio_id = input("\n🆔 Enter ID to delete (or press Enter to cancel): ").strip()
        
        if not physio_id:
            print("\n⚠️ Deletion cancelled.")
            return
        
        # Read current data
        df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
        df['ID_str'] = df.iloc[:, 0].astype(str).str.strip()
        
        # Check if ID exists
        if physio_id not in df['ID_str'].values:
            print(f"\n❌ ERROR: ID '{physio_id}' not found in the system!")
            return
        
        # Confirm deletion
        matching_row = df[df['ID_str'] == physio_id].iloc[0]
        first_name = str(matching_row.iloc[1]) if pd.notna(matching_row.iloc[1]) else "N/A"
        last_name = str(matching_row.iloc[2]) if pd.notna(matching_row.iloc[2]) else "N/A"
        
        print(f"\n⚠️ You are about to delete:")
        print(f"   🆔 ID: {physio_id}")
        print(f"   👤 Name: {first_name} {last_name}")
        
        confirm = input("\n⚠️ Are you sure? Type 'yes' to confirm: ").strip().lower()
        
        if confirm != 'yes':
            print("\n⚠️ Deletion cancelled.")
            return
        
        # Delete the row
        df_original = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
        df_original['ID_str'] = df_original.iloc[:, 0].astype(str).str.strip()
        df_updated = df_original[df_original['ID_str'] != physio_id].copy()
        df_updated = df_updated.drop(columns=['ID_str'])
        
        # Save to Excel
        with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='w') as writer:
            df_updated.to_excel(writer, sheet_name=SHEET_NAME, index=False)
        
        print("\n" + "="*60)
        print(f"✅ SUCCESS! Physiotherapist with ID '{physio_id}' has been deleted.")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File '{EXCEL_FILE}' not found!")
    except Exception as e:
        print(f"\n❌ ERROR: Could not delete physiotherapist: {e}")


def main():
    """Main program loop"""
    while True:
        show_menu()
        choice = input("\n👉 Enter your choice (1-4): ").strip()
        
        if choice == '1':
            view_physiotherapists()
            input("\n📌 Press Enter to continue...")
        
        elif choice == '2':
            add_physiotherapist()
            input("\n📌 Press Enter to continue...")
        
        elif choice == '3':
            delete_physiotherapist()
            input("\n📌 Press Enter to continue...")
        
        elif choice == '4':
            print("\n" + "="*60)
            print("👋 Goodbye! Exiting Physiotherapist Management System.")
            print("="*60)
            break
        
        else:
            print("\n❌ Invalid choice! Please enter a number between 1 and 4.")
            input("\n📌 Press Enter to continue...")


if __name__ == "__main__":
    # Check if Excel file exists
    if not os.path.exists(EXCEL_FILE):
        print(f"\n❌ ERROR: '{EXCEL_FILE}' not found!")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Please make sure you're running this script from the correct folder.")
    else:
        main()


import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import re
import math
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import Excel
import Settings as s
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import fitz  # PyMuPDF
from io import BytesIO



def reverse_hebrew_sequence_in_text(text):
    # Regular expression to detect Hebrew sequences (words and spaces between them)
    hebrew_sequence_pattern = re.compile(r'([\u0590-\u05FF\s]+)')

    # Replace Hebrew sequences with their reversed version
    def reverse_match(match):
        return match.group()[::-1]

    return hebrew_sequence_pattern.sub(reverse_match, text)

def create_table_for_patients_email():
    # Create table data
    table_data = {}
    for exercise, rep_num in s.ex_list.items():
        formatted_ex_1_name = Excel.get_name_by_exercise(exercise)
        # Reverse only the Hebrew sequences in the text
        formatted_ex_1_name = reverse_hebrew_sequence_in_text(formatted_ex_1_name)
        table_data[formatted_ex_1_name] = str(rep_num)

    if table_data:
        # Define the name of the file for saving the table image
        timestamp = s.starts_and_ends_of_stops[0]
        formatted_dt = datetime.fromtimestamp(timestamp).strftime("%d-%m-%Y %H-%M-%S")


        # Create and open the folder to save the tables
        folder_path = f'Patients/{s.chosen_patient_ID}/Table_to_Patient_Email'
        Excel.create_and_open_folder(folder_path)

        # Prepare data for the table
        table_df = pd.DataFrame(list((value, key) for key, value in table_data.items()), columns=[f'מספר חזרות מוצלחות מתוך {str(s.rep)[::-1]}'[::-1], 'שם התרגיל'[::-1]])

        # Dynamically adjust the figure size based on content
        max_text_width = max([len(str(value)) for value in table_data.values()] + [len(str(key)) for key in table_data.keys()])
        max_including_title = max(len(f'מספר חזרות מוצלחות מתוך {str(s.rep)[::-1]}'[::-1]), max_text_width)
        fig_width = max(4, max_including_title * 0.2 + 1)  # Adjust figure width based on max text length
        fig_height = max(1, len(table_df) * 0.5 + 1)  # Adjust figure height based on number of rows

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')

        # Add the table to the figure
        table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        # Dynamically adjust column widths
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header row
                cell.set_text_props(weight='bold', fontsize=12)  # Set bold and slightly larger font size
                cell.set_facecolor('#f0f0f0')  # Optional: Add a light gray background for headers
            cell.set_width(1.0 / len(table_df.columns))  # Dynamically set cell width

        timestamp = s.starts_and_ends_of_stops[0]
        start_dt = datetime.fromtimestamp(timestamp).strftime("%d-%m-%Y %H-%M-%S")

        # Save the table as an image
        filename = f'{folder_path}/{start_dt}.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        print(f"Table saved as {filename}")
        return filename  # Return the file path
    else:
        print("There is no data")
        return None  # Explicitly return None if no data


def email_to_patient():

    # Generate the table file
    table_file_path = create_table_for_patients_email()

    if table_file_path is None:
        print("Error: Table file creation failed.")
        return

    # Email configuration
    sender_email = 'yyy'
    receiver_email = s.email_of_patient
    password = ''  # Replace with your actual password or app password

    if s.email_of_patient is not None:
        try:
            with open(table_file_path, 'rb') as table_file:
                table_data = table_file.read()

            # Create message container
            message = MIMEMultipart("related")
            message['From'] = sender_email
            message['To'] = receiver_email
            message['Subject'] = 'לגימי יש משהו לומר לך!'
            name = 'יעל'

            # Create HTML content with embedded image
            html_content = f'''
            <html>
              <body style="direction: rtl; font-family: Arial, sans-serif; font-size: 14px; color: black;">
                <p>{name}, כל הכבוד על האימון היום! </p>
                <p>האימון שביצעת היום כלל {str(len(s.ex_list))} תרגילים</p>
                <p>דירגת את האימון בדרגת קושי:  {str(s.effort)} </p>
                <p>סיכום האימון:</p>
                <img src="cid:image" alt="Image" style="display: block; margin: 0 auto;">
                <br><br>
                <p style="font-family: Arial, sans-serif; font-size: 20px; font-weight: bold;">יישר כוח!</p>
              </body>
            </html>
            '''

            # Attach HTML content
            message.attach(MIMEText(html_content, 'html'))

            # Attach the PNG image as inline content
            png_image = MIMEImage(table_data, 'png')
            png_image.add_header('Content-ID', '<image>')
            message.attach(png_image)

            # Connect to SMTP server and send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()  # Secure the connection
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, message.as_string())
                print('Email sent successfully!')

        except FileNotFoundError:
            print("Error: Table file not found. Please check the file path.")
        except smtplib.SMTPAuthenticationError:
            print("Error: Unable to log in. Please check your email credentials.")
        except Exception as e:
            print(f"An unexpected error occurred while sending the email: {e}")
    else:
        print("Error: No email address provided for the patient.")

    # Continue execution even if email sending fails
    print("Proceeding with the next steps of the program.")



def get_ordered_images(image_list):
    """
    Reorder the image list according to the pattern requested:
    - 2 images: 1, 2
    - 4 images: 1, 3, 2, 4
    - 6 images: 1, 4, 2, 5, 3, 6
    """
    n = len(image_list)
    if n == 2:
        return [image_list[0], image_list[1]]
    elif n == 4:
        return [image_list[0], image_list[2], image_list[1], image_list[3]]
    elif n == 6:
        return [image_list[0], image_list[2], image_list[4], image_list[1], image_list[3], image_list[5]]
    else:
        return image_list  # If the number of images is not 2, 4, or 6, return as-is


def collect_images_from_folders(patient_id, exercise_name, timestamp):
    """
    Collect all the image file paths from the 'Graphs' and 'Tables' folders for a given exercise.
    The images are ordered by their number suffix as described.
    """
    # Define the base paths for the graphs and tables folders
    base_path_graphs = f"Patients/{patient_id}/Graphs/{exercise_name}/{timestamp}"
    base_path_tables = f"Patients/{patient_id}/Tables/{exercise_name}/{timestamp}"


    for attempt in range(3):
        try:
            # Collect all .jpeg images from the Graphs folder and .png images from the Tables folder
            graph_images = sorted([os.path.join(base_path_graphs, img) for img in os.listdir(base_path_graphs) if img.endswith('.jpeg')], key=lambda x: int(re.search(r'(\d+)\.jpeg$', x).group(1)))
            table_images = sorted([os.path.join(base_path_tables, img) for img in os.listdir(base_path_tables) if img.endswith('.png')], key=lambda x: int(re.search(r'(\d+)\.png$', x).group(1)))

            if graph_images and table_images:
                # Combine and reorder the images
                graph_images_ordered = get_ordered_images(graph_images)
                table_images_ordered = get_ordered_images(table_images)
                return table_images_ordered, graph_images_ordered

        except Exception:
            pass

        time.sleep(2)

    return [], []



def create_pdf():
    """
    Create a PDF with a global header, section headers, and fixed-size images, with the images
    ordered for specific layouts depending on the number of images in each section.
    """

    pdf_path, image_groups, section_headers, global_header_line2, global_header_line3, global_header_line4, global_header_line5, global_header_line6, global_header_line7, global_header_line8, global_header_line9= data_creation_to_create_pdf()

    # Register the Hebrew-compatible font
    pdfmetrics.registerFont(TTFont('Hebrew', "arial.ttf-master/arial.ttf"))

    # Create a canvas to generate the PDF
    pdf_canvas = canvas.Canvas(pdf_path, pagesize=letter)

    # Set the PDF page dimensions (letter size)
    width, height = letter

    number_of_headers = 8
    # Calculate the total height needed for the global header (3 lines of text with intervals)
    line_height = 0.5 * inch  # Height of each line plus some interval
    total_header_height = 3 * line_height

    # Calculate the starting Y position to vertically center the global header
    start_y_position = (height + total_header_height) / 2  + 200 # This centers the 3 lines of text

    # Set the font for the global header
    pdf_canvas.setFont("Hebrew", 20)


    # Line 2 (with interval)
    text_width_line2 = pdf_canvas.stringWidth(global_header_line2, "Hebrew", 20)
    x_position_line2 = (width - text_width_line2) / 2  # Center the second line horizontally
    pdf_canvas.drawString(x_position_line2, start_y_position , global_header_line2)

    # Line 3 (with interval)
    text_width_line3 = pdf_canvas.stringWidth(global_header_line3, "Hebrew", 20)
    x_position_line3 = (width - text_width_line3) / 2  # Center the third line horizontally
    pdf_canvas.drawString(x_position_line3, start_y_position - 1.5 * line_height, global_header_line3)

    # Line 4 (with interval)
    text_width_line4 = pdf_canvas.stringWidth(global_header_line4, "Hebrew", 20)
    x_position_line4 = (width - text_width_line4) / 2  # Center the third line horizontally
    pdf_canvas.drawString(x_position_line4, start_y_position - 3 * line_height, global_header_line4)

    # Line 5 (with interval)
    text_width_line5 = pdf_canvas.stringWidth(global_header_line5, "Hebrew", 20)
    x_position_line5 = (width - text_width_line5) / 2  # Center the third line horizontally
    pdf_canvas.drawString(x_position_line5, start_y_position - 4.5 * line_height, global_header_line5)

    # Line 6 (with interval)
    text_width_line6 = pdf_canvas.stringWidth(global_header_line6, "Hebrew", 20)
    x_position_line6 = (width - text_width_line6) / 2  # Center the third line horizontally
    pdf_canvas.drawString(x_position_line6, start_y_position - 6 * line_height, global_header_line6)

    pdf_canvas.setFont("Hebrew", 16)

    text_width_line7 = pdf_canvas.stringWidth(global_header_line7, "Hebrew", 16)
    x_position_line7 = (width - text_width_line7) / 2  # Center the third line horizontally
    pdf_canvas.drawString(x_position_line7, start_y_position - 7.5 * line_height, global_header_line7)

    if global_header_line8:
        number_of_headers+=1

        text_width_line8 = pdf_canvas.stringWidth(global_header_line8, "Hebrew", 16)
        x_position_line8 = (width - text_width_line8) / 2  # Center the third line horizontally
        pdf_canvas.drawString(x_position_line8, start_y_position - 9 * line_height, global_header_line8)

        text_width_line9 = pdf_canvas.stringWidth(global_header_line9, "Hebrew", 16)
        x_position_line9 = (width - text_width_line9) / 2  # Center the third line horizontally
        pdf_canvas.drawString(x_position_line9, start_y_position - 10.5 * line_height, global_header_line9)

    else:
        text_width_line9 = pdf_canvas.stringWidth(global_header_line9, "Hebrew", 16)
        x_position_line9 = (width - text_width_line9) / 2  # Center the third line horizontally
        pdf_canvas.drawString(x_position_line9, start_y_position - 9 * line_height, global_header_line9)

    # Start placing content after the header
    current_y_position = start_y_position - number_of_headers * line_height - inch  # Adjust Y position after the header

    # Set a consistent padding/margin between images and sections
    image_padding = 0.1 * inch  # Reduced padding between images
    section_padding = 0.25 * inch  # Padding between sections

    # Calculate the image size based on a 3-images-per-row layout
    images_per_row = 3
    margin_x = 0.4 * inch  # Reduced margin to make images bigger
    image_width = (width - 2 * margin_x - (images_per_row - 1) * image_padding) / images_per_row
    image_height = image_width * 2 / 3  # Maintain aspect ratio (3:2) for the images

    # Iterate over each section, placing the section header and corresponding images
    for section_index, image_paths in enumerate(image_groups):
        # Draw the section header only once
        section_header_printed = False

        # Determine the number of images in the current section
        total_images = len(image_paths)

        if total_images == 12:
            # Split the image list into two halves
            first_half = image_paths[:6]
            second_half = image_paths[6:]

            # Reorder the images to match the requested layout for 12 images
            reordered_images = [
                first_half[0], first_half[1], first_half[2],  # First row
                second_half[0], second_half[1], second_half[2],  # Second row
                first_half[3], first_half[4], first_half[5],  # Third row
                second_half[3], second_half[4], second_half[5]  # Fourth row
            ]
            images_per_row = 3

        elif total_images == 8:
            # Split the image list into two halves
            first_half = image_paths[:4]
            second_half = image_paths[4:]

            # Reorder the images to match the requested layout for 8 images
            reordered_images = [
                first_half[0], first_half[1],  # First row
                second_half[0], second_half[1],  # Second row
                first_half[2], first_half[3],  # Third row
                second_half[2], second_half[3]  # Fourth row
            ]
            images_per_row = 2  # For layout, but image size remains based on 3 per row

        elif total_images == 4:
            # Reorder the images to match the requested layout for 4 images
            reordered_images = [
                image_paths[0], image_paths[1],  # First row
                image_paths[2], image_paths[3]  # Second row
            ]
            images_per_row = 2  # For layout, but image size remains based on 3 per row

        else:
            # If the image count doesn't match the specific cases, keep the original order
            reordered_images = image_paths
            images_per_row = 3 if total_images >= 3 else total_images

        # Calculate the height needed for this section (including images and section header)

        rows_in_section = math.ceil(len(reordered_images) / images_per_row)
        total_section_height = rows_in_section * (image_height + image_padding) + section_padding + 20  # 20 for header

        # Check if there's enough space to fit the entire section on the current page
        if current_y_position - total_section_height < inch:
            # Not enough space, move to the next page
            pdf_canvas.showPage()  # Create a new page
            current_y_position = height - 1 * inch  # Reset Y position for the new page (starting from the top)
            section_header_printed = False  # Ensure the header is printed once when the new page is started

        # Print section header and images
        for i, image_path in enumerate(reordered_images):
            # Draw the section header from the right side of the page
            if not section_header_printed:
                # Set the Hebrew font and size for the section header
                pdf_canvas.setFont("Hebrew", 14)

                # Define the section header text
                section_header = section_headers[section_index]  # Example: 'כותרת'

                # Define the X position for right alignment (e.g., 1 inch from the right margin)
                right_margin = 1 * inch  # Margin from the right edge
                x_position = letter[0] - right_margin  # Page width minus the margin

                # Draw the text aligned to the right
                pdf_canvas.drawRightString(x_position, current_y_position, section_header)

                section_header_printed = True  # Mark that the section header has been printed
                current_y_position -= section_padding  # Adjust Y position after the section header

            # Calculate the total width of images in the current row to center them
            row_start_index = i // images_per_row * images_per_row  # Starting index of the current row
            images_in_row = min(images_per_row, len(reordered_images) - row_start_index)
            total_row_width = images_in_row * image_width + (images_in_row - 1) * image_padding
            x_start_position = (width - total_row_width) / 2  # Center the row of images

            # Calculate x and y positions for the image
            x_position = x_start_position + (i % images_per_row) * (image_width + image_padding)  # Adjust x to center images
            y_position_offset = (i // images_per_row) * (image_height + image_padding)  # Adjust y based on row index

            # Insert the image at the calculated position
            pdf_canvas.drawImage(image_path, x_position, current_y_position - y_position_offset - image_height,
                                 width=image_width, height=image_height)

            # Draw a border around the image
            pdf_canvas.rect(x_position, current_y_position - y_position_offset - image_height,
                            image_width, image_height, stroke=1, fill=0)

        # Adjust Y position for the next section
        current_y_position -= rows_in_section * (image_height + image_padding) + 50

    # Finalize the PDF
    pdf_canvas.save()
    return pdf_path


def data_creation_to_create_pdf():
    exercises = s.exercises_by_order
    image_groups = []
    section_headers = []

    training_length = round(Excel.calculate_training_length(),3)
    success_percent = float(s.number_of_repetitions_in_training) / s.max_repetitions_in_training

    timestamp = s.starts_and_ends_of_stops[0]
    formatted_dt = datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y %H:%M:%S")

    # Global header with 3 lines
    global_header_line2 = f'מספר מטופל: {s.chosen_patient_ID[::-1]}'[::-1]
    global_header_line3 = f'זמן האימון: {formatted_dt[::-1]}'[::-1]
    global_header_line4 = f'אורך האימון בדקות: {str(training_length)[::-1]}'[::-1]
    global_header_line5 = f'אחוז הצלחה באימון: {str(success_percent)[::-1]}'[::-1]
    global_header_line6= f'דירוג קושי של האימון: {str(s.effort)[::-1]}'[::-1]

    # Set up initial variables for email content
    did_paused = "לא"
    did_stopped = "הושלם"

    # Determine if the training was paused or stopped
    if s.stop_requested:
        did_stopped = "הופסק באמצע"

    if s.number_of_pauses > 0:
        did_paused = "כן"

    global_header_line7 = f"האם בוצעה הפסקה באימון? {did_paused}"[::-1]

    global_header_line8 = None
    if s.number_of_pauses != 0:
        global_header_line8 = f" בוצעו {str(s.number_of_pauses)[::-1]} הפסקות באימון "[::-1]

    global_header_line9 = f" האם האימון הופסק באמצע או הושלם? {did_stopped}"[::-1]


    for exercise_name in exercises:
        # Collect the images for each exercise
        start_dt = s.starts_and_ends_of_stops[0]
        formatted_dt = datetime.fromtimestamp(start_dt).strftime("%d-%m-%Y %H-%M-%S")

        table_images, graph_images = collect_images_from_folders(s.chosen_patient_ID, exercise_name,
                                                                 formatted_dt)


        # Add images and section headers to the lists (Tables first, then Graphs)
        image_groups.append(table_images + graph_images)  # Add all images for the exercise
        # formatted_ex_name= reverse_hebrew_sequence_in_text(Excel.get_name_by_exercise(exercise_name))
        exercise_name_reversed = reverse_hebrew_sequence_in_text(Excel.get_name_by_exercise(exercise_name))
        exercise_label_reversed = "שם התרגיל:"[::-1]
        exercise_equipment = f"{Excel.get_equipment(exercise_name)}"[::-1]
        equipment_label_reversed = "אביזר:"[::-1]
        successful_reps_label = reverse_hebrew_sequence_in_text(f":מספר חזרות מוצלחות")

        section_headers.append(
            f"{str(s.ex_list[exercise_name])} {successful_reps_label}   {exercise_equipment} {equipment_label_reversed}   {exercise_name_reversed} {exercise_label_reversed}")  # Section header for the exercise

    # Define the directory path where you want to save the PDF
    output_directory = f'Patients/{s.chosen_patient_ID}/PDF_to_Therapist_Email'

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define the full path including the filename
    output_path = os.path.join(output_directory, f'{formatted_dt}.pdf')

    # Create the PDF with images, headers, and a global title
    return output_path, image_groups, section_headers, global_header_line2, global_header_line3, global_header_line4, global_header_line5, global_header_line6, global_header_line7, global_header_line8, global_header_line9


def create_pdf_preview(pdf_path):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Check if there is a second page (page 1 in zero-indexed format)
    if pdf_document.page_count < 2:
        print("The PDF does not have a second page. No preview created.")
        pdf_document.close()
        return None  # Return None if there's no second page

    # Load the second page (page index 1)
    page = pdf_document.load_page(1)

    # Render page to a pixmap (an image representation)
    pix = page.get_pixmap()

    # Convert the pixmap to a PIL image
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Save the image to a BytesIO object
    image_io = BytesIO()
    image.save(image_io, format='JPEG')
    image_io.seek(0)

    # Close the PDF
    pdf_document.close()

    return image_io



def email_to_physical_therapist():
    # Define the PDF file path
    pdf_file_path = create_pdf()

    # Generate the PDF preview
    preview_image_io = create_pdf_preview(pdf_file_path)

    # Email configuration
    sender_email = 'yaelszn@gmail.com'
    receiver_emails_str = Excel.find_value_by_colName_and_userID("Patients.xlsx", "patients_details",
                                                                 s.chosen_patient_ID, "email of therapist")

    # Handle the case where receiver_emails_str is empty
    if not receiver_emails_str or receiver_emails_str.strip() == "":
        print("No recipient email found. Skipping email sending.")
        return  # Exit function without sending email

    # Convert the comma-separated string into a list
    receiver_emails = [email.strip() for email in receiver_emails_str.split(',') if email.strip()]

    # Email should not be sent if there are no valid recipients
    if not receiver_emails:
        print("No valid recipient emails found. Skipping email sending.")
        return

    password = ''

    try:
        # Create message container
        message = MIMEMultipart("related")
        message['From'] = sender_email
        message['To'] = ", ".join(receiver_emails)  # Display recipients as comma-separated
        message['Subject'] = f'{s.full_name} סיכום אימון - {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'

        # Set up initial variables for email content
        did_paused = "לא"
        did_stopped = "הושלם"

        # Determine if the training was paused or stopped
        if s.stop_requested:
            did_stopped = "הופסק באמצע"

        if s.number_of_pauses > 0:
            did_paused = "כן"

        html_content = f'''
        <html>
          <body style="direction: rtl; font-family: Arial, sans-serif; font-size: 14px; color: black;">
            <p> סיכום אימון של <b style="color: black;">{s.full_name}</b>, </p>
            <p> האם בוצעה הפסקה באימון?  <b style="color: black;">{did_paused}</b> </p>
        '''
        if s.number_of_pauses != 0:
            html_content += f'<p> בוצעו  <b style="color: black;">{s.number_of_pauses}</b> הפסקות באימון </p>'

        html_content += f'''
            <p> האם האימון הושלם או הופסק באמצע? <b style="color: black;">{did_stopped}</b> </p>
            <br><br>
            <p style="font-family: Arial, sans-serif; font-weight: bold; color: black;">סיכום האימון מופיע בקובץ המצורף</p>
            <p style="color: black;">תצוגה מקדימה של סיכום האימון:</p>
            <img src="cid:preview_image" alt="PDF Preview">
            <div style="height: 20px;"></div> <!-- Empty row with height 20px -->
          </body>
        </html>
        '''

        # Attach HTML content
        message.attach(MIMEText(html_content, 'html'))

        # Attach the PDF preview image as an inline image if it exists
        if preview_image_io:
            preview_image_io.seek(0)  # Reset the file pointer to the beginning
            preview_image = MIMEImage(preview_image_io.read(), _subtype="jpeg")
            preview_image.add_header('Content-ID', '<preview_image>')
            message.attach(preview_image)

        # Attach the PDF file
        with open(pdf_file_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
            pdf_attachment = MIMEApplication(pdf_data, _subtype="pdf")
            pdf_attachment.add_header('Content-Disposition', 'attachment', filename='summary.pdf')
            message.attach(pdf_attachment)

        # Connect to SMTP server and send the email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_emails, message.as_string())
            print('Email sent successfully!')

    except FileNotFoundError:
        print("Error: PDF file not found. Please check the file path.")
    except smtplib.SMTPAuthenticationError:
        print("Error: Unable to log in. Please check your email credentials.")
    except Exception as e:
        print(f"An unexpected error occurred while sending the email: {e}")

    # Continue execution even if email sending fails
    print("Proceeding with the next steps of the program.")



if __name__ == '__main__':
    s.rep=10
    s.email_of_patient = "yaelszn@gmail.com"
    s.full_name = "יעל שניידמן"
    s.chosen_patient_ID= "314808981"
    s.stop_requested = True
    s.starts_and_ends_of_stops=[datetime.now()]
    s.ex_list = {"ball_bend_elbows": 3,
                 "ball_raise_arms_above_head": 4,
                 "ball_open_arms_and_forward": 5}

    s.effort= 5

    email_to_patient()
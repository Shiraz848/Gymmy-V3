def __init__():

    # classes pointers
    global training
    global camera
    global robot
    global screen
    global continuous_audio
    global zed_camera

    global volume
    # global participant_code
    global training_workbook
    global training_workbook_path
    global ex_list
    # global Last_workbook #the workbook of the last time the patient did every exercise
    global success_worksheet
    global number_of_repetitions_in_training #how many successful repetitions the patient did in the whole training
    global patient_repetitions_counting_in_exercise #save the repetitions so Gymmy will also be able to know the number of repetition done
    global max_repetitions_in_training
    global current_level_of_patient
    global points_in_current_level_before_training

    # training variables
    global exercise_amount
    global rep
    global req_exercise
    global finish_workout
    global waved
    global success_exercise
    global calibration
    global gymmy_done
    global camera_done
    global robot_count
    global exercises_start
    global waved_has_tool #the participant has the tool in his hand and is ready to start
    #global demo_finish #the first demo of the exercise is done
    global ex_in_training
    global effort #list of chosen effort to each exercise
    global finished_effort #finished to grade the efforts of exercises
    global email_of_patient
    global is_second_repetition_or_more #is it the second time or more of the training in the same session?
    global stop_requested
    global starts_and_ends_of_stops #variables that has the times of the starts and the ends of the stops
    global choose_continue_or_not #did the user chose whether to continue or not, or did not do it yet
    global another_training_requested #did the patient requested for another training
    global did_training_paused #did the training was paused in the middle (but not stopped)

    # audio variables
    global audio_path

    # screen variables
    global picture_path
    global camera_num

    global chosen_patient_ID #the id of the patient the therpist chose to enter
    global excel_file_path_Patient
    global finished_training_adding_to_excel

    global exercises_by_order

    global ball_exercises_number
    global explanation_over
    global gymmy_finished_demo
    global full_name
    global robot_counter

    global reached_max_limit




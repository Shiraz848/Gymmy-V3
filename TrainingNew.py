import os
import threading
import time
import re
import random
import copy
import cv2
import pygame

import Email
from Camera import Camera
from Gymmy import Gymmy
from ScreenNew import Screen, FullScreenApp, Ball, Rubber_Band, Stick, NoTool, StartOfTraining, GoodbyePage, \
    EffortScale, EntrancePage, ExplanationPage, ExercisePage, Repeat_training_or_not, \
    Number_of_good_repetitions_page, ClappingPage, Weights, CalibrationScreen
import Settings as s
import Excel
import random
from Audio import say, get_wav_duration, ContinuousAudio, AdditionalAudio
from datetime import datetime



class Training(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):

        self.selected_exercises = []
        while not s.finish_program:
            s.exercise_name_repeated_explanation = None

            self.training_session()
            if s.finish_program:
                break

            self.finish_training()

        print("Training Done")

    def select_exercises(self, options_for_ex_in_training, exercise_pairs, max_exercises= 10,  max_per_category=4):
        
        # List of exercises that have Gymmy robot implementations
        # This prevents crashes when trying to run exercises without robot support
        # List of ALL exercises that have implementations in Camera.py and ExerciseConfig.py
        # Updated to include all exercises from ExerciseConfig
        SUPPORTED_EXERCISES = [
            # Ball exercises
            "ball_bend_elbows",
            "ball_raise_arms_above_head",
            "ball_switch",
            "ball_open_arms_and_forward",
            # Band exercises
            "band_open_arms",
            "band_open_arms_and_up",
            "band_up_and_lean",
            "band_straighten_left_arm_elbows_bend_to_sides",
            "band_straighten_right_arm_elbows_bend_to_sides",
            # Stick exercises
            "stick_bend_elbows",
            "stick_bend_elbows_and_up",
            "stick_raise_arms_above_head",
            "stick_switch",
            "stick_up_and_lean",
            # Weights exercises
            "weights_open_arms_and_forward",
            "weights_abduction",
            # No-tool exercises
            "notool_hands_behind_and_lean",
            "notool_right_hand_up_and_bend",
            "notool_left_hand_up_and_bend",
            "notool_raising_hands_diagonally",
            "notool_right_bend_left_up_from_side",
            "notool_left_bend_right_up_from_side",
        ]
        
        def filter_supported_exercises(exercises):
            """Filter out exercises that don't have Gymmy robot implementations"""
            filtered = []
            unsupported = []
            for ex in exercises:
                if ex in SUPPORTED_EXERCISES:
                    filtered.append(ex)
                else:
                    unsupported.append(ex)
            
            if unsupported:
                print(f"[Training] ⚠️  Skipping {len(unsupported)} unsupported exercises:")
                for ex in unsupported:
                    print(f"             - {ex} (no Gymmy robot implementation)")
            
            return filtered
        
        # Check if ROM Assessment mode is active
        if s.is_rom_assessment_mode:
            print("[Training] ROM Mode: Loading patient's selected exercises for ROM assessment")
            
            # Import here to avoid circular imports
            import ExerciseConfig
            
            # Get exercises that the physiotherapist selected for this patient
            patient_exercises = ExerciseConfig.get_patient_selected_exercises(s.chosen_patient_ID)
            
            if patient_exercises:
                # Filter out unsupported exercises
                patient_exercises = filter_supported_exercises(patient_exercises)
                
                if patient_exercises:
                    print(f"[Training] ROM Mode: Using {len(patient_exercises)} supported exercises:")
                    for ex in patient_exercises:
                        print(f"  - {ex}")
                    return patient_exercises
                else:
                    print("[Training] ROM Mode: No supported exercises found, using defaults")
            else:
                # Fallback to default exercises if no exercises found
                print("[Training] ROM Mode: No exercises found for patient, using defaults")
            
            # Default exercises (all verified to have Gymmy implementations)
            return [
                "ball_raise_arms_above_head",
                "band_open_arms",
                "ball_bend_elbows",
                "stick_switch",
                "notool_hands_behind_and_lean"
            ]

        # Filter exercises for regular training too (not just ROM mode)
        options_for_ex_in_training = filter_supported_exercises(options_for_ex_in_training)
        
        if len(options_for_ex_in_training) <= max_exercises:
            return options_for_ex_in_training  # If 10 or fewer exercises exist, use all

        selected_exercises = []
        exercise_counts_in_training = {category: 0 for category in ["ball", "band", "stick", "weights", "notool"]}
        pairs_list = {"ball": [], "band": [], "stick": [], "weights": [], "notool": []}
        count_pairs = 0
        not_pairs_list = {"ball": [], "band": [], "stick": [], "weights": [], "notool": []}
        count_not_pairs = 0

        exercise_counts_in_list_before_training = {category: 0 for category in
                                                   ["ball", "band", "stick", "weights", "notool"]}
        ex_in_training_copy = copy.deepcopy(options_for_ex_in_training)

        for ex in options_for_ex_in_training:
            was_in_pair = False
            if ex in ex_in_training_copy:  # If it was not removed because it was in a couple
                for pair in exercise_pairs:
                    if ex in pair:
                        was_in_pair = True

                        other_exercise = pair[1] if pair[0] == ex else pair[0]

                        if other_exercise in options_for_ex_in_training:
                            ex_in_training_copy.remove(ex)
                            ex_in_training_copy.remove(other_exercise)
                            count_pairs += 1

                            category = ex.split("_")[0]
                            exercise_counts_in_list_before_training[category] += 2
                            pairs_list[category].append((ex, other_exercise))


                        else:
                            ex_in_training_copy.remove(ex)
                            category = ex.split("_")[0]
                            exercise_counts_in_list_before_training[category] += 1
                            not_pairs_list[category].append(ex)
                            count_not_pairs += 1

                if not was_in_pair:
                    ex_in_training_copy.remove(ex)
                    category = ex.split("_")[0]
                    not_pairs_list[category].append(ex)
                    exercise_counts_in_list_before_training[category] += 1
                    count_not_pairs += 1

        while len(selected_exercises) < max_exercises:
            rnd_num = random.randint(1, (count_not_pairs + count_pairs))
            
            total_exercises_other_categories = sum(count for cat, count in exercise_counts_in_list_before_training.items() if cat != category)
            
            
            if rnd_num <= count_not_pairs:  # We will randomly choose from the exercises not in pairs
                all_not_pairs = [not_pair for not_pairs in not_pairs_list.values() for not_pair in not_pairs]
                exercise = random.choice(all_not_pairs)
                category = exercise.split("_")[0]

                if exercise_counts_in_training[category] < max_per_category:
                    selected_exercises.append(exercise)
                    not_pairs_list[category].remove(exercise)
                    exercise_counts_in_list_before_training[category] -= 1
                    exercise_counts_in_training[category] += 1
                    count_not_pairs -= 1

                else:
    

                    if total_exercises_other_categories == 0:
                        selected_exercises.append(exercise)
                        not_pairs_list[category].remove(exercise)
                        exercise_counts_in_list_before_training[category] -= 1
                        exercise_counts_in_training[category] += 1
                        count_not_pairs -= 1

                    else:
                        pass


            else:
                all_pairs = [pair for pairs in pairs_list.values() for pair in pairs]
                pair = random.choice(all_pairs)
                category = pair[0].split("_")[0]

                all_not_pairs = [not_pair for not_pairs in not_pairs_list.values() for not_pair in not_pairs]

                if len(selected_exercises) + 2 <= max_exercises:
                    if exercise_counts_in_training[category] + 1 < max_per_category:
                        selected_exercises.append(pair[0])
                        selected_exercises.append(pair[1])
                        pairs_list[category].remove(pair)
                        exercise_counts_in_list_before_training[category] -= 2
                        exercise_counts_in_training[category] += 2
                        count_pairs -= 1

                    else:
                        if total_exercises_other_categories == 0:
                            selected_exercises.append(pair[0])
                            selected_exercises.append(pair[1])
                            pairs_list[category].remove(pair)
                            exercise_counts_in_list_before_training[category] -= 2
                            exercise_counts_in_training[category] += 2
                            count_pairs -= 1


                elif len(all_not_pairs) == 0 and total_exercises_other_categories == 0:
                    # If remained only one or zero room, and no exercises from other categories are left, and not single exercises from this category are left
                    # Last resort: no singles anywhere and nothing in other categories.
                    # We got here precisely because there's no room for two.
                    remaining_slots = max_exercises - len(selected_exercises)

                    if remaining_slots == 1:
                        # Pick ONE member of the pair at random
                        chosen = random.choice(pair)
                        selected_exercises.append(chosen)

                        # Remove the pair, convert the leftover to a single
                        pairs_list[category].remove(pair)

                        # Update counters: consumed 1; one remains available as a single
                        exercise_counts_in_list_before_training[category] -= 1
                        exercise_counts_in_training[category] += 1
                        count_pairs -= 1

                    else:
                        # No slots left → stop to avoid overshooting
                        break

                else:
                    pass

        return selected_exercises

    def shuffle_exercises(self, selected_exercises):
        # **Shuffle but keep special pair together**
        special_pair = ("notool_right_bend_left_up_from_side", "notool_left_bend_right_up_from_side")

        if all(ex in selected_exercises for ex in special_pair):
            # Remove special pair from list before shuffling
            selected_exercises = [ex for ex in selected_exercises if ex not in special_pair]
            random.shuffle(selected_exercises)  # Shuffle the remaining exercises

            # Insert the special pair at a random position while keeping them together
            insert_index = random.randint(0, len(selected_exercises))
            selected_exercises[insert_index:insert_index] = list(special_pair)  # Insert the pair back together
        else:
            # If the special pair is not both present, just shuffle normally
            random.shuffle(selected_exercises)

        for ex in selected_exercises:
            print(ex)

        return selected_exercises


    def training_session(self):
        while s.ex_in_training==[] and not getattr(s, "is_rom_assessment_mode", False):
            time.sleep(0.1)

            if s.finish_program:
                break

        time.sleep(1)

        if not s.finish_program:
            s.turn_camera_on= True
            s.starts_and_ends_of_stops= []
            s.stop_requested = False

            time.sleep(1) #wait another second so that s.ex_in_training will contain all of the exercises
            # הגדרת זוגות התרגילים
            exercise_pairs = [
                ("band_straighten_left_arm_elbows_bend_to_sides", "band_straighten_right_arm_elbows_bend_to_sides"),
                ("notool_right_hand_up_and_bend", "notool_left_hand_up_and_bend"),
                ("notool_right_bend_left_up_from_side", "notool_left_bend_right_up_from_side"),
            ]


            if not s.is_second_repetition_or_more and not s.finish_program:

                selected_exercises_before_shuffle = self.select_exercises(s.ex_in_training, exercise_pairs)
                self.selected_exercises = self.shuffle_exercises(selected_exercises_before_shuffle)

                    # שמירת התרגילים שנבחרו
                s.ex_in_training = self.selected_exercises

                # הדפסת התרגילים שנבחרו
                for item in self.selected_exercises:
                    print(item)

                while not s.explanation_over:
                    time.sleep(0.5)

            s.starts_and_ends_of_stops.append(time.time())

            categories = ["ball", "stick", "notool", "band", "weights"]
            random.shuffle(categories)

            Excel.create_workbook_for_training() #create workbook in excel for this session

            s.exercises_by_order=[]


            s.req_exercise = "calibration"
            while not s.finished_calibration:
                time.sleep(0.0001)

                if s.stop_requested or s.finish_program:
                    break

            s.req_exercise = ""
            time.sleep(get_wav_duration("end_calibration"))

            s.general_sayings = self.get_motivation_file_names()

            for i in categories:
                if s.stop_requested or s.finish_program:
                    break

                self.exercises_in_category = [category for category in s.ex_in_training if i in category] #search for the exercises that are in the specific category
                s.waved_has_tool= False
                if self.exercises_in_category!=[]:
                    self.show_screen_category(i)
                    while not s.waved_has_tool:
                         time.sleep(0.0001)

                    self.first_coordination_ex = True
                    j=0

                    while j < len(self.exercises_in_category):
                        s.can_comment_robot= False
                        s.skipped_exercise = False
                        s.last_entry_angles = None
                        s.hand_not_good = False
                        s.not_reached_max_limit_rest_rules_ok = False
                        s.reached_max_limit = False
                        s.all_rules_ok = False
                        s.was_in_first_condition = False
                        e = self.exercises_in_category[j]

                        # if not s.finished_calibration:
                        #     calibration_help_variable = True
                        #
                        # while not s.finished_calibration:
                        #     time.sleep(0.0001)
                        #
                        # if calibration_help_variable: #אם הייתה קליברציה
                        #     calibration_help_variable = False
                        #     time.sleep(get_wav_duration("welcome"))


                        # if s.try_again_calibration:
                        #     say("welcome")
                        #     time.sleep(get_wav_duration("welcome"))


                        s.was_in_first_condition = False
                        if e in ["band_straighten_left_arm_elbows_bend_to_sides", "band_straighten_right_arm_elbows_bend_to_sides"]:
                            s.zed_camera.set_detection_model_to_accurate()

                        s.direction = None
                        s.skip = False
                        s.exercises_by_order.append(e)
                        s.gymmy_done= False
                        s.camera_done= False
                        #s.demo_finish = False
                        exercise = e
                        # s.number_of_repetitions_in_training =0
                        # s.max_repetitions_in_training =0
                        s.patient_repetitions_counting_in_exercise = 0
                        s.num_exercises_started +=1
                        self.run_exercise(e)

                        # if not (exercise == "notool_right_bend_left_up_from_side" or exercise == "notool_left_bend_right_up_from_side") or not self.first_coordination_ex:
                        #     self.end_exercise()


                        if (exercise == "notool_right_bend_left_up_from_side" or exercise == "notool_left_bend_right_up_from_side") and (not s.try_again_calibration and not s.repeat_explanation):
                            if self.first_coordination_ex == True:
                                    self.first_coordination_ex = False
                            else:
                                self.first_coordination_ex = True


                        if e in ["band_straighten_left_arm_elbows_bend_to_sides", "band_straighten_right_arm_elbows_bend_to_sides"]:
                            s.zed_camera.set_detection_model_to_medium()


                        if not s.try_again_calibration and not s.repeat_explanation:
                            j+=1
                            self.end_exercise()

                        elif s.try_again_calibration:
                            # say("start_not_first_time_calibration")
                            # time.sleep(int(get_wav_duration("start_not_first_time_calibration") + 2))
                            s.req_exercise = "calibration"

                            s.screen_finished_counting = False
                            s.asked_for_measurement = True
                            # s.camera_done = True
                            # s.gymmy_done = True

                            s.screen.switch_frame(CalibrationScreen)
                            s.num_exercises_started -= 1


                            while not s.finished_calibration:
                                time.sleep(0.0001)

                            time.sleep(get_wav_duration("end_not_first_time_calibration"))

                        elif s.repeat_explanation:
                            s.req_exercise = ""
                            s.num_exercises_started -= 1

                        if s.stop_requested or s.finish_program:
                            break

                        while not s.gymmy_done or not s.camera_done:
                            time.sleep(0.001)


                if s.stop_requested or s.finish_program:
                    break

    def show_screen_category(self, category):
        if category=="ball":
            s.screen.switch_frame(Ball)
        elif category=="stick":
            s.screen.switch_frame(Stick)
        elif category=="band":
            s.screen.switch_frame(Rubber_Band)
        elif category == "weights":
            s.screen.switch_frame(Weights)
        else:
            s.screen.switch_frame(NoTool)

    def get_motivation_file_names(self):
        """
        Retrieves all file names in a directory that:
        - Start with 'faster_' followed by a number.
        - Start with 'motivation_' followed by a number and optionally '_start', '_middle', '_end', or '_end_good'.

        Returns:
        - List[str]: A list of matching file names without extensions.
        """
        # Updated regex pattern
        pattern = r'^(faster_\d+|motivation_\d+_(start|middle|end|end_good|not_start|small_gap|large_gap|gap_and_many_rep)|dont_recognize_comment_\d+)\.\w+$'

        # Initialize a list to store matching file names
        matching_file_names = []

        # Verify the directory exists
        if not os.path.exists(s.audio_path):
            print(f"Directory does not exist: {s.audio_path}")
            return matching_file_names

        # Check all files in the directory
        for file_name in os.listdir(s.audio_path):
            # Match the file name against the pattern
            if re.match(pattern, file_name):
                # Remove the file extension
                name_without_extension, _ = os.path.splitext(file_name)
                matching_file_names.append(name_without_extension)

        return matching_file_names


    def finish_training(self):
        # ==================== RESET ROM MODE ====================
        # Critical: Reset ROM assessment mode when training ends
        # This ensures normal feedback behavior in subsequent trainings
        if getattr(s, 'is_rom_assessment_mode', False):
            print("[Training] Resetting ROM assessment mode")
            s.is_rom_assessment_mode = False
        # =========================================================
        
        #time.sleep(3)
        s.starts_and_ends_of_stops.append(time.time())

        s.finish_workout= True
        s.finished_effort= False

        s.screen.switch_frame(EffortScale)


        while not s.finished_effort:
            time.sleep(0.0001)

        if not s.stop_requested:
            s.screen.switch_frame(Repeat_training_or_not)

            while not s.choose_continue_or_not:
                time.sleep(0.0001)

        else:
            s.screen.switch_frame(ClappingPage)

        print(s.starts_and_ends_of_stops)
        Excel.find_and_add_training_to_patient()
        Excel.close_workbook()
        time.sleep(2)

        email_thread = threading.Thread(target=Email.create_pdf)
        email_thread.start()
        email_thread.join()

        if s.another_training_requested:

            # Email.email_to_patient()
            # Email.email_to_physical_therapist()

            s.req_exercise = ""
            s.waved = False
            s.success_exercise = False
            s.calibration = False
            s.finish_workout = False
            s.gymmy_done = False
            s.camera_done = False
            s.robot_count = False
            # s.demo_finish = False
            # s.exercises_start=False
            s.waved_has_tool = True  # True just in order to go through the loop in Gymmy
            # Excel variable
            s.ex_list = {}
            s.stop_requested = False
            s.choose_continue_or_not= False
            s.another_training_requested= False
            s.is_second_repetition_or_more= True
            s.finished_effort= False
            s.finished_calibration = True
            s.effort= None
            s.patient_repetitions_counting_in_exercise=0
            s.number_of_repetitions_in_training=0
            s.did_training_paused= False
            s.starts_and_ends_of_stops= []
            s.general_sayings = []
            s.num_exercises_started = 0
            s.number_of_pauses = 0
            s.needs_first_position = False
            s.exercises_skipped= {}
            s.skip = False
            s.additional_audio_playing= False
            s.volume = 0.3
            s.play_song = False
            s.asked_for_measurement = False
            s.screen_finished_counting = False
            s.explanation_over = False
            s.gymmy_finished_demo = False
            s.last_saying_time = time.time()
            s.robot_counter= 0
            s.was_in_first_condition = False
            s.time_of_change_position = None
            s.try_again_calibration = False
            s.not_reached_max_limit_rest_rules_ok = False
            s.repeat_explanation = False
            s.shoulder_problem_calibration = False
            s.elbow_problem_calibration = False
            s.hand_not_good = False
            s.exercise_name_repeated_explanation = None
            s.suggest_repeat_explanation = False
            s.last_entry_angles = None
            s.skipped_exercise = False
            s.shoulders_not_good = False
            s.can_comment_robot= False
            s.change_in_trend = [False]
            time.sleep(2)

        else:

            # Email.email_to_patient()
            # self.check_points_and_send_email()
            # Email.email_to_physical_therapist()
            print("TRAINING DONE")
            self.reset()


    def end_exercise(self):
        s.screen.switch_frame(Number_of_good_repetitions_page)
        time.sleep(0.5)
        say(f'{s.patient_repetitions_counting_in_exercise}_successful_rep')
        time.sleep(get_wav_duration(f"{s.patient_repetitions_counting_in_exercise}_successful_rep")+0.1)

        if self.selected_exercises and not s.num_exercises_started ==  len(self.selected_exercises) and self.first_coordination_ex and not s.stop_requested:
            continue_files = Excel.get_files_names_by_start_word("continue")
            chosen_continue = random.choice(continue_files)
            say(chosen_continue)
            time.sleep(get_wav_duration(chosen_continue) + 1)


    def run_exercise(self, name):
        s.robot_counter = 0
        s.success_exercise = False
        s.patient_repetitions_counting_in_exercise = 0

        print("TRAINING: Exercise ", name, " start")
        s.explanation_over = False
        s.req_exercise = name

        if self.first_coordination_ex:
            s.screen.switch_frame(ExplanationPage, exercise= name)
            # if name == "notool_right_bend_left_up_from_side" or name == "notool_left_bend_right_up_from_side":
            #     name = "notool_arm_bend_arm_up_from_side"
            # time.sleep(get_wav_duration(name)) #wait the time of the audio

            while not s.explanation_over or not s.gymmy_finished_demo:
                time.sleep(0.001)

            say(str(f'{s.rep}_times'))
            time.sleep(get_wav_duration(f'{str(s.rep)}_times')+0.5)
            # name = s.req_exercise
            #s.screen.switch_frame(ExercisePage)
            self.which_exercise_page()
        else:
            self.which_exercise_page()
            #s.screen.switch_frame(ExercisePage)
            say("notool_arm_bend_arm_up_from_side_continue")
            time.sleep(get_wav_duration("notool_arm_bend_arm_up_from_side_continue") + 0.5)
            # say(str(f'{s.rep}_times'))
            # time.sleep(get_wav_duration(f'{str(s.rep)}_times')+0.5)
            s.explanation_over = True


        while s.req_exercise == name and not s.try_again_calibration and not s.repeat_explanation and not s.skipped_exercise:
            time.sleep(0.00000001)

        if s.try_again_calibration:
            s.req_exercise = ""
            s.finished_calibration = False

        if s.skipped_exercise:
            s.req_exercise = ""



        print("TRAINING: Exercise ", name, " done")
        s.gymmy_finished_demo = False
        # time.sleep(1)



    def get_video_duration(self, exercise):
        video_file = f'Videos//{exercise}_vid.mp4'
        video_path = os.path.join(os.getcwd(), video_file)
        self.cap = cv2.VideoCapture(video_path)

        fps = self.cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Get the total number of frames
        duration = total_frames / fps  # Duration in seconds

        return duration

    def reset(self):
        # s.screen.switch_frame(GoodbyePage)
        time.sleep(get_wav_duration('goodbye') + 2)

        s.req_exercise = ""
        s.waved = False
        s.success_exercise = False
        s.calibration = False
        s.finish_workout = False
        s.gymmy_done = False
        s.camera_done = False
        s.robot_count = False
        #s.demo_finish = False
        s.ex_in_training = []
        # s.exercises_start=False
        s.waved_has_tool = True  # True just in order to go through the loop in Gymmy
        # Excel variable
        s.ex_list = {}
        s.stop_requested = False
        s.choose_continue_or_not=False
        s.another_training_requested= False
        s.is_second_repetition_or_more= False
        s.finished_effort= False
        s.effort = None
        s.was_in_first_condition = False
        s.chosen_patient_ID = None
        s.patient_repetitions_counting_in_exercise=0
        s.number_of_repetitions_in_training=0
        s.did_training_paused= False
        s.starts_and_ends_of_stops= []
        
        # ==================== ROM MODE RESET ====================
        s.is_rom_assessment_mode = False  # Critical: reset for next training
        # =========================================================
        
        time.sleep(2)
        s.general_sayings = []
        s.num_exercises_started = 0
        s.dist_between_shoulders = 0
        s.number_of_pauses = 0
        s.needs_first_position = False
        s.exercises_skipped = {}
        s.screen.switch_frame(EntrancePage)
        s.skip = False
        s.additional_audio_playing = False
        s.volume = 0
        s.play_song = False
        s.asked_for_measurement = False
        s.screen_finished_counting = False
        s.finished_calibration = False
        s.len_left_arm = None
        s.len_right_arm = None
        s.dist_between_wrists = None
        s.dist_between_shoulders = None
        s.time_of_change_position = None
        s.average_dist = None
        s.rep = 0
        s.audio_path = None
        s.rate = "moderate"
        s.explanation_over = False
        s.gymmy_finished_demo = False
        s.last_saying_time = time.time()
        s.robot_counter = 0
        s.try_again_calibration = False
        s.not_reached_max_limit_rest_rules_ok = False
        s.repeat_explanation = False
        s.shoulder_problem_calibration = False
        s.elbow_problem_calibration = False
        s.hand_not_good = False
        s.exercise_name_repeated_explanation = None
        s.suggest_repeat_explanation = False
        s.last_entry_angles = None
        s.skipped_exercise = False
        s.shoulders_not_good = False
        s.can_comment_robot = False
        s.change_in_trend = [False]

    def get_dynamic_bar_params(self, exercise_name, default_min, default_max, reverse_bar=False):
        """
        SIMPLIFIED: Get bar parameters using UNIFIED thresholds.
        
        Uses the same unified threshold logic as the counting logic in Camera.py
        This ensures bar and counting are always in sync.
        
        For exercises where bar fills when angle INCREASES (reverse_bar=False):
        - min_distance = down_ub (upper bound of DOWN range)
        - max_distance = up_lb (lower bound of UP range)
        
        For exercises where bar fills when angle DECREASES (reverse_bar=True):
        - min_distance = up_ub (upper bound of UP range)  
        - max_distance = down_lb (lower bound of DOWN range)
        """
        if not hasattr(s, 'patient_rom_limits') or not s.patient_rom_limits:
            return default_min, default_max
        
        # Get values from both sides
        r_down_ub = s.patient_rom_limits.get(f"{exercise_name}_right_down_ub")
        l_down_ub = s.patient_rom_limits.get(f"{exercise_name}_left_down_ub")
        r_down_lb = s.patient_rom_limits.get(f"{exercise_name}_right_down_lb")
        l_down_lb = s.patient_rom_limits.get(f"{exercise_name}_left_down_lb")
        r_up_lb = s.patient_rom_limits.get(f"{exercise_name}_right_up_lb")
        l_up_lb = s.patient_rom_limits.get(f"{exercise_name}_left_up_lb")
        r_up_ub = s.patient_rom_limits.get(f"{exercise_name}_right_up_ub")
        l_up_ub = s.patient_rom_limits.get(f"{exercise_name}_left_up_ub")
        
        # UNIFY thresholds: min of lbs, max of ubs (same as Camera.get_unified_thresholds)
        down_ub = max(r_down_ub or 0, l_down_ub or 0) if (r_down_ub or l_down_ub) else None
        down_lb = min(r_down_lb or 180, l_down_lb or 180) if (r_down_lb or l_down_lb) else None
        up_lb = min(r_up_lb or 180, l_up_lb or 180) if (r_up_lb or l_up_lb) else None
        up_ub = max(r_up_ub or 0, l_up_ub or 0) if (r_up_ub or l_up_ub) else None
        
        if reverse_bar:
            # Bar fills when angle DECREASES
            dynamic_min = up_ub if up_ub is not None else default_min
            dynamic_max = down_lb if down_lb is not None else default_max
        else:
            # Bar fills when angle INCREASES
            dynamic_min = down_ub if down_ub is not None else default_min
            dynamic_max = up_lb if up_lb is not None else default_max
        
        # Ensure min < max with at least 10 degrees difference
        if dynamic_min >= dynamic_max - 10:
            dynamic_min = dynamic_max - 20
        
        print(f"[BAR] {exercise_name}: min={dynamic_min:.0f}° max={dynamic_max:.0f}° [reverse={reverse_bar}]")
        
        return dynamic_min, dynamic_max

    def which_exercise_page(self):

        #average_len_arms = (s.len_left_arm + s.len_right_arm)/2

        if s.req_exercise in ["ball_bend_elbows", "stick_bend_elbows"]:
            min_d, max_d = self.get_dynamic_bar_params(s.req_exercise, 65, 130, reverse_bar=True)
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_height_y", reverse_color=True, reverse_bar=True, min_distance=min_d, max_distance=max_d, which_side = "both")
        elif s.req_exercise in ["ball_raise_arms_above_head", "stick_raise_arms_above_head"]:
            min_d, max_d = self.get_dynamic_bar_params(s.req_exercise, 30, 105)
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_height_y", reverse_color=False, reverse_bar=False, min_distance=min_d, max_distance=max_d, which_side = "both")
        elif s.req_exercise in ["ball_switch", "stick_switch"]:
            s.screen.switch_frame(ExercisePage, exercise_type="shoulders_distance_x", reverse_color=True, reverse_bar=True, min_distance=(s.dist_between_shoulders-s.dist_between_shoulders/4), max_distance= s.dist_between_shoulders)
        elif s.req_exercise in ["ball_open_arms_and_forward", "weights_open_arms_and_forward"]:
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_distance_x", reverse_color=False, reverse_bar=False, min_distance=90, max_distance= 140)
        elif s.req_exercise in ["ball_open_arms_above_head", "stick_bend_elbows_and_up"]:
            min_d, max_d = self.get_dynamic_bar_params(s.req_exercise, 30, 110)
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_height_y", reverse_color=False, reverse_bar=False, min_distance=min_d, max_distance=max_d, which_side = "both")
        elif s.req_exercise == "band_open_arms":
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_distance_x", reverse_color=False, reverse_bar=False, min_distance=90, max_distance= 135)
        elif s.req_exercise == "band_open_arms_and_up":
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_height_y_distance_x", reverse_color=False, reverse_bar=False, min_distance=50, max_distance= 120, which_side = None, min_distance_x= 30 , max_distance_x=  70)
        elif s.req_exercise in ["band_up_and_lean", "stick_up_and_lean", "notool_hands_behind_and_lean"]:
            s.screen.switch_frame(ExercisePage, exercise_type="shoulders_distance_x", reverse_color=True, reverse_bar=True, min_distance= s.dist_between_shoulders-50, max_distance= s.dist_between_shoulders, is_lean = True)
        elif s.req_exercise in ["band_straighten_left_arm_elbows_bend_to_sides", "band_straighten_right_arm_elbows_bend_to_sides"]:
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_distance_x", reverse_color=False, reverse_bar=False, min_distance= 50, max_distance= 160)
        elif s.req_exercise in ["notool_right_hand_up_and_bend", "notool_left_hand_up_and_bend"]:
            s.screen.switch_frame(ExercisePage, exercise_type="shoulders_distance_x", reverse_color=True, reverse_bar=True, min_distance= s.dist_between_shoulders-50, max_distance= s.dist_between_shoulders)
        elif s.req_exercise in ["weights_abduction"]:
            min_d, max_d = self.get_dynamic_bar_params(s.req_exercise, 20, 80)
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_height_y", reverse_color=False, reverse_bar=False, min_distance=min_d, max_distance=max_d, which_side = "both")
        elif s.req_exercise in ["notool_right_bend_left_up_from_side"]:
            min_d, max_d = self.get_dynamic_bar_params(s.req_exercise, 0, 165)
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_height_y", reverse_color=False, reverse_bar=False, min_distance=min_d, max_distance=max_d, which_side = "left")
        elif s.req_exercise in ["notool_left_bend_right_up_from_side"]:
            min_d, max_d = self.get_dynamic_bar_params(s.req_exercise, 0, 165)
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_height_y", reverse_color=False, reverse_bar=False, min_distance=min_d, max_distance=max_d, which_side="right")
        elif s.req_exercise in ["notool_raising_hands_diagonally"]:
            min_d, max_d = self.get_dynamic_bar_params(s.req_exercise, 50, 185)
            s.screen.switch_frame(ExercisePage, exercise_type="wrist_height_y", reverse_color=False, reverse_bar=False, min_distance=min_d, max_distance=max_d, which_side="both")


if __name__ == "__main__":
    s.gender= "Female"
    s.audio_path = f'audio files/Hebrew/{s.gender}/'

    current_time = datetime.now()
    s.participant_code = str(current_time.day) + "." + str(current_time.month) + " " + str(current_time.hour) + "." + \
                         str(current_time.minute) + "." + str(current_time.second)
    s.waved = False
    s.finish_workout = False
    s.finish_program= False
    s.asked_for_measurement= False
    s.rep = 5
    s.time_of_change_position = None
    s.fps = 0
    s.ex_in_training = ["notool_left_hand_up_and_bend"]
    s.skipped_exercise = False
    s.change_in_trend = [False]

    #,"ball_switch" ,"ball_open_arms_and_forward" , "ball_open_arms_above_head"] "ball_bend_elbows" ,
        # ["band_open_arms",  "band_up_and_lean", "band_open_arms_and_up"]
    #s.ex_in_training =  ["ball_raise_arms_above_head"]
    # s.ex_in_training = ["notool_right_bend_left_up_from_side", "notool_left_bend_right_up_from_side"]
    # s.ex_in_training= [, "band_open_arms_and_up", "band_up_and_lean", "band_straighten_left_arm_elbows_bend_to_sides", "band_straighten_right_arm_elbows_bend_to_sides"]
                    #"band_open_arms", "band_open_arms_and_up", "band_up_and_lean", "band_straighten_left_arm_elbows_bend_to_sides", "band_straighten_right_arm_elbows_bend_to_sides"
    #
                    # "stick_bend_elbows", "stick_bend_elbows_and_up", "stick_raise_arms_above_head", "stick_switch", "stick_up_and_lean"
                    # weights_right_hand_up_and_bend, weights_left_hand_up_and_bend, weights_open_arms_and_forward, weights_abduction
    # notool_hands_behind_and_lean
    # notool_right_hand_up_and_bend
    # notool_left_hand_up_and_bend
    # notool_raising_hands_diagonally
    # notool_right_bend_left_up_from_side
    # notool_left_bend_right_up_from_side
    s.direction = None
    s.chosen_patient_ID="3333"
    s.req_exercise=""
    s.ex_list = {}
    s.finished_calibration = False
    s.repeat_explanation = False
    s.last_entry_angles = None

    s.len_left_arm = None
    s.len_right_arm = None
    s.dist_between_wrists = None
    s.dist_between_shoulders = None
    s.len_left_upper_arm = None
    s.len_right_upper_arm = None
    s.shoulder_problem_calibration = False
    s.hand_not_good = False
    s.exercise_name_repeated_explanation = None

    s.was_in_first_condition = False
    s.screen_finished_counting = False
    s.is_second_repetition_or_more =False
    #s.demo_finish = False
    s.all_rules_ok = False
    s.reached_max_limit = False
    s.latest_keypoints = {}
    s.did_training_paused = False
    s.volume = 0
    s.additional_audio_playing = False
    s.gymmy_finished_demo = False
    s.shoulders_not_good = False
    s.robot_counter = 0
    s.last_saying_time = time.time()
    s.rate= "moderate"
    s.num_exercises_started = 0
    pygame.mixer.init()
    s.full_name = "יעל שניידמן"
    s.play_song = False
    s.explanation_over = False
    s.suggest_repeat_explanation = False
    s.number_of_repetitions_in_training = 0
    s.max_repetitions_in_training = 0

    s.another_training_requested= False
    s.choose_continue_or_not = False
    s.email_of_patient = "yaelszn@gmail.com"
    s.try_again_calibration = False
    s.number_of_pauses=0
    # Start continuous audio in a separate thread
    s.screen = Screen()
    s.camera = Camera()
    s.training = Training()
    s.robot = Gymmy()
    s.camera.start()
    s.training.start()
    s.robot.start()
    s.audio_manager = AdditionalAudio()
    s.continuous_audio = ContinuousAudio()
    s.continuous_audio.start()
    s.screen.switch_frame(StartOfTraining)
    app = FullScreenApp(s.screen)
    s.screen.mainloop()

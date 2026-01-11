import time
import Settings as s
from Audio import ContinuousAudio, AdditionalAudio
from Camera import Camera
from Gymmy import Gymmy
from TrainingNew import Training
from ScreenNew import Screen, FullScreenApp, EntrancePage
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    s.exercises_skipped = {}
    s.skip = False
    s.additional_audio_playing = False
    s.volume = 0 #Training variables initialization
    s.ball_exercises_number = 4
    s.band_exercises_number = 5
    s.stick_exercises_number = 5
    s.weights_exercises_number = 2
    s.no_tool_exercises_number = 6


    # ==================== CRITICAL FIX: Initialize ALL measurement attributes ====================
    # These MUST be initialized to None (not commented out) to prevent AttributeError crashes
    s.len_left_arm = None
    s.len_right_arm = None
    s.dist_between_wrists = None
    s.dist_between_shoulders = None
    s.len_left_upper_arm = None
    s.len_right_upper_arm = None
    # =============================================================================================

    # ==================== ROM Assessment Mode Initialization ====================
    s.is_rom_assessment_mode = False
    s.patient_rom_limits = {}
    # ============================================================================


    s.screen_finished_counting = False
    s.finished_calibration = False
    s.skipped_exercise = False
    s.time_of_change_position = None
    s.average_dist = None
    s.rep = 5
    s.req_exercise = ""
    s.audio_path = None
    s.finish_workout = False
    s.waved = False
    s.success_exercise = False
    s.shoulders_not_good = False
    s.gymmy_done = False
    s.camera_done = False
    # s.robot_count = False
    s.demo_finish= False
    s.needs_first_position = False
    s.num_exercises_started = 0
    s.ex_in_training=[]
    s.finish_program= False #will turn to true only when the user will press on the exit button
    #s.exercises_start=False
    s.waved_has_tool= True # True just in order to go through the loop in Gymmy
    # s.finished_training_adding_to_excel= False
    s.side = None
    s.asked_for_measurement = False
    s.ex_list = {}
    s.reached_max_limit = False
    s.latest_keypoints = {}
    s.effort=None
    s.starts_and_ends_of_stops =[]
    s.stop_requested = False
    s.is_second_repetition_or_more=False
    s.another_training_requested= False
    s.choose_continue_or_not= False
    s.did_training_paused= False
    s.rate= "moderate"
    s.explanation_over= False
    s.gymmy_finished_demo = False
    s.hand_not_good = False
    s.exercise_name_repeated_explanation = None
    s.suggest_repeat_explanation = False
    s.last_entry_angles = None
    s.can_comment_robot = False
    s.number_of_repetitions_in_training = 0
    s.max_repetitions_in_training = 0
    s.last_saying_time = time.time()
    s.robot_counter = 0
    s.general_sayings = []
    s.dist_between_shoulders = 0
    s.number_of_pauses = 0
    s.not_reached_max_limit_rest_rules_ok = False
    s.try_again_calibration = False
    s.repeat_explanation = False
    s.name_of_exercise_repeated_explanation = None
    s.shoulder_problem_calibration = False
    s.all_rules_ok = False
    s.fps = 0
    s.change_in_trend = [False]

    # Create all components
    s.camera = Camera()
    s.training = Training()
    s.robot = Gymmy()


    s.play_song = False

    s.audio_manager = AdditionalAudio()
    # Start continuous audio in a separate thread
    s.continuous_audio = ContinuousAudio()
    s.continuous_audio.start()

    s.screen = Screen()
    #s.screen.switch_frame(HelloPage)

    # Start all threads
    s.camera.start()
    s.training.start()
    s.robot.start()


    s.screen.switch_frame(EntrancePage)
    app = FullScreenApp(s.screen)
    s.screen.mainloop()

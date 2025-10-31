import threading

import pygame
from pypot.creatures import PoppyTorso
import time
import Settings as s
from Audio import say, get_wav_duration, ContinuousAudio, AdditionalAudio
from ScreenNew import ExercisePage, Screen, EntrancePage, FullScreenApp, ExplanationPage


class Gymmy(threading.Thread):

    ################################################# INITIALIZATION ###########################################
    def __init__(self):
        threading.Thread.__init__(self)

        #self.gymmy = PoppyTorso(camera="dummy", port= "COM4")  # for real robot
        self.gymmy = PoppyTorso(simulator='vrep')  # for simulator
        print("ROBOT INITIALIZATION")
        #self.gymmy.abs_z.goto_position(0, 1, wait=True)

        for m in self.gymmy.motors:  # motors need to be initialized, False=stiff, True=loose
            m.compliant = False
            time.sleep(0.1)

        self.init_robot()

    def init_robot(self):
        self.gymmy.abs_z.goto_position(0, 2, wait=True)
        self.gymmy.l_shoulder_x.goto_position(30, 2, wait=False)
        self.gymmy.r_shoulder_x.goto_position(-30, 2, wait=True)


        for m in self.gymmy.motors:
            if not m.name == 'head_y' and not m.name == 'r_shoulder_x' and not m.name== 'l_shoulder_x':
                    m.goto_position(0, 2, wait=False)

        self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
        self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
        self.gymmy.head_y.goto_position(5, 2, wait=False)

        time.sleep(1)

    ########################################################### RUN ##########################################
    def run(self):
        print("ROBOT START")
        self.first_coordination_ex = True
        while not s.finish_program:
            self.last_time_suggestion_screen_popped_up = None

            while not s.finish_workout:
                time.sleep(0.00001)  # Prevents the MP to stuck
                if s.req_exercise != "" and not s.req_exercise=="hello_waving":
                    ex = s.req_exercise

                    print("ROBOT: Exercise ", ex, " start")
                    self.exercise_demo(ex)
                    print("ROBOT: Exercise ", ex, " done")

                    s.req_exercise = ""
                    s.gymmy_done = True

                elif s.finish_program:
                    break

                else:
                    time.sleep(0.01)  # Prevents the MP to stuck

        print("Robot Done")



    def exercise_demo(self, ex):
        if ex == "hello_waving":
            self.hello_waving()

        elif ex == "calibration":
            time.sleep(get_wav_duration("start_calibration") - 3)
            self.calibration()

        else:
            # self.faster_sayings = ['pick_up_pace', 'faster']
            if self.first_coordination_ex == True:
                self.i = "demo"
                getattr(self, ex)("slow")

                while not s.explanation_over:
                    time.sleep(0.00001)

                s.gymmy_finished_demo = True

                time.sleep(get_wav_duration(f'{str(s.rep)}_times')+1)

            self.i = 0
            s.needs_first_position = False
            s.gymmy_finished_demo = True

            # time.sleep(2)
            while self.i < s.rep:
                if not s.did_training_paused:
                    self.did_init = False

                if s.finish_program:
                    break

                if s.did_training_paused and not s.stop_requested:
                    if not self.i ==0:
                        self.i -= 1

                    s.needs_first_position = True

                    if not self.did_init:
                        self.init_robot()

                    self.did_init = False

                while s.did_training_paused and not s.stop_requested:
                    time.sleep(0.01)

                if not s.skipped_exercise:
                    if s.stop_requested:
                        break


                    s.robot_counter = self.i

                    if s.req_exercise == "" and not self.did_init:
                        self.init_robot()
                        break

                    getattr(self, ex)(s.rate)

                    if not (s.did_training_paused or s.stop_requested):
                        if s.req_exercise not in ["ball_switch", "band_up_and_lean", "stick_switch", "stick_up_and_lean", "notool_hands_behind_and_lean", "notool_raising_hands_diagonally"]:
                            if not self.i == s.rep:
                                self.i += 1


                    if self.i - s.patient_repetitions_counting_in_exercise >=4 and not self.i + 1 >= s.rep:
                        if s.exercise_name_repeated_explanation is None or not s.exercise_name_repeated_explanation == s.req_exercise and \
                                (s.last_time_suggestion_screen_popped_up is None or s.num_exercises_started - s.last_time_suggestion_screen_popped_up > 3):
                            s.suggest_repeat_explanation = True
                            s.last_time_suggestion_screen_popped_up = s.num_exercises_started
                            time.sleep(0.1)

                    s.needs_first_position = False

                    if s.req_exercise == "" and not self.did_init:
                        self.init_robot()
                        break


                    if s.req_exercise == "notool_right_bend_left_up_from_side" or s.req_exercise == "notool_left_bend_right_up_from_side": #if this is the fist of the 2, turn into false, and then in the next iteration it will skip the demonstration
                        if  self.first_coordination_ex== False:
                                self.first_coordination_ex = True
                        if  self.first_coordination_ex== True:
                                self.first_coordination_ex = False

                    print("robot count: "+str(self.i))


                    if (s.success_exercise and self.i != s.rep) and not self.did_init:
                        self.init_robot()
                        break


                else:
                    self.init_robot()
                    break



    ################################################################ WAVING EXERCISES ###############################33
    def hello_waving(self):
        if not s.waved:
            self.gymmy.r_shoulder_x.goto_position(-90, 1.5, wait=False)
            self.gymmy.r_elbow_y.goto_position(-20, 1.5, wait=False)
            self.gymmy.r_arm_z.goto_position(-80, 1.5, wait=False)
            for i in range(3):
                self.gymmy.r_arm[3].goto_position(-35, 0.6, wait=False)
                self.gymmy.r_arm[3].goto_position(35, 0.6, wait=False)
            self.finish_waving()

    def finish_waving(self):
        self.gymmy.r_shoulder_x.goto_position(0, 1.5, wait=False)
        self.gymmy.r_elbow_y.goto_position(90, 1.5, wait=False)
        self.gymmy.r_arm_z.goto_position(0, 1.5, wait=False)



################################################### Physiotherapy exercises ##########################################################

# ----------------------------------------------------- ball exercises Video No 1 ------------------------------

    def calibration(self):
        self.gymmy.l_shoulder_x.goto_position(90, 1, wait=False)
        self.gymmy.r_shoulder_x.goto_position(-90, 1, wait=True)

        while not s.finished_calibration:
            time.sleep(0.0001)

        self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
        self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)

    # EX1
    def ball_bend_elbows(self, rate):
        if self.i=="demo":
            self.gymmy.l_shoulder_y.goto_position(-50, 1, wait=False)
            self.gymmy.r_shoulder_y.goto_position(-50, 1, wait=False)
            time.sleep(1)
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(5, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-5, 1, wait=False)
                time.sleep(1)

            if not s.explanation_over:
                self.gymmy.r_elbow_y.goto_position(-135, 1.75, wait=False)
                self.gymmy.l_elbow_y.goto_position(-135, 1.75, wait=True)
                time.sleep(0.75)


                self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=True)
                time.sleep(0.75)

            # init
            self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=False)
            self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=True)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_y.goto_position(-50, 1, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-50, 1, wait=False)
                time.sleep(1)
                self.gymmy.l_shoulder_x.goto_position(5, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-5, 1, wait=False)
                time.sleep(1)
                s.can_comment_robot = True


            if (rate=="fast"):
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
                        time.sleep(0.25)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-135, 1.25, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
                    time.sleep(0.25)


            elif (rate=="moderate"):
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=True)
                        time.sleep(0.5)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-135, 1.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 1.5, wait=True)
                    time.sleep(0.5)


            else:
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=True)
                        time.sleep(0.75)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-135, 1.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 1.75, wait=True)
                    time.sleep(0.75)


            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                # init
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)
                self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=True)
                self.did_init = True




    # EX2
    def ball_raise_arms_above_head(self, rate):
        if self.i=="demo":
            self.gymmy.l_shoulder_y.goto_position(-85, 1.75, wait=False)
            self.gymmy.r_shoulder_y.goto_position(-85, 1.75, wait=True)
            self.gymmy.l_shoulder_x.goto_position(-15, 1.75, wait=False)
            self.gymmy.r_shoulder_x.goto_position(15, 1.75, wait=True)

            if not s.explanation_over:
                self.gymmy.l_arm_z.goto_position(30, 1.75, wait=False)
                self.gymmy.r_arm_z.goto_position(-30, 1.75, wait=False)
                self.gymmy.l_shoulder_x.goto_position(20, 1.75, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-20, 1.75, wait=False)
                self.gymmy.l_shoulder_y.goto_position(-170, 1.75, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-170, 1.75, wait=True)

            if not s.explanation_over:
                time.sleep(1)
                self.gymmy.l_shoulder_y.goto_position(-85, 1.75, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-85, 1.75, wait=False)
                self.gymmy.l_shoulder_x.goto_position(-15, 1.75, wait=False)
                self.gymmy.r_shoulder_x.goto_position(15, 1.75, wait=False)
                self.gymmy.l_arm_z.goto_position(0, 1.75, wait=False)
                self.gymmy.r_arm_z.goto_position(0, 1.75, wait=True)

                # init
            self.gymmy.l_arm_z.goto_position(0, 2, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 2, wait=False)

        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_y.goto_position(-85, 1, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-85, 1, wait=True)
                self.gymmy.l_shoulder_x.goto_position(-15, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(15, 1, wait=False)
                time.sleep(0.5)
                s.can_comment_robot = True


            if (rate=="fast"):
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        time.sleep(0.5)
                        self.gymmy.l_shoulder_y.goto_position(-85, 1.25, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-85, 1.25, wait=False)
                        self.gymmy.l_shoulder_x.goto_position(-15, 1.25, wait=False)
                        self.gymmy.r_shoulder_x.goto_position(15, 1.25, wait=False)
                        self.gymmy.l_arm_z.goto_position(0, 1.25, wait=False)
                        self.gymmy.r_arm_z.goto_position(0, 1.25, wait=True)

                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(0.5)
                    self.gymmy.l_arm_z.goto_position(30, 1.25, wait=False)
                    self.gymmy.r_arm_z.goto_position(-30, 1.25, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(20, 1.25, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-20, 1.25, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-170, 1.25, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-170, 1.25, wait=True)

            elif (rate=="moderate"):
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-85, 1.5, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-85, 1.5, wait=False)
                        self.gymmy.l_shoulder_x.goto_position(-15, 1.5, wait=False)
                        self.gymmy.r_shoulder_x.goto_position(15, 1.5, wait=False)
                        self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
                        self.gymmy.r_arm_z.goto_position(0, 1.5, wait=True)

                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(0.75)
                    self.gymmy.l_arm_z.goto_position(30, 1.5, wait=False)
                    self.gymmy.r_arm_z.goto_position(-30, 1.5, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(20, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-20, 1.5, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-170, 1.5, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-170, 1.5, wait=True)
                    time.sleep(0.75)

            else:
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        time.sleep(1)
                        self.gymmy.l_shoulder_y.goto_position(-85, 1.75, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-85, 1.75, wait=False)
                        self.gymmy.l_shoulder_x.goto_position(-15, 1.75, wait=False)
                        self.gymmy.r_shoulder_x.goto_position(15, 1.75, wait=False)
                        self.gymmy.l_arm_z.goto_position(0, 1.75, wait=False)
                        self.gymmy.r_arm_z.goto_position(0, 1.75, wait=True)

                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(1)
                    self.gymmy.l_arm_z.goto_position(30, 1.75, wait=False)
                    self.gymmy.r_arm_z.goto_position(-30, 1.75, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(20, 1.75, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-20, 1.75, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-170, 1.75, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-170, 1.75, wait=True)

        if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
            # init
            if self.i == (s.rep - 1):
                self.i = s.rep
                s.robot_counter = self.i

            self.gymmy.l_arm_z.goto_position(0, 2, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 2, wait=False)
            self.did_init = True

    # EX3
    def ball_switch(self, rate):
        if self.i == "demo":
            self.gymmy.l_shoulder_y.goto_position(-100, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(-100, 1.5, wait=True)
            self.gymmy.l_shoulder_x.goto_position(-10, 1, wait=False)
            self.gymmy.r_shoulder_x.goto_position(10, 1, wait=True)
            time.sleep(1.5)

            if not s.explanation_over:
                time.sleep(1.5)

            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(-60, 1.75, wait=True)
                time.sleep(0.5)

            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(0, 1.75, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(60, 1.75, wait=True)

                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(0, 1.75, wait=True)
                time.sleep(0.5)

            self.gymmy.abs_z.goto_position(0, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 2, wait=True)

        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_y.goto_position(-100, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-100, 1.5, wait=True)
                self.gymmy.l_shoulder_x.goto_position(-10, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(10, 1, wait=True)
                time.sleep(1)
                s.can_comment_robot = True

            if (rate=="fast"):
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(-60, 1, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.75)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1, wait=True)
                        time.sleep(0.75)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(60, 1, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.75)
                        self.i += 1
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1, wait=True)
                        time.sleep(0.75)

            elif (rate=="moderate"):
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(-60, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)

                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1.25, wait=True)
                        time.sleep(1)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(60, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1.25, wait=True)
                        time.sleep(1)

            else:
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(-60, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1.75, wait=True)
                        time.sleep(1)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(60, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1.75, wait=True)
                        time.sleep(1)


            if self.i >= s.rep or s.did_training_paused or s.stop_requested:
                # init
                self.gymmy.abs_z.goto_position(0, 2, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
                time.sleep(2)
                self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 2, wait=True)
                self.did_init = True


# ------------------------------------------------------ ball exercises Video No 2 ------------------------------------

    # EX4
    def ball_open_arms_and_forward(self, rate):
        if self.i == "demo":
            time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-100, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-100, 1.5, wait=True)
                self.gymmy.l_shoulder_x.goto_position(75, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-75, 1.5, wait=True)

            if not s.explanation_over:
                time.sleep(1)
                self.gymmy.l_shoulder_x.goto_position(-10, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(10, 1.5, wait=True)

            if not s.explanation_over:
                time.sleep(1)
                self.gymmy.l_shoulder_x.goto_position(75, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-75, 1.5, wait=True)

            if not s.explanation_over:
                time.sleep(0.25)
                self.gymmy.l_shoulder_x.goto_position(-10, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(10, 1.5, wait=True)

            if not s.explanation_over:
                time.sleep(1)
                self.gymmy.l_shoulder_x.goto_position(75, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-75, 1.5, wait=True)

            # init
            time.sleep(1)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 2, wait=True)

        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_y.goto_position(-100, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-100, 1.5, wait=True)
                self.gymmy.l_shoulder_x.goto_position(75, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-75, 1.5, wait=True)
                time.sleep(0.5)
                s.can_comment_robot = True

            if (rate=="fast"):
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(0.5)
                    self.gymmy.l_shoulder_x.goto_position(-10, 1, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(10, 1, wait=True)
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(0.5)
                    self.gymmy.l_shoulder_x.goto_position(75, 1, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-75, 1, wait=True)

            elif (rate=="moderate"):
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(1)
                    self.gymmy.l_shoulder_x.goto_position(-10, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(10, 1.5, wait=True)
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(1)
                    self.gymmy.l_shoulder_x.goto_position(75, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-75, 1.5, wait=True)

            else:
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(1)
                    self.gymmy.l_shoulder_x.goto_position(-10, 2, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(10, 2, wait=True)
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(1)
                    self.gymmy.l_shoulder_x.goto_position(75, 2, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-75, 2, wait=True)


            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                # init
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                time.sleep(1)
                self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
                self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 2, wait=True)
                self.did_init = True



    # # EX5
    # def ball_open_arms_above_head(self, i, rate):
    #     if i == "demo":
    #         time.sleep(2)
    #         if not s.explanation_over:
    #             self.gymmy.r_shoulder_x.goto_position(-90, 2, wait=False)
    #             self.gymmy.l_shoulder_x.goto_position(90, 2, wait=False)
    #             self.gymmy.r_shoulder_y.goto_position(-170, 2, wait=False)
    #             self.gymmy.l_shoulder_y.goto_position(-170, 2, wait=True)
    #             time.sleep(1)
    #
    #         if not s.explanation_over:
    #             self.gymmy.r_shoulder_x.goto_position(-10, 1.75, wait=False)
    #             self.gymmy.l_shoulder_x.goto_position(10, 1.75, wait=True)
    #             time.sleep(0.75)
    #         if not s.explanation_over:
    #             self.gymmy.r_shoulder_x.goto_position(-90, 1.75, wait=False)
    #             self.gymmy.l_shoulder_x.goto_position(90, 1.75, wait=True)
    #             time.sleep(0.75)
    #
    #         if not s.explanation_over:
    #             for _ in range (2):
    #                 self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
    #                 self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=True)
    #                 time.sleep(0.5)
    #                 if s.explanation_over:
    #                     break
    #                 self.gymmy.r_shoulder_x.goto_position(-90, 1.5, wait=False)
    #                 self.gymmy.l_shoulder_x.goto_position(90, 1.5, wait=True)
    #                 time.sleep(0.5)
    #                 if s.explanation_over:
    #                     break
    #
    #         # init
    #         self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
    #         self.gymmy.r_shoulder_y.goto_position(0, 2, wait=True)
    #         self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
    #         self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
    #
    #     else:
    #         if i==0 or s.needs_first_position:
    #             self.gymmy.r_shoulder_x.goto_position(-90, 2, wait=False)
    #             self.gymmy.l_shoulder_x.goto_position(90, 2, wait=False)
    #             self.gymmy.r_shoulder_y.goto_position(-170, 2, wait=False)
    #             self.gymmy.l_shoulder_y.goto_position(-170, 2, wait=True)
    #             time.sleep(1)
    #
    #
    #         if (rate=="fast"):
    #             if not s.did_training_paused and not s.stop_requested:
    #                 self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=False)
    #                 self.gymmy.l_shoulder_x.goto_position(10, 1, wait=True)
    #                 time.sleep(0.5)
    #             if not s.did_training_paused and not s.stop_requested:
    #                 self.gymmy.r_shoulder_x.goto_position(-90, 1, wait=False)
    #                 self.gymmy.l_shoulder_x.goto_position(90, 1, wait=True)
    #                 time.sleep(0.5)
    #
    #         elif (rate == "moderate"):
    #             if not s.did_training_paused and not s.stop_requested:
    #                 self.gymmy.r_shoulder_x.goto_position(-10, 1.25, wait=False)
    #                 self.gymmy.l_shoulder_x.goto_position(10, 1.25, wait=True)
    #                 time.sleep(1)
    #             if not s.did_training_paused and not s.stop_requested:
    #                 self.gymmy.r_shoulder_x.goto_position(-90, 1.25, wait=False)
    #                 self.gymmy.l_shoulder_x.goto_position(90, 1.25, wait=True)
    #                 time.sleep(1)
    #
    #         else:
    #             if not s.did_training_paused and not s.stop_requested:
    #                 self.gymmy.r_shoulder_x.goto_position(-10, 1.75, wait=False)
    #                 self.gymmy.l_shoulder_x.goto_position(10, 1.75, wait=True)
    #                 time.sleep(1.25)
    #             if not s.did_training_paused and not s.stop_requested:
    #                 self.gymmy.r_shoulder_x.goto_position(-90, 1.75, wait=False)
    #                 self.gymmy.l_shoulder_x.goto_position(90, 1.75, wait=True)
    #                 time.sleep(1.25)
    #
    #
    #         if i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
    #             # init
    #             self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
    #             self.gymmy.r_shoulder_y.goto_position(0, 2, wait=True)
    #             self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
    #             self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)



# -------------------------------------- Rubber band exercises --------------------------------------


    # EX6
    def band_open_arms(self, rate):
        if self.i=="demo":
            if not s.explanation_over:
                self.gymmy.r_arm_z.goto_position(90, 1, wait=False)
                self.gymmy.l_arm_z.goto_position(-90, 1, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                    self.gymmy.l_shoulder_y.goto_position(-100, 2, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-100, 2, wait=True)
                    time.sleep(1)

            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(40, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-40, 1, wait=True)
                time.sleep(2.5)
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(5, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-5, 1, wait=True)
                time.sleep(1)

            # init
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
            self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 1.5, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=True)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.r_arm_z.goto_position(90, 1, wait=False)
                self.gymmy.l_arm_z.goto_position(-90, 1, wait=True)
                self.gymmy.l_shoulder_y.goto_position(-100, 2, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-100, 2, wait=True)
                time.sleep(1)
                s.can_comment_robot = True


            if (rate=="fast"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(5, 1, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-5, 1, wait=True)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(40, 1, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-40, 1, wait=True)
                    time.sleep(0.5)

            elif (rate == "moderate"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(5, 1.25, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-5, 1.25, wait=True)
                    time.sleep(0.75)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(40, 1.25, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-40, 1.25, wait=True)
                    time.sleep(0.75)

            else:
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(5, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-5, 1.5, wait=True)
                    time.sleep(1)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(40, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-40, 1.5, wait=True)
                    time.sleep(1)


            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                # init
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
                self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
                self.gymmy.r_arm_z.goto_position(0, 1.5, wait=True)
                self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=True)
                self.did_init = True

    # EX7
    def band_open_arms_and_up(self, rate):
        if self.i =="demo":
            if not s.explanation_over:
                self.gymmy.r_arm_z.goto_position(90, 1, wait=False)
                self.gymmy.l_arm_z.goto_position(-90, 1, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-95, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-95, 1.5, wait=True)
                time.sleep(0.5)
                self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)

            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(30, 1.25, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-30, 1.25, wait=False)
                self.gymmy.r_elbow_y.goto_position(-20, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(-20, 1.25, wait=True)
                time.sleep(1.5)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-170, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-170, 1.5, wait=True)
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-95, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-95, 1.5, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(10, 1.25, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1.25, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
                time.sleep(0.75)
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(30, 1.25, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-30, 1.25, wait=False)
                self.gymmy.r_elbow_y.goto_position(-20, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(-20, 1.25, wait=True)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-170, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-170, 1.5, wait=True)
                time.sleep(0.5)

            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-95, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-95, 1.5, wait=True)
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(10, 1.25, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1.25, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
                time.sleep(1)

            # init
            self.gymmy.l_shoulder_x.goto_position(30, 1.25, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-30, 1.25, wait=False)
            self.gymmy.r_elbow_y.goto_position(-20, 1.25, wait=False)
            self.gymmy.l_elbow_y.goto_position(-20, 1.25, wait=True)
            self.gymmy.l_shoulder_y.goto_position(-95, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(-95, 1.5, wait=True)
            self.gymmy.r_elbow_y.goto_position(0, 1, wait=False)
            self.gymmy.l_elbow_y.goto_position(0, 1, wait=True)
            self.gymmy.l_arm_z.goto_position(0, 1, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 1, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.r_arm_z.goto_position(90, 0.5, wait=False)
                self.gymmy.l_arm_z.goto_position(-90, 0.5, wait=True)
                self.gymmy.l_shoulder_y.goto_position(-95, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-95, 1.5, wait=True)
                time.sleep(0.5)
                s.can_comment_robot = True

            if (rate=="fast"):
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-95, 1, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-95, 1, wait=True)
                        self.gymmy.l_shoulder_x.goto_position(10, 0.5, wait=False)
                        self.gymmy.r_shoulder_x.goto_position(-10, 0.5, wait=False)
                        self.gymmy.r_elbow_y.goto_position(0, 0.5, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 0.5, wait=True)
                        time.sleep(0.75)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(30, 0.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-30, 0.5, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-20, 0.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-20, 0.5, wait=True)
                    self.gymmy.l_shoulder_y.goto_position(-170, 1, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-170, 1, wait=True)
                    time.sleep(0.75)


            elif (rate == "moderate"):
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-95, 1.25, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-95, 1.25, wait=True)
                        self.gymmy.l_shoulder_x.goto_position(10, 0.75, wait=False)
                        self.gymmy.r_shoulder_x.goto_position(-10, 0.75, wait=False)
                        self.gymmy.r_elbow_y.goto_position(0, 0.75, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 0.75, wait=True)
                        time.sleep(0.75)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(30, 0.75, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-30, 0.75, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-20, 0.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-20, 0.75, wait=True)
                    self.gymmy.l_shoulder_y.goto_position(-170, 1.25, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-170, 1.25, wait=True)
                    time.sleep(1)


            else:
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-95, 1.5, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-95, 1.5, wait=True)
                        self.gymmy.l_shoulder_x.goto_position(10, 1.25, wait=False)
                        self.gymmy.r_shoulder_x.goto_position(-10, 1.25, wait=False)
                        self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
                        time.sleep(0.75)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(30, 1.25, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-30, 1.25, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-20, 1.25, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-20, 1.25, wait=True)
                    self.gymmy.l_shoulder_y.goto_position(-170, 1.5, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-170, 1.5, wait=True)
                    time.sleep(1)


            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                # init
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                self.gymmy.l_shoulder_x.goto_position(30, 1.25, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-30, 1.25, wait=False)
                self.gymmy.r_elbow_y.goto_position(-20, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(-20, 1.25, wait=True)
                self.gymmy.l_shoulder_y.goto_position(-95, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-95, 1.5, wait=True)
                self.gymmy.r_elbow_y.goto_position(0, 1, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1, wait=True)
                self.gymmy.l_arm_z.goto_position(0, 1, wait=False)
                self.gymmy.r_arm_z.goto_position(0, 1, wait=True)
                self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)
                self.did_init = True




    # EX8
    def band_up_and_lean(self, rate):
        if self.i == "demo":
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(30, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-30, 1, wait=False)
                self.gymmy.r_elbow_y.goto_position(-20, 1, wait=False)
                self.gymmy.l_elbow_y.goto_position(-20, 1, wait=True)
                self.gymmy.r_arm_z.goto_position(90, 1, wait=False)
                self.gymmy.l_arm_z.goto_position(-90, 1, wait=True)
            if not s.explanation_over:
                time.sleep(1)
                self.gymmy.l_shoulder_y.goto_position(-80, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-80, 1.5, wait=True)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-170, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-170, 1.5, wait=True)
                time.sleep(1)

            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(-30, 1.75, wait=True)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)

            # init
            self.gymmy.bust_x.goto_position(0, 1.5, wait=True)
            self.gymmy.l_shoulder_y.goto_position(-90, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(-90, 1.5, wait=True)
            self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 1.5, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 1, wait=False)
            self.gymmy.l_elbow_y.goto_position(0, 1, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=True)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_x.goto_position(30, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-30, 1, wait=False)
                self.gymmy.r_elbow_y.goto_position(-20, 1, wait=False)
                self.gymmy.l_elbow_y.goto_position(-20, 1, wait=True)
                self.gymmy.r_arm_z.goto_position(90, 0.5, wait=False)
                self.gymmy.l_arm_z.goto_position(-90, 0.5, wait=False)
                time.sleep(1)
                self.gymmy.l_shoulder_y.goto_position(-170, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-170, 1.5, wait=False)
                time.sleep(1.5)
                s.can_comment_robot = True

            if (rate=="fast"):
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 0.75, wait=True)
                        time.sleep(0.75)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(30, 0.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.75)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 0.75, wait=True)
                        time.sleep(0.75)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(-30, 0.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.75)


            elif (rate == "moderate"):
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.25, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(30, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.25, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(-30, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)

            else:
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(-30, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)

            if self.i >= s.rep or s.did_training_paused or s.stop_requested:
                # init
                self.gymmy.bust_x.goto_position(0, 1.5, wait=True)
                self.gymmy.l_shoulder_y.goto_position(-90, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-90, 1.5, wait=True)
                self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
                self.gymmy.r_arm_z.goto_position(0, 1.5, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 1, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1, wait=True)
                self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=True)
                self.did_init = True


    # EX9
    def band_straighten_left_arm_elbows_bend_to_sides(self, rate):
        if self.i == "demo":
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(95, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-95, 1.5, wait=True)
                self.gymmy.r_elbow_y.goto_position(-135, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
            if not s.explanation_over:
                time.sleep(1.5)

            if not s.explanation_over:
                for _ in range (4):
                    self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=True)
                    time.sleep(0.75)
                    if s.explanation_over:
                        break
                    self.gymmy.r_elbow_y.goto_position(-135, 1.25, wait=True)
                    time.sleep(0.75)
                    if s.explanation_over:
                        break


            self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=False)
            self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)

        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_x.goto_position(95, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-95, 1.5, wait=True)
                self.gymmy.r_elbow_y.goto_position(-135, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
                time.sleep(1)
                s.can_comment_robot = True

            if (rate == "fast"):
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(-135, 1, wait=True)
                        time.sleep(0.75)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(0, 1, wait=True)
                    time.sleep(0.75)


            elif (rate == "moderate"):
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(-135, 1.25, wait=True)
                        time.sleep(1)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=True)
                    time.sleep(1)


            else:
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(-135, 1.75, wait=True)
                        time.sleep(1.5)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=True)
                    time.sleep(1.5)


            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
                self.did_init = True


    # EX10
    def band_straighten_right_arm_elbows_bend_to_sides(self, rate):
        if self.i == "demo":
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(95, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-95, 1.5, wait=True)
                self.gymmy.r_elbow_y.goto_position(-135, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
            if not s.explanation_over:
                time.sleep(1.5)

            if not s.explanation_over:
                for _ in range(4):
                    self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
                    time.sleep(0.75)
                    if s.explanation_over:
                        break
                    self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
                    time.sleep(0.75)
                    if s.explanation_over:
                        break

            self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=False)
            self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_x.goto_position(95, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-95, 1.5, wait=True)
                self.gymmy.r_elbow_y.goto_position(-135, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
                time.sleep(1)
                s.can_comment_robot = True

            if (rate == "fast"):
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_elbow_y.goto_position(-135, 1, wait=True)
                        time.sleep(0.75)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_elbow_y.goto_position(0, 1, wait=True)
                    time.sleep(0.75)

            elif (rate == "moderate"):
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
                        time.sleep(1)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
                    time.sleep(1)

            else:
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_elbow_y.goto_position(-135, 1.75, wait=True)
                        time.sleep(1.5)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=True)
                    time.sleep(1.5)

            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
                self.did_init = True



    # -------------------------------------- Stick exercises ------------------------------------------------------------------

    # EX11
    def stick_bend_elbows(self, rate):
        if self.i == "demo":
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-50, 1, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-50, 1, wait=False)
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)
                time.sleep(1)

            if not s.explanation_over:
                for _ in range (3):
                    self.gymmy.r_elbow_y.goto_position(-135, 1.25, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
                    time.sleep(0.75)
                    if s.explanation_over:
                        break
                    self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=False)
                    self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
                    time.sleep(0.75)
                    if s.explanation_over:
                        break

            # init
            self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=True)

        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_y.goto_position(-50, 1, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-50, 1, wait=False)
                time.sleep(2)
                self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)
                time.sleep(1)
                s.can_comment_robot = True

            if (rate=="fast"):
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(0, 1.25, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
                        time.sleep(0.25)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-135, 1.25, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
                    time.sleep(0.25)

            elif (rate=="moderate"):
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=True)
                        time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-135, 1.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 1.5, wait=True)
                    time.sleep(0.5)

            else:
                if self.i!=0 and not s.needs_first_position:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=True)
                        time.sleep(0.75)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-135, 1.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 1.75, wait=True)
                    time.sleep(0.75)

            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                # init
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)
                self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=True)
                self.did_init = True


    # EX12
    def stick_bend_elbows_and_up(self, rate):
        if self.i == "demo":
            if not s.explanation_over:
                for _ in range(2):
                    self.gymmy.r_elbow_y.goto_position(-120, 1.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-120, 1.75, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-20, 1.75, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-20, 1.75, wait=True)
                    time.sleep(0.5)
                    if s.explanation_over:
                        break

                    self.gymmy.r_elbow_y.goto_position(-10, 1.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-10, 1.75, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(20, 1.75, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-20, 1.75, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-160, 1.75, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-160, 1.75, wait=True)
                    time.sleep(0.5)
                    if s.explanation_over:
                        break


            if not s.explanation_over:
                self.gymmy.r_elbow_y.goto_position(-120, 1.5, wait=False)
                self.gymmy.l_elbow_y.goto_position(-120, 1.5, wait=False)
                self.gymmy.l_shoulder_y.goto_position(-20, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-20, 1.5, wait=True)
                time.sleep(0.5)

            # init
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)

        else:
            s.can_comment_robot = True

            if (rate=="fast"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-120, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-120, 1, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-20, 1, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-20, 1, wait=True)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-10, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-10, 1, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(20, 1, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-20, 1, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-160, 1, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-160, 1, wait=True)
                    time.sleep(0.5)


            elif (rate == "moderate"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-120, 1.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-120, 1.5, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-20, 1.5, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-20, 1.5, wait=True)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-10, 1.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-10, 1.5, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(20, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-20, 1.5, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-160, 1.5, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-160, 1.5, wait=True)
                    time.sleep(0.5)

            else:
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-120, 1.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-120, 1.75, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-20, 1.75, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-20, 1.75, wait=True)
                    time.sleep(0.75)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(-10, 1.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-10, 1.75, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(20, 1.75, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-20, 1.75, wait=False)
                    self.gymmy.l_shoulder_y.goto_position(-160, 1.75, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-160, 1.75, wait=True)
                    time.sleep(0.75)


            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                # init
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)
                self.did_init = True


    # EX13
    def stick_raise_arms_above_head(self, rate):
        if self.i == "demo":
            self.gymmy.l_arm_z.goto_position(-90, 1, wait=False)
            self.gymmy.r_arm_z.goto_position(90, 1, wait=False)
            self.gymmy.l_shoulder_x.goto_position(30, 1, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-30, 1, wait=True)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-50, 1.75, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-50, 1.75, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-170, 2.25, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-170, 2.25, wait=True)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-50, 2.25, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-50, 2.25, wait=True)
            if not s.explanation_over:
                time.sleep(1)

            # init
            self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 1.5, wait=False)
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
            time.sleep(1)
            self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=False)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_arm_z.goto_position(-90, 1, wait=False)
                self.gymmy.r_arm_z.goto_position(90, 1, wait=False)
                self.gymmy.l_shoulder_x.goto_position(30, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-30, 1, wait=False)
                s.can_comment_robot = True


            if (rate=="fast"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_y.goto_position(-50, 1.25, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-50, 1.25, wait=True)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_y.goto_position(-170, 1.25, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-170, 1.25, wait=True)
                    time.sleep(0.5)

            elif (rate == "moderate"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_y.goto_position(-50, 1.5, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-50, 1.5, wait=True)
                    time.sleep(0.75)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_y.goto_position(-170, 1.5, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-170, 1.5, wait=True)
                    time.sleep(0.75)

            else:
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_y.goto_position(-50, 1.75, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-50, 1.75, wait=True)
                    time.sleep(1)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_y.goto_position(-170, 1.75, wait=False)
                    self.gymmy.r_shoulder_y.goto_position(-170, 1.75, wait=True)
                    time.sleep(1)



            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                # init
                self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
                self.gymmy.r_arm_z.goto_position(0, 1.5, wait=True)
                self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=True)
                self.did_init = True



    # EX14
    def stick_switch(self, rate):
        if self.i == "demo":
            time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-100, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-100, 1.5, wait=False)
                self.gymmy.l_arm_z.goto_position(-90, 1, wait=False)
                self.gymmy.r_arm_z.goto_position(90, 1, wait=True)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(-60, 1.75, wait=True)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(0, 1.75, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(60, 1.75, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(0, 1.25, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(-60, 1.5, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(0, 1.5, wait=True)
            if not s.explanation_over:
                time.sleep(2)


            # init
            self.gymmy.abs_z.goto_position(0, 1.5, wait=False)
            self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 1.5, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=True)



        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_y.goto_position(-100, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-100, 1.5, wait=False)
                self.gymmy.l_arm_z.goto_position(-90, 1, wait=False)
                self.gymmy.r_arm_z.goto_position(90, 1, wait=True)
                time.sleep(1)
                s.can_comment_robot = True

            if (rate=="fast"):
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(-60, 1, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.75)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1, wait=True)
                        time.sleep(0.75)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(60, 1, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.75)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1, wait=True)
                        time.sleep(0.75)

            elif (rate == "moderate"):
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(-60, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1.25, wait=True)
                        time.sleep(1)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(60, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1.25, wait=True)
                        time.sleep(1)

            else:
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(-60, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1.75, wait=True)
                        time.sleep(1)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(60, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(0, 1.75, wait=True)
                        time.sleep(1)


            if self.i >= s.rep or s.did_training_paused or s.stop_requested:
                # init
                self.gymmy.abs_z.goto_position(0, 1.5, wait=False)
                self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
                self.gymmy.r_arm_z.goto_position(0, 1.5, wait=True)
                self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=True)
                self.did_init = True


    # EX15
    def stick_up_and_lean(self, rate):
        if self.i == "demo":
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(30, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-30, 1, wait=False)
                self.gymmy.r_elbow_y.goto_position(-20, 1, wait=False)
                self.gymmy.l_elbow_y.goto_position(-20, 1, wait=False)
                self.gymmy.r_arm_z.goto_position(90, 0.5, wait=False)
                self.gymmy.l_arm_z.goto_position(-90, 0.5, wait=True)
                time.sleep(1)

            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-170, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-170, 1.5, wait=True)
                time.sleep(1)

            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(30, 1.25, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(0, 1.25, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(-30, 1.25, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(0, 1.25, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(30, 1.25, wait=True)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(0, 1.25, wait=True)


            # init
            self.gymmy.bust_x.goto_position(0, 1.5, wait=True)
            self.gymmy.l_shoulder_y.goto_position(-90, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(-90, 1.5, wait=True)
            self.gymmy.r_elbow_y.goto_position(0, 1, wait=False)
            self.gymmy.l_elbow_y.goto_position(0, 1, wait=False)
            self.gymmy.l_arm_z.goto_position(0, 0.75, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 0.75, wait=True)
            self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=True)

        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_x.goto_position(30, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-30, 1, wait=False)
                self.gymmy.r_elbow_y.goto_position(-20, 1, wait=False)
                self.gymmy.l_elbow_y.goto_position(-20, 1, wait=False)
                self.gymmy.r_arm_z.goto_position(90, 0.5, wait=False)
                self.gymmy.l_arm_z.goto_position(-90, 0.5, wait=True)
                time.sleep(1)
                self.gymmy.l_shoulder_y.goto_position(-170, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-170, 1.5, wait=True)
                time.sleep(1)
                s.can_comment_robot = True

            if (rate == "fast"):
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 0.75, wait=True)
                        time.sleep(0.75)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(30, 0.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.75)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 0.75, wait=True)
                        time.sleep(0.75)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(-30, 0.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.75)


            elif (rate == "moderate"):
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.25, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(30, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.25, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(-30, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)

            else:
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(-30, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)


            if self.i >= s.rep or s.did_training_paused or s.stop_requested:
                # init
                self.gymmy.bust_x.goto_position(0, 1.5, wait=True)
                self.gymmy.l_shoulder_y.goto_position(-90, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-90, 1.5, wait=True)
                self.gymmy.r_elbow_y.goto_position(0, 1, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1, wait=False)
                self.gymmy.l_arm_z.goto_position(0, 0.75, wait=False)
                self.gymmy.r_arm_z.goto_position(0, 0.75, wait=True)
                self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
                self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=True)
                self.did_init = True



    # -------------------------------------- Weights exercises ------------------------------------------------------------------

    # EX16
    def weights_right_hand_up_and_bend(self, rate):
        if self.i == "demo":
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_arm_z.goto_position(85, 1, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-60, 1, wait=False)
                self.gymmy.r_elbow_y.goto_position(-60, 1, wait=False)
                self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_elbow_y.goto_position(-20, 1.75, wait=False)
                self.gymmy.l_shoulder_x.goto_position(150, 2, wait=True)
            if not s.explanation_over:
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(20, 3, wait=False)
                self.gymmy.bust_x.goto_position(0, 2.25, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 2.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 2.25, wait=False)
                time.sleep(0.5)
                self.gymmy.r_elbow_y.goto_position(0, 2.25, wait=True)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.r_shoulder_x.goto_position(-60, 1, wait=False)
                self.gymmy.r_elbow_y.goto_position(-60, 1, wait=False)
                self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
            if not s.explanation_over:
                self.gymmy.l_elbow_y.goto_position(-20, 1.75, wait=False)
                self.gymmy.l_shoulder_x.goto_position(150, 2, wait=True)
            if not s.explanation_over:
                time.sleep(1)

            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(20, 3, wait=False)
                self.gymmy.bust_x.goto_position(0, 2.25, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 2.25, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 2.25, wait=False)
                time.sleep(0.5)
                self.gymmy.r_elbow_y.goto_position(0, 2.25, wait=True)

            # init
            self.gymmy.bust_x.goto_position(0, 2, wait=True)
            self.gymmy.l_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.l_arm_z.goto_position(0, 2, wait=True)

        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_arm_z.goto_position(85, 1, wait=False)
                s.can_comment_robot = True

            if (rate=="fast"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-60, 0.5, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-60, 0.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-20, 1, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(150, 1.5, wait=False)
                    self.gymmy.bust_x.goto_position(30, 1, wait=True)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(20, 1.5, wait=False)
                    self.gymmy.bust_x.goto_position(0, 1, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(0, 1, wait=False)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(0, 1, wait=True)
                    time.sleep(0.5)

            elif (rate == "moderate"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-60, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-60, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-20, 1.5, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(150, 2, wait=False)
                    self.gymmy.bust_x.goto_position(30, 1.5, wait=True)
                    time.sleep(1)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(20, 2, wait=False)
                    self.gymmy.bust_x.goto_position(0, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=False)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=True)
                    time.sleep(1)

            else:
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-60, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-60, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-20, 1.75, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(150, 2, wait=False)
                    self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
                    time.sleep(1.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(20, 2, wait=False)
                    self.gymmy.bust_x.goto_position(0, 1.75, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-10, 1.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=False)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=True)
                    time.sleep(1.5)

        if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
            if self.i == (s.rep - 1):
                self.i = s.rep
                s.robot_counter = self.i

            # init
            self.gymmy.bust_x.goto_position(0, 2, wait=True)
            self.gymmy.l_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.l_arm_z.goto_position(0, 2, wait=True)
            self.did_init = True

    # EX17
    def weights_left_hand_up_and_bend(self, rate):
        if self.i == "demo":
            time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
                self.gymmy.r_arm_z.goto_position(-85, 1, wait=False)

            if not s.explanation_over:
                for _ in range (4):
                    self.gymmy.l_shoulder_x.goto_position(60, 1.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-60, 1.75, wait=False)
                    self.gymmy.bust_x.goto_position(-30, 1.75, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-20, 1.75, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-150,1.75, wait=True)
                    time.sleep(1)
                    if s.explanation_over:
                        break

                    self.gymmy.r_shoulder_x.goto_position(-20, 1.75, wait=False)
                    self.gymmy.bust_x.goto_position(0, 1.75, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(10, 1.75, wait=False)
                    self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=False)
                    time.sleep(0.5)
                    self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=True)
                    time.sleep(1)
                    if s.explanation_over:
                        break



            # init
            self.gymmy.bust_x.goto_position(0, 2, wait=True)
            self.gymmy.l_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 2, wait=True)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.r_arm_z.goto_position(-85, 1, wait=False)
                s.can_comment_robot = True

            if (rate == "fast"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(60, 0.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-60, 0.5, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-20, 1, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-150, 1.5, wait=False)
                    self.gymmy.bust_x.goto_position(-30, 1, wait=True)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-20, 1.5, wait=False)
                    self.gymmy.bust_x.goto_position(0, 1, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(0, 1, wait=False)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_elbow_y.goto_position(0, 1, wait=True)
                    time.sleep(0.5)

            elif (rate == "moderate"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(60, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-60, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-20, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-150, 2, wait=False)
                    self.gymmy.bust_x.goto_position(-30, 1.5, wait=True)
                    time.sleep(1)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-20, 2, wait=False)
                    self.gymmy.bust_x.goto_position(0, 1.5, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
                    self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=False)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=True)
                    time.sleep(1)

            else:
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(60, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-60, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-20, 1.75, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-150, 2, wait=False)
                    self.gymmy.bust_x.goto_position(-30, 1.75, wait=True)
                    time.sleep(1.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-20, 2, wait=False)
                    self.gymmy.bust_x.goto_position(0, 1.75, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(10, 1.75, wait=False)
                    self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=False)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=True)
                    time.sleep(1.5)

            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                # init
                self.gymmy.bust_x.goto_position(0, 2, wait=True)
                self.gymmy.l_elbow_y.goto_position(0, 2, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 2, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
                self.gymmy.r_arm_z.goto_position(0, 2, wait=True)
                self.did_init = True


    # EX18
    def weights_open_arms_and_forward(self, rate):
        if self.i == "demo":
            time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-100, 2, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-100,2, wait=True)
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(75, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-75, 1.5, wait=True)
                time.sleep(0.5)

            if not s.explanation_over:
                for _ in range (3):
                    self.gymmy.l_shoulder_x.goto_position(-10, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(10, 1.5, wait=True)
                    time.sleep(0.5)
                    if s.explanation_over:
                        break
                    self.gymmy.l_shoulder_x.goto_position(75, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-75, 1.5, wait=True)
                    time.sleep(0.5)
                    if s.explanation_over:
                        break


            # init
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 2, wait=True)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_y.goto_position(-100, 2, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-100, 2, wait=True)
                self.gymmy.l_shoulder_x.goto_position(75, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-75, 1.5, wait=True)
                time.sleep(1)
                s.can_comment_robot = True

            if (rate == "fast"):
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(0.5)
                    self.gymmy.l_shoulder_x.goto_position(-10, 1, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(10, 1, wait=True)
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(0.5)
                    self.gymmy.l_shoulder_x.goto_position(75, 1, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-75, 1, wait=True)

            elif (rate == "moderate"):
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(1)
                    self.gymmy.l_shoulder_x.goto_position(-10, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(10, 1.5, wait=True)
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(1)
                    self.gymmy.l_shoulder_x.goto_position(75, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-75, 1.5, wait=True)

            else:
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(1)
                    self.gymmy.l_shoulder_x.goto_position(-10, 2, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(10, 2, wait=True)
                if not s.did_training_paused and not s.stop_requested:
                    time.sleep(1)
                    self.gymmy.l_shoulder_x.goto_position(75, 2, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-75, 2, wait=True)

            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                # init
                time.sleep(1)
                self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=True)
                self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 2, wait=True)
                self.did_init = True


    # EX19
    def weights_abduction(self, rate):
        if self.i == "demo":
            time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(1)

            if not s.explanation_over:
                for _ in range (2):
                    self.gymmy.r_shoulder_x.goto_position(-100, 2, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(100, 2, wait=True)
                    time.sleep(0.75)
                    if s.explanation_over:
                        break
                    self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(10, 2, wait=True)
                    time.sleep(0.75)
                    if s.explanation_over:
                        break

            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=True)
            time.sleep(1)

        else:
            s.can_comment_robot = True

            if (rate == "fast"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(10, 1, wait=True)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-100, 1, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(100, 1, wait=True)
                    time.sleep(0.5)


            elif (rate == "moderate"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=True)
                    time.sleep(0.75)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-100, 1.5, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(100, 1.5, wait=True)
                    time.sleep(0.75)

            else:
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(10, 2, wait=True)
                    time.sleep(1)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-100, 2, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(100, 2, wait=True)
                    time.sleep(1)

            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 2, wait=True)
                self.did_init = True


    # -------------------------------------- No equipment exercises ------------------------------------------------------------------

    # EX20
    def notool_hands_behind_and_lean(self, rate):
        if self.i == "demo":
            time.sleep(2)
            if not s.explanation_over:
                time.sleep(1)
            if not s.explanation_over:
                self.gymmy.l_arm_z.goto_position(85, 1.5, wait=False)
                self.gymmy.r_arm_z.goto_position(-85, 1.5, wait=False)
                self.gymmy.l_shoulder_x.goto_position(120, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-120, 1.5, wait=False)
                self.gymmy.l_elbow_y.goto_position(-120, 1.5, wait=False)
                self.gymmy.r_elbow_y.goto_position(-120, 1.5, wait=True)

            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
                time.sleep(1.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(-30, 1.75, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
            if not s.explanation_over:
                time.sleep(1.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
            if not s.explanation_over:
                time.sleep(1.5)
            if not s.explanation_over:
                self.gymmy.bust_x.goto_position(0, 1.5, wait=True)


            # init
            self.gymmy.bust_x.goto_position(0, 1.5, wait=True)
            self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=True)
            self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
            self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 1.5, wait=False)
            self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=True)

        else:
            if self.i ==0 or s.needs_first_position:
                self.gymmy.l_arm_z.goto_position(85, 1.5, wait=False)
                self.gymmy.r_arm_z.goto_position(-85, 1.5, wait=False)
                self.gymmy.l_shoulder_x.goto_position(120, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-120, 1.5, wait=False)
                self.gymmy.l_elbow_y.goto_position(-120, 1.5, wait=False)
                self.gymmy.r_elbow_y.goto_position(-120, 1.5, wait=True)
                s.can_comment_robot = True

            if (rate=="fast"):
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 0.75, wait=True)
                        time.sleep(0.75)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(30, 0.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.75)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 0.75, wait=True)
                        time.sleep(0.75)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(-30, 0.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.75)

            elif (rate == "moderate"):
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.25, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(30, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.25, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(-30, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)


            else:
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.75, wait=True)
                        time.sleep(1)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(-30, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(1)


            if self.i >= s.rep or s.did_training_paused or s.stop_requested:
                # init
                self.gymmy.bust_x.goto_position(0, 1.5, wait=True)
                self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=True)
                self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
                self.gymmy.l_arm_z.goto_position(0, 1.5, wait=False)
                self.gymmy.r_arm_z.goto_position(0, 1.5, wait=False)
                self.gymmy.l_shoulder_y.goto_position(0, 1.5, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 1.5, wait=True)
                self.did_init = True


    # EX21
    def notool_right_hand_up_and_bend(self, rate):
        if self.i == "demo":
            if not s.explanation_over:
                self.gymmy.l_arm_z.goto_position(85, 1, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(150, 2, wait=False)
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.r_shoulder_x.goto_position(-40, 1, wait=False)
                self.gymmy.r_elbow_y.goto_position(-40, 1, wait=False)
                self.gymmy.l_elbow_y.goto_position(-20, 1.75, wait=False)
                self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                    time.sleep(2)

            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(20, 2, wait=False)
                self.gymmy.bust_x.goto_position(0, 1.75, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1.75, wait=False)
                self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=False)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=True)
                time.sleep(1.5)

            # init
            self.gymmy.bust_x.goto_position(0, 2, wait=True)
            self.gymmy.l_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.l_arm_z.goto_position(0, 2, wait=True)

        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_arm_z.goto_position(85, 1, wait=False)
                s.can_comment_robot = True

            if (rate=="fast"):
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1, wait=False)
                        self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 1, wait=False)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(0, 1, wait=True)
                        time.sleep(0.5)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-40, 0.5, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-40, 0.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-20, 1, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(150, 2, wait=False)
                    self.gymmy.bust_x.goto_position(30, 1, wait=True)
                    time.sleep(0.5)

            elif (rate == "moderate"):
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.5, wait=False)
                        self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=False)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=True)
                        time.sleep(1)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-40, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-40, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-20, 1.5, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(150, 2, wait=False)
                    self.gymmy.bust_x.goto_position(30, 1.5, wait=True)
                    time.sleep(1)


            else:
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.75, wait=False)
                        self.gymmy.r_shoulder_x.goto_position(-10, 1.75, wait=False)
                        self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=False)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=True)
                        time.sleep(1.5)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-40, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-40, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-20, 1.75, wait=False)
                    self.gymmy.l_shoulder_x.goto_position(150, 2, wait=False)
                    self.gymmy.bust_x.goto_position(30, 1.75, wait=True)
                    time.sleep(1.5)


        if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
            if self.i == (s.rep - 1):
                self.i = s.rep
                s.robot_counter = self.i

            # init
            self.gymmy.bust_x.goto_position(0, 2, wait=True)
            self.gymmy.l_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.l_arm_z.goto_position(0, 2, wait=True)
            self.did_init = True

    # EX22
    def notool_left_hand_up_and_bend(self, rate):
        if self.i == "demo":
            if not s.explanation_over:
                self.gymmy.r_arm_z.goto_position(-85, 1, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.r_shoulder_x.goto_position(-150, 2, wait=False)
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(40, 1, wait=False)
                self.gymmy.l_elbow_y.goto_position(-40, 1, wait=False)
                self.gymmy.r_elbow_y.goto_position(-20, 1.75, wait=False)
                self.gymmy.bust_x.goto_position(-30, 1.75, wait=True)
                time.sleep(1)
            if not s.explanation_over:
                time.sleep(2)

            if not s.explanation_over:
                self.gymmy.r_shoulder_x.goto_position(-20, 2, wait=False)
                self.gymmy.bust_x.goto_position(0, 1.75, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 1.75, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=False)
                time.sleep(0.5)
            if not s.explanation_over:
                self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=True)
                time.sleep(1.5)

            # init
            self.gymmy.bust_x.goto_position(0, 2, wait=True)
            self.gymmy.l_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
            self.gymmy.r_arm_z.goto_position(0, 2, wait=True)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.r_arm_z.goto_position(-85, 1, wait=False)
                s.can_comment_robot = True

            if (rate=="fast"):
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1, wait=False)
                        self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                        self.gymmy.r_elbow_y.goto_position(0, 1, wait=False)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_elbow_y.goto_position(0, 1, wait=True)
                        time.sleep(0.5)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(40, 0.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-40, 0.5, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-20, 1, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-150, 1.5, wait=False)
                    self.gymmy.bust_x.goto_position(-30, 1, wait=True)
                    time.sleep(0.5)


            elif (rate == "moderate"):
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.5, wait=False)
                        self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
                        self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=False)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=True)
                        time.sleep(1)

                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(40, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-40, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-20, 1.5, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-150, 2, wait=False)
                    self.gymmy.bust_x.goto_position(-30, 1.5, wait=True)
                    time.sleep(1)


            else:
                if self.i != 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.bust_x.goto_position(0, 1.75, wait=False)
                        self.gymmy.l_shoulder_x.goto_position(10, 1.75, wait=False)
                        self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=False)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_elbow_y.goto_position(0, 1.75, wait=True)
                        time.sleep(1.5)


                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(40, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-40, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-20, 1.75, wait=False)
                    self.gymmy.r_shoulder_x.goto_position(-150, 2, wait=False)
                    self.gymmy.bust_x.goto_position(-30, 1.75, wait=True)
                    time.sleep(1.5)



            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                # init
                self.gymmy.bust_x.goto_position(0, 2, wait=True)
                self.gymmy.l_elbow_y.goto_position(0, 2, wait=False)
                self.gymmy.r_elbow_y.goto_position(0, 2, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 2, wait=False)
                self.gymmy.r_arm_z.goto_position(0, 2, wait=True)
                self.did_init = True



    # EX 23
    def notool_raising_hands_diagonally(self, rate):
        if self.i=="demo":
            self.gymmy.r_arm_z.goto_position(20, 1.25, wait=False)
            self.gymmy.l_arm_z.goto_position(-20, 1.25, wait=False)
            self.gymmy.l_shoulder_y.goto_position(-65, 1.25, wait=False)
            self.gymmy.r_shoulder_y.goto_position(-65, 1.25, wait=True)
            self.gymmy.r_shoulder_x.goto_position(10, 1.25, wait=False)
            self.gymmy.l_shoulder_x.goto_position(-10, 1.25, wait=True)

            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-65, 1.75, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-65, 1.75, wait=False)
                self.gymmy.abs_z.goto_position(-25, 1.75, wait=True)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                time.sleep(1)
            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(60, 1.75, wait=False)
                self.gymmy.l_shoulder_y.goto_position(-130, 1.75, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-130, 1.75, wait=True)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-65, 1.75, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-65, 1.75, wait=False)
                self.gymmy.abs_z.goto_position(-25, 1.75, wait=True)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-65, 1.75, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-65, 1.75, wait=False)
                self.gymmy.abs_z.goto_position(25, 1.75, wait=True)
            if not s.explanation_over:
                time.sleep(2)
            if not s.explanation_over:
                self.gymmy.abs_z.goto_position(-60, 1.75, wait=False)
                self.gymmy.l_shoulder_y.goto_position(-130, 1.75, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-130, 1.75, wait=True)
            if not s.explanation_over:
                time.sleep(1.5)
            if not s.explanation_over:
                self.gymmy.l_shoulder_y.goto_position(-65, 1.75, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-65, 1.75, wait=False)
                self.gymmy.abs_z.goto_position(25, 1.75, wait=True)
                time.sleep(0.5)

            self.gymmy.abs_z.goto_position(0, 2, wait=True)
            self.gymmy.r_arm_z.goto_position(0, 0.75, wait=False)
            self.gymmy.l_arm_z.goto_position(0, 0.75, wait=True)
            self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 2, wait=True)
            self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 2, wait=True)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.r_arm_z.goto_position(20, 1.25, wait=False)
                self.gymmy.l_arm_z.goto_position(-20, 1.25, wait=False)
                self.gymmy.l_shoulder_y.goto_position(-65, 1.25, wait=False)
                self.gymmy.r_shoulder_y.goto_position(-65, 1.25, wait=True)
                self.gymmy.r_shoulder_x.goto_position(10, 1.25, wait=False)
                self.gymmy.l_shoulder_x.goto_position(-10, 1.25, wait=True)
                s.can_comment_robot = True

            if rate == "fast":
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.25, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.25, wait=False)
                        self.gymmy.abs_z.goto_position(-25, 1.25, wait=True)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(60, 1.25, wait=False)
                        self.gymmy.l_shoulder_y.goto_position(-130, 1.25, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-130, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.25, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.25, wait=False)
                        self.gymmy.abs_z.goto_position(-25, 1.25, wait=True)
                        time.sleep(0.5)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.25, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.25, wait=False)
                        self.gymmy.abs_z.goto_position(25, 1.25, wait=True)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(-60, 1.25, wait=False)
                        self.gymmy.l_shoulder_y.goto_position(-130, 1.25, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-130, 1.25, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.25, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.25, wait=False)
                        self.gymmy.abs_z.goto_position(25, 1.25, wait=True)
                        time.sleep(0.5)

            elif rate == "moderate":
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.5, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.5, wait=False)
                        self.gymmy.abs_z.goto_position(-25, 1.5, wait=True)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(60, 1.5, wait=False)
                        self.gymmy.l_shoulder_y.goto_position(-130, 1.5, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-130, 1.5, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.5, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.5, wait=False)
                        self.gymmy.abs_z.goto_position(-25, 1.5, wait=True)
                        time.sleep(0.5)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.5, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.5, wait=False)
                        self.gymmy.abs_z.goto_position(25, 1.5, wait=True)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(-60, 1.5, wait=False)
                        self.gymmy.l_shoulder_y.goto_position(-130, 1.5, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-130, 1.5, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.5, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.5, wait=False)
                        self.gymmy.abs_z.goto_position(25, 1.5, wait=True)
                        time.sleep(0.5)

            else:
                if self.i % 2 == 0:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.75, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.75, wait=False)
                        self.gymmy.abs_z.goto_position(-25, 1.75, wait=True)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(60, 1.75, wait=False)
                        self.gymmy.l_shoulder_y.goto_position(-130, 1.75, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-130, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.75, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.75, wait=False)
                        self.gymmy.abs_z.goto_position(-25, 1.75, wait=True)
                        time.sleep(0.5)

                if self.i < s.rep:
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.75, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.75, wait=False)
                        self.gymmy.abs_z.goto_position(25, 1.75, wait=True)
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.abs_z.goto_position(-60, 1.75, wait=False)
                        self.gymmy.l_shoulder_y.goto_position(-130, 1.75, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-130, 1.75, wait=True)
                        self.i += 1
                        s.robot_counter = self.i
                        time.sleep(0.5)
                    if not s.did_training_paused and not s.stop_requested:
                        self.gymmy.l_shoulder_y.goto_position(-65, 1.75, wait=False)
                        self.gymmy.r_shoulder_y.goto_position(-65, 1.75, wait=False)
                        self.gymmy.abs_z.goto_position(25, 1.75, wait=True)
                        time.sleep(0.5)

            if self.i >= s.rep or s.did_training_paused or s.stop_requested:
                self.gymmy.abs_z.goto_position(0, 2, wait=True)
                self.gymmy.r_arm_z.goto_position(0, 0.75, wait=False)
                self.gymmy.l_arm_z.goto_position(0, 0.75, wait=True)
                self.gymmy.r_shoulder_x.goto_position(-10, 2, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 2, wait=True)
                self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 2, wait=True)
                self.did_init = True

    # EX 24
    def notool_right_bend_left_up_from_side(self, rate):
        if self.i == "demo":
            self.gymmy.l_shoulder_y.goto_position(-20, 1, wait=True)
            time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
            if not s.explanation_over:
                self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
            if not s.explanation_over:
                self.gymmy.l_elbow_y.goto_position(-135, 1.25, wait=True)
            if not s.explanation_over:
                self.gymmy.l_elbow_y.goto_position(0, 1.25, wait=True)
            if not s.explanation_over:
                self.gymmy.r_shoulder_x.goto_position(-100, 1.25, wait=True)
            if not s.explanation_over:
                self.gymmy.r_shoulder_x.goto_position(-10, 1.25, wait=True)

            if not s.explanation_over:
                for _ in range (2):
                    self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 1, wait=True)
                    time.sleep(0.25)
                    if s.explanation_over:
                        break
                    self.gymmy.r_shoulder_x.goto_position(-100, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(0, 1, wait=True)
                    time.sleep(0.25)
                    if s.explanation_over:
                        break


            self.gymmy.l_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_x.goto_position(-10, 1.25, wait=True)

        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.l_shoulder_y.goto_position(-20, 1, wait=True)
                time.sleep(0.5)
                s.can_comment_robot = True

            if (rate == "fast"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-10, 0.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 0.75, wait=True)
                    time.sleep(0.25)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-100, 0.75, wait=False)
                    self.gymmy.l_elbow_y.goto_position(0, 0.75, wait=True)
                    time.sleep(0.25)

            elif (rate == "moderate"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-10, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 1, wait=True)
                    time.sleep(0.25)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-100, 1, wait=False)
                    self.gymmy.l_elbow_y.goto_position(0, 1, wait=True)
                    time.sleep(0.25)


            else:
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-10, 1.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(-135, 1.5, wait=True)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.r_shoulder_x.goto_position(-100, 1.5, wait=False)
                    self.gymmy.l_elbow_y.goto_position(0, 1.5, wait=True)
                    time.sleep(0.5)


            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                self.gymmy.l_elbow_y.goto_position(0, 2, wait=False)
                self.gymmy.l_shoulder_y.goto_position(0, 2, wait=False)
                self.gymmy.r_shoulder_x.goto_position(-10, 1.75, wait=True)
                self.did_init = True


    # EX 25
    def notool_left_bend_right_up_from_side(self, rate):
        if self.i == "demo":
            self.gymmy.r_shoulder_y.goto_position(-20, 1, wait=True)
            time.sleep(2)
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(100, 1.75, wait=True)
            if not s.explanation_over:
                self.gymmy.l_shoulder_x.goto_position(10, 1.75, wait=True)
            if not s.explanation_over:
                self.gymmy.r_elbow_y.goto_position(-135, 1.75, wait=True)
            if not s.explanation_over:
                self.gymmy.r_elbow_y.goto_position(0, 1.75, wait=True)
                time.sleep(1)

            if not s.explanation_over:
                for _ in range(2):
                    self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-135, 1, wait=True)
                    time.sleep(0.25)
                    if s.explanation_over:
                        break
                    self.gymmy.l_shoulder_x.goto_position(100, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(0, 1, wait=True)
                    time.sleep(0.25)
                    if s.explanation_over:
                        break


            self.gymmy.r_elbow_y.goto_position(0, 2, wait=False)
            self.gymmy.r_shoulder_y.goto_position(0, 2, wait=False)
            self.gymmy.l_shoulder_x.goto_position(10, 1.25, wait=True)


        else:
            if self.i==0 or s.needs_first_position:
                self.gymmy.r_shoulder_y.goto_position(-20, 1, wait=True)
                time.sleep(0.5)
                s.can_comment_robot = True

            if (rate == "fast"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(10, 0.75, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-135, 0.75, wait=True)
                    time.sleep(0.25)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(100, 0.75, wait=False)
                    self.gymmy.r_elbow_y.goto_position(0, 0.75, wait=True)
                    time.sleep(0.25)

            elif (rate == "moderate"):
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(10, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-135, 1, wait=True)
                    time.sleep(0.25)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(100, 1, wait=False)
                    self.gymmy.r_elbow_y.goto_position(0, 1, wait=True)
                    time.sleep(0.25)


            else:
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(10, 1.5, wait=False)
                    self.gymmy.r_elbow_y.goto_position(-135, 1.5, wait=True)
                    time.sleep(0.5)
                if not s.did_training_paused and not s.stop_requested:
                    self.gymmy.l_shoulder_x.goto_position(100, 1.5, wait=False)
                    self.gymmy.r_elbow_y.goto_position(0, 1.5, wait=True)
                    time.sleep(0.5)


            if self.i == (s.rep - 1) or s.did_training_paused or s.stop_requested:
                if self.i == (s.rep - 1):
                    self.i = s.rep
                    s.robot_counter = self.i

                self.gymmy.r_elbow_y.goto_position(0, 2, wait=False)
                self.gymmy.r_shoulder_y.goto_position(0, 2, wait=False)
                self.gymmy.l_shoulder_x.goto_position(10, 1.25, wait=True)
                self.did_init = True



if __name__ == "__main__":
    s.rep = 5
    s.success_exercise = False
    s.finish_workout = False

    #########################למחוק
    s.robot_count = False
    language = 'Hebrew'
    gender = 'Male'
    s.audio_path = 'audio files/' + language + '/' + gender + '/'
    #s.picture_path = 'audio files/' + language + '/' + gender + '/'
    ###########################################################
    s.waved=True
    s.finish_workout=False
    s.did_training_paused=False
    s.stop_requested=False
    #######################################################################
    #s.req_exercise ="bend_elbows_ball"
    #s.req_exercise = "raise_arms_above_head_ball"
    #s.req_exercise = "raise_arms_forward_turn_ball"
    #s.req_exercise = "open_arms_and_forward_ball"
    #s.req_exercise = "open_arms_above_head_ball"
    #s.req_exercise = "open_arms_with_band"
    #s.req_exercise = "open_arms_and_up_with_band"
    #s.req_exercise = "up_with_band_and_lean"
    #s.req_exercise = "bend_elbows_stick"
    #s.req_exercise = "bend_elbows_and_up_stick"
    #s.req_exercise = "arms_up_and_down_stick"
    #s.req_exercise = "switch_with_stick"
    #s.req_exercise = "hands_behind_and_lean_notool"
    #s.req_exercise="right_hand_up_and_bend_notool"
    #s.req_exercise = "left_hand_up_and_bend_notool"
    #s.req_exercise = "raising_hands_diagonally_notool"
    #s.req_exercise = "notool_bending_forward"
    #s.req_exercise = "weights_hands_up_and_bend_backwards"
    #s.req_exercise = "weights_right_hand_to_side_left_up"
    #s.req_exercise = "band_open_arms"
    #s.req_exercise = "weights_abduction"
    s.req_exercise = "notool_right_hand_up_and_bend"
    s.rate="moderate"
    s.additional_audio_playing = False
    s.volume = 0
    s.play_song = False
    s.gender = "Female"
    s.finish_program = False
    s.explanation_over = False
    s.skipped_exercise = False
    robot = Gymmy()
    s.patient_repetitions_counting_in_exercise=0
    #mp=MP()
    #mp.start()
    robot.start()
    pygame.mixer.init()
    s.stop_song = False
    s.audio_manager = AdditionalAudio()
    # Start continuous audio in a separate thread
    s.continuous_audio = ContinuousAudio()
    s.continuous_audio.start()

    s.screen = Screen()

    s.screen.switch_frame(ExplanationPage, exercise=s.req_exercise)

    app = FullScreenApp(s.screen)
    s.screen.mainloop()




    #robot.join()
    #robot.exercise_demo("bend_elbows_ball")

   # s.robot.exercise_demo("open_and_close_arms_90")
    #s.robot.exercise_demo("raise_arms_forward")
    # robot.start()
    #time.sleep(10)

    # robot.poppy.l_shoulder_y.goto_position(-90, 1.5, wait=False)
    # robot.poppy.r_shoulder_y.goto_position(-90, 1.5, wait=True)
    # robot.poppy.l_shoulder_x.goto_position(90, 1, wait=False)
    # # robot.poppy.r_shoulder_x.goto_position(-90, 1, wait=True)
    # robot.poppy.l_elbow_y.goto_position(0, 1.5, wait=False)
    # robot.poppy.r_elbow_y.goto_position(0, 1.5, wait=True)

    #robot.join()
    #time.sleep(5)
    #s.finish_workout=True

    #sys.exit()
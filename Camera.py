import copy
import json
import math
import random

import pygame
import pyzed.sl as sl
import threading
import socket
from Audio import say, get_wav_duration
from Joint_zed import Joint
from PyZedWrapper import PyZedWrapper
import Settings as s
import time
import Excel
from scipy.signal import lfilter


import numpy as np

# class KalmanFilter:
#     def __init__(self, initial_state=None):
#         self.x = np.zeros(3) if initial_state is None else np.array(initial_state)  # Initial state
#         self.P = np.eye(3) * 10  # Initial covariance
#         self.F = np.eye(3)  # State transition matrix
#         self.H = np.eye(3)  # Observation matrix
#         self.R = np.eye(3) * 0.5  # Measurement noise covariance
#         self.Q = np.eye(3) * 0.5 # Process noise covariance
#
#     def predict(self):
#         self.x = self.F @ self.x  # Predict state
#         self.P = self.F @ self.P @ self.F.T + self.Q  # Predict covariance
#
#     def update(self, z):
#         if z is None or np.any(np.isnan(z)) or len(z) != 3:  # Check for null or invalid measurements
#             print("Invalid or missing measurement, using prediction only.")
#             self.predict()  # Use prediction step only
#             return self.x  # Return the predicted state
#
#         z = np.array(z)
#         y = z - (self.H @ self.x)  # Measurement residual
#         S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
#         K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
#         self.x = self.x + (K @ y)  # Update state
#         self.P = (np.eye(len(self.P)) - K @ self.H) @ self.P  # Update covariance
#         return self.x



from scipy.signal import butter, filtfilt


class ButterworthFilter:
    def __init__(self, order=2, cutoff=5, min_samples=5):
        self.b, self.a = butter(order, cutoff / (0.5 * s.fps), btype='low', analog=False)
        self.history = []  # Stores previous positions for filtering
        self.min_samples = min_samples  # Avoid filtfilt errors

    def update(self, measurement):
        """Apply Butterworth filter with interpolation for missing values."""
        measurement = np.array(measurement, dtype=np.float32)

        # If measurement is invalid, try interpolation
        if measurement is None or np.any(np.isnan(measurement)) or np.all(measurement == 0):
            if len(self.history) > 1:
                # Interpolate missing value
                measurement = self.interpolate_missing_value()
            else:
                return np.zeros(3)

        # Store last `min_samples` values
        self.history.append(measurement)
        if len(self.history) > self.min_samples:
            self.history.pop(0)

        # If not enough data, return raw measurement
        if len(self.history) < self.min_samples:
            return measurement

        # Apply Butterworth filter
        data = np.array(self.history)
        try:
            filtered_x = lfilter(self.b, self.a, data[:, 0])[-1]
            filtered_y = lfilter(self.b, self.a, data[:, 1])[-1]
            filtered_z = lfilter(self.b, self.a, data[:, 2])[-1]
            return np.array([filtered_x, filtered_y, filtered_z])

        except ValueError:
            return measurement  # Return unfiltered value if error occurs

    def interpolate_missing_value(self):
        """Interpolates missing values using linear interpolation."""
        if len(self.history) < 2:
            return np.zeros(3)

        last_valid = np.array(self.history[-1])
        second_last_valid = np.array(self.history[-2])

        # Linear interpolation
        interpolated = last_valid + (last_valid - second_last_valid) * 0.5
        return interpolated



class MovingAverageFilter:
    def __init__(self, window_size=3, max_null_extrapolation=500, max_jump=100.0):
        self.window_size = window_size
        self.max_null_extrapolation = max_null_extrapolation
        self.max_jump = max_jump
        self.previous_positions = []
        self.consecutive_invalid_measurements = 0
        self.last_valid_position = None
        self.last_velocity = None

    def update(self, measurement):
        # Convert measurement to NumPy array
        measurement = np.array(measurement, dtype=np.float32)

        # Check for invalid measurements
        if measurement is None or np.any(np.isnan(measurement)) or np.all(measurement == 0):
            self.consecutive_invalid_measurements += 1
            if self.last_velocity is not None and self.consecutive_invalid_measurements < self.max_null_extrapolation:
                measurement = self.extrapolate_position()
            else:
                measurement = self.last_valid_position if self.last_valid_position is not None else np.zeros(3)
        else:
            # Handle sudden jumps
            if self.last_valid_position is not None:
                # Ensure both are NumPy arrays for subtraction
                last_position = np.array(self.last_valid_position, dtype=np.float32)
                if np.linalg.norm(measurement - last_position) > self.max_jump:
                    measurement = last_position
                else:
                    self.consecutive_invalid_measurements = 0
                    self.last_velocity = self.calculate_velocity(measurement)
                    self.last_valid_position = measurement

        # Add to window and trim
        self.previous_positions.append(measurement)
        if len(self.previous_positions) > self.window_size:
            self.previous_positions.pop(0)

        return self.calculate_moving_average()

    def calculate_velocity(self, measurement):
        if self.last_valid_position is None:
            return np.zeros(3)
        # Ensure both are NumPy arrays for subtraction
        last_position = np.array(self.last_valid_position, dtype=np.float32)
        return measurement - last_position

    def extrapolate_position(self):
        if self.last_velocity is None or self.last_valid_position is None:
            return np.zeros(3)
        return self.last_valid_position + self.last_velocity

    def calculate_moving_average(self):
        if len(self.previous_positions) == 0:
            return np.zeros(3)
        return np.mean(np.array(self.previous_positions), axis=0)


class Camera(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('localhost', 7000)
        self.sock.bind(self.server_address)
        print("CAMERA INITIALIZATION")
        self.frame_count = 0
        self.start_time = None
        self.joints = {}
        self.previous_angles = {}
        self.max_angle_jump = 15

        self.first_coordination_ex = True
        # Define the keys
        keys = ["nose", "neck", "R_shoulder", "R_elbow", "R_wrist", "L_shoulder", "L_elbow", "L_wrist",
                "R_hip", "R_knee", "R_ankle", "L_hip", "L_knee", "L_ankle", "R_eye", "L_eye", "R_ear", "L_ear"]

        # Create the dictionary with empty lists
        self.body_parts_dict = {key: [] for key in keys}

    def calc_angle_3d(self, joint1, joint2, joint3, joint_name="default"):
        a = np.array([joint1.x, joint1.y, joint1.z], dtype=np.float32)
        b = np.array([joint2.x, joint2.y, joint2.z], dtype=np.float32)
        c = np.array([joint3.x, joint3.y, joint3.z], dtype=np.float32)

        ba = a - b  # Vector from joint2 to joint1
        bc = c - b  # Vector from joint2 to joint3

        try:
            # Compute cosine of the angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

            # ✅ Fix: Clamp cosine value between -1 and 1 to prevent NaN errors
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

            # Convert to degrees
            angle = np.degrees(np.arccos(cosine_angle))

            # ✅ Handle cases where the angle might get stuck at 180° due to straight alignment
            if np.isclose(cosine_angle, -1.0, atol=1e-3):
                angle -= 0.1  # Small shift to prevent it from sticking

            # ✅ Ensure angle smoothing to avoid sudden jumps
            if joint_name in self.previous_angles:
                angle = self.limit_angle_jump(angle, joint_name)

            self.previous_angles[joint_name] = angle

            return round(angle, 2)

        except Exception as e:
            print(f"⚠️ Could not calculate the angle for {joint_name}: {e}")
            return None



    def limit_angle_jump(self, angle, joint_name):
        previous_angle = self.previous_angles[joint_name]
        if abs(angle - previous_angle) > self.max_angle_jump:
            direction = 1 if angle > previous_angle else -1
            angle = previous_angle + direction * self.max_angle_jump
        return angle

    def safe_mean(self, values, name):
        if not values:  # If the list is empty
            print(f"⚠️ Warning: {name} list is empty! Returning None.")
            return None

        # Convert to NumPy array and check if all elements are NaN
        values_array = np.array(values, dtype=np.float32)

        if np.isnan(values_array).all():  # If all values are NaN, return None
            print(f"⚠️ Warning: All values in {name} are NaN! Returning None.")
            return None

        return float(np.nanmean(values))  # Compute mean safely

    import math

    def euclidean_distance(self, p1, p2):
        if None in p1 or None in p2:
            return None
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2 +
            (p1[2] - p2[2]) ** 2
        )


    def run(self):
        print("CAMERA START")
        s.zed_camera = PyZedWrapper()
        s.zed_camera.start()

        # self.zed = PyZedWrapper.get_zed(s.zed_camera)

        while not s.finish_program:
            time.sleep(0.0001)
            if s.asked_for_measurement and not s.finished_calibration and not s.screen_finished_counting:
                time.sleep(1)
                s.zed_camera.set_detection_model_to_accurate()
                while not s.screen_finished_counting and not s.finish_program:
                    time.sleep(0.001)

                # Initialize lists for each measurement
                left_arm_lengths = []
                right_arm_lengths = []
                wrist_distances = []
                shoulder_distances = []
                left_upper_arm_lengths = []
                right_upper_arm_lengths = []

                j = 0
                # Define the keys
                keys = ["nose", "neck", "R_shoulder", "R_elbow", "R_wrist", "L_shoulder", "L_elbow", "L_wrist",
                        "R_hip", "R_knee", "R_ankle", "L_hip", "L_knee", "L_ankle", "R_eye", "L_eye", "R_ear", "L_ear"]

                # Create the dictionary with empty lists
                self.body_parts_dict = {key: [] for key in keys}
                self.previous_angles = {}

                # Collect 20 readings
                while j < 20:
                    self.get_skeleton_data_for_measurements()

                    # Extract X-coordinates
                    L_shoulder_x = self.body_parts_dict["L_shoulder"][-1][0]
                    L_elbow_x = self.body_parts_dict["L_elbow"][-1][0]
                    L_wrist_x = self.body_parts_dict["L_wrist"][-1][0]
                    R_shoulder_x = self.body_parts_dict["R_shoulder"][-1][0]
                    R_elbow_x = self.body_parts_dict["R_elbow"][-1][0]
                    R_wrist_x = self.body_parts_dict["R_wrist"][-1][0]



                    # Compute distances only if values are valid (not None)
                    if None not in (L_shoulder_x, L_wrist_x):
                        left_arm_lengths.append(abs(L_shoulder_x - L_wrist_x))
                    if None not in (R_shoulder_x, R_wrist_x):
                        right_arm_lengths.append(abs(R_shoulder_x - R_wrist_x))
                    if None not in (L_wrist_x, R_wrist_x):
                        wrist_distances.append(abs(L_wrist_x - R_wrist_x))
                    if None not in (L_shoulder_x, R_shoulder_x):
                        shoulder_distances.append(abs(L_shoulder_x - R_shoulder_x))
                    if None not in (L_shoulder_x, L_elbow_x):
                        left_upper_arm_lengths.append(abs(L_shoulder_x - L_elbow_x))
                    if None not in (R_shoulder_x, R_elbow_x):
                        right_upper_arm_lengths.append(abs(R_shoulder_x - R_elbow_x))

                    j+=1

                self.process_joints_from_body_parts_dict()

                s.zed_camera.set_detection_model_to_medium()

                s.len_left_arm = self.safe_mean(left_arm_lengths, "Left Arm")
                s.len_right_arm = self.safe_mean(right_arm_lengths, "Right Arm")
                s.dist_between_wrists = self.safe_mean(wrist_distances, "Wrist Distance")
                s.dist_between_shoulders = self.safe_mean(shoulder_distances, "Shoulder Distance")
                s.len_left_upper_arm = self.safe_mean(left_upper_arm_lengths, "Left Upper Arm")
                s.len_right_upper_arm = self.safe_mean(right_upper_arm_lengths, "Right Upper Arm")


                # Print results
                print(f"Average Left Arm Length (X-axis): {s.len_left_arm}")
                print(f"Average Right Arm Length (X-axis): {s.len_right_arm}")
                print(f"Average Wrist Distance (X-axis): {s.dist_between_wrists}")
                print(f"Average Shoulder Distance (X-axis): {s.dist_between_shoulders}")
                print(f"Average Upper Left Arm Length (X-axis): {s.len_left_upper_arm}")
                print(f"Average Upper Right Arm Length (X-axis): {s.len_right_upper_arm}")



            elif s.req_exercise != "" and not s.req_exercise == "calibration":
                ex = s.req_exercise


                if s.req_exercise != "hello_waving":
                    s.max_repetitions_in_training += s.rep

                    if self.first_coordination_ex:
                        while not s.explanation_over or not s.gymmy_finished_demo:
                            time.sleep(0.001)

                        time.sleep(get_wav_duration(f'{s.rep}_times')+0.5)

                if s.req_exercise == "notool_right_bend_left_up_from_side" or s.req_exercise == "notool_left_bend_right_up_from_side":  # if this is the fist of the 2, turn into false, and then in the next iteration it will skip the demonstration
                    if self.first_coordination_ex == False:
                        self.first_coordination_ex = True
                    if self.first_coordination_ex == True:
                        self.first_coordination_ex = False


                print("CAMERA: Exercise ", ex, " start")
                self.joints = {}
                self.previous_angles = {}
                self.count_not_good_range = 0
                getattr(self, ex)()
                self.count_not_good_range = 0
                print("CAMERA: Exercise ", ex, " done")
                # s.req_exercise = ""
                s.camera_done = True

            else:
                time.sleep(1)

        print("Camera Done")

    def process_joints_from_body_parts_dict(self):
        right_shoulder_angle_not_in_range_count = 0
        left_shoulder_angle_not_in_range_count = 0

        for i in range(len(self.body_parts_dict["R_shoulder"])):  # iterate over time

            # --- RIGHT SIDE ---
            r_hip_coords = self.body_parts_dict["R_hip"][i]
            r_shoulder_coords = self.body_parts_dict["R_shoulder"][i]
            r_elbow_coords = self.body_parts_dict["R_elbow"][i]

            r_hip = Joint("R_hip", r_hip_coords)
            r_shoulder = Joint("R_shoulder", r_shoulder_coords)
            r_elbow = Joint("R_elbow", r_elbow_coords)

            right_shoulder_angle = self.calc_angle_3d(r_hip, r_shoulder, r_elbow, "R_2")

            # --- LEFT SIDE ---
            l_hip_coords = self.body_parts_dict["L_hip"][i]
            l_shoulder_coords = self.body_parts_dict["L_shoulder"][i]
            l_elbow_coords = self.body_parts_dict["L_elbow"][i]

            l_hip = Joint("L_hip", l_hip_coords)
            l_shoulder = Joint("L_shoulder", l_shoulder_coords)
            l_elbow = Joint("L_elbow", l_elbow_coords)

            left_shoulder_angle = self.calc_angle_3d(l_hip, l_shoulder, l_elbow, "L_2")

            if not (50 < right_shoulder_angle < 130):
                right_shoulder_angle_not_in_range_count +=1
            if not (50 < left_shoulder_angle < 130):
                left_shoulder_angle_not_in_range_count +=1


        if right_shoulder_angle_not_in_range_count >= 2 or left_shoulder_angle_not_in_range_count >= 2:
            s.shoulder_problem_calibration = True
            print("shoulder_problem_calibration : True")




    def get_skeleton_data_for_measurements(self):
        bodies = sl.Bodies()
        body_runtime_param = sl.BodyTrackingRuntimeParameters()
        body_runtime_param.detection_confidence_threshold = 40

        # Define keypoint names in order corresponding to their index
        arr_organs = ["nose", "neck", "R_shoulder", "R_elbow", "R_wrist", "L_shoulder", "L_elbow", "L_wrist",
                      "R_hip", "R_knee", "R_ankle", "L_hip", "L_knee", "L_ankle", "R_eye", "L_eye", "R_ear",
                      "L_ear"]

        if s.zed_camera.zed.grab() == sl.ERROR_CODE.SUCCESS:
            s.zed_camera.zed.retrieve_bodies(bodies, body_runtime_param)
            body_array = bodies.body_list

            if body_array:
                body = body_array[0]  # Get the first detected body

                for i, organ in enumerate(arr_organs):
                    if i < len(body.keypoint) and body.keypoint[i] is not None:
                        self.body_parts_dict[organ].append(tuple(body.keypoint[i]))  # Store (x, y, z)
                    else:
                        self.body_parts_dict[organ].append((None, None, None))  # Append (-1, -1, -1) for missing keypoints
            else:
                # No bodies detected, append (-1, -1, -1) for all organs
                for organ in arr_organs:
                    self.body_parts_dict[organ].append((None, None, None))


    def get_skeleton_data(self):
        """
        Capture 3D joint positions from ZED camera and apply both Butterworth and Moving Average filters.
        If one filter fails, use the other. If both work, take the average.
        """
        bodies = sl.Bodies()
        body_runtime_param = sl.BodyTrackingRuntimeParameters()
        body_runtime_param.detection_confidence_threshold = 50
        time.sleep(0.001)

        if s.zed_camera.zed.grab() == sl.ERROR_CODE.SUCCESS:
            s.zed_camera.zed.retrieve_bodies(bodies, body_runtime_param)
            body_array = bodies.body_list
            if body_array:
                body = bodies.body_list[0]

                arr_organs = ["nose", "neck", "R_shoulder", "R_elbow", "R_wrist", "L_shoulder", "L_elbow", "L_wrist",
                              "R_hip", "R_knee", "R_ankle", "L_hip", "L_knee", "L_ankle", "R_eye", "L_eye", "R_ear",
                              "L_ear"]

                for i, kp_3d in enumerate(body.keypoint):
                    organ = arr_organs[i]


                    if organ in self.joints:

                        if organ == "L_shoulder" and s.req_exercise == "band_straighten_left_arm_elbows_bend_to_sides":
                            # if kp_3d is None or np.isnan(kp_3d).any() or np.all(kp_3d == 0):
                            right_shoulder = self.joints.get("R_shoulder")  # Use .get() to avoid KeyError
                            if right_shoulder:
                                kp_3d = np.array([
                                    right_shoulder.x + s.dist_between_shoulders,
                                    # Mirror L_shoulder from R_shoulder
                                    right_shoulder.y,
                                    right_shoulder.z
                                ], dtype=np.float32)


                        if organ == "R_shoulder" and s.req_exercise == "band_straighten_right_arm_elbows_bend_to_sides":
                            # if kp_3d is None or np.isnan(kp_3d).any() or np.all(kp_3d == 0):
                            left_shoulder = self.joints.get("L_shoulder")  # Use .get() to avoid KeyError
                            if left_shoulder:
                                kp_3d = np.array([
                                    left_shoulder.x - s.dist_between_shoulders,
                                    # Mirror L_shoulder from R_shoulder
                                    left_shoulder.y,
                                    left_shoulder.z
                                ], dtype=np.float32)

                        if organ == "L_elbow" and s.req_exercise == "band_straighten_left_arm_elbows_bend_to_sides":
                            left_shoulder = self.joints.get("L_shoulder")  # Get left shoulder safely
                            right_elbow = self.joints.get("R_elbow")

                            if left_shoulder and left_shoulder.x is not None:  # Ensure it's valid
                                if right_elbow and right_elbow.y is not None and right_elbow.z is not None:
                                    # Use kp_3d if it exists, otherwise default to left_shoulder
                                    y_value = kp_3d[1] if kp_3d is not None else left_shoulder.y
                                    z_value = kp_3d[2] if kp_3d is not None else left_shoulder.z

                                # else:
                                #     y_value = kp_3d[1] if kp_3d is not None else left_shoulder.y
                                #     z_value = kp_3d[2] if kp_3d is not None else left_shoulder.z

                                # Estimate the elbow position using upper arm length
                                kp_3d = np.array([
                                    left_shoulder.x + s.len_left_upper_arm,  # Shoulder + upper arm length
                                    y_value,  # Preserve Y coordinate
                                    z_value  # Preserve Z coordinate
                                ], dtype=np.float32)

                        if organ == "R_elbow" and s.req_exercise == "band_straighten_right_arm_elbows_bend_to_sides":
                            right_shoulder = self.joints.get("R_shoulder")  # Get left shoulder safely
                            left_elbow = self.joints.get("L_elbow")

                            if right_shoulder and right_shoulder.x is not None:  # Ensure it's valid
                                if left_elbow and left_elbow.y is not None and left_elbow.z is not None:
                                    # Use kp_3d if it exists, otherwise default to left_shoulder
                                    y_value = kp_3d[1] if kp_3d is not None else right_shoulder.y
                                    z_value = kp_3d[2] if kp_3d is not None else right_shoulder.z

                                # else:
                                #     y_value = kp_3d[1] if kp_3d is not None else left_shoulder.y
                                #     z_value = kp_3d[2] if kp_3d is not None else left_shoulder.z

                                # Estimate the elbow position using upper arm length
                                kp_3d = np.array([
                                    right_shoulder.x - s.len_right_upper_arm,  # Shoulder + upper arm length
                                    y_value,  # Preserve Y coordinate
                                    z_value  # Preserve Z coordinate
                                ], dtype=np.float32)

                        butter_filtered = self.joints[organ].butter_filter.update(kp_3d)
                        moving_avg_filtered = self.joints[organ].moving_avg_filter.update(kp_3d)

                        # Handling NaNs: If one filter returns NaN, use the other
                        valid_butter = not np.any(np.isnan(butter_filtered))
                        valid_moving_avg = not np.any(np.isnan(moving_avg_filtered))

                        # if valid_butter and valid_moving_avg:
                        #     kp_3d_new = (butter_filtered + moving_avg_filtered) / 2  # Average if both valid
                        if valid_butter:
                            kp_3d_new = butter_filtered  # Use Butterworth if Moving Average failed
                        elif valid_moving_avg:
                            print("Moving avg")
                            kp_3d_new = moving_avg_filtered  # Use Moving Average if Butterworth failed
                        else:
                            kp_3d_new = kp_3d  # If both fail, return raw data

                        self.joints[organ].x, self.joints[organ].y, self.joints[organ].z = kp_3d_new
                    else:
                        joint = Joint(organ, kp_3d)
                        joint.butter_filter = ButterworthFilter()  # Initialize Butterworth filter
                        joint.moving_avg_filter = MovingAverageFilter()  # Initialize Moving Average filter
                        joint.position = kp_3d  # Store raw position initially
                        self.joints[organ] = joint

                s.latest_keypoints = self.joints
                return self.joints
            else:
                time.sleep(0.01)
                return None
        else:
            return None


    def sayings_generator(self, counter):
        if s.robot_counter < s.rep - 1 and s.robot_counter >= 2 and counter < s.rep - 1:
            random_number_for_general_saying = random.randint(1, s.rep*30)  # Random frame condition

            if (random_number_for_general_saying in range (1,5)) and \
                    time.time() - s.last_saying_time >= 7:
                if s.general_sayings:  # Ensure the list is not empty
                    # Filter sayings based on the counter condition
                    num = random.randint(2, 5)
                    filtered_sayings = s.general_sayings
                    if s.robot_counter <=2 :
                        filtered_sayings = []

                    else: # אם הרובוט גדול מ-2
                        if s.rep - s.robot_counter > 3:# רובוט לא לקראת הסוף
                                if s.rep- counter > 3: #מטופל לא לקראת הסוף
                                    filtered_sayings = [saying for saying in filtered_sayings if not saying.endswith(("_end", "_end_good"))]
                        else:
                            if s.rep - counter > 4:#אם לקראת הסוף ולא הצלחתי הרבה
                                filtered_sayings = [saying for saying in filtered_sayings if not saying.endswith(("_end_good"))]

                        if counter <= 3 or (s.rep - s.robot_counter) >= 4: #אם לא עשיתי הרבה חזרות
                            filtered_sayings = [saying for saying in filtered_sayings if not saying.endswith("_middle")]

                        if s.robot_counter - num < counter: #אם הספירה של הרובוט קרובה לספירה של האדם
                            filtered_sayings = [saying for saying in filtered_sayings if not saying.startswith("faster")]

                        if abs(s.robot_counter - counter) > 1 or counter <= 3:
                            filtered_sayings = [saying for saying in filtered_sayings if not saying.endswith("_small_gap")]

                        if counter < 3:
                            filtered_sayings = [saying for saying in filtered_sayings if not saying.endswith("_not_start")]

                        if abs(s.robot_counter - counter) <= 3 or counter < 4:
                            filtered_sayings = [saying for saying in filtered_sayings if not saying.endswith("_gap_and_many_rep")]
                            filtered_sayings = [saying for saying in filtered_sayings if not saying.endswith("_large_gap")]

                        if abs(s.robot_counter - counter) <= 4:
                            filtered_sayings = [saying for saying in filtered_sayings if not saying.startswith("comment_dont_recognize")]


                    if filtered_sayings:  # Ensure the filtered list is not empty
                        random_saying_name = random.choice(filtered_sayings)
                        s.general_sayings.remove(random_saying_name)  # Remove the selected saying
                        say(random_saying_name)  # Call the function to say it
                        s.last_saying_time = time.time()
                        
                        
                else:
                    pass

    def fill_null_joint_list(self):
        arr_organs = ["nose", "neck", "R_shoulder", "R_elbow", "R_wrist", "L_shoulder", "L_elbow", "L_wrist",
                      "R_hip", "R_knee", "R_ankle", "L_hip", "L_knee", "L_ankle", "R_eye", "L_eye", "R_ear", "L_ear"]

        for organ in arr_organs:
            joint = Joint(organ, [math.nan, math.nan, math.nan])
            self.joints[organ] = joint

        return self.joints



    def exercise_two_angles_3d(self, exercise_name, joint1, joint2, joint3, up_lb, up_ub, down_lb, down_ub,
                                   joint4, joint5, joint6, up_lb2, up_ub2, down_lb2, down_ub2, use_alternate_angles=False, left_right_differ=False):

            list_first_angle=[]
            list_second_angle=[]
            list_second_angle=[]
            flag = True
            counter = 0
            list_joints = []
            s.time_of_change_position = time.time()
            s.last_entry_angles = [0,0,0,0]
            s.change_in_trend = [False] * 4
            increasing_decreasing = [[-1] * 40 for _ in range(len(s.last_entry_angles))]

            while s.req_exercise == exercise_name:
                while s.did_training_paused and not s.stop_requested:
                    time.sleep(0.01)
                    if self.joints != {}:
                        self.joints = {}

                    if self.previous_angles != {}:
                        self.previous_angles = {}


                self.sayings_generator(counter)
                joints = self.get_skeleton_data()
                if joints is not None:
                    right_angle = self.calc_angle_3d(joints[str("R_" + joint1)], joints[str("R_" + joint2)],
                                                     joints[str("R_" + joint3)], "R_1")
                    left_angle = self.calc_angle_3d(joints[str("L_" + joint1)], joints[str("L_" + joint2)],
                                                    joints[str("L_" + joint3)], "L_1")
                    if use_alternate_angles:
                        right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                         joints[str("L_" + joint6)], "R_2")
                        left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                         joints[str("R_" + joint6)], "L_2")

                        new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                     joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                     joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)],
                                     joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)],
                                     right_angle, left_angle, right_angle2, left_angle2]

                        if flag == False:
                            s.information = [[str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), up_lb, up_ub],
                                             [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), up_lb, up_ub],
                                             [str("R_" + joint4), str("R_" + joint5), str("L_" + joint6), up_lb2, up_ub2],
                                             [str("L_" + joint4), str("L_" + joint5), str("R_" + joint6), up_lb2, up_ub2]]
                        else:
                            s.information = [
                                [str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), down_lb, down_ub],
                                [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), down_lb, down_ub],
                                [str("R_" + joint4), str("R_" + joint5), str("L_" + joint6), down_lb2, down_ub2],
                                [str("L_" + joint4), str("L_" + joint5), str("R_" + joint6), down_lb2, down_ub2]]

                    else:
                        right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                       joints[str("R_" + joint6)], "R_2")
                        left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                      joints[str("L_" + joint6)], "L_2")

                        new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                     joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                     joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                     joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                     right_angle, left_angle, right_angle2, left_angle2]



                        if flag == False:
                            s.information = [[str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), up_lb, up_ub],
                                             [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), up_lb, up_ub],
                                             [str("R_" + joint4), str("R_" + joint5), str("R_" + joint6), up_lb2,up_ub2],
                                             [str("L_" + joint4), str("L_" + joint5), str("L_" + joint6), up_lb2, up_ub2]]

                            if s.req_exercise in ["ball_bend_elbows", "ball_raise_arms_above_head", "stick_bend_elbows", "stick_bend_elbows_and_up", "stick_raise_arms_above_head", "weights_abduction"]:
                                s.direction = "up"

                            elif s.req_exercise in ["band_open_arms"]:
                                s.direction = "out"

                            elif s.req_exercise in ["ball_open_arms_above_head"]:
                                s.direction = "in"


                        else:
                            s.information = [
                                    [str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), down_lb, down_ub],
                                    [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), down_lb, down_ub],
                                    [str("R_" + joint4), str("R_" + joint5), str("R_" + joint6), down_lb2, down_ub2],
                                    [str("L_" + joint4), str("L_" + joint5), str("L_" + joint6), down_lb2, down_ub2]]

                            if s.req_exercise in ["ball_bend_elbows", "ball_raise_arms_above_head", "stick_bend_elbows", "stick_bend_elbows_and_up", "stick_raise_arms_above_head", "weights_abduction"]:
                                s.direction = "down"

                            elif s.req_exercise in ["band_open_arms"]:
                                s.direction = "in"

                            elif s.req_exercise in ["ball_open_arms_above_head"]:
                                s.direction = "out"


                    s.last_entry_angles = [right_angle, left_angle, right_angle2, left_angle2]

                    ##############################################################################
                    print(str(joints[str("R_" + joint1)]))
                    print(str(joints[str("R_" + joint2)]))
                    print(str(joints[str("R_" + joint3)]))
                    print(str(joints[str("R_" + joint4)]))
                    print(str(joints[str("R_" + joint5)]))
                    print(str(joints[str("R_" + joint6)]))
                    print(str(joints[str("L_" + joint1)]))
                    print(str(joints[str("L_" + joint2)]))
                    print(str(joints[str("L_" + joint3)]))
                    print(str(joints[str("L_" + joint4)]))
                    print(str(joints[str("L_" + joint5)]))
                    print(str(joints[str("L_" + joint6)]))


                    print(left_angle, " ", right_angle)
                    print(left_angle2, " ", right_angle2)

                    list_first_angle.append(left_angle)
                    list_second_angle.append(left_angle2)

                    ##############################################################################

                    # if list_joints:
                    #     previous_entry = list_joints[-1]
                    #
                    list_joints.append(copy.deepcopy(new_entry))
                    #
                    # self.tred_func(previous_entry, increasing_decreasing)


                    if right_angle is not None and left_angle is not None and \
                            right_angle2 is not None and left_angle2 is not None:
                        print("first angle mean: ", np.nanmean(list_first_angle))
                        print("first angle stdev: ", np.nanstd(list_first_angle))
                        print("second angle mean: ", np.nanmean(list_second_angle))
                        print("second angle stdev: ", np.nanstd(list_second_angle))


                        if left_right_differ:
                            if (up_lb < right_angle < up_ub) and (down_lb < left_angle < down_ub) and \
                                    (up_lb2 < right_angle2 < up_ub2) and (down_lb2 < left_angle2 < down_ub2) and (not flag):

                                    if s.reached_max_limit:
                                        flag = True
                                        counter += 1
                                        s.number_of_repetitions_in_training += 1
                                        s.patient_repetitions_counting_in_exercise+=1
                                        print("counter:"+ str(counter))
                                        s.all_rules_ok = True
                                        self.count_not_good_range = 0
                                        s.time_of_change_position = time.time()
                                        s.not_reached_max_limit_rest_rules_ok = False
                                        # s.change_in_trend = [False] * 4
                                        # increasing_decreasing = [[0] * 40 for _ in range(len(s.last_entry_angles))]

                                    else:
                                        s.not_reached_max_limit_rest_rules_ok = True


                            elif (down_lb < right_angle < down_ub) and (up_lb < left_angle < up_ub) and \
                                    (down_lb2 < right_angle2 < down_ub2) and (up_lb2 < left_angle2 < up_ub2) and (flag):
                                flag = False
                                s.all_rules_ok = False
                                s.was_in_first_condition = True
                                self.count_not_good_range = 0
                                s.time_of_change_position = time.time()
                                # s.change_in_trend = [False] * 4
                                # increasing_decreasing = [[0] * 40 for _ in range(len(s.last_entry_angles))]

                            # elif time.time() - s.time_of_change_position > 3:
                            #     if not s.reached_max_limit and (up_lb < right_angle < up_ub) and (down_lb < left_angle < down_ub) and \
                            #         (up_lb2 < right_angle2 < up_ub2) and (down_lb2 < left_angle2 < down_ub2):
                            #         self.count_not_good_range += 1
                            #
                            #         if self.count_not_good_range >= 20:
                            #             s.try_again_calibration = True

                        else:
                            if (up_lb < right_angle < up_ub) and (up_lb < left_angle < up_ub) and \
                                    (up_lb2 < right_angle2 < up_ub2) and (up_lb2 < left_angle2 < up_ub2) and (not flag):

                                if s.reached_max_limit:
                                    flag = True
                                    counter += 1
                                    s.number_of_repetitions_in_training += 1
                                    s.patient_repetitions_counting_in_exercise+=1
                                    print("counter:" + str(counter))
                                    s.time_of_change_position = time.time()
                                    self.count_not_good_range = 0
                                    s.all_rules_ok = True
                                    s.not_reached_max_limit_rest_rules_ok = False
                                    # s.change_in_trend = [False] * 4

                                else:
                                    s.not_reached_max_limit_rest_rules_ok = True

                            elif (down_lb < right_angle < down_ub) and (down_lb < left_angle < down_ub) and \
                                    (down_lb2 < right_angle2 < down_ub2) and (down_lb2 < left_angle2 < down_ub2) and (flag):
                                flag = False
                                s.all_rules_ok = False
                                s.was_in_first_condition = True
                                self.count_not_good_range = 0
                                s.time_of_change_position = time.time()
                                # s.change_in_trend = [False] * 4

                            # elif time.time() - s.time_of_change_position > 3:
                            #     if not s.reached_max_limit and (up_lb < right_angle < up_ub) and (down_lb < left_angle < down_ub) and \
                            #         (up_lb2 < right_angle2 < up_ub2) and (down_lb2 < left_angle2 < down_ub2):
                            #         self.count_not_good_range += 1
                            #
                            #         if self.count_not_good_range >= 20:
                            #             s.try_again_calibration = True

                if counter == s.rep:
                    s.req_exercise = ""
                    s.success_exercise = True
                    break

            if len(list_joints) == 0:
                joints = self.fill_null_joint_list()
                if use_alternate_angles:
                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                             joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                             joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)],
                             joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)],
                             None, None, None, None]
                else:
                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                 None, None, None, None]
                list_joints.append(copy.deepcopy(new_entry))

            if not s.try_again_calibration and not s.repeat_explanation:
                s.ex_list.update({exercise_name: counter})

                if s.stop_requested or s.num_exercises_started == len(s.ex_in_training):
                    Excel.wf_joints(exercise_name, list_joints)
                else:
                    threading.Thread(target=Excel.wf_joints, args=(exercise_name, list_joints), daemon=True).start()

            else:
                s.number_of_repetitions_in_training -= s.patient_repetitions_counting_in_exercise
                s.max_repetitions_in_training -= s.rep

    def exercise_two_angles_3d_one_side(self, exercise_name, joint1, joint2, joint3, up_lb_right, up_ub_right, down_lb_right, down_ub_right, up_lb_left, up_ub_left, down_lb_left, down_ub_left,
                                   joint4, joint5, joint6, up_lb_right2, up_ub_right2, down_lb_right2, down_ub_right2 , up_lb_left2, up_ub_left2, down_lb_left2, down_ub_left2, use_alternate_angles=False):

            list_first_angle=[]
            list_second_angle=[]
            flag = True
            counter = 0
            list_joints = []
            s.time_of_change_position = time.time()

            while s.req_exercise == exercise_name:
                while s.did_training_paused and not s.stop_requested:
                    time.sleep(0.01)

                    if self.joints != {}:
                        self.joints = {}

                    if self.previous_angles != {}:
                        self.previous_angles = {}

                self.sayings_generator(counter)

                #for i in range (1,200):
                joints = self.get_skeleton_data()
                if joints is not None:
                    right_angle = self.calc_angle_3d(joints[str("R_" + joint1)], joints[str("R_" + joint2)],
                                                     joints[str("R_" + joint3)], "R_1")
                    left_angle = self.calc_angle_3d(joints[str("L_" + joint1)], joints[str("L_" + joint2)],
                                                    joints[str("L_" + joint3)], "L_1")


                    if use_alternate_angles:
                        right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                         joints[str("L_" + joint6)], "R_2")
                        left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                         joints[str("R_" + joint6)], "L_2")

                        new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                     joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                     joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)],
                                     joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)],
                                     right_angle, left_angle, right_angle2, left_angle2]


                    else:

                        right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                          joints[str("R_" + joint6)], "R_2")
                        left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                         joints[str("L_" + joint6)], "L_2")

                        new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                     joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                     joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                     joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                     right_angle, left_angle, right_angle2, left_angle2]

                        if flag == False:
                            s.information = [
                                [str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), up_lb_right, up_ub_right],
                                [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), up_lb_left, up_ub_left],
                                [str("R_" + joint4), str("R_" + joint5), str("R_" + joint6), up_lb_right2, up_ub_right2],
                                [str("L_" + joint4), str("L_" + joint5), str("L_" + joint6), up_lb_left2, up_ub_left2]]

                            if s.req_exercise == "band_straighten_left_arm_elbows_bend_to_sides":
                                s.direction = "left"

                            elif s.req_exercise == "notool_right_bend_left_up_from_side":
                                s.direction = "up_down"

                            elif s.req_exercise == "notool_left_bend_right_up_from_side":
                                s.direction = "down_up"

                            else:
                                s.direction = "right"

                        else:
                            s.information = [
                                [str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), down_lb_right, down_ub_right],
                                [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), down_lb_left, down_ub_left],
                                [str("R_" + joint4), str("R_" + joint5), str("R_" + joint6), down_lb_right2, down_ub_right2],
                                [str("L_" + joint4), str("L_" + joint5), str("L_" + joint6), down_lb_left2, down_ub_left2]]

                            if s.req_exercise == "band_straighten_left_arm_elbows_bend_to_sides":
                                s.direction = "right"

                            elif s.req_exercise == "notool_right_bend_left_up_from_side":
                                s.direction = "down_up"

                            elif s.req_exercise == "notool_left_bend_right_up_from_side":
                                s.direction = "up_down"

                            else:
                                s.direction = "left"



                    s.last_entry_angles = [right_angle, left_angle, right_angle2, left_angle2]

                    ##############################################################################
                    print(str(joints[str("R_" + joint1)]))
                    print(str(joints[str("R_" + joint2)]))
                    print(str(joints[str("R_" + joint3)]))
                    print(str(joints[str("R_" + joint4)]))
                    print(str(joints[str("R_" + joint5)]))
                    print(str(joints[str("R_" + joint6)]))
                    print(str(joints[str("L_" + joint1)]))
                    print(str(joints[str("L_" + joint2)]))
                    print(str(joints[str("L_" + joint3)]))
                    print(str(joints[str("L_" + joint4)]))
                    print(str(joints[str("L_" + joint5)]))
                    print(str(joints[str("L_" + joint6)]))





                    print(left_angle, " ", right_angle)
                    print(left_angle2, " ", right_angle2)

                    list_first_angle.append(left_angle)
                    list_second_angle.append(left_angle2)

                    ##############################################################################

                    list_joints.append(copy.deepcopy(new_entry))

                    #print(str(i))
                    if right_angle is not None and left_angle is not None and \
                            right_angle2 is not None and left_angle2 is not None:
                        print("first angle mean: ", np.nanmean(list_first_angle))
                        print("first angle stdev: ", np.nanstd(list_first_angle))
                        print("second angle mean: ", np.nanmean(list_second_angle))
                        print("second angle stdev: ", np.nanstd(list_second_angle))





                    if (up_lb_right < right_angle < up_ub_right) and (up_lb_left < left_angle < up_ub_left) and \
                            (up_lb_right2 < right_angle2 < up_ub_right2) and (up_lb_left2 < left_angle2 < up_ub_left2) and (not flag):

                        if s.reached_max_limit:
                            flag = True
                            counter += 1
                            s.number_of_repetitions_in_training += 1
                            s.patient_repetitions_counting_in_exercise+=1
                            #self.change_count_screen(counter)
                            print("counter:" + str(counter))
                            s.all_rules_ok = True
                            s.time_of_change_position = time.time()
                            self.count_not_good_range = 0
                            s.not_reached_max_limit_rest_rules_ok = False

                        else:
                            s.not_reached_max_limit_rest_rules_ok = True

                        #  if not s.robot_count:
                        # say(str(counter))
                    elif (down_lb_right < right_angle < down_ub_right) and (down_lb_left < left_angle < down_ub_left) and \
                            (down_lb_right2 < right_angle2 < down_ub_right2) and (down_lb_left2 < left_angle2 < down_ub_left2) and (flag):
                        flag = False
                        s.all_rules_ok = False
                        s.was_in_first_condition = True
                        s.time_of_change_position = time.time()
                        self.count_not_good_range = 0


                    # elif time.time() - s.time_of_change_position > 3:
                    #     if not s.reached_max_limit and (up_lb_right < right_angle < up_ub_right) and (up_lb_left < left_angle < up_ub_left) and \
                    #         (up_lb_right2 < right_angle2 < up_ub_right2) and (up_lb_left2 < left_angle2 < up_ub_left2):
                    #         self.count_not_good_range += 1
                    #
                    #         if self.count_not_good_range >= 20:
                    #             s.try_again_calibration = True

                if counter == s.rep:
                    s.req_exercise = ""
                    s.success_exercise = True
                    break

            if len(list_joints) == 0:
                joints = self.fill_null_joint_list()
                if use_alternate_angles:
                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)],
                                 None, None, None, None]
                else:
                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                 None, None, None, None]
                list_joints.append(copy.deepcopy(new_entry))

            if not s.try_again_calibration and not s.repeat_explanation:
                s.ex_list.update({exercise_name: counter})

                if s.stop_requested or s.num_exercises_started == len(s.ex_in_training):
                    Excel.wf_joints(exercise_name, list_joints)
                else:
                    threading.Thread(target=Excel.wf_joints, args=(exercise_name, list_joints), daemon=True).start()

            else:
                s.number_of_repetitions_in_training -= s.patient_repetitions_counting_in_exercise
                s.max_repetitions_in_training -= s.rep

    def exercise_two_angles_3d_with_axis_check(self, exercise_name, joint1, joint2, joint3, up_lb, up_ub, down_lb, down_ub,
                               joint4, joint5, joint6, up_lb2, up_ub2, down_lb2, down_ub2, use_alternate_angles=False,
                               left_right_differ=False, wrist_check = False):

        list_first_angle = []
        list_second_angle = []
        flag = True
        counter = 0
        list_joints = []
        s.was_in_first_condition = True
        s.time_of_change_position = time.time() + 2


        while s.req_exercise == exercise_name:
            while s.did_training_paused and not s.stop_requested:
                time.sleep(0.01)

                if self.joints != {}:
                    self.joints = {}

                if self.previous_angles != {}:
                    self.previous_angles = {}

            self.sayings_generator(counter)

            #for i in range (1,100):
            joints = self.get_skeleton_data()
            if joints is not None:


                right_angle = self.calc_angle_3d(joints[str("R_" + joint1)], joints[str("R_" + joint2)],
                                                 joints[str("R_" + joint3)], "R_1")
                left_angle = self.calc_angle_3d(joints[str("L_" + joint1)], joints[str("L_" + joint2)],
                                                joints[str("L_" + joint3)], "L_1")
                if use_alternate_angles:

                    if left_right_differ:

                        if flag == False:
                            s.information = [[str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), up_lb, up_ub],
                                             [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), down_lb, down_ub],
                                             [str("R_" + joint4), str("R_" + joint5), str("L_" + joint6), up_lb2, up_ub2],
                                             [str("L_" + joint4), str("L_" + joint5), str("R_" + joint6), down_lb2, down_ub2]]

                            s.direction = "right"

                            if s.req_exercise == "ball_switch":
                                left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)], "L_2")
                                right_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)],joints[str("R_" + joint5)], joints[str("L_" + joint6)], "R_2")

                            else:
                                left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)], "L_2")
                                right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)], "R_2")


                        else:

                            s.information = [[str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), down_lb, down_ub],
                                             [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), up_lb, up_ub],
                                             [str("R_" + joint4), str("R_" + joint5), str("L_" + joint6), down_lb2, down_ub2],
                                             [str("L_" + joint4), str("L_" + joint5), str("R_" + joint6), up_lb2, up_ub2]]

                            s.direction = "left"

                            if s.req_exercise == "ball_switch":
                                right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)], "R_2")
                                left_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)], "L_2")

                            else:
                                right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)], "R_2")
                                left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)], "L_2")


                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                     joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                     joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)],
                                     joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)],
                                     right_angle, left_angle, right_angle2, left_angle2]


                else:
                    right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                      joints[str("R_" + joint6)], "R_2")
                    left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                     joints[str("L_" + joint6)], "L_2")

                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                 right_angle, left_angle, right_angle2, left_angle2]


                    if left_right_differ:
                        if flag == False:
                            s.information = [[str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), up_lb, up_ub],
                                             [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), down_lb,down_ub],
                                             [str("R_" + joint4), str("R_" + joint5), str("R_" + joint6), up_lb2, up_ub2],
                                             [str("L_" + joint4), str("L_" + joint5), str("L_" + joint6), down_lb2, down_ub2]]

                            if s.req_exercise == "notool_raising_hands_diagonally":
                                s.direction = "left_diagonal"

                        else:

                            s.information = [
                                [str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), down_lb, down_ub],
                                [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), up_lb, up_ub],
                                [str("R_" + joint4), str("R_" + joint5), str("R_" + joint6), down_lb2, down_ub2],
                                [str("L_" + joint4), str("L_" + joint5), str("L_" + joint6), up_lb2, up_ub2]]

                            if s.req_exercise == "notool_raising_hands_diagonally":
                                s.direction = "right_diagonal"



                s.last_entry_angles = [right_angle, left_angle, right_angle2, left_angle2]

                ##############################################################################
                print(left_angle, " ", right_angle)
                print(left_angle2, " ", right_angle2)

                list_first_angle += [left_angle]
                list_second_angle += [left_angle2]

                print("left shoulder", joints["L_shoulder"].__str__())
                print("right shoulder", joints["R_shoulder"].__str__())
                print(str(abs(joints["L_shoulder"].x - joints["R_shoulder"].x)))


                ##############################################################################


                #print(i)
                list_joints.append(copy.deepcopy(new_entry))

                if right_angle is not None and left_angle is not None and \
                        right_angle2 is not None and left_angle2 is not None:
                    print("first angle mean: ", np.nanmean(list_first_angle))
                    print("first angle stdev: ", np.nanstd(list_first_angle))
                    print("second angle mean: ", np.nanmean(list_second_angle))
                    print("second angle stdev: ", np.nanstd(list_second_angle))
                    print("distance between shoulders: "+str(abs(joints["L_shoulder"].x - joints["R_shoulder"].x)))
                    if left_right_differ:

                        if wrist_check:
                            if ((down_lb < right_angle < down_ub) and (up_lb < left_angle < up_ub) and \
                                    (down_lb2 < right_angle2 < down_ub2) and (up_lb2 < left_angle2 < up_ub2) and (not flag)):


                                    if s.reached_max_limit:
                                        s.not_reached_max_limit_rest_rules_ok = False

                                        if (joints["R_wrist"].x - joints["L_shoulder"].x > 50):
                                            s.hand_not_good = False
                                            flag = True
                                            counter += 1
                                            s.number_of_repetitions_in_training += 1
                                            s.patient_repetitions_counting_in_exercise += 1
                                            print("counter:" + str(counter))
                                            s.all_rules_ok = True
                                            s.time_of_change_position = time.time()
                                            self.count_not_good_range = 0

                                        else:
                                            s.hand_not_good = True

                                    else:
                                        s.not_reached_max_limit_rest_rules_ok = True

                            elif (up_lb < right_angle < up_ub) and (down_lb < left_angle < down_ub) and \
                                    (up_lb2 < right_angle2 < up_ub2) and (down_lb2 < left_angle2 < down_ub2) and (flag):

                                    if s.reached_max_limit:
                                        if joints["R_shoulder"].x - joints["L_wrist"].x > 50:
                                            s.hand_not_good = False

                                            counter += 1
                                            s.number_of_repetitions_in_training += 1
                                            s.patient_repetitions_counting_in_exercise += 1
                                            print("counter:" + str(counter))

                                            flag = False
                                            s.all_rules_ok = True
                                            s.time_of_change_position = time.time()
                                            self.count_not_good_range = 0
                                            s.not_reached_max_limit_rest_rules_ok = False

                                        else:
                                            s.hand_not_good = True

                                    else:
                                        s.not_reached_max_limit_rest_rules_ok = True


                            # elif time.time() - s.time_of_change_position > 1.5:
                            #     if not s.reached_max_limit and \
                            #            (((down_lb < right_angle < down_ub) and (up_lb < left_angle < up_ub) and (down_lb2 < right_angle2 < down_ub2) and (up_lb2 < left_angle2 < up_ub2) and (joints["R_wrist"].x - joints["L_shoulder"].x > 50)) or \
                            #             ((up_lb < right_angle < up_ub) and (down_lb < left_angle < down_ub) and (up_lb2 < right_angle2 < up_ub2) and (down_lb2 < left_angle2 < down_ub2) and (joints["R_shoulder"].x - joints["L_wrist"].x > 50))):
                            #         self.count_not_good_range += 1
                            #
                            #         if self.count_not_good_range >= 20:
                            #             s.try_again_calibration = True
                            #             exercise_name = "s"


                        else:
                            if (up_lb < right_angle < up_ub) and (down_lb < left_angle < down_ub) and \
                                    (up_lb2 < right_angle2 < up_ub2) and (down_lb2 < left_angle2 < down_ub2) and (not flag):

                                if s.reached_max_limit:
                                    flag = True
                                    counter += 1
                                    s.number_of_repetitions_in_training += 1
                                    s.patient_repetitions_counting_in_exercise += 1
                                    print("counter:" + str(counter))
                                    s.all_rules_ok = True
                                    s.time_of_change_position = time.time() + 3
                                    self.count_not_good_range = 0

                                    s.not_reached_max_limit_rest_rules_ok = False

                                else:
                                    s.not_reached_max_limit_rest_rules_ok = True


                            elif (down_lb < right_angle < down_ub) and (up_lb < left_angle < up_ub) and \
                                    (down_lb2 < right_angle2 < down_ub2) and (up_lb2 < left_angle2 < up_ub2) and (flag):

                                if s.reached_max_limit:
                                    counter += 1
                                    s.number_of_repetitions_in_training += 1
                                    s.patient_repetitions_counting_in_exercise += 1
                                    print("counter:" + str(counter))

                                    flag = False
                                    s.all_rules_ok = True
                                    s.time_of_change_position = time.time() + 3
                                    self.count_not_good_range = 0
                                    s.not_reached_max_limit_rest_rules_ok = False

                                else:
                                    s.not_reached_max_limit_rest_rules_ok = True

                            # elif time.time() - s.time_of_change_position > 1.5:
                            #     if not s.reached_max_limit and (((up_lb < right_angle < up_ub) and (down_lb < left_angle < down_ub) and (up_lb2 < right_angle2 < up_ub2) and (down_lb2 < left_angle2 < down_ub2)) or
                            #        ((down_lb < right_angle < down_ub) and (up_lb < left_angle < up_ub) and (down_lb2 < right_angle2 < down_ub2) and (up_lb2 < left_angle2 < up_ub2))) and \
                            #         abs(joints["L_shoulder"].z - joints["R_shoulder"].z)>100:
                            #
                            #         self.count_not_good_range += 1
                            #         print("count_not_good_range: " + str(self.count_not_good_range))
                            #
                            #         if self.count_not_good_range >= 20:
                            #             s.try_again_calibration = True
                            #             exercise_name = "s"


                    else:
                        if (up_lb < right_angle < up_ub) and (up_lb < left_angle < up_ub) and \
                                (up_lb2 < right_angle2 < up_ub2) and (up_lb2 < left_angle2 < up_ub2) and (not flag):

                            if s.reached_max_limit:
                                flag = True
                                counter += 1
                                s.number_of_repetitions_in_training += 1
                                s.patient_repetitions_counting_in_exercise += 1
                                print("counter:" + str(counter))

                                s.all_rules_ok = True
                                s.time_of_change_position = time.time() + 2
                                self.count_not_good_range = 0
                                s.not_reached_max_limit_rest_rules_ok = False

                            else:
                                s.not_reached_max_limit_rest_rules_ok = True


                        elif (down_lb < right_angle < down_ub) and (down_lb < left_angle < down_ub) and \
                                (down_lb2 < right_angle2 < down_ub2) and (down_lb2 < left_angle2 < down_ub2) and (flag):

                            if s.reached_max_limit:
                                counter += 1
                                s.number_of_repetitions_in_training += 1
                                s.patient_repetitions_counting_in_exercise += 1
                                print("counter:" + str(counter))

                                flag = False
                                s.all_rules_ok = True
                                s.time_of_change_position = time.time() + 2
                                self.count_not_good_range = 0
                                s.not_reached_max_limit_rest_rules_ok = False

                            else:
                                s.not_reached_max_limit_rest_rules_ok = True

                        # elif time.time() - s.time_of_change_position > 1.5:
                        #     if not s.reached_max_limit and \
                        #            (((up_lb < right_angle < up_ub) and (up_lb < left_angle < up_ub) and (up_lb2 < right_angle2 < up_ub2) and (up_lb2 < left_angle2 < up_ub2)) or \
                        #         ((down_lb < right_angle < down_ub) and (down_lb < left_angle < down_ub) and (down_lb2 < right_angle2 < down_ub2) and (down_lb2 < left_angle2 < down_ub2))):
                        #
                        #         self.count_not_good_range += 1
                        #
                        #         if self.count_not_good_range >= 20:
                        #             s.try_again_calibration = True
                        #             exercise_name = "s"

            if counter == s.rep:
                s.req_exercise = ""
                s.success_exercise = True
                break

        if len(list_joints) == 0:
            joints = self.fill_null_joint_list()
            if use_alternate_angles:
                new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                             joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                             joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)],
                             joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)],
                             None, None, None, None]
            else:
                new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                             joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                             joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                             joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                             None, None, None, None]
            list_joints.append(copy.deepcopy(new_entry))

        if not s.try_again_calibration and not s.repeat_explanation:
            s.ex_list.update({exercise_name: counter})

            if s.stop_requested or s.num_exercises_started == len(s.ex_in_training):
                Excel.wf_joints(exercise_name, list_joints)
            else:
                threading.Thread(target=Excel.wf_joints, args=(exercise_name, list_joints), daemon=True).start()

        else:
            s.number_of_repetitions_in_training -= s.patient_repetitions_counting_in_exercise
            s.max_repetitions_in_training -= s.rep


    def exercise_three_angles_3d(self, exercise_name, joint1, joint2, joint3, up_lb, up_ub, down_lb, down_ub,
                               joint4, joint5, joint6, up_lb2, up_ub2, down_lb2, down_ub2,
                                joint7, joint8, joint9, up_lb3, up_ub3, down_lb3, down_ub3, use_alternate_angles=False, use_alternate_for_second =False):

        if s.req_exercise.endswith("open_arms_and_forward"): # שלא יספור את החזרה הראשונה של הפתיחת ידיים כי היא לא נחשבת
            flag = False
            opened_arms = False

        else:
            flag = True
            opened_arms = True

        counter = 0
        list_joints = []
        s.time_of_change_position = time.time()
        # s.last_entry_angles = [0, 0, 0, 0, 0, 0]
        # s.change_in_trend = [False] * 6
        # increasing_decreasing = [[0] * 30 for _ in range(len(s.last_entry_angles))]
        # previous_entry = None

        while s.req_exercise == exercise_name:
            while s.did_training_paused and not s.stop_requested:
                time.sleep(0.01)

                if self.joints != {}:
                    self.joints = {}

                if self.previous_angles != {}:
                    self.previous_angles = {}

            self.sayings_generator(counter)

            #for i in range (1,100):
            joints = self.get_skeleton_data()
            if joints is not None:
                right_angle = self.calc_angle_3d(joints[str("R_" + joint1)], joints[str("R_" + joint2)],
                                                 joints[str("R_" + joint3)], "R_1")
                left_angle = self.calc_angle_3d(joints[str("L_" + joint1)], joints[str("L_" + joint2)],
                                                joints[str("L_" + joint3)], "L_1")

                angle_1_R_joints = joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)]
                angle_1_L_joints= joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)]

                if flag == False:
                    information_temp = [[str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), up_lb, up_ub],
                                     [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), up_lb, up_ub]]

                else:
                    information_temp = [[str("R_" + joint1), str("R_" + joint2), str("R_" + joint3), down_lb, down_ub],
                                     [str("L_" + joint1), str("L_" + joint2), str("L_" + joint3), down_lb, down_ub]]


                if use_alternate_for_second:
                    right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                      joints[str("L_" + joint6)], "R_2")
                    left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                     joints[str("R_" + joint6)], "L_2")

                    angle_2_R_joints = joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)]
                    angle_2_L_joints = joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)]

                    if flag == False:
                        information_temp.extend([[str("R_" + joint4), str("R_" + joint5), str("L_" + joint6), up_lb2, up_ub2],
                                         [str("L_" + joint4), str("L_" + joint5), str("R_" + joint6), up_lb2, up_ub2]])

                    else:
                        information_temp.extend([[str("R_" + joint4), str("R_" + joint5), str("L_" + joint6), down_lb2, down_ub2],
                                         [str("L_" + joint4), str("L_" + joint5), str("R_" + joint6), down_lb2, down_ub2]])


                else:
                    right_angle2 = self.calc_angle_3d(joints[str("R_" + joint4)], joints[str("R_" + joint5)],
                                                     joints[str("R_" + joint6)], "R_2")
                    left_angle2 = self.calc_angle_3d(joints[str("L_" + joint4)], joints[str("L_" + joint5)],
                                                    joints[str("L_" + joint6)], "L_2")

                    angle_2_R_joints = joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)]
                    angle_2_L_joints = joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)]

                    if flag == False:
                        information_temp.extend([[str("R_" + joint4), str("R_" + joint5), str("R_" + joint6), up_lb2, up_ub2],
                                         [str("L_" + joint4), str("L_" + joint5), str("L_" + joint6), up_lb2, up_ub2]])

                    else:
                        information_temp.extend([[str("R_" + joint4), str("R_" + joint5), str("R_" + joint6), down_lb2, down_ub2],
                                         [str("L_" + joint4), str("L_" + joint5), str("L_" + joint6), down_lb2, down_ub2]])


                if use_alternate_angles:
                    right_angle3 = self.calc_angle_3d(joints[str("R_" + joint7)], joints[str("R_" + joint8)],
                                                      joints[str("L_" + joint9)], "R_3")
                    left_angle3 = self.calc_angle_3d(joints[str("L_" + joint7)], joints[str("L_" + joint8)],
                                                     joints[str("R_" + joint9)], "L_3")

                    angle_3_R_joints = joints[str("R_" + joint7)], joints[str("R_" + joint8)], joints[str("L_" + joint9)]
                    angle_3_L_joints = joints[str("L_" + joint7)], joints[str("L_" + joint8)], joints[str("R_" + joint9)]

                    if flag == False:
                        information_temp.extend([[str("R_" + joint7), str("R_" + joint8), str("L_" + joint9), up_lb3, up_ub3],
                                         [str("L_" + joint7), str("L_" + joint8), str("R_" + joint9), up_lb3, up_ub3]])


                        if not s.req_exercise ==  "band_open_arms_and_up":
                            s.direction = "out"
                        else:
                            if s.direction == "out" or s.direction == None or s.direction == "in":
                                s.direction = "out"
                            else:
                                pass


                    else:
                        information_temp.extend([[str("R_" + joint7), str("R_" + joint8), str("L_" + joint9), down_lb3, down_ub3],
                                         [str("L_" + joint7), str("L_" + joint8), str("R_" + joint9), down_lb3, down_ub3]])

                        if not s.req_exercise ==  "band_open_arms_and_up":
                            s.direction = "in"
                        else:
                            if s.direction == "up" or s.direction == "down":
                                s.direction = "down"
                            elif s.direction == None:
                                s.direction = "in"
                            else:
                                pass

                else:
                    right_angle3 = self.calc_angle_3d(joints[str("R_" + joint7)], joints[str("R_" + joint8)],
                                                      joints[str("R_" + joint9)], "R_3")
                    left_angle3 = self.calc_angle_3d(joints[str("L_" + joint7)], joints[str("L_" + joint8)],
                                                     joints[str("L_" + joint9)], "L_3")

                    angle_3_R_joints = joints[str("R_" + joint7)], joints[str("R_" + joint8)], joints[str("R_" + joint9)]
                    angle_3_L_joints = joints[str("L_" + joint7)], joints[str("L_" + joint8)], joints[str("L_" + joint9)]

                    if flag == False:
                        information_temp.extend([[str("R_" + joint7), str("R_" + joint8), str("R_" + joint9), up_lb3, up_ub3],
                             [str("L_" + joint7), str("L_" + joint8), str("L_" + joint9), up_lb3, up_ub3]])


                    else:
                        information_temp.extend([[str("R_" + joint7), str("R_" + joint8), str("R_" + joint9), down_lb3, down_ub3],
                                         [str("L_" + joint7), str("L_" + joint8), str("L_" + joint9), down_lb3, down_ub3]])


                s.information = information_temp
                new_entry = [*angle_1_R_joints, *angle_1_L_joints, *angle_2_R_joints, *angle_2_L_joints, *angle_3_R_joints, *angle_3_L_joints,
                             right_angle, left_angle, right_angle2, left_angle2, right_angle3, left_angle3]

                s.last_entry_angles = [right_angle, left_angle, right_angle2, left_angle2, right_angle3, left_angle3]


                ##############################################################################
                print(left_angle, " ", right_angle)
                print(left_angle2, " ", right_angle2)
                print(left_angle3, " ", right_angle3)
                print("#######################################")
                ##############################################################################


                # if list_joints:
                #     previous_entry = list_joints[-1][-6:]
                #
                list_joints.append(copy.deepcopy(new_entry))
                #
                # if previous_entry:
                #     self.tred_func(previous_entry, increasing_decreasing)

                if right_angle is not None and left_angle is not None and \
                        right_angle2 is not None and left_angle2 is not None and \
                        right_angle3 is not None and left_angle3 is not None:

                    if (up_lb < right_angle < up_ub) and (up_lb < left_angle < up_ub) and \
                            (up_lb2 < right_angle2 < up_ub2) and (up_lb2 < left_angle2 < up_ub2) and \
                            (up_lb3 < right_angle3 < up_ub3) and (up_lb3 < left_angle3 < up_ub3) and (not flag):

                        if not opened_arms: #happens only for "__open_arms_and_forward" exercises where we want to ignore the first opening
                            opened_arms = True
                            flag = True
                            s.time_of_change_position = time.time()
                            self.count_not_good_range = 0
                            s.all_rules_ok = True
                            s.not_reached_max_limit_rest_rules_ok = False

                        else:
                            if s.reached_max_limit:
                                flag = True
                                counter += 1
                                s.number_of_repetitions_in_training += 1
                                s.patient_repetitions_counting_in_exercise += 1
                                print("counter:" + str(counter))
                                s.all_rules_ok = True
                                s.time_of_change_position = time.time()
                                self.count_not_good_range = 0
                                s.not_reached_max_limit_rest_rules_ok = False
                                # s.change_in_trend = [False] * 6
                                # increasing_decreasing = [[0] * 30 for _ in range(len(s.last_entry_angles))]

                            else:
                                s.not_reached_max_limit_rest_rules_ok = True

                    elif (down_lb < right_angle < down_ub) and (down_lb < left_angle < down_ub) and \
                            (down_lb2 < right_angle2 < down_ub2) and (down_lb2 < left_angle2 < down_ub2) and \
                            (down_lb3 < right_angle3 < down_ub3) and (down_lb3 < left_angle3 < down_ub3) and (flag):
                        flag = False
                        s.all_rules_ok = False
                        s.was_in_first_condition = True
                        s.time_of_change_position = time.time()
                        self.count_not_good_range = 0
                        # s.change_in_trend = [False] * 6
                        # increasing_decreasing = [[0] * 30 for _ in range(len(s.last_entry_angles))]

                    # elif time.time() - s.time_of_change_position > 3:
                    #     if not s.reached_max_limit and (up_lb < right_angle < up_ub) and (up_lb < left_angle < up_ub) and \
                    #         (up_lb2 < right_angle2 < up_ub2) and (up_lb2 < left_angle2 < up_ub2) and \
                    #         (up_lb3 < right_angle3 < up_ub3) and (up_lb3 < left_angle3 < up_ub3):
                    #         self.count_not_good_range += 1
                    #
                    #         if self.count_not_good_range >= 20:
                    #             s.try_again_calibration = True

            if counter == s.rep:
                s.req_exercise = ""
                s.success_exercise = True
                break


        if len(list_joints) == 0:
            joints = self.fill_null_joint_list()
            if use_alternate_angles:
                if use_alternate_for_second:
                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)],
                                 joints[str("R_" + joint7)], joints[str("R_" + joint8)], joints[str("L_" + joint9)],
                                 joints[str("L_" + joint7)], joints[str("L_" + joint8)], joints[str("R_" + joint9)],
                                 None, None, None, None,  None, None]

                else:
                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                 joints[str("R_" + joint7)], joints[str("R_" + joint8)], joints[str("L_" + joint9)],
                                 joints[str("L_" + joint7)], joints[str("L_" + joint8)], joints[str("R_" + joint9)],
                                 None, None, None, None,  None, None]
            else:
                if use_alternate_for_second:
                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("L_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("R_" + joint6)],
                                 joints[str("R_" + joint7)], joints[str("R_" + joint8)], joints[str("R_" + joint9)],
                                 joints[str("L_" + joint7)], joints[str("L_" + joint8)], joints[str("L_" + joint9)],
                                 None, None, None, None,  None, None]

                else:
                    new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                                 joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                                 joints[str("R_" + joint4)], joints[str("R_" + joint5)], joints[str("R_" + joint6)],
                                 joints[str("L_" + joint4)], joints[str("L_" + joint5)], joints[str("L_" + joint6)],
                                 joints[str("R_" + joint7)], joints[str("R_" + joint8)], joints[str("R_" + joint9)],
                                 joints[str("L_" + joint7)], joints[str("L_" + joint8)], joints[str("L_" + joint9)],
                                 None, None, None, None,  None, None]

            list_joints.append(copy.deepcopy(new_entry))

        if not s.try_again_calibration and not s.repeat_explanation:
            s.ex_list.update({exercise_name: counter})

            if s.stop_requested or s.num_exercises_started == len(s.ex_in_training):
                Excel.wf_joints(exercise_name, list_joints)
            else:
                threading.Thread(target=Excel.wf_joints, args=(exercise_name, list_joints), daemon=True).start()

        else:
            s.number_of_repetitions_in_training -= s.patient_repetitions_counting_in_exercise
            s.max_repetitions_in_training -= s.rep


    def hand_up_and_bend_angles(self, exercise_name, joint1, joint2, joint3, one_lb, one_ub, two_lb, two_ub, side):
        flag = True
        counter = 0
        list_joints = []
        s.time_of_change_position = time.time()


        while s.req_exercise == exercise_name:
            while s.did_training_paused and not s.stop_requested:
                time.sleep(0.01)
                if self.joints != {}:
                    self.joints = {}

                if self.previous_angles != {}:
                    self.previous_angles = {}

            self.sayings_generator(counter)
            joints = self.get_skeleton_data()
            if joints is not None:
                right_angle= self.calc_angle_3d(joints[str("R_" + joint1)], joints[str("R_" + joint2)],
                                          joints[str("L_" + joint3)], "R_1")

                left_angle = self.calc_angle_3d(joints[str("L_" + joint1)], joints[str("L_" + joint2)],
                                         joints[str("R_" + joint3)], "L_1")

                right_angle_2 = self.calc_angle_3d(joints[str("R_wrist")], joints[str("R_elbow")],
                                          joints[str("R_shoulder")], "R_2")

                left_angle_2 = self.calc_angle_3d(joints[str("L_wrist")], joints[str("L_elbow")],
                                          joints[str("L_shoulder")], "L_2")

                new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("L_" + joint3)],
                             joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("R_" + joint3)],
                             joints[str("R_wrist")], joints[str("R_elbow")], joints[str("R_shoulder")],
                             joints[str("L_wrist")], joints[str("L_elbow")], joints[str("L_shoulder")],
                             right_angle, left_angle, right_angle_2, left_angle_2]


                if side == "right":
                    s.last_entry_angles = [right_angle, right_angle_2]
                    if flag == False:
                        s.information = [[str("R_" + joint1), str("R_" + joint2), str("L_" + joint3), one_lb, one_ub],
                                         [str("R_wrist"), str("R_elbow"), str("R_shoulder"), 120, 180]]
                        s.direction = "left"


                    else:
                        s.information = [[str("R_" + joint1), str("R_" + joint2), str("L_" + joint3), two_lb, two_ub],
                                         [str("R_wrist"), str("R_elbow"), str("R_shoulder"), 0, 180]]

                        s.direction = "right"

                else:
                    s.last_entry_angles = [left_angle, left_angle_2]
                    if flag == False:
                        s.information = [[str("L_" + joint1), str("L_" + joint2), str("R_" + joint3), one_lb, one_ub],
                                         [str("L_wrist"), str("L_elbow"), str("L_shoulder"), 120, 180]]
                        s.direction = "right"


                    else:
                        s.information = [[str("L_" + joint1), str("L_" + joint2), str("R_" + joint3), two_lb, two_ub],
                                         [str("L_wrist"), str("L_elbow"), str("L_shoulder"), 0, 180]]
                        s.direction = "left"


                list_joints.append(copy.deepcopy(new_entry))


                ##############################################################################
                print(left_angle, " ", right_angle)
                print("second angle: ", left_angle_2, " ", right_angle_2)

                print("left wrist x: ", joints[str("R_wrist")].x)
                print("right wrist x: ", joints[str("L_shoulder")].x)
                print("nose: ", joints[str("nose")].y)


                ##############################################################################

                if side == 'right':
                    if right_angle is not None and right_angle_2 is not None:
                        if (one_lb < right_angle < one_ub) and (120 < right_angle_2< 180) and (not flag):
                            if joints["R_wrist"].x - joints["L_shoulder"].x > 60:
                                s.hand_not_good = False

                                if s.reached_max_limit:
                                    flag = True
                                    counter += 1
                                    s.patient_repetitions_counting_in_exercise += 1
                                    s.number_of_repetitions_in_training += 1
                                    print("counter:" + str(counter))
                                    s.all_rules_ok = True
                                    s.time_of_change_position = time.time()
                                    s.not_reached_max_limit_rest_rules_ok = False

                                else:
                                    s.not_reached_max_limit_rest_rules_ok = True

                            else:
                                s.hand_not_good = True

                                if s.reached_max_limit:
                                    s.not_reached_max_limit_rest_rules_ok = False

                                else:
                                    s.not_reached_max_limit_rest_rules_ok = True


                        elif (two_lb < right_angle < two_ub) and (flag):

                            if (abs(joints["L_shoulder"].x - joints["R_shoulder"].x) > (s.dist_between_shoulders - 30)) or\
                                (abs(joints["L_shoulder"].y - joints["R_shoulder"].y) < 15) or \
                                    joints["L_shoulder"].x - joints["R_wrist"].x > 0 :
                                s.hand_not_good = False
                                flag = False
                                s.all_rules_ok = False
                                s.was_in_first_condition = True
                                s.time_of_change_position = time.time()

                            else:
                                s.hand_not_good = True


                    # elif time.time() - s.time_of_change_position > 3:
                    #     if not s.reached_max_limit and (one_lb < right_angle < one_ub) and\
                    #     (135 < right_angle_2< 180):
                    #         self.count_not_good_range += 1
                    #
                    #         if self.count_not_good_range >= 20:
                    #             s.try_again_calibration = True

                else:
                    if right_angle is not None and left_angle is not None:
                        if (one_lb < left_angle < one_ub) and  (120 < left_angle_2< 180) and (not flag):

                            if joints["R_shoulder"].x - joints["L_wrist"].x > 60:
                                s.hand_not_good = False

                                if s.reached_max_limit:
                                    flag = True
                                    counter += 1
                                    s.number_of_repetitions_in_training += 1
                                    s.patient_repetitions_counting_in_exercise += 1
                                    print("counter:" + str(counter))
                                    s.all_rules_ok = True
                                    s.time_of_change_position = time.time()
                                    self.count_not_good_range = 0
                                    s.not_reached_max_limit_rest_rules_ok = False

                                else:
                                    s.not_reached_max_limit_rest_rules_ok = True


                            else:
                                s.hand_not_good = True

                                if s.reached_max_limit:
                                    s.not_reached_max_limit_rest_rules_ok = False

                                else:
                                    s.not_reached_max_limit_rest_rules_ok = True


                        elif (two_lb < left_angle < two_ub) and (flag):

                            if (abs(joints["L_shoulder"].x - joints["R_shoulder"].x) > s.dist_between_shoulders - 30) or\
                                (abs(joints["L_shoulder"].y - joints["R_shoulder"].y) < 15) or \
                                    joints["L_wrist"].x - joints["R_shoulder"].x > 0:

                                s.hand_not_good = False
                                flag = False
                                s.all_rules_ok = False
                                s.was_in_first_condition = True
                                s.time_of_change_position = time.time()
                                self.count_not_good_range = 0

                            else:
                                s.hand_not_good = True


                        #
                        # elif time.time() - s.time_of_change_position > 3:
                        #     if not s.reached_max_limit and (one_lb < left_angle < one_ub) and (135 < left_angle_2< 180):
                        #         self.count_not_good_range += 1
                        #
                        #         if self.count_not_good_range >= 20:
                        #             s.try_again_calibration = True

            if counter == s.rep:
                s.req_exercise = ""
                s.success_exercise = True
                break

        if len(list_joints) == 0:
            joints = self.fill_null_joint_list()
            new_entry = [joints[str("R_" + joint1)], joints[str("R_" + joint2)], joints[str("R_" + joint3)],
                         joints[str("L_" + joint1)], joints[str("L_" + joint2)], joints[str("L_" + joint3)],
                         joints[str("R_wrist")], joints[str("R_elbow")], joints[str("R_shoulder")],
                         joints[str("L_wrist")], joints[str("L_elbow")], joints[str("L_shoulder")],
                         None, None, None, None]
            list_joints.append(copy.deepcopy(new_entry))

        if not s.try_again_calibration and not s.repeat_explanation:
            s.ex_list.update({exercise_name: counter})

            if s.stop_requested or s.num_exercises_started == len(s.ex_in_training):
                Excel.wf_joints(exercise_name, list_joints)
            else:
                threading.Thread(target=Excel.wf_joints, args=(exercise_name, list_joints), daemon=True).start()

        else:
            s.number_of_repetitions_in_training -= s.patient_repetitions_counting_in_exercise
            s.max_repetitions_in_training -= s.rep


    def hello_waving(self):  # check if the participant waved
        while s.req_exercise == "hello_waving":
            joints = self.get_skeleton_data()
            if joints is not None:
                right_shoulder = joints[str("R_shoulder")]
                right_wrist = joints[str("R_wrist")]
                if right_shoulder.y > right_wrist.y != 0:
                    s.waved_has_tool = True
                    s.req_exercise = ""


######################################################### First set of ball exercises

    def ball_bend_elbows(self):  # EX1
        self.exercise_two_angles_3d("ball_bend_elbows", "shoulder", "elbow", "wrist", 10, 65, 95, 180,
                                    "elbow", "shoulder", "hip", 0, 85, 0, 85)

    def ball_raise_arms_above_head(self):  # EX2
        self.exercise_two_angles_3d("ball_raise_arms_above_head", "hip", "shoulder", "elbow", 100, 180, 0, 70,
                                    "shoulder", "elbow", "wrist", 120, 180, 120, 180)


    def ball_switch(self):  # EX3
        self.exercise_two_angles_3d_with_axis_check("ball_switch", "shoulder", "elbow","wrist", 0, 180, 135, 180,
                                    "wrist", "hip", "hip",100,160,40,70, True, True)
                                    #"wrist", "hip", "hip",95 ,135 , 35, 70, True, True)


######################################################### Second set of ball exercises

    def ball_open_arms_and_forward(self):  # EX4
        self.exercise_three_angles_3d("ball_open_arms_and_forward", "hip", "shoulder", "elbow", 60, 120,20, 110,
                                    "shoulder", "elbow", "wrist", 150, 180 , 0, 180,
                                    "wrist", "shoulder", "wrist", 140,180, 0, 105 ,True)

    def ball_open_arms_above_head(self):  # EX5
        self.exercise_two_angles_3d("ball_open_arms_above_head", "elbow", "shoulder", "hip", 145,180, 80, 100,
                                   "shoulder", "elbow", "wrist", 130, 180, 130, 180)


########################################################### Set with a rubber band

    def band_open_arms(self):  # EX6
        self.exercise_three_angles_3d("band_open_arms","hip", "shoulder", "wrist", 65, 120, 40, 120,
                                    "shoulder", "elbow", "wrist", 135, 180, 0, 180,
                                    "wrist", "shoulder", "wrist", 135,180,0,120,True)

        #"wrist", "shoulder", "shoulder", 100, 160,75, 95, True)

    def band_open_arms_and_up(self):  # EX7
        self.exercise_three_angles_3d("band_open_arms_and_up", "shoulder", "elbow", "wrist", 135,180,0,180,
                                      "elbow", "shoulder", "hip", 120, 180, 0, 105,
                                    "wrist", "shoulder", "wrist", 70, 170, 20, 130, True, True)

    def band_up_and_lean(self):  # EX8
        self.exercise_two_angles_3d_with_axis_check("band_up_and_lean", "shoulder", "elbow", "wrist", 110, 180, 90,180,
                                   "elbow", "hip", "hip", 120, 170, 50, 115, True, True)

    def band_straighten_left_arm_elbows_bend_to_sides(self):  # EX9
        self.exercise_two_angles_3d_one_side("band_straighten_left_arm_elbows_bend_to_sides", "shoulder", "elbow", "wrist", 0, 75, 0,75, 135,180, 0, 75,
                                   "elbow", "shoulder", "hip", 60, 130, 60, 130, 60, 130,60,130)


    def band_straighten_right_arm_elbows_bend_to_sides(self):  # EX10
        self.exercise_two_angles_3d_one_side("band_straighten_right_arm_elbows_bend_to_sides", "shoulder", "elbow", "wrist", 135, 180, 0,75, 0,75, 0, 75,
                                   "elbow", "shoulder", "hip", 60, 120, 60, 120, 60, 120,60,120)

######################################################  Set with a stick
    def stick_bend_elbows(self):  # EX11
        self.exercise_two_angles_3d("stick_bend_elbows", "shoulder", "elbow", "wrist",10, 70, 95, 180,
                                    "elbow", "shoulder", "hip", 0, 70, 0, 70)

    def stick_bend_elbows_and_up(self):  # EX12
        self.exercise_two_angles_3d("stick_bend_elbows_and_up", "hip", "shoulder", "elbow", 110, 180, 0, 70,
                                 "shoulder", "elbow", "wrist", 120, 180, 0, 75)

    def stick_raise_arms_above_head(self):  # EX13
        self.exercise_two_angles_3d("stick_raise_arms_above_head", "hip", "shoulder", "elbow", 105, 180, 10, 70,
                                    "wrist", "elbow", "shoulder", 125,180,125,180)

    def stick_switch(self):  # EX14
        # self.exercise_two_angles_3d("stick_switch", "shoulder", "elbow", "wrist", 0, 180, 140, 180,
        #                             "wrist", "hip", "hip", 95, 140, 35, 70, True, True)
        self.exercise_two_angles_3d_with_axis_check("stick_switch", "shoulder", "elbow","wrist", 0, 180, 135, 180,
                                    "wrist", "hip", "hip",85,160,10,70, True, True)


    # def stick_bending_forward(self):
    #     self.exercise_two_angles_3d("stick_bending_forward", "wrist", "elbow", "shoulder", 120,180,120,180,
    #                                  "shoulder", "hip", "knee",40,90,105,150)....

    def stick_up_and_lean(self):  # EX15
        self.exercise_two_angles_3d_with_axis_check("stick_up_and_lean", "shoulder", "elbow", "wrist", 110, 180, 90, 180,
                                                    "elbow", "hip", "hip", 115, 170, 50, 110, True, True)

    ######################################################  Set with a weights

    # def weights_right_hand_up_and_bend(self):  # EX16
    #     self.hand_up_and_band_angles("weights_right_hand_up_and_bend", "hip", "shoulder", "wrist", 120, 160, 140, 180, "right")
    #
    #
    # def weights_left_hand_up_and_bend(self):  # EX17
    #     self.hand_up_and_band_angles("weights_left_hand_up_and_bend", "hip", "shoulder", "wrist", 120, 160, 140,
    #                                         180, "left")

    def weights_open_arms_and_forward(self):  # EX18
        self.exercise_three_angles_3d("weights_open_arms_and_forward", "hip", "shoulder", "elbow", 60, 120, 20, 110,
                                      "shoulder", "elbow", "wrist", 150, 180, 0, 180,
                                      "wrist", "shoulder", "wrist", 140, 180, 0, 105, True)

    def weights_abduction(self):  # EX19
        self.exercise_two_angles_3d("weights_abduction" , "hip", "shoulder", "elbow", 80,120,0,55,
                                    "shoulder", "elbow", "wrist", 140,180,140,180)

    ################################################# Set of exercises without equipment
    def notool_hands_behind_and_lean(self): # EX20
        self.exercise_two_angles_3d_with_axis_check("notool_hands_behind_and_lean", "shoulder", "elbow", "wrist", 10,70,10,70,
                                    "elbow", "hip", "hip", 115, 170, 80, 115,True, True)
                                    # "elbow", "hip", "hip", 30, 100, 125, 170,True, True)

    def notool_right_hand_up_and_bend(self):  # EX21
        self.hand_up_and_bend_angles("notool_right_hand_up_and_bend", "wrist", "hip", "hip", 20, 100, 90, 180, "right")

    def notool_left_hand_up_and_bend(self): #EX22
        self.hand_up_and_bend_angles("notool_left_hand_up_and_bend", "wrist", "hip", "hip", 20, 100, 90, 180, "left")

    def notool_raising_hands_diagonally(self): # EX23
        self.exercise_two_angles_3d_with_axis_check("notool_raising_hands_diagonally", "wrist", "shoulder", "hip", 80, 135, 105, 150,
                                    #"elbow", "shoulder", "shoulder", 0, 180, 40, 75, True, True)\
                                    "shoulder", "elbow", "wrist", 0,180, 120, 180, False,  True, True)


    def notool_right_bend_left_up_from_side(self):# EX24
        self.exercise_two_angles_3d_one_side("notool_right_bend_left_up_from_side", "wrist", "elbow", "shoulder", 95, 170, 0,50, 140, 180, 140, 180,
                                             "hip", "shoulder", "elbow", 0, 60, 0, 60, 70, 120, 0, 50)

    def notool_left_bend_right_up_from_side(self):# EX25
        self.exercise_two_angles_3d_one_side("notool_left_bend_right_up_from_side", "wrist", "elbow","shoulder", 140, 180, 140, 180,95, 170, 0, 40,
                                             "hip", "shoulder", "elbow", 80, 120, 0, 50, 0, 60, 0, 60)



if __name__ == '__main__':
    s.camera_num = 1  # 0 - webcam, 2 - second USB in maya's computer
    s.robot_counter = 0
    # Audio variables initialization
    language = 'Hebrew'
    gender = 'Male'
    s.audio_path = 'audio files/' + language + '/' + gender + '/'
    s.picture_path = 'audio files/' + language + '/' + gender + '/'
    # s.str_to_say = ""
    # current_time = datetime.datetime.now()
    # s.participant_code = str(current_time.day) + "." + str(current_time.month) + " " + str(current_time.hour) + "." + \
    # str(current_time.minute) + "." + str(current_time.second)

    s.gymmy_finished_demo = True
    # Training variables initialization
    s.rep = 5
    s.waved = False
    s.success_exercise = False
    s.calibration = False
    s.finish_workout = False
    s.gymmy_done = False
    s.camera_done = False
    s.robot_count = False
    s.demo_finish = False
    s.list_effort_each_exercise = {}
    s.ex_in_training = []
    s.finish_program= False
    # s.exercises_start=False
    s.waved_has_tool = True  # True just in order to go through the loop in Gymmy
    s.max_repetitions_in_training=0
    s.did_training_paused= False
    # Excel variable
    ############################# להוריד את הסולמיות
    s.ex_list = {}
    s.chosen_patient_ID="3333"
    s.req_exercise = "band_straighten_left_arm_elbows_bend_to_sides"
    s.explanation_over = True
    time.sleep(2)
    s.asked_for_measurement = False
    # Create all components
    s.camera = Camera()
    s.number_of_repetitions_in_training=0
    s.patient_repetitions_counting_in_exercise=0
    s.starts_and_ends_of_stops=[]
    s.starts_and_ends_of_stops.append(time.time())
    s.dist_between_shoulders =280
    pygame.mixer.init()
    # Start all threads
    s.camera.start()
    Excel.create_workbook_for_training()  # create workbook in excel for this session
    time.sleep(30)
    s.req_exercise=""
    # Excel.find_and_add_training_to_patient()
    Excel.close_workbook()


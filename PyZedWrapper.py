import threading
import time
import cv2
import pyzed.sl as sl
import Settings as s  # Controlled shutdown


class PyZedWrapper(threading.Thread):
    def __init__(self):
        """Initialize the ZED Camera Thread."""
        super().__init__()
        print("MP INITIALIZATION")
        self.zed = sl.Camera()
        self.frame_count = 0
        self.start_time = time.time()
        self.gui_enabled = False
        self.latest_frame = None
        self.latest_keypoints = None
        self.lock = threading.Lock()
        self.gui_update_callback = None
        self.running = True  # Flag to control thread execution

    def stop(self):
        """Stop the thread safely."""
        self.running = False
        s.finish_program = True

    def get_latest_frame(self):
        """Retrieve the latest frame safely for GUI updating."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_latest_keypoints(self):
        """Retrieve the latest detected keypoints."""
        with self.lock:
            return s.latest_keypoints.copy() if s.latest_keypoints is not None else None

    def set_gui_enabled(self, enabled):
        """Enable or disable GUI updates dynamically."""
        self.gui_enabled = enabled

    def set_gui_update_callback(self, callback):
        """Set a callback function to update the GUI with new frames."""
        self.gui_update_callback = callback

    def get_zed(self):
        return self.zed

    def set_detection_model(self, model):
        """Change the body tracking model dynamically."""
        self.zed.disable_body_tracking()  # Stop tracking
        self.body_params.detection_model = model  # Change model
        self.zed.enable_body_tracking(self.body_params)  # Restart tracking
        print(f"Switched to detection model: {model}")

    def set_detection_model_to_accurate(self):
        self.set_detection_model(sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE)

    def set_detection_model_to_medium(self):
        self.set_detection_model(sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM)


    def run(self):
        """Main thread function to capture frames and detect keypoints."""
        print("MP START")

        # Initialize camera parameters
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.depth_stabilization = True
        init.coordinate_units = sl.UNIT.MILLIMETER
        init.camera_fps = 60

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

        positional_tracking_parameters = sl.PositionalTrackingParameters()
        positional_tracking_parameters.set_as_static = True
        self.zed.enable_positional_tracking(positional_tracking_parameters)

        self.body_params = sl.BodyTrackingParameters()
        self.body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
        self.body_params.enable_tracking = True
        self.body_params.enable_body_fitting = True
        self.body_params.body_format = sl.BODY_FORMAT.BODY_18
        self.body_params.prediction_timeout_s = 0.2

        if self.zed.enable_body_tracking(self.body_params) != sl.ERROR_CODE.SUCCESS:
            print("Error enabling body tracking")
            self.zed.close()
            return

        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)

        # bodies = sl.Bodies()
        # runtime = sl.RuntimeParameters()
        # selected_keypoints = list(range(18))

        runtime = sl.RuntimeParameters()
        runtime.enable_depth = True  # Ensure depth sensing is enabled

        image = sl.Mat()

        while self.zed.is_opened() and not s.finish_program and self.running:
            if self.zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                time.sleep(0.00001)

                # self.zed.retrieve_bodies(bodies, sl.BodyTrackingRuntimeParameters())
                # body_list = bodies.body_list

                if s.asked_for_measurement:
                    self.zed.retrieve_image(image, sl.VIEW.LEFT)  # Get the left image from the ZED camera

                    # Convert to OpenCV format
                    frame = image.get_data()  # Get frame as NumPy array
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

                    # Lock and store the latest frame for potential GUI use
                    with self.lock:
                        self.latest_frame = frame

                else:
                    self.latest_frame = None


                keypoints_list = []
                # for body in body_list:
                #     if body.keypoint_2d is not None:
                #         keypoints = np.array(body.keypoint_2d, dtype=int)
                #         for idx in selected_keypoints:
                #             if len(keypoints) > idx:
                #                 kp = keypoints[idx]
                #                 if 0 <= kp[0] < 1280 and 0 <= kp[1] < 720:
                #                     keypoints_list.append(kp)
                #                 else:
                #                     keypoints_list.append([-1, -1])

                with self.lock:
                    s.latest_keypoints = keypoints_list

                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time >= 1:
                    fps = self.frame_count / elapsed_time
                    s.fps = fps
                    print(f"FPS: {fps:.2f}")
                    self.frame_count = 0
                    self.start_time = time.time()

                # Stop camera if 'q' is pressed or finish_program is triggered
                key = cv2.waitKey(10)
                if s.finish_program or key == ord('q'):
                    self.stop()
                    break

        self.zed.close()
        print("Camera closed")


if __name__ == '__main__':
    s.stop = False
    s.finish_program = False
    s.finish_workout = False
    s.asked_for_measurement = True

    mediap = PyZedWrapper()
    mediap.start()
    try:
        while not s.finish_program:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping thread...")
        mediap.stop()
        mediap.join()

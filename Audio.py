import threading
import time
import pygame.mixer
import wave
import Settings as s
import random
import queue


class ContinuousAudio(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        random_number = random.randint(1, 5)
        self.file_name = f'song_{random_number}'
        self.current_volume = s.volume
        self.last_additional_audio_time = 0
        self.audio_playing = False

    def run(self):
        while s.audio_path is None and not s.finish_program:
            time.sleep(1)

        if s.finish_program:
            return

        audio_path = 'audio files/Songs/' + self.file_name + '.wav'
        total_length = self.get_audio_length(audio_path)

        min_start_time = 0
        max_start_time = max(total_length - 30 * 60, 0)
        start_time = random.uniform(min_start_time, max_start_time)

        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.set_volume(s.volume)

        while not s.finish_program:
            if s.play_song and s.audio_path is not None:
                if not self.audio_playing:
                    pygame.mixer.music.play(start=0)
                    pygame.mixer.music.set_pos(start_time)
                    self.audio_playing = True

                current_time = time.time()

                if s.additional_audio_playing:
                    pygame.mixer.music.set_volume(0.1 if s.volume >= 0.1 else s.volume)
                    self.last_additional_audio_time = current_time
                else:
                    if current_time - self.last_additional_audio_time >= 1:
                        if pygame.mixer.music.get_volume() != s.volume:
                            pygame.mixer.music.set_volume(s.volume)
                            self.current_volume = s.volume

                if not s.additional_audio_playing:
                    pygame.mixer.music.set_volume(s.volume)

                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(start=0)
                    pygame.mixer.music.set_pos(start_time)

            else:
                if self.audio_playing:
                    pygame.mixer.music.stop()
                    self.audio_playing = False

                time.sleep(0.1)

        pygame.mixer.music.stop()

    def stop(self):
        s.finish_workout = True
        s.finish_program = True
        pygame.mixer.music.stop()

    def get_audio_length(self, file_path):
        with wave.open(file_path, 'rb') as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            duration = frames / float(rate)
            return duration


class AdditionalAudio(threading.Thread):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.daemon = True
        pygame.mixer.init()
        self.current_chanel = None
        self.something_added_to_queue = False
        self.time_added = None
        self.start()

    def run(self):
        while not s.finish_program:
            try:
                file_name, is_explanation, is_effort , is_popping_screen= self.queue.get(timeout=0.5)
                self.play_audio(file_name, is_explanation, is_effort, is_popping_screen)

            except queue.Empty:
                continue

    def play_audio(self, file_name, is_explanation, is_effort, is_popping_screen):
        s.additional_audio_playing = True
        channel= pygame.mixer.find_channel()

        sound = pygame.mixer.Sound(s.audio_path + file_name + '.wav')
        channel.play(sound)
        self.current_chanel = channel

        while channel.get_busy() and not s.finish_program:
            if (is_explanation and s.explanation_over) or (is_effort and s.finished_effort) or self.something_added_to_queue and self.time_added and (time.time() - self.time_added >=2) \
                   or (is_popping_screen and not s.suggest_repeat_explanation):
                channel.stop()
                self.something_added_to_queue = False

            pygame.time.Clock().tick(10)

        self.something_added_to_queue = False
        self.time_added = None

        s.additional_audio_playing = False
        time.sleep(0.2)  # Optional small delay between clips

    def add_to_queue(self, file_name, is_explanation=False, is_effort=False, is_popping_screen= False):
        if self.current_chanel and self.current_chanel.get_busy():
            self.time_added = time.time()
            self.something_added_to_queue = True

        self.queue.put((file_name, is_explanation, is_effort, is_popping_screen))


def say(str_to_say, is_explanation=False, is_effort=False, is_popping_screen= False):
    if not s.finish_program:
        s.audio_manager.add_to_queue(str_to_say, is_explanation, is_effort, is_popping_screen)


def get_wav_duration(str_to_say):
    str_with_path = s.audio_path + str_to_say + '.wav'
    with wave.open(str_with_path, 'rb') as wav_file:
        num_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        duration = num_frames / float(frame_rate)
        return round(duration, 3)


if __name__ == '__main__':
    language = 'Hebrew'
    gender = 'Male'
    s.audio_path = 'audio files/' + language + '/' + gender + '/'
    s.finish_workout = False
    s.finish_program = False
    s.additional_audio_playing = False
    s.volume = 0.5
    s.play_song = True
    s.explanation_over = False
    s.finished_effort = False

    pygame.mixer.init()
    s.req_exercise = "bbb"
    s.stop_song = False

    # Initialize Audio Manager
    audio_manager = AudioManager()

    # Start background music
    continuous_audio = ContinuousAudio()
    continuous_audio.start()

    # Simulate use of say()
    time.sleep(5)
    say("welcome")
    say("welcome")
    say("welcome")

    # Simulate ending
    time.sleep(10)
    print("Stopping program...")
    s.finish_program = True
    continuous_audio.stop()
    continuous_audio.join()

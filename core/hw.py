import datetime
import time

import RPi.GPIO as GPIO
import pygame
from picamera import PiCamera
from pygame import mixer
from path_support import *

#btn_scan = 10
#btn_specific_card = 13
#btn_table = 15
#btn_s = 21
#btn_c = 22
#btn_d = 23
#btn_h = 24
#led1 = 16
#led2 = 18
import detection_support
import input_support

mixer.init()


def init_hardware():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(16, GPIO.OUT)
    GPIO.setup(18, GPIO.OUT)
    GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    camera = PiCamera()
    camera.resolution = (1920, 1080)
    camera.framerate = 30
    return camera


def record_video(camera, is_recording):
    if not is_recording:
        is_recording = True
        print("Started Recording")
        camera.start_recording(f"/home/pi/Videos/{datetime.datetime.now()}.h264")
        time.sleep(0.15)
    else:
        is_recording = False
        try:
            camera.stop_recording()
            time.sleep(0.15)
            print("Stopped Recording")
        except Exception as e:
            print(str(e))
    return is_recording

def is_button_pressed(button):
    if GPIO.input(button) == GPIO.HIGH:
        return True
    return False

def suit_button_press():
    if GPIO.input(21) == GPIO.HIGH:
        pin = "s"
    if GPIO.input(22) == GPIO.HIGH:
        pin = "c"
    if GPIO.input(23) == GPIO.HIGH:
        pin = "h"
    if GPIO.input(24) == GPIO.HIGH:
        pin = "d"
    return pin

def capture_input(camera, path, recording=None):
    if recording:
        print(f"{datetime.datetime.now()} : Capturing Input to {path}")
        camera.wait_recording(2)
        camera.capture(path, use_video_port=True)
    else:
        print(f"{datetime.datetime.now()} : Capturing Input to {path}")
        camera.capture(path)

    camera.capture(path)
    camera.stop_preview()

def wait_for_audio():
    while pygame.mixer.music.get_busy() == True:
        continue


def play_audio(dir):
    mixer.music.load(dir)
    mixer.music.play()
    wait_for_audio()


def on_led():
    GPIO.setup(16, True)
    GPIO.setup(18, True)


def off_led():
    GPIO.setup(16, False)
    GPIO.setup(18, False)

def handle_check_for_suit(suit, detected_cards):
    print(f"Checking for suit {suit}")
    play_audio(f"{PATH_TO_SOUNDFILES}/finding.mp3")
    play_audio(f"{PATH_TO_SOUNDFILES}/{suit}.mp3")
    detection_support.chk_for_suit(suit, detected_cards)
    input_support.ready_for_next()

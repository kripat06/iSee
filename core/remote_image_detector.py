import datetime
import time

import support
from pygame import mixer

import detection_support
import hw
import input_support
from path_support import *

input_arguments = support.parse_input_arguments()

# Starting the mixer
print(f"{datetime.datetime.now()} : mixer.init()")
mixer.init()

### START HERE

print(f"{datetime.datetime.now()} : h1.init_hardware()")
camera = hw.init_hardware()

input_image_path = input_arguments.imgpath
api_ip = input_arguments.api_ip
print(f"{datetime.datetime.now()} : Input image path: {input_image_path}.")
print(f"{datetime.datetime.now()} : initiating loop.")

print(f"{datetime.datetime.now()} : Click button to detect cards.")

detected_cards = []
is_recording_on = False

while True:
    if hw.is_button_pressed(13):
        print("Scanning Cards!")
        hw.play_audio(f"{PATH_TO_SOUNDFILES}/scanning_cards_in_hand.mp3")
        try:
            camera.wait_recording(0.25)
        except Exception as e:
            print(str(e))
        input_support.scan_card(camera, input_image_path)
        detected_cards = detection_support.detect_cards(input_image_path, api_ip)
        input_support.ready_for_next()
    elif hw.is_button_pressed(21):
        suit = "spades"
        hw.handle_check_for_suit(suit, detected_cards)
        input_support.ready_for_next()
    elif hw.is_button_pressed(22):
        suit = "clubs"
        hw.handle_check_for_suit(suit, detected_cards)
        input_support.ready_for_next()
    elif hw.is_button_pressed(23):
        suit = "hearts"
        hw.handle_check_for_suit(suit, detected_cards)
        input_support.ready_for_next()
    elif hw.is_button_pressed(15):
        suit = "diamonds"
        hw.handle_check_for_suit(suit, detected_cards)
        input_support.ready_for_next()
    elif hw.is_button_pressed(10):
        is_recording_on = hw.record_video(camera, is_recording_on)
        input_support.ready_for_next()
    elif hw.is_button_pressed(24):
        print("Scanning Table!")
        hw.play_audio(f"{PATH_TO_SOUNDFILES}/scanning_cards_in_play.mp3")
        try:
            camera.wait_recording(0.25)
        except Exception as e:
            print(str(e))
        input_support.scan_table(camera, input_image_path)
        detection_support.detect_cards_on_table(input_image_path, api_ip)
        input_support.ready_for_next()

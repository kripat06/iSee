from pygame import mixer
import datetime
import argparse

import hw
import support
from image_processor import ImageProcessor

input_arguments = support.parse_input_arguments()

parser = argparse.ArgumentParser()
parser.add_argument('--confidence', help='Minimum confidence threshold for displaying detected objects', default=10)

args = parser.parse_args()
min_confidence = int(input_arguments.confidence)


# Starting the mixer
print(f"{datetime.datetime.now()} : mixer.init()")
mixer.init()

### START HERE
#min_confidence = 10
PATH_TO_CKPT = "/home/pi/tflite1/tflite/Sample_TFLite_model/detect.tflite"
PATH_TO_LABELS = "/home/pi/tflite1/tflite/Sample_TFLite_model/labelmap.txt"
PATH_TO_SOUNDFILES = "/home/pi/tflite1/audio"
print(f"{datetime.datetime.now()} : h1.init_hardware()")
camera = hw.init_hardware()

image_processor = ImageProcessor(PATH_TO_CKPT, PATH_TO_LABELS, min_confidence, PATH_TO_SOUNDFILES)
input_image_path = "/home/pi/tflite1/tmp/input.jpg"
print(f"{datetime.datetime.now()} : initiating loop.")

print(f"{datetime.datetime.now()} : Click button to detect cards.")
while True:
    camera.start_preview()
    if hw.is_button_pressed():
        hw.capture_input(camera, input_image_path)
    else:
        camera.stop_preview()
        continue

    detected_cards = image_processor.process(input_image_path)
    print(f"{datetime.datetime.now()} : Detected Cards: {detected_cards}")
    print(f"{datetime.datetime.now()} : Ready for next Input.")
    print(f"{datetime.datetime.now()} : Click button to detect cards again.")

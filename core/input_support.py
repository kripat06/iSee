import datetime

import hw
from path_support import *


def scan_card(camera, input_image_path):
    # hw.on_led()
    hw.capture_input(camera, input_image_path)
    # hw.off_led()

def scan_table(camera, input_image_path):
    hw.capture_input(camera, input_image_path)


def ready_for_next():
    print(f"{datetime.datetime.now()} : Ready for next Input.")
    print(f"{datetime.datetime.now()} : Click button to detect cards again.")

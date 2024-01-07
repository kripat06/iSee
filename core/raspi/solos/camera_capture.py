from picamera import PiCamera
from time import sleep
import RPi.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(38, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
camera = PiCamera()
camera.start_preview()

while True:
    if GPIO.input(38) == GPIO.HIGH:
        camera.capture('/tmp/input.jpg')
        camera.stop_preview()
        print("Image Captured!")
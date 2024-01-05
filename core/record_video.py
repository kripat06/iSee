import time
from picamera import PiCamera

camera = PiCamera()

def record_video():
    print('Please Wait...')
    data= time.strftime("%d_%b_%Y\%H:%M:%S")
    camera.start_preview()
    camera.start_recording('/home/pi/Desktop/Visitors/rec.h264')
    time.sleep(30)
    print(data)
    camera.stop_recording()
    camera.stop_preview()
    print('Video recorded successfully')
    time.sleep(2)

record_video()
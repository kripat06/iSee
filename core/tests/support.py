import argparse


def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidence', help='Minimum confidence threshold for displaying detected objects', default=10)
    parser.add_argument('--imgpath', help='Image Path', default="/home/pi/tflite1/tmp/input.jpg")
    parser.add_argument('--api_ip', help='API IP')

    args = parser.parse_args()
    return args

import datetime

import cv2
import numpy as np
from pygame import mixer

import support


class ImageProcessor():
    def __init__(self, path_to_check_point, path_to_labels, min_confidence, path_to_sound_files):
        self._path_to_checkpoint = path_to_check_point
        # self._path_to_labels = path_to_labels
        self._min_confidence = min_confidence
        self._path_to_sound_files = path_to_sound_files
        self._interpreter = support.prepare_tf(self._path_to_checkpoint)
        self._labels = support.load_labels(path_to_labels)
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._height = self._input_details[0]['shape'][1]
        self._width = self._input_details[0]['shape'][2]

    def process(self, image_path):
        # Process the input image
        mixer.music.load(f"{self._path_to_sound_files}/proscessing.mp3")
        mixer.music.play()
        print(f"{datetime.datetime.now()} : Processing")
        img = cv2.imread(image_path)
        print(f"{datetime.datetime.now()} : Image read from {image_path}")
        detected_cards = self.detect_cards(img)
        self.describe_detected_cards(detected_cards)
        return detected_cards

    def process_table(self, image_path):
        ### @TODO: Implement Proscess Table Using Appropriate Model
        return self.proscess(image_path)

    def detect_cards(self, image):
        name_of_duplicate = set()
        imH, imW, _ = image.shape
        input_data = self.prepare_image_data(image)
        self.invoke_interpreter(input_data)
        boxes, classes, scores = self.retrieve_detection_results()
        cards = []
        for i in range(len(scores)):
            score = scores[i] * 100
            if self._min_confidence <= score <= 100:
                xmin = int(max(1, (boxes[i][1] * imW)))
                object_name = self._labels[int(classes[i])]
                if object_name not in name_of_duplicate:
                    support.decorate_image(boxes[i], classes[i], scores[i], imH, imW, image, object_name)
                    name_of_duplicate.add(object_name)
                    dc = self.prepare_card_info(xmin, object_name, score)
                    cards.append(dc)
        sorted(cards, key=lambda k: k["loc"])
        support.show_image("Detected Cards", image)
        return cards

    def prepare_image_data(self, img):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self._width, self._height))
        image_data = np.expand_dims(image_resized, axis=0)
        return image_data

    def invoke_interpreter(self, image_input_data):
        # Perform the actual detection by running the model with the image as input
        print(f"{datetime.datetime.now()} : Invoking interpreter")
        self._interpreter.set_tensor(self._input_details[0]['index'], image_input_data)
        self._interpreter.invoke()
        print(f"{datetime.datetime.now()} : Interpreter Invoked")

    def retrieve_detection_results(self):
        # Retrieve detection results
        box_list = self._interpreter.get_tensor(self._output_details[0]['index'])[0]
        class_list = self._interpreter.get_tensor(self._output_details[1]['index'])[0]
        score_list = self._interpreter.get_tensor(self._output_details[2]['index'])[0]
        # num = interpreter.get_tensor(interpreter_output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        print(f"{datetime.datetime.now()} : Detection results generated. Count: {len(score_list)}")
        return box_list, class_list, score_list

    def prepare_card_info(self, x_min, object_name, score):
        card_info = {}
        print(f"{datetime.datetime.now()} : Name: {object_name}, X-Min: {x_min}")
        card_info["card"] = object_name
        card_info["score"] = score
        card_info["loc"] = x_min
        return card_info

    def describe_detected_cards(self, cards):
        if len(cards) == 0:
            mixer.music.load(f"{self._path_to_sound_files}/please_scan_again.mp3")
            mixer.music.play()
            print(f"{datetime.datetime.now()} : Please Scan Again")
        else:
            i = 1
            for dc in cards:
                print(" " + str(i) + ": " + dc["card"])
                mixer.music.load(f"{self._path_to_sound_files}/{dc['card']}.mp3")
                mixer.music.play()
                i = i + 1

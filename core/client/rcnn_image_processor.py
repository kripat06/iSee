import datetime
import math
import time

import cv2
import numpy as np
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util


class RCNNImageProcessor():
    def __init__(self, path_to_labels, path_to_ckpt):
        self._min_score_thresh = 0.5
        self._label_map = label_map_util.load_labelmap(path_to_labels)
        self._categories = label_map_util.convert_label_map_to_categories(self._label_map, max_num_classes=52,
                                                                          use_display_name=True)
        self._category_index = label_map_util.create_category_index(self._categories)
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self._sess = tf.Session(graph=detection_graph)
        self._image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self._detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self._detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self._detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self._num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        self._card_map = {}
        self._time_data = {}
        for key in self._category_index.keys():
            tmp = self._category_index[key]
            self._card_map[tmp["id"]] = tmp["name"]

    def process(self, image_path): #ACTUAL RCNN DETECT CODE
        frame = cv2.imread(image_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
        (boxes, scores, classes, num) = self._sess.run(
            [self._detection_boxes, self._detection_scores, self._detection_classes, self._num_detections],
            feed_dict={self._image_tensor: frame_expanded})
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #     frame,
        #     np.squeeze(boxes),
        #     np.squeeze(classes).astype(np.int32),
        #     np.squeeze(scores),
        #     self._category_index,
        #     use_normalized_coordinates=True,
        #     line_thickness=3,
        #     min_score_thresh=self._min_score_thresh)
        # tmp_path = f"D:/projects/tflite/tmp/{round(time.time() * 1000)}.jpg"
        # print(f"Tmp img saved @ {tmp_path}")
        # cv2.imwrite(tmp_path, frame)
        detected_cards = []

        card_added = set()
        xmin_added = set()

        for i in range(len(classes[0])):
            card_index = classes[0][i]
            score = scores[0][i]
            xmin = boxes[0][i][1]
            # print(score)
            if score >= self._min_score_thresh:
                card_name = self._card_map[int(card_index)]
                xmin_val = float(xmin)
                card_added.add(card_name)
                xmin_added.add(xmin_val)
                dc = {"card": card_name, "score": float(score), "loc": xmin_val}
                detected_cards.append(dc)
        # self.describe_cards(detected_cards, "orig")
        detected_cards = self.filter_duplicate_cards(detected_cards)
        # self.describe_cards(detected_cards, "after card filter")
        detected_cards = self.filter_same_loc_cards_orig(detected_cards)
        # self.describe_cards(detected_cards, "after loc filter")
        detected_cards = sorted(detected_cards, key=lambda k: k["loc"])
        # self.describe_cards(detected_cards, "after loc sort")
        #cv2.imshow('Object detector', frame)
        # Press any key to close the image
        #cv2.waitKey(0)

        # print(f"Lapsed time array: {self._time_data}")
        return detected_cards

    def filter_duplicate_cards(self, cards):
        cards = sorted(cards, key=lambda k: k["score"], reverse=True)
        # cards = support.multikeysort(cards, ["card", "score"])
        filtered_cards = []
        card_map = set()
        for card in cards:
            if card["card"] not in card_map:
                card_map.add(card["card"])
                filtered_cards.append(card)
        return filtered_cards

    def filter_same_loc_cards(self, cards):
        cards = sorted(cards, key=lambda k: k["loc"], reverse=True)
        prev_loc = 0.0
        filtered_cards = []
        for card in cards:
            tmp = card["loc"]
            loc = (math.floor(tmp * 100)) / 100.0
            if prev_loc > 0.0 and loc - prev_loc > 0.05:
                filtered_cards.append(card)
        filtered_cards = sorted(filtered_cards, key=lambda k: k["score"], reverse=True)
        return filtered_cards

    def filter_same_loc_cards_orig(self, cards):
        cards = sorted(cards, key=lambda k: k["score"], reverse=True)
        filtered_cards = []
        loc_map = set()
        for card in cards:
            tmp = card["loc"]
            loc = (math.floor(tmp * 100)) / 100.0
            if loc not in loc_map:
                loc_map.add(loc)
                filtered_cards.append(card)
        return filtered_cards

    def describe_cards(self, cards, label):
        print(label + ": ")
        for card in cards:
            print(f" {card}")

    def match_card_to_img(self, img_dir, images, sort, index):
        match_cnt = 0
        total_cnt = 0
        total = 1
        td_list = []
        for image in images:
            start_time = time.time()
            cards = self.process(f"{img_dir}/{image}")
            end_time = time.time()
            time_lapsed = end_time - start_time
            td_list.append(time_lapsed)
            val = image.split(".")[0]
            tokens = val.split("_")
            sorted(tokens)
            val = "_".join(tokens)

            if total % 25 == 0:
                print(f"{index}: {total}    ")

            if "_" in val:
                total_cnt = total_cnt + 1
                val2 = self.cards_to_string(cards, sort)
                if val == val2:
                    #print(f"    {val} matches")
                    match_cnt = match_cnt + 1
                # else:
                #     print(f"    {val} doesn't match {val2}")
            total = total + 1
        accuracy = (match_cnt * 100) / total_cnt

        result = {
            "total": total_cnt,
            "match_count": match_cnt,
            "accuracy": accuracy,
            "time_data": td_list
        }

        return result

    def sort_by_split(self, str):
        val = str.split(".")[0]
        tokens = val.split("_")
        sorted(tokens)
        val = "_".join(tokens)
        return val

    def cards_to_string(self, cards, sort):
        tmp = []
        if sort:
            sorted(cards, key=lambda k: k["loc"])
        for card in cards:
            tmp.append(card["card"])
        val2 = "_".join(tmp)
        return val2

    def find_accuracy(self, img_dir, images, sort, index):
        result = self.match_card_to_img(img_dir, images, sort, index)
        print(f"{index} complete")
        return result

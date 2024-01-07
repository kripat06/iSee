import os
import threading

from flask import Flask, request, jsonify

import support
from rcnn_image_processor import RCNNImageProcessor

app = Flask(__name__)

model_type = "rcnn"  ## ssd, rcnn

path_to_ckpt = f"D:/projects/tflite/models/{model_type}/frozen_inference_graph.pb"
path_to_labels = f"D:/projects/tflite/models/{model_type}/labelmap.pbtxt"
image_processor = RCNNImageProcessor(path_to_labels, path_to_ckpt)
input_img_path = f"D:/projects/tflite/tmp/input.jpg"


@app.route('/')
def hello_world():
    """Print 'Hello, world!' as the response body."""
    return 'Hello, world!'


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    file.save(input_img_path)
    detected_cards = image_processor.process(input_img_path)
    for card in detected_cards:
        print(card)

    print(f"Total cards (in hand) detected: {len(detected_cards)}")
    response = jsonify(detected_cards)
    return response


@app.route('/detect_prev')
def detect_prev():
    detected_cards = image_processor.process(input_img_path)
    for card in detected_cards:
        print(card)

    print(f"Total cards (in hand) detected: {len(detected_cards)}")
    response = jsonify(detected_cards)
    return response



@app.route('/scan_table', methods=['POST'])
def scan_table():
    file = request.files['file']
    file.save(input_img_path)
    detected_cards = image_processor.process(input_img_path)
    for card in detected_cards:
        print(card)

    print(f"Total cards (on table) detected: {len(detected_cards)}")
    response = jsonify(detected_cards)
    return response


@app.route('/find_accuracy')
def find_accuracy():
    img_dir = "D:\projects/tflite/test_data/detection_9"
    images = os.listdir(img_dir)
    chunks = support.chunkIt(images, 50)
    threads = []
    results = {}
    index = 0
    for chunk in chunks:
        x = threading.Thread(target=call_accuracy, args=(img_dir, chunk, False, results, index))
        threads.append(x)
        x.start()
        index = index + 1

    for thread in threads:
        thread.join()

    total_count = 0
    match_count = 0
    time_data = []
    for i in range(len(chunks)):
        result = results[i]
        total_count = total_count + result["total"]
        match_count = match_count + result["match_count"]
        time_data = time_data + result["time_data"]

    accuracy = (match_count * 100) / total_count
    time_avg = support.average(time_data)

    return jsonify({
        "total": total_count,
        "match_count": match_count,
        "accuracy": accuracy,
        "time_taken": sum(time_data),
        "average_time": time_avg
    })


def call_accuracy(img_dir, images, sort, results, index):
    result = image_processor.find_accuracy(img_dir, images, sort, index)

    results[index] = result

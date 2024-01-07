import importlib.util
from functools import cmp_to_key
from operator import itemgetter as i

import cv2


def prepare_tf(path_to_checkpoint):
    # Import TensorFlow libraries
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    intp = Interpreter(model_path=path_to_checkpoint)
    intp.allocate_tensors()
    return intp


def load_labels(path):
    # Load the label map
    with open(path, 'r') as f:
        label_list = [line.strip() for line in f.readlines()]
    return label_list


def decorate_image(boxes, classes, scores, imH, imW, image, object_name):
    ymin = int(max(1, (boxes[0] * imH)))
    xmin = int(max(1, (boxes[1] * imW)))
    ymax = int(min(imH, (boxes[2] * imH)))
    xmax = int(min(imW, (boxes[3] * imW)))

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

    label = '%s: %d%%' % (object_name, int(scores * 100))  # Example: 'person: 72%'
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
    label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
    cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10),
                  (255, 255, 255), cv2.FILLED)  # Draw white box to put label text in
    cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text


def show_image(label, image):
    cv2.imshow(label, image)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


def cmp(x, y):
    return (x > y) - (x < y)


def multikeysort(items, columns):
    comparers = [
        ((i(col[1:].strip()), -1) if col.startswith('-') else (i(col.strip()), 1))
        for col in columns
    ]

    def comparer(left, right):
        comparer_iter = (
            cmp(fn(left), fn(right)) * mult
            for fn, mult in comparers
        )
        return next((result for result in comparer_iter if result), 0)

    return sorted(items, key=cmp_to_key(comparer))


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def average(array):
    return sum(array) / len(array)

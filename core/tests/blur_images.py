import os
import threading

import cv2
import numpy as np

img_dir = "C:/tensorflow1/models/research/object_detection/images/extra_files"
def blur_images(images):
    for image in images:
        #while idx < 1:
        if str(image).endswith(".jpg"):
            img_path = "C:/tensorflow1/models/research/object_detection/images/extra_files"
            try:
                src = cv2.imread(img_path)
                #src = cv2.imread(f"{image}")
                blur = cv2.blur(src, (5, 5))
                #blur2 = cv2.bilateralFilter(blur, 9, 10, 10)
                cv2.imwrite(image, blur)
               #idx = idx + 1
            except Exception as e:
                os.remove(img_path)
                print(f"Error Blurring: {image}")

        else:
            continue

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
def blur_chunk_image(img_dir):
    os.chdir(img_dir)
    images = os.listdir(img_dir)
    chunks = chunkIt(images, 70)
    threads = []
    index = 0
    for chunk in chunks:
        x = threading.Thread(target=blur_images, args=(chunk,))
        threads.append(x)
        x.start()
        index = index + 1

    for thread in threads:
        thread.join()


blur_chunk_image(img_dir)
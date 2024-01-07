import os
import shutil
import threading

#import cv2
#img_dir = "C:/tensorflow1/models/research/object_detection/images/train"
img_dir = "D:/DeepFaceLab/biden_audio/"
dest_dir = "D:\DeepFaceLab\DeepFaceLab_NVIDIA_RTX2080Ti_and_earlier\workspace\data_dst/blur"
cnt = 4
def delete_bad_files(img_dir, cnt):
    cards = os.listdir(img_dir)
    for card in cards:
        if len(card.split("_")) == cnt:
            os.remove(f"{img_dir}/{card}")

def delete_xml_files(img_dir):
    cards = os.listdir(img_dir)
    for card in cards:
        if str(card).endswith(".xml"):
            os.remove(f"{img_dir}/{card}")

def detect_total_cards_in_dir(img_dir):
    cards = os.listdir(img_dir)
    num_of_3 = 0
    num_of_4 = 0
    num_of_5 = 0
    for card in cards:
        if len(card.split("_")) == 3:
             num_of_3 = num_of_3 + 1
        elif len(card.split("_")) == 4:
            num_of_4 = num_of_4 + 1
        elif len(card.split("_")) == 5:
            num_of_5 = num_of_5 + 1
    print(f"3-card: {num_of_3/2} "
          f"4-card: {num_of_4/2} "
          f"5card: {num_of_5/2}"
          f"Total: {(num_of_3 + num_of_4 + num_of_5)/2}")

def add_file_extention(img_dir):
    files = os.listdir(img_dir)
    num_of_files = 0
    for file in files:
        conv = file.split(".")[0]
        conv2 = conv.split("/")[3]
        if conv2.split("_")[1] == "1" or "2" or "3" or "4" or "5" or "6" or "7" or "8" or "9":
            os.rename(f"D:/DeepFaceLab/biden_audio/{file}", f"D:/DeepFaceLab/biden_audio/biden_00{conv2.split('_')[1]}.wav")
        elif len(conv2.split("_")[1]) == 2:
            os.rename(f"D:/DeepFaceLab/biden_audio/{file}", f"D:/DeepFaceLab/biden_audio/biden_0{conv2.split('_')[1]}.wav")
        num_of_files = num_of_files + 1
    print(f"files renamed: {num_of_files}")

def move_files(img_dir, dest_dir, cnt, file_number):
    cards = os.listdir(img_dir)
    num_moved = 0
    for card in cards:
        if num_moved <= file_number:
            if len(card.split("_")) == cnt:
                try:
                    shutil.move(f"{img_dir}/{card}", dest_dir)
                    xml = card.split(".")
                    shutil.move(f"{img_dir}/{xml[0]}.xml", dest_dir)
                    num_moved = num_moved + 1
                except Exception as e:
                    print(str(e))
def detect_for_blur(images, img_dir, dest_dir):
    for image in images:
        if str(image).endswith(".jpg"):
            try:
                src = cv2.imread(f"{img_dir}/{image}")
                gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                fm = variance_of_laplacian(gray)
                if fm < 100:
                    shutil.move(f"{img_dir}/{image}", dest_dir)
            except Exception as e:
                print(f"Error reading: {image}")
                os.remove(f"{img_dir}/{image}")


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def detect_chunk_blur(img_dir, dest_dir):
    os.chdir(img_dir)
    images = os.listdir(img_dir)
    chunks = chunkIt(images, 70)
    threads = []
    index = 0
    for chunk in chunks:
        x = threading.Thread(target=detect_for_blur, args=(chunk,img_dir, dest_dir))
        threads.append(x)
        x.start()
        index = index + 1

    for thread in threads:
        thread.join()



def run_in_parallel(func, chunks):
    threads = []
    index = 0
    for chunk in chunks:
        x = threading.Thread(target=func, args=(chunk,))
        threads.append(x)
        x.start()
        index = index + 1

    for thread in threads:
        thread.join()

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

add_file_extention(img_dir)
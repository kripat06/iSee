import datetime
import json
import os

import requests

import hw
from path_support import *


def detect_cards(input_image_path, api_ip):
    detected_cards = []
    try:
        response = requests.post(f'http://{api_ip}:5000/detect', files={"file": open(input_image_path, "rb")}, timeout=8)
        detected_cards = json.loads(response.text)
        # print(f"{datetime.datetime.now()} : Detected Cards:")
        # for card in detected_cards:
        #     print(f" {card}")
    except Exception as e:
        print(str(e))
    if len(detected_cards) == 0:
        hw.play_audio(f"{PATH_TO_SOUNDFILES}/please_scan_again.mp3")
        print(f"{datetime.datetime.now()} : Please Scan Again")
    else:
        i = 1
        print(f"Total cards detected: {len(detected_cards)}")
        hw.play_audio(f"{PATH_TO_SOUNDFILES}/you_have.mp3")
        for dc in detected_cards:
            print(f' {str(i)}: {dc["card"]} , {dc["score"]}, {dc["loc"]}')
            hw.play_audio(f"{PATH_TO_SOUNDFILES}/{dc['card']}.mp3")
            i = i + 1
    os.remove(input_image_path)
    return detected_cards


def detect_cards_on_table(input_image_path, api_ip):
    cards_on_table = []
    try:
        response = requests.post(f'http://{api_ip}:5000/scan_table', files={"file": open(input_image_path, "rb")})
        cards_on_table = json.loads(response.text)
        print(f"{datetime.datetime.now()} : Detected Cards: {cards_on_table}")
    except Exception as e:
        print(str(e))
    if len(cards_on_table) == 0:
        hw.play_audio(f"{PATH_TO_SOUNDFILES}/please_scan_again.mp3")
        print(f"{datetime.datetime.now()} : Please Scan Again")
    else:
        i = 1
        print(f"Total cards found: {len(cards_on_table)}")
        hw.play_audio(f"{PATH_TO_SOUNDFILES}/cards_in_play_are.mp3")
        for dc in cards_on_table:
            print(" " + str(i) + ": " + dc["card"])
            hw.play_audio(f"{PATH_TO_SOUNDFILES}/{dc['card']}.mp3")
            i = i + 1
    os.remove(input_image_path)


def chk_for_suit(suit, detected_cards):
    found_card = []
    idx = 1
    for card in detected_cards:
        if card["card"].endswith(suit[0]):
            found_card.append((card["card"], idx))
        idx = idx + 1
    if len(found_card) > 0:
        hw.play_audio(f"{PATH_TO_SOUNDFILES}/found.mp3")
        for card in found_card:
            hw.play_audio(f"{PATH_TO_SOUNDFILES}/{card[0]}.mp3")
            hw.play_audio(f"{PATH_TO_SOUNDFILES}/at_position.mp3")
            hw.play_audio(f"{PATH_TO_SOUNDFILES}/{card[1]}.mp3")
    else:
        print("No Card(s) found.")
        hw.play_audio(f"{PATH_TO_SOUNDFILES}/no_cards_found.mp3")


def find_a_card(card_val, detected_cards):
    found = False
    found_card = None
    for card in detected_cards:
        # hw.play_audio(f"{PATH_TO_SOUNDFILES}/you_have.mp3")
        if card["card"] == card_val:
            found = True
            found_card = card["card"]
            break
            # hw.play_audio(f"{PATH_TO_SOUNDFILES}/you_have.mp3")
    return found, found_card


def which_suit():
    suit = hw.suit_button_press()
    return suit

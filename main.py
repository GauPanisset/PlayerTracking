import playerPath
from number import numberDetection
import cv2
import os
import pyautogui
from ray import Reader

import numpy as np
import time
import math
from matplotlib import pyplot as plt


def read_replays(rep, player_name):
    _list = os.listdir(rep)
    replays = []
    for element in _list:
        if element[-7:] == ".replay":
            replays.append(element)

    for replay_name in replays:
        with Reader(rep + "/" + replay_name) as replay:
            for elim in replay.eliminations:
                print(elim)
            replay.export_to_json("res/kill"+player_name+replay_name[-27:]+".json", player_name)


def mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err


def record_replays(rep, player_name):
    _list = os.listdir(rep)
    replays = []
    for element in _list:
        if element[-7:] == ".replay":
            replays.append(element)

    my_map = cv2.imread("images/TestMap.png", 0)
    first_mini = cv2.imread("images/FirstMini.png", 0)
    redif_img = cv2.imread("images/Redif.png", 0)

    for i in range(len(replays)):

        # Clic on replay, read and confirm
        pyautogui.moveTo(1800, 220 + i * 40)
        pyautogui.click()
        pyautogui.moveTo(975, 1040)
        pyautogui.click()
        pyautogui.moveTo(1200, 790)
        pyautogui.click()
        # Wait for the end of loading
        mini = playerPath.screen_record((1625, 15, 1625 + 279, 15 + 279))
        while mse(first_mini, mini) > 1000:
            mini = playerPath.screen_record((1625, 15, 1625 + 279, 15 + 279))
        # x4
        pyautogui.moveTo(1125, 960)
        pyautogui.click()
        pyautogui.click()
        (my_path, my_path_time) = playerPath.get_path_auto(my_map, 3)
        playerPath.path_to_json(my_path, my_path_time, "res/path"+player_name+replays[i][-27:]+".json")

        # Exit replay
        pyautogui.press('esc')
        pyautogui.moveTo(1640, 285)
        pyautogui.click()
        pyautogui.moveTo(1155, 790)
        pyautogui.click()
        # Wait for the end of loading
        tmp_img = playerPath.screen_record((32, 112, 32 + 170, 112 + 48))
        while mse(redif_img, tmp_img) > 1000:
            tmp_img = playerPath.screen_record((32, 112, 32 + 170, 112 + 48))


# read_replays("C:/Users/Gauthier/AppData/Local/FortniteGame/Saved/Demos", "Aerocus")



start = time.time()
last = time.time()

while time.time() - start < 3:
    pass

record_replays("C:/Users/Gauthier/AppData/Local/FortniteGame/Saved/Demos", 'Aerocus')


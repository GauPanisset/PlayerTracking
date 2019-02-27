import os
import cv2
import numpy as np
from PIL import ImageGrab
from matplotlib import pyplot as plt
import time
import math
from number import numberDetection

MAX_SPEED = {"zipline": 14, "plane": 20, "quad": 12, "feet": 6, "glider": 19}
MAX = 25 * 4  # pixel/second with x4 recording
SIZE = 279


def distance(start, end):
    if not start or not end:
        return 0
    return np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)


def template_matching(src, template):
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    w, h = template.shape[::-1]

    method = eval(methods[0])
    res = cv2.matchTemplate(src, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    return top_left[0] + w // 2, top_left[1] + h // 2


def image_sort(list_img):
    res = [0 for i in range(len(list_img))]
    for img in list_img:
        res[int(img[5:-4])] = img
    i = 0
    while i < len(res):
        if res[i] == 0:
            del res[i]
        else:
            i += 1
    return res


def get_path_from_dir(map_img, rep):
    """
    Return path from mini-maps stored in rep.
    :param map_img: image of the map
    :param rep: directory in which mini maps are stored
    :return: list of nodes that are the position of the mini map in the fortnite map
    """

    path = []
    count = 0

    list_mini = os.listdir(rep)
    list_mini = image_sort(list_mini)
    for mini in list_mini:
        print(mini)
        template = cv2.imread(rep + '/' + mini, 0)
        if len(path) > 0:
            if count < 5:
                node = template_matching(map_img, template)

                # plt.subplot(121), plt.imshow(map_img,  cmap='gray')
                # plt.title('Map'), plt.xticks([]), plt.yticks([])

                if distance(path[-1], node) < MAX:
                    count += 1
                    path.append(node)
                else:
                    count = 0
                    path.append(node)
            else:

                x1 = path[-1][0] - int(MAX + SIZE // 1.9) if path[-1][0] - (MAX + SIZE//1.9) > 0 else 0
                x2 = path[-1][0] + int(MAX + SIZE // 1.9) if path[-1][0] + (MAX + SIZE // 1.9) > 0 else 0
                y1 = path[-1][1] - int(MAX + SIZE // 1.9) if path[-1][1] - (MAX + SIZE // 1.9) > 0 else 0
                y2 = path[-1][1] + int(MAX + SIZE // 1.9) if path[-1][1] + (MAX + SIZE // 1.9) > 0 else 0

                _node = template_matching(map_img[y1:y2, x1:x2], template)
                node = (_node[0] + x1, _node[1] + y1)

                # plt.subplot(121), plt.imshow(map_img[y1:y2, x1:x2],  cmap='gray')
                # plt.title('Map'), plt.xticks([]), plt.yticks([])

                path.append(node)

            # plt.subplot(122), plt.imshow(template, cmap='gray')
            # plt.title('Template : '+mini), plt.xticks([]), plt.yticks([])
            # plt.show()
            print(distance(path[-1], path[-2]), path[-1])
        else:
            path.append(template_matching(map_img, template))
            print(path[-1])

    return path


def check_node(node, prev, freq):
    """
    Check if a node seems to have the good position
    :param node: current node
    :param prev: previous node
    :param freq: image/second
    :return: True if the distance between the nodes seems correct. Else fasle
    """
    return distance(prev, node) < MAX / freq


def get_path(map_img, timer, freq):
    """
    Return path.
    :param map_img: image of the map
    :param timer: recording time in seconds
    :param freq: image/second
    :return: list of nodes that are the position of the mini map in the fortnite map
    """
    path = []
    path_time = []
    count = 0
    last_node = None

    while True:
        cv2.imshow('img', np.array([[1, 2], [3, 4]]))
        if cv2.waitKey(33) == 32:
            cv2.destroyAllWindows()
            start_time = time.time()
            last_time = time.time()
            while time.time() - start_time < timer:
                if time.time() - last_time > 1 / freq:

                    template = screen_record((1625, 15, 1625 + SIZE, 15 + SIZE))

                    if count < 5:
                        node = template_matching(map_img, template)

                        # plt.subplot(121), plt.imshow(map_img,  cmap='gray')
                        # plt.title('Map'), plt.xticks([]), plt.yticks([])

                        if distance(last_node, node) < MAX:
                            count += 1
                            last_node = node
                            path.append(node)
                            path_time.append((last_time - start_time) * 4)
                        else:
                            count = 0
                            last_node = node
                            path = []
                    else:

                        x1 = path[-1][0] - int(MAX/freq + SIZE // 1.9) if path[-1][0] - (MAX/freq + SIZE // 1.9) > 0 else 0
                        x2 = path[-1][0] + int(MAX/freq + SIZE // 1.9) if path[-1][0] + (MAX/freq + SIZE // 1.9) > 0 else 0
                        y1 = path[-1][1] - int(MAX/freq + SIZE // 1.9) if path[-1][1] - (MAX/freq + SIZE // 1.9) > 0 else 0
                        y2 = path[-1][1] + int(MAX/freq + SIZE // 1.9) if path[-1][1] + (MAX/freq + SIZE // 1.9) > 0 else 0

                        _node = template_matching(map_img[y1:y2, x1:x2], template)
                        node = (_node[0] + x1, _node[1] + y1)

                        # plt.subplot(121), plt.imshow(map_img[y1:y2, x1:x2],  cmap='gray')
                        # plt.title('Map'), plt.xticks([]), plt.yticks([])

                        path.append(node)
                        path_time.append((last_time - start_time) * 4)
                    # plt.subplot(122), plt.imshow(template, cmap='gray')
                    # plt.title('Template : '+mini), plt.xticks([]), plt.yticks([])
                    # plt.show()

                    last_time = time.time()
            break
    return path, path_time


def clock_start(clock_img, clock_type):
    """
    Return True if the clock image is the clock type
    :param clock_img: image of the clock icon
    :param clock_type: clock type "zone" or "bus"
    :return: True if the type of the clock icon is clock_type
    """
    w, h = clock_img.shape[::-1]
    bool_lines = {"zone":
                      {5: [False, True, False],
                       15: [False, True, False, True, False, True, False],
                       25: [False, True, False, True, False]},
                  "bus":
                      {5: [False],
                       15: [False, True, False],
                       25: [False]},
                  "jump":
                      {7: [False, True, False],
                       15: [False, True, False, True, False],
                       26: [False, True, False]}}
    for i in bool_lines[clock_type].keys():
        bool_array = [clock_img[i][0] > 240]
        for j in range(1, w):
            if bool_array[-1] != (clock_img[i][j] > 240):
                bool_array.append(clock_img[i][j] > 240)
        if not (bool_array == bool_lines[clock_type][i] or bool_array[1:-1] == bool_lines[clock_type][i]):
            return False
    return True


def get_death_time(replay_length):
    death_marker = cv2.imread("images/DeathMarker.png", 0)
    time_line = screen_record((120, 855, 120 + 1680, 855 + 40))
    _, death_marker = cv2.threshold(death_marker, 220, 255, cv2.THRESH_BINARY_INV)
    _, time_line = cv2.threshold(time_line, 220, 255, cv2.THRESH_BINARY_INV)
    x, y = template_matching(time_line, death_marker)
    return x*replay_length/1680


def get_path_auto(map_img, freq):
    """
    Return path.
    :param map_img: image of the map
    :param freq: image/second
    :return: list of nodes that are the position of the mini map in the fortnite map
    """
    path = []
    path_time = []
    replay_length = []
    count = 0
    death_time = 400
    record = False
    offset = None
    model = numberDetection.setup_model("number")
    tmp_length = numberDetection.detect_number(model, screen_record((1825, 890, 1895, 916)))
    tmp_length = math.ceil(numberDetection.array_to_second(tmp_length, 4)/4) if numberDetection.array_to_second(tmp_length, 4) > 0 else 400
    replay_length.append(tmp_length)
    last_node = None

    start_time = time.time()
    last_time = time.time()
    while time.time() - start_time < (max(replay_length, key=replay_length.count) if death_time >= 400 else math.ceil(death_time)):
        if not offset and clock_start(screen_record((1637, 304, 1674, 340)), "zone"):
            offset = numberDetection.detect_number(model, screen_record((25, 890, 95, 916)))
            offset = numberDetection.array_to_second(offset)
            if not record:
                record = True
        if not record and clock_start(screen_record((1637, 304, 1674, 340)), "jump"):
            death_time = get_death_time(max(replay_length, key=replay_length.count))
            record = True
        if len(replay_length) < 20 or max(replay_length, key=replay_length.count) > 400:
            tmp_length = numberDetection.detect_number(model, screen_record((1825, 890, 1895, 916)))
            tmp_length = math.ceil(numberDetection.array_to_second(tmp_length, 4) / 4) if numberDetection.array_to_second(tmp_length,4) > 0 else 400
            replay_length.append(tmp_length)

        if record and time.time() - last_time > 1 / freq:

            template = screen_record((1625, 15, 1625 + SIZE, 15 + SIZE))

            if count < 5:
                node = template_matching(map_img, template)

                if distance(last_node, node) < MAX:
                    count += 1
                    last_node = node
                    path.append(node)
                    path_time.append((last_time - start_time) * 4)
                else:
                    count = 0
                    last_node = node
                    path = []
            else:

                x1 = path[-1][0] - int(MAX/freq + SIZE // 1.9) if path[-1][0] - (MAX/freq + SIZE // 1.9) > 0 else 0
                x2 = path[-1][0] + int(MAX/freq + SIZE // 1.9) if path[-1][0] + (MAX/freq + SIZE // 1.9) > 0 else 0
                y1 = path[-1][1] - int(MAX/freq + SIZE // 1.9) if path[-1][1] - (MAX/freq + SIZE // 1.9) > 0 else 0
                y2 = path[-1][1] + int(MAX/freq + SIZE // 1.9) if path[-1][1] + (MAX/freq + SIZE // 1.9) > 0 else 0

                _node = template_matching(map_img[y1:y2, x1:x2], template)
                node = (_node[0] + x1, _node[1] + y1)

                path.append(node)
                path_time.append((last_time - start_time) * 4)

            last_time = time.time()
    return path, path_time


def display_path(map_img, path):
    """
    Display a path on map_img
    :param map_img: image of the map
    :param path: list of path nodes
    :return: None
    """
    for node in path:
        cv2.circle(map_img, node, 20, (255, 255, 255), -1)
    plt.imshow(map_img, cmap='gray')
    plt.show()


def screen_record(rect, disp=False, rec=(False, "", "")):
    """
    Get the mini-map from the screen
    :param rect: rectangle (x1, y1, x2, y2) in which it records
    :param disp: display the image (default: False)
    :param rec: save the image in directory rec[1] with the name rec[2](default: False)
    :return: grayscale image in numpy array
    """
    printscreen = np.array(ImageGrab.grab(bbox=(rect[0], rect[1], rect[2], rect[3])))
    printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
    if disp:
        plt.imshow(printscreen, cmap='gray')
        plt.show()
    if rec[0]:
        cv2.imwrite(rec[1] + rec[2], printscreen)
    return printscreen


def record_mini(dest, timer, freq):
    """
    Save mini maps in the directory dest. It saves one image every second
    :param dest: destination directory
    :param timer: recording time in seconds
    :param freq: image/second
    :return: None
    """
    count = 0
    start_time = time.time()
    last_time = time.time()
    while time.time() - start_time < timer:
        if time.time() - last_time > 1/freq:
            screen_record((1625, 15, 1625 + SIZE, 15 + SIZE), False, (True, dest, "/mini-" + str(count) + ".png"))
            last_time = time.time()
            count += 1


def path_to_json(path, path_time, dest):
    """
    Convert and save the path in a json file
    :param path: list of nodes
    :param path_time: list of time nodes
    :param dest: path to the destination file
    :return: None
    """
    alpha_x = 1.075
    alpha_y = 1.069
    file = open(dest, "w")
    json_path = "["
    for index in range(len(path)):
        speed = distance(path[index], path[index - 1])/(path_time[index] - path_time[index - 1]) if index != 0 else 0
        speed = 20 if speed > 20 else speed
        json_path += "{\"x\":" + str(int((path[index][0])/alpha_x) + 35) + ", \"y\":" + str(int((path[index][1])/alpha_y) + 85) + ", \"time\":" + str(int(path_time[index])) + ", \"speed\":" + str(int(speed)) + "},"
    json_path = json_path[:-1] + "]"

    file.write(json_path)
    file.close()




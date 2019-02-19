import os
import cv2
import numpy as np
from PIL import ImageGrab
from matplotlib import pyplot as plt
import time


MAX_SPEED = {"zipline": 14, "plane": 20, "quad": 12, "feet": 6, "glider": 19}
MAX = 25 * 4  # pixel/second with x4 recording
SIZE = 279


def distance(start, end):
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

    while True:
        cv2.imshow('img', np.array([[1, 2], [3, 4]]))
        if cv2.waitKey(33) == 32:
            cv2.destroyAllWindows()
            start_time = time.time()
            last_time = time.time()
            while time.time() - start_time < timer:
                if time.time() - last_time > 1 / freq:

                    path_time.append(last_time - start_time)

                    template = screen_record()

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

                            x1 = path[-1][0] - int(MAX/freq + SIZE // 1.9) if path[-1][0] - (MAX/freq + SIZE // 1.9) > 0 else 0
                            x2 = path[-1][0] + int(MAX/freq + SIZE // 1.9) if path[-1][0] + (MAX/freq + SIZE // 1.9) > 0 else 0
                            y1 = path[-1][1] - int(MAX/freq + SIZE // 1.9) if path[-1][1] - (MAX/freq + SIZE // 1.9) > 0 else 0
                            y2 = path[-1][1] + int(MAX/freq + SIZE // 1.9) if path[-1][1] + (MAX/freq + SIZE // 1.9) > 0 else 0

                            _node = template_matching(map_img[y1:y2, x1:x2], template)
                            node = (_node[0] + x1, _node[1] + y1)

                            # plt.subplot(121), plt.imshow(map_img[y1:y2, x1:x2],  cmap='gray')
                            # plt.title('Map'), plt.xticks([]), plt.yticks([])

                            path.append(node)

                        # plt.subplot(122), plt.imshow(template, cmap='gray')
                        # plt.title('Template : '+mini), plt.xticks([]), plt.yticks([])
                        # plt.show()
                    else:
                        path.append(template_matching(map_img, template))

                    last_time = time.time()
            break
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
    plt.imshow(my_map, cmap='gray')
    plt.show()


def screen_record(disp=False, rec=(False, "", "")):
    """
    Get the mini-map from the screen
    :param disp: display the image (default: False)
    :param rec: save the image in directory rec[1] with the name rec[2](default: False)
    :return: grayscale image in numpy array
    """
    x1 = 1625
    y1 = 15
    size = SIZE
    printscreen = np.array(ImageGrab.grab(bbox=(x1, y1, x1 + size, y1 + size)))
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
            screen_record(False, (True, dest, "/mini-" + str(count) + ".png"))
            last_time = time.time()
            count += 1


def path_to_json(path, path_time):
    """
    Convert and save the path in a json file
    :param path: list of nodes
    :param path_time: list of time nodes
    :return: None
    """
    alpha_x = 1.075
    alpha_y = 1.069
    file = open("path.json", "w")
    json_path = "["
    for index in range(len(path)):
        json_path += "{\"x\":" + str(int((path[index][0])/alpha_x) + 35) + ", \"y\":" + str(int((path[index][1])/alpha_y) + 85) + ", \"time\":" + str(int(path_time[index])) + "},"
    json_path = json_path[:-1] + "]"

    file.write(json_path)
    file.close()


my_map = cv2.imread("TestMap.png", 0)

# record_mini("game2", 315, 1)

(my_path, my_path_time) = get_path(my_map, 330, 3)
path_to_json(my_path, my_path_time)
print(len(my_path))
print(len(my_path_time))
display_path(my_map, my_path)

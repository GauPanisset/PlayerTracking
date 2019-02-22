import playerPath
import cv2
from ray import Reader

with Reader("UnsavedReplay-2019.02.16-16.57.37.replay") as replay:
    for elim in replay.eliminations:
        print(elim)
    replay.export_to_json("killCompani.json", "Compani")

my_map = cv2.imread("TestMap.png", 0)

(my_path, my_path_time) = playerPath.get_path(my_map, 235, 3)
playerPath.path_to_json(my_path, my_path_time, "pathCompani.json")
print(len(my_path))
print(len(my_path_time))
playerPath.display_path(my_map, my_path)


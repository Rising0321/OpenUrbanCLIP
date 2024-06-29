import os
import numpy as np
import time

place = "Guangzhou"
walk_dir = f"../task_data/{place}"
out_dir = f"./{place}"

coords = []
for root, dirs, files in os.walk(walk_dir):
    for file in files:
        if file.endswith(".npy"):
            temp = np.load(os.path.join(root, file))
            for item in temp:
                coords.append(["%.4lf" % item[0], "%.4lf" % item[3], "%.4lf" % item[2], "%.4lf" % item[1]])

# excecute from os
for i in range(len(coords)):
    coord = coords[i]
    '''
    print(
        f"getmap.exe -p1 {coord[0]},{coord[1]} -p2 {coord[2]},{coord[3]} -s amap -z 16 -f {out_dir}/{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}.png")
    '''
    os.system(
        f"getmap.exe -p1 {coord[0]},{coord[1]} -p2 {coord[2]},{coord[3]} -s amap -z 16 -f {out_dir}/{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}.png")
    time.sleep(0.2)
    print(i)

place = "Shanghai"
walk_dir = f"../task_data/{place}"
out_dir = f"./{place}"

coords = []
for root, dirs, files in os.walk(walk_dir):
    for file in files:
        if file.endswith(".npy"):
            temp = np.load(os.path.join(root, file))
            for item in temp:
                coords.append(["%.4lf" % item[0], "%.4lf" % item[3], "%.4lf" % item[2], "%.4lf" % item[1]])

# excecute from os
for i in range(len(coords)):
    coord = coords[i]
    '''s
    print(
        f"getmap.exe -p1 {coord[0]},{coord[1]} -p2 {coord[2]},{coord[3]} -s amap -z 16 -f {out_dir}/{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}.png")
    '''
    os.system(
        f"getmap.exe -p1 {coord[0]},{coord[1]} -p2 {coord[2]},{coord[3]} -s amap -z 16 -f {out_dir}/{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}.png")
    time.sleep(0.2)
    print(i)

place = "Shenzhen"
walk_dir = f"../task_data/{place}"
out_dir = f"./{place}"

coords = []
for root, dirs, files in os.walk(walk_dir):
    for file in files:
        if file.endswith(".npy"):
            temp = np.load(os.path.join(root, file))
            for item in temp:
                coords.append(["%.4lf" % item[0], "%.4lf" % item[3], "%.4lf" % item[2], "%.4lf" % item[1]])

# excecute from os
for i in range(len(coords)):
    coord = coords[i]
    '''
    print(
        f"getmap.exe -p1 {coord[0]},{coord[1]} -p2 {coord[2]},{coord[3]} -s amap -z 16 -f {out_dir}/{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}.png")
    '''
    os.system(
        f"getmap.exe -p1 {coord[0]},{coord[1]} -p2 {coord[2]},{coord[3]} -s amap -z 16 -f {out_dir}/{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}.png")
    time.sleep(0.2)
    print(i)

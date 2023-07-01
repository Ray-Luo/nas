import os
import shutil

path = "/home/luoleyouluole/val_sharp/val/val_sharp/"
des_path ="/home/luoleyouluole/val/"
folders = os.listdir(path)

for folder in folders:
    loc = os.path.join(path,folder)
    images = os.listdir(loc)
    for image in images:
        location = os.path.join(loc, image)
        new_name = folder + "_" + image
        shutil.copy(location, os.path.join(des_path, new_name))

import os
import shutil
from tqdm import tqdm

path = "/home/luoleyouluole/train/train_sharp/"
des_path ="/home/luoleyouluole/train/"
folders = os.listdir(path)

for folder in tqdm(folders):
    loc = os.path.join(path,folder)
    images = os.listdir(loc)
    for image in tqdm(images):
        location = os.path.join(loc, image)
        new_name = "train_" + folder + "_" + image
        shutil.copy(location, os.path.join(des_path, new_name))

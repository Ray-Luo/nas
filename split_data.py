import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import shutil
from tqdm import tqdm
import random
seed_value = 42
random.seed(seed_value)

PERCENT = [0.6, 0.2, 0.2]

train_path = "/home/luoleyouluole/reds/train"
valid_path = "/home/luoleyouluole/reds/valid"
test_path = "/home/luoleyouluole/reds/test"

for folder in tqdm([train_path, valid_path, test_path]):
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.mkdir(folder)

folder_list = [
    "/home/luoleyouluole/val",
]
folder_list.sort()


for folder_path in tqdm(folder_list):
    file_list = os.listdir(folder_path)
    file_list.sort()
    random.shuffle(file_list)

    number_train = int(PERCENT[0] * len(file_list))
    number_valid = int(PERCENT[1] * len(file_list))
    train = file_list[0:number_train]
    test = file_list[number_train:number_train + number_valid]
    valid = file_list[number_train + number_valid:]

    for i in train:
        shutil.copy(os.path.join(folder_path, i), train_path)

    for i in valid:
        shutil.copy(os.path.join(folder_path, i), valid_path)

    for i in test:
        shutil.copy(os.path.join(folder_path, i), test_path)



"""
607 + 667 + 472 + 495 + 223

"""

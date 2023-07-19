import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
import bisect
import warnings
from PIL import Image
import numpy as np
import string


random.seed(0)

def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im

env = lmdb.open(
            "/home/luoleyouluole/TextZoom/train1",
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

txn = env.begin(write=False)

nSamples = int(txn.get(b'num-samples'))
print(nSamples)

for index in range(nSamples):
    index = index + 1
    label_key = b'label-%09d' % index
    word = str(txn.get(label_key).decode())
    print(word)

    img_HR_key = b'image_hr-%09d' % index
    img_lr_key = b'image_lr-%09d' % index
    img_HR = buf2PIL(txn, img_HR_key, 'RGB')
    img_lr = buf2PIL(txn, img_lr_key, 'RGB')
    print(type(img_HR))
    img_HR.save('./HR.png')
    img_lr.save('./LR.png')
    break

import os
import sys
from shutil import copyfile
import numpy as np

src_dir = "/data/websci2020_yansen/processed_profile_images"
dest_dir = "/usr0/home/yansenwa/tumblr/data/processed_1000"
if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

dirs = os.listdir(src_dir)
for dname in dirs:
    sub_dir = os.path.join(src_dir, dname)
    print("processing directory %s" % sub_dir)
    imgs = os.listdir(sub_dir)
    for iname in imgs:
        img_id = iname.split('.')[0]
        arr = np.load(os.path.join(sub_dir, iname))
#        arr = np.reshape(arr, (4, -1))
#        arr = np.mean(arr, axis=0)
        np.save(os.path.join(dest_dir, "%s.npy" % img_id), arr)

import cv2
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
# import glob
# import os.path as osp
import torch
import argparse
import os.path as osp
import sys
import cv2
from loguru import logger

import torch

import random
import os
import numpy as np

import cv2
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
# import torch.nn as nn 
import time
import math


path_gt = "/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/anno/UAV123/car1_s.txt" 
img_files_path = glob.glob("/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/*")
img_files_path.sort()
frame = cv2.imread(img_files_path[0], cv2.IMREAD_COLOR)


my_file = open(path_gt)

line = my_file.readline()
line = [int(l) for l in line[:-1].split(',')]
my_file.close()

box = line

q_size = 200
target_sz_area = box[2]*box[3]
search_area_factor = 4.0
search_area = search_area_factor*search_area_factor*target_sz_area
target_scale = np.sqrt(search_area)/q_size

output_sz = np.array([q_size, q_size])
sample_sz = target_scale * output_sz

im = frame

pos = np.array([box[1] + box[3]/2.0, box[0] + box[2]/2.0])

posl = pos.astype(np.int).copy()

resize_factor = np.min(sample_sz.astype(np.float) / output_sz.astype(np.float)).item()
df = int(max(int(resize_factor - 0.1), 1))

sz = sample_sz.astype(np.float) / df

os = posl % df  # offset
posl = (posl - os) // df  # new position

im2 = im[os[0].item()::df, os[1].item()::df, :]  # downsample

szl = np.maximum(np.round(sz), 2.0).astype(np.int)

tl = posl - (szl - 1) // 2
br = posl + szl // 2 + 1





# cv2.rectangle(im2, (tl[1], tl[0]),
# 				(br[1] , br[0]),
# 				(255,0,0), 2)

cv2.imshow("im2", im2)
# cv2.waitKey(0)
crop_xyxy = np.array([tl[1], tl[0], br[1], br[0]])


# warpAffine transform matrix
M_13 = crop_xyxy[0]
M_23 = crop_xyxy[1]
M_11 = (crop_xyxy[2] - M_13) / (output_sz[0] - 1)
M_22 = (crop_xyxy[3] - M_23) / (output_sz[1] - 1)
mat2x3 = np.array([
    M_11,
    0,
    M_13,
    0,
    M_22,
    M_23,
]).reshape(2, 3)


im_patch = cv2.warpAffine(im2, mat2x3, 
                          (output_sz[0], output_sz[1]),
                          flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0,0,0))


cv2.imshow("im_patch", im_patch)
cv2.waitKey(0)


print(tl)


		






from main.paths import ROOT_PATH  # isort:skip
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
# import torch.nn.functional as F

from utils import *
from tracker import *
from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.data import builder as dataloader_builder
from videoanalyst.engine import builder as engine_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.optim import builder as optim_builder
from videoanalyst.utils import Timer, complete_path_wt_root_in_cfg, ensure_dir
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.engine.monitor.monitor_impl.tensorboard_logger import TensorboardLogger
from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import complete_path_wt_root_in_cfg

cv2.setNumThreads(1)



# parsed_args_config = 'experiments/stmtrack/train/got10k/stmtrack-effnet-trn.yaml'
parsed_args_config = '/home/meysam/test-apps/STMTrack/experiments/stmtrack/test/got10k/stmtrack-effnet-got.yaml'



exp_cfg_path = osp.realpath(parsed_args_config)

root_cfg.merge_from_file(exp_cfg_path)

root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)

task, task_cfg = specify_task(root_cfg.test)

task_cfg.freeze()

log_dir = osp.join(task_cfg.exp_save, task_cfg.exp_name, "logs")

ensure_dir(log_dir)


logger.configure(
        handlers=[
            dict(sink=sys.stderr, level="INFO"),
            dict(sink=osp.join(log_dir, "train_log.txt"),
                 enqueue=True,
                 serialize=True,
                 diagnose=True,
                 backtrace=True,
                 level="INFO")
        ],
        extra={"common_to_all": "default"},
    )

# backup config
logger.info("Load experiment configuration at: %s" % exp_cfg_path)
logger.info(
    "Merged with root_cfg imported from videoanalyst.config.config.cfg")
cfg_bak_file = osp.join(log_dir, "%s_bak.yaml" % task_cfg.exp_name)

with open(cfg_bak_file, "w") as f:
    f.write(task_cfg.dump())

logger.info("Task configuration backed up at %s" % cfg_bak_file)

# if task_cfg.device == "cuda":
#     world_size = task_cfg.num_processes
#     assert torch.cuda.is_available(), "please check your devices"
#     assert torch.cuda.device_count(
#     ) >= world_size, "cuda device {} is less than {}".format(
#         torch.cuda.device_count(), world_size)
#     devs = ["cuda:{}".format(i) for i in range(world_size)]
# else:
#     devs = ["cpu"]

# build model
model = model_builder.build(task, task_cfg.model)
# model.set_device(devs[0])


# print(model)

# model = STMTrack(backbone_m, backbone_q, neck_m, neck_q, head)
# model.update_params()

# Convert BatchNorm to SyncBatchNorm 
# task_model = convert_model(task_model)

# model_file = "new-epoch-19.pkl"
# model_file = "/home/meysam/test-apps/STMTrack/epoch-19_got10k.pkl"
# model_file = "/home/meysam/test-apps/STMTrack/snapshots/stmtrack-effnet-got-train/epoch-1.pkl"
model_file = "/home/meysam/test-apps/STMTrack/snapshots/stmtrack-effnet-adam-got-train-lr/epoch-16.pkl"
# model_file = "epoch-19.pkl"



model_state_dict = torch.load(model_file,
                        map_location=torch.device("cpu"))

model = model_builder.build("track", task_cfg.model)
# build pipeline
pipeline_tracker = pipeline_builder.build("track", task_cfg.pipeline, model)

# model.load_state_dict(model_state_dict['model_state_dict'])

# pipeline_tracker = pipeline_builder.build("track", task_cfg.pipeline, model)

# pipeline_tracker = STMTrackTracker(model)
# pipeline_tracker.update_params()

dev = torch.device('cuda:0')
pipeline_tracker.set_device(dev)




g = "car1_s"
# /media/meysam/hdd/dataset/Dataset_UAV123/UAV123
path_gt = "/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/anno/UAV123/car1_s.txt" 
img_files_path = glob.glob("/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/*")
img_files_path.sort()

img_files = []
for i in img_files_path:
	frame = cv2.imread(i, cv2.IMREAD_COLOR)
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	img_files.append(frame)

my_file = open(path_gt)

line = my_file.readline()
line = [int(l) for l in line[:-1].split(',')]
my_file.close()


box = line
frame_num = len(img_files)
boxes = np.zeros((frame_num, 4))
boxes[0] = box
times = np.zeros(frame_num)
# my_file = open('output/'+g+'.txt','w+')
for f, img_file in enumerate(img_files):
	image = img_file
	start_time = time.time()
	if f == 0:
		pipeline_tracker.init(image, box)
	else:
		boxes[f, :] = pipeline_tracker.update(image)
		# print(pipeline_tracker._state['pscores'][-1])
		times[f] = time.time() - start_time

		# visualiation         
		# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		pred = boxes[f,:].astype(int)
		

		# image = cv2.resize(image,(1920,1080))
		# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		cv2.rectangle(image, (pred[0], pred[1]),
				(pred[0] + pred[2], pred[1] + pred[3]),
				(255,0,0), 2)
				
		# line = str(pred[0])+','+str(pred[1])+','+str(pred[2])+','+str(pred[3])+'\n'
		# print(line)
		# my_file.writelines(line)
		cv2.imshow(g, image)
		print("FPS: ",1/times[f])
		# cv2.waitKey(0)        
		if cv2.waitKey(1)  == 27:
			break

# my_file.close()
cv2.destroyAllWindows()





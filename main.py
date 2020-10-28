import numpy as np
import os
from pathlib import Path
import time
import warnings

from sklearn.metrics import accuracy_score
import scrapbook as sb
import torch
import torchvision

# from utils_cv.action_recognition.data import Urls
from dataset import VideoDataset
# from utils_cv.action_recognition.model import VideoLearner
# from utils_cv.common.gpu import system_info
# from utils_cv.common.data import data_path, unzip_url


# Number of consecutive frames used as input to the DNN. Use: 32 for high accuracy, 8 for inference speed.
MODEL_INPUT_SIZE = 8

# Number of training epochs
EPOCHS = 16

# Batch size. Reduce if running out of memory.
BATCH_SIZE = 8

# Learning rate
LR = 0.0001

data = VideoDataset('D:\\video_data_loader\\data\\EmotiW', batch_size=BATCH_SIZE, sample_length=MODEL_INPUT_SIZE, train_pct=1.0, video_ext='mp4')

print(data)

x = next(enumerate(data.train_dl))
"""
This is a version of https://github.com/HongguLiu/Deepfake-Detection/blob/master/network/mesonet.py
In case anyone wants to use it, please cite the original author.
Credits:
    Human Analog
    https://github.com/HongguLiu/Deepfake-Detection/blob/master/network/mesonet.py
"""

import os
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

import os, sys, time
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
frame_h = 5
frame_l = 5
len(test_videos)

class MesoInception4(nn.Module):
    def __init__(self, num_classes=2):
        super(MesoInception4, self).__init__()
        self.num_classes = num_classes
        #InceptionLayer1
        self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)


        #InceptionLayer2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        #Normal Layer
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16*8*8, 16)
        self.fc2 = nn.Linear(16, num_classes)


    #InceptionLayer
    def InceptionLayer1(self, input):
        x1 = self.Incption1_conv1(input)
        x2 = self.Incption1_conv2_1(input)
        x2 = self.Incption1_conv2_2(x2)
        x3 = self.Incption1_conv3_1(input)
        x3 = self.Incption1_conv3_2(x3)
        x4 = self.Incption1_conv4_1(input)
        x4 = self.Incption1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)

        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        x2 = self.Incption2_conv2_1(input)
        x2 = self.Incption2_conv2_2(x2)
        x3 = self.Incption2_conv3_1(input)
        x3 = self.Incption2_conv3_2(x3)
        x4 = self.Incption2_conv4_1(input)
        x4 = self.Incption2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)

        return y

    def forward(self, input):
        x = self.InceptionLayer1(input) #(Batch, 11, 128, 128)
        x = self.InceptionLayer2(x) #(Batch, 12, 64, 64)

        x = self.conv1(x) #(Batch, 16, 64 ,64)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x) #(Batch, 16, 32, 32)

        x = self.conv2(x) #(Batch, 16, 32, 32)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x) #(Batch, 16, 8, 8)

        x = x.view(x.size(0), -1) #(Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x) #(Batch, 16)
        x = x.view(1, -1)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(1, -1)

        return x
    
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
sys.path.insert(0, "/kaggle/input/deepfakes-inference-demo")
from blazeface import BlazeFace
facedet = BlazeFace().to(gpu)
facedet.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")
_ = facedet.train(False)
from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor

frames_per_video = 64 #frame_h * frame_l
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)
input_size = 224
from torchvision.transforms import Normalize

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)
def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

model = MesoInception4()
_ = model.eval()

def predict_on_video(video_path, batch_size):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)
        
        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.                    
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = make_square_image(resized_face)

                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
                    
                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    # x[n] = cv2.flip(resized_face, 1)
                    # n += 1

            if n > 0:
                x = torch.tensor(x, device=gpu).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i])

                # Make a prediction, then take the average.
                with torch.no_grad():
                    y_pred = model(x)
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    return y_pred[:n].mean().item()

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5

from concurrent.futures import ThreadPoolExecutor

def predict_on_video_set(videos, num_workers):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(os.path.join(test_dir, filename), batch_size=32)
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))

    return list(predictions)

predictions = predict_on_video_set(test_videos, num_workers=4)
submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
submission_df.to_csv("submission.csv", index=False)
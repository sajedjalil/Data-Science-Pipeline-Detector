# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import csv
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn
import imageio
import ignite
import torchvision.ops


class ResidualModule(torch.nn.Module):
    def __init__(self, num_channels):
        super(ResidualModule, self).__init__()

        self.conv0 = torch.nn.Conv2d(in_channels=num_channels,
                                     out_channels=num_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=num_channels,
                                     out_channels=num_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv0_relu = torch.nn.functional.relu(conv0)
        conv1 = self.conv1(conv0_relu)
        residual = x + conv1
        residual_relu = torch.nn.functional.relu(residual)

        return residual_relu


class YOLOv3Module(torch.nn.Module):
    def __init__(self, num_channels):
        super(YOLOv3Module, self).__init__()

        self.conv0 = torch.nn.Conv2d(in_channels=num_channels,
                                     out_channels=num_channels // 2,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=num_channels // 2,
                                     out_channels=num_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=True)

        self.residual = ResidualModule(num_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv0_relu = torch.nn.functional.relu(conv0)
        conv1 = self.conv1(conv0_relu)
        conv1_relu = torch.nn.functional.relu(conv1)
        residual = self.residual(conv1_relu)

        return residual


class WheatDetector2(torch.nn.Module):
    def __init__(self):
        super(WheatDetector2, self).__init__()

        self.resizer = torch.nn.Conv2d(in_channels=3,
                                       out_channels=1,
                                       kernel_size=9,
                                       stride=4,
                                       padding=4,
                                       bias=False)

        self.conv0 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo2 = YOLOv3Module(64)

        self.conv3 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo4 = YOLOv3Module(128)
        self.yolo5 = YOLOv3Module(128)

        self.conv6 = torch.nn.Conv2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=True)

        self.yolo7 = YOLOv3Module(256)
        self.yolo8 = YOLOv3Module(256)
        self.yolo9 = YOLOv3Module(256)
        self.yolo10 = YOLOv3Module(256)
        self.yolo11 = YOLOv3Module(256)
        self.yolo12 = YOLOv3Module(256)
        self.yolo13 = YOLOv3Module(256)
        self.yolo14 = YOLOv3Module(256)

        self.conv15 = torch.nn.Conv2d(in_channels=256,
                                      out_channels=512,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo16 = YOLOv3Module(512)
        self.yolo17 = YOLOv3Module(512)
        self.yolo18 = YOLOv3Module(512)
        self.yolo19 = YOLOv3Module(512)
        self.yolo20 = YOLOv3Module(512)
        self.yolo21 = YOLOv3Module(512)
        self.yolo22 = YOLOv3Module(512)
        self.yolo23 = YOLOv3Module(512)

        self.conv24 = torch.nn.Conv2d(in_channels=512,
                                      out_channels=1024,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=True)

        self.yolo25 = YOLOv3Module(1024)
        self.yolo26 = YOLOv3Module(1024)
        self.yolo27 = YOLOv3Module(1024)
        self.yolo28 = YOLOv3Module(1024)

        #---------------------------------------------------------------------

        self.tconv29 = torch.nn.ConvTranspose2d(in_channels=1024,
                                                out_channels=512,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv30 = torch.nn.ConvTranspose2d(in_channels=512,
                                                out_channels=256,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv31 = torch.nn.ConvTranspose2d(in_channels=256,
                                                out_channels=128,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.tconv32 = torch.nn.ConvTranspose2d(in_channels=128,
                                                out_channels=64,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        #---------------------------------------------------------------------

        self.tconv33 = torch.nn.ConvTranspose2d(in_channels=64,
                                                out_channels=256,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0,
                                                bias=True)

        self.scores = torch.nn.ConvTranspose2d(in_channels=256,
                                               out_channels=2,
                                               kernel_size=4,
                                               stride=4,
                                               padding=0,
                                               bias=True)

        self.bboxes = torch.nn.ConvTranspose2d(in_channels=256,
                                               out_channels=4,
                                               kernel_size=4,
                                               stride=4,
                                               padding=0,
                                               bias=True)

    def forward(self, x):
        #print(x)
        resizer = self.resizer(x)

        conv0 = self.conv0(resizer)
        conv0_relu = torch.nn.functional.relu(conv0)

        conv1 = self.conv1(conv0_relu)
        conv1_relu = torch.nn.functional.relu(conv1)

        yolo2 = self.yolo2(conv1_relu)

        conv3 = self.conv3(yolo2)
        conv3_relu = torch.nn.functional.relu(conv3)

        yolo4 = self.yolo4(conv3_relu)
        yolo5 = self.yolo5(yolo4)

        conv6 = self.conv6(yolo5)
        conv6_relu = torch.nn.functional.relu(conv6)

        yolo7 = self.yolo7(conv6_relu)
        yolo8 = self.yolo8(yolo7)
        yolo9 = self.yolo9(yolo8)
        #yolo10 = self.yolo10(yolo9)
        #yolo11 = self.yolo11(yolo10)
        #yolo12 = self.yolo12(yolo11)
        #yolo13 = self.yolo13(yolo12)
        #yolo14 = self.yolo14(yolo13)
        yolo14 = yolo9

        conv15 = self.conv15(yolo14)
        conv15_relu = torch.nn.functional.relu(conv15)

        yolo16 = self.yolo16(conv15_relu)
        yolo17 = self.yolo17(yolo16)
        yolo18 = self.yolo18(yolo17)
        #yolo19 = self.yolo19(yolo18)
        #yolo20 = self.yolo20(yolo19)
        #yolo21 = self.yolo21(yolo20)
        #yolo22 = self.yolo22(yolo21)
        #yolo23 = self.yolo23(yolo22)
        yolo23 = yolo18

        conv24 = self.conv24(yolo23)
        conv24_relu = torch.nn.functional.relu(conv24)

        yolo25 = self.yolo25(conv24_relu)
        yolo26 = self.yolo26(yolo25)
        #yolo27 = self.yolo27(yolo26)
        #yolo28 = self.yolo28(yolo27)
        yolo28 = yolo26

        #---------------------------------------------------------------------

        tconv29 = self.tconv29(yolo28)
        tconv29_relu = torch.nn.functional.relu(tconv29)
        tconv29_residual = yolo23 - tconv29_relu
        tconv29_residual_relu = torch.nn.functional.relu(tconv29_residual)

        tconv30 = self.tconv30(tconv29_residual_relu)
        tconv30_relu = torch.nn.functional.relu(tconv30)
        tconv30_residual = yolo14 - tconv30_relu
        tconv30_residual_relu = torch.nn.functional.relu(tconv30_residual)

        tconv31 = self.tconv31(tconv30_residual_relu)
        tconv31_relu = torch.nn.functional.relu(tconv31)
        tconv31_residual = yolo5 - tconv31_relu
        tconv31_residual_relu = torch.nn.functional.relu(tconv31_residual)

        tconv32 = self.tconv32(tconv31_residual_relu)
        tconv32_relu = torch.nn.functional.relu(tconv32)
        tconv32_residual = yolo2 - tconv32_relu
        tconv32_residual_relu = torch.nn.functional.relu(tconv32_residual)

        ##---------------------------------------------------------------------

        tconv33 = self.tconv33(tconv32_residual_relu)
        tconv33_relu = torch.nn.functional.relu(tconv33)

        scores = self.scores(tconv33_relu)
        scores = torch.softmax(scores, dim=1)

        scores = torch.nn.functional.relu(scores)
        bboxes = self.bboxes(tconv33_relu)
        #bboxes = scores[:, 0, :, :] * bboxes

        if torch.any(torch.isnan(scores)) or torch.any(torch.isnan(bboxes)):
            raise Exception("Nan values")

        return scores[:, 1, :, :].unsqueeze(1), bboxes


def select_bboxes2(bboxes, scores, center_threshold, shape_threshold,
                  score_threshold):
    bboxes = bboxes.clone()
    scores = scores.clone()
    kept = scores >= score_threshold

    l2dist = lambda x, y: torch.sqrt(torch.sum((x - y)**2, 2))
    for b in range(kept.shape[0]):
        for h in range(kept.shape[2]):
            for w in range(kept.shape[3]):
                if kept[b, 0, h, w]:
                    center_dist = l2dist(bboxes[b, :, :, 0:2],
                                         bboxes[b, h, w, 0:2])
                    shape_dist = l2dist(bboxes[b, :, :, 2:4],
                                        bboxes[b, h, w, 2:4])

                    same = center_dist < center_threshold
                    same = torch.logical_and(same,
                                             (shape_dist < shape_threshold))
                    same = torch.logical_and(same, kept[b, 0, :, :])

                    if same.any():
                        same = torch.logical_and(
                                same, (scores[b, 0, :, :] < scores[b, 0, h, w]))
                        if same.any():
                            kept[b, 0, :, :][same] = False

    return kept

def select_bboxes_nms(bboxes, scores, iou_theshold=0.95):
    bboxes_t = bboxes.clone().flatten(1, 2)
    bboxes_t[:, :, 2:4] = bboxes_t[:, :, 2:4] + bboxes_t[:, :, 0:2]
    scores = scores.clone().flatten(1, 3)
    keep = torchvision.ops.nms(bboxes_t[0], scores[0], iou_theshold)

    return keep


def decode_bboxes(encoded_bboxes, mean, std):
    _, _, height, width = encoded_bboxes.shape
    position = torch.cartesian_prod(torch.arange(height), torch.arange(width))
    position = position.reshape(height, width, 2)
    bboxes = encoded_bboxes.clone().permute(0, 2, 3, 1)

    bboxes[:, :, :, 0:2] = bboxes[:, :, :, 0:2] * torch.tensor([height, width
                                                                ]) + position
    bboxes[:, :, :, 2:4] = (bboxes[:, :, :, 2:4] * torch.tensor(
        [std[1], std[0]])) + torch.tensor([mean[1], mean[0]])

    bboxes[:, :, :, 0:2] = bboxes[:, :, :, 0:2] - bboxes[:, :, :, 2:4] / 2

    #y_min, x_min, height, width
    return bboxes


test_dir = '/kaggle/input/global-wheat-detection/test'
test_images = os.listdir(test_dir)
wheat_net = WheatDetector2()
wheat_net.load_state_dict(torch.load("../input/cputrainedsmallarea/wheat_detector_2_model_6158.pth", map_location=torch.device('cpu')))
mean = torch.tensor([84.4350, 76.9273])
std = torch.tensor([35.5535, 33.8532])

with open('/kaggle/working/submission.csv', 'w', newline='') as submission:
    submission_writer = csv.writer(submission, delimiter=',')
    submission_writer.writerow(["image_id","PredictionString"])
    for img_name in test_images:
        image_id = img_name.split('.')[0]
        img = torch.tensor(
            imageio.imread(os.path.join(f"{test_dir}",
                                        f"{img_name}"))).permute(2, 0, 1)
        img = img.float() / 255
        img = img.unsqueeze(0)
        padded_image = torch.zeros((1, 3, 1024, 1024))
        _, _, img_height, img_width = img.shape
        padded_image[:, :, 0:min(1024, img_height), 0:min(1024, img_width)] = img[:, :, 0:min(1024, img_height), 0:min(1024, img_width)]

        with torch.no_grad():
            scores, bboxes = wheat_net(padded_image)

        decoded_bboxes = decode_bboxes(bboxes, mean, std)
        decoded_bboxes = torch.clamp(decoded_bboxes, 0, 1023)
        decoded_bboxes = decoded_bboxes[:, 0:1024:4, 0:1024:4, :]
        scores = scores[:, :, 0:1024:4, 0:1024:4]
        selected = select_bboxes2(decoded_bboxes, scores, 65, 65, 0.5)
        #selected = select_bboxes_nms(decoded_bboxes, scores, 0.4)

        submission_row = ""
        for y in range(decoded_bboxes.shape[1]):
            for x in range(decoded_bboxes.shape[2]):
                #if (y*decoded_bboxes.shape[1] + x) in selected and scores[0, 0, y, x] > 0.45:
                if selected[0, 0, y, x]:
                    y_min, x_min, height, width = decoded_bboxes[0, y, x]
                    submission_row = submission_row + f"{scores[0, 0, y, x].item():.2} {int(x_min.item())} {int(y_min.item())} {int(width.item())} {int(height.item())} "
        submission_writer.writerow([image_id, submission_row])

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
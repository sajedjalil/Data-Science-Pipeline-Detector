# %% [markdown]
# # Pytorch starter - FasterRCNN Train
# In this notebook I enabled the GPU and the Internet access (needed for the pre-trained weights). We can not use Internet during inference, so I'll create another notebook for commiting. Stay tuned!
# 
# You can find the [inference notebook here](https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-inference)
# 
# - FasterRCNN from torchvision
# - Use Resnet50 backbone
# - Albumentation enabled (simple flip for now)

# %% [code]
import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

from tqdm import tqdm
from pathlib import Path

DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

# %% [code]
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
train_df.shape

# %% [code]
train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

# %% [code]
image_ids = train_df['image_id'].unique()
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]

# %% [code]
valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]

# %% [code]
valid_df.shape, train_df.shape

# %% [code]
class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

# %% [code]
# Albumentations
def get_train_transform():
    return A.Compose(
        [
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
#             A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0)
        ], 
        p=1.0, 
        bbox_params={
            'format':'pascal_voc',
            'label_fields':['labels']
        }
    )

# def get_third_train_transform():
#     return A.Compose([
#         A.Rotate(p=1),
#         A.RandomBrightness(p=1),
#         ToTensorV2(p=1.0)
#     ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# def get_fourth_train_transform():
#     return A.Compose([
#         A.Rotate(p=1),
#         A.HueSaturationValue(p=1),
#         ToTensorV2(p=1.0)
#     ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# %% [markdown]
# # Create the model

# %% [code]
# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

# %% [code]
num_classes = 2  # 1 class (wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# %% [code]
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

# %% [code]
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())


# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# %% [code]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %% [markdown]
# # Sample

# %% [code]
# images, targets, image_ids = next(iter(train_data_loader))
# images = list(image.to(device) for image in images)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# %% [code]
# boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)
# sample = images[2].permute(1,2,0).cpu().numpy()

# %% [code]
# fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# for box in boxes:
#     cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2], box[3]),
#                   (220, 0, 0), 3)
    
# ax.set_axis_off()
# ax.imshow(sample)

# %% [markdown]
# # Train

# %% [code]
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

num_epochs = 50

# %% [code]
loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()
    
    for images, targets, image_ids in train_data_loader:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")

# %% [code]
# images, targets, image_ids = next(iter(valid_data_loader))

# %% [code]
# images = list(img.to(device) for img in images)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# %% [code]
# boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
# sample = images[1].permute(1,2,0).cpu().numpy()

# %% [code]
# model.eval()
# cpu_device = torch.device("cpu")

# outputs = model(images)
# outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

# %% [code]
# fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# for box in boxes:
#     cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2], box[3]),
#                   (220, 0, 0), 3)
    
# ax.set_axis_off()
# ax.imshow(sample)

# %% [code]
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn_trained.pth')

# %% [code]
# submission = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')


# root_image = Path("../input/global-wheat-detection/test")
# test_images = [root_image / f"{img}.jpg" for img in submission.image_id]

# submission = []
# model.eval()

# def model_prediction(image_path,model,device):
#     model.eval()
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#     image /= 255.0
#     images = torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0).to(device)
#     outputs = model(images)


#     outputs = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in outputs]
#     boxes = outputs[0]["boxes"]
#     scores = outputs[0]["scores"]
#     valid_boxes = boxes[scores > 0.5]
#     valid_scores = scores[scores > 0.5]
#     return valid_boxes, valid_scores

# for image in tqdm(test_images):
#     boxes, scores = model_prediction(str(image),model,device)
#     prediction_string = []
#     for (x_min,y_min,x_max,y_max),s in zip(boxes,scores):
#         x = round(x_min)
#         y = round(y_min)
#         h = round(x_max-x_min)
#         w = round(y_max-y_min)
#         prediction_string.append(f"{s} {x} {y} {h} {w}")
#     prediction_string = " ".join(prediction_string)
    
#     submission.append([image.name[:-4],prediction_string])

# sample_submission = pd.DataFrame(submission, columns=["image_id","PredictionString"])
# sample_submission.to_csv('submission.csv', index=False)
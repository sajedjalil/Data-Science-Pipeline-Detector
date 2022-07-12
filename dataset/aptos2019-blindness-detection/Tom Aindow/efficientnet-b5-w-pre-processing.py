# Versions

# v1 - Initial Commit

# References

# Pytorch implementation: https://github.com/lukemelas/EfficientNet-PyTorch
# Pre-trained weights: http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth
# Early stopping: https://github.com/Bjarten/early-stopping-pytorch
# Apex: https://github.com/NVIDIA/apex
# Borrowed a lot from abhishek, so give him an upvote: https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59

# Parameters

lr = 2e-5
img_size = 224
batch_size = 32
n_epochs = 100
n_freeze = 1
patience = 10
coef = [0.5, 1.5, 2.5, 3.5]

# Libraries

import torch
import os
import gc
import sys
import subprocess
import numpy as np
import pandas as pd
from albumentations import Compose, HorizontalFlip, VerticalFlip, Rotate, Resize
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import cv2
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold
from tqdm import tqdm

package_dir = '../input/earlystoppingpytorch/early-stopping-pytorch/early-stopping-pytorch'
sys.path.append(package_dir)
from pytorchtools import EarlyStopping

package_dir = '../input/efficientnetpytorchrepo/efficientnet-pytorch/EfficientNet-PyTorch'
sys.path.append(package_dir)
from efficientnet_pytorch import EfficientNet

# Install Apex for mixed precision and faster training

print('Starting Apex installation ...')

FNULL = open(os.devnull, 'w')
process = subprocess.Popen(
    'pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidia-apex/apex/apex',
    shell=True, 
    stdout=FNULL, stderr=subprocess.STDOUT)
process.wait()

if process.returncode==0:
    print('Apex successfully installed')
    
from apex import amp

# Functions

def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'), device=device)


def crop_image_from_gray(img, tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop(img):
    """
    Create circular crop around image centre
    """

    img = crop_image_from_gray(img)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img

class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/train_images', self.data.loc[idx, 'id_code'] + '.png')
        im = cv2.imread(img_name)
        im = circle_crop(im)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        if self.transform:
            augmented = self.transform(image=im)
            im = augmented['image']
        return {'image': im, 'labels': label}
  
class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')
        im = cv2.imread(img_name)
        im = circle_crop(im)
        if self.transform:
            augmented = self.transform(image=im)
            im = augmented['image']
        return {'image': im}
        
def train_model(model, patience, n_epochs):

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):

        if epoch == n_freeze:

            for param in model.parameters():
                param.requires_grad = True

        model.train()

        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        tr_loss = 0

        for step, batch in enumerate(data_loader_train):
            inputs = batch["image"]
            labels = batch["labels"].view(-1, 1)

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            tr_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()


        epoch_loss = tr_loss / len(data_loader_train)
        print('Training Loss: {:.4f}'.format(epoch_loss))

        model.eval()
        eval_loss = 0
        eval_kappa = 0
        nb_eval_steps = 0

        for step, batch in enumerate(data_loader_valid):

            inputs = batch["image"]
            labels = batch["labels"].view(-1, 1)

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            with torch.no_grad():
                outputs = model(inputs)

                
            y_hat = torch.Tensor.cpu(outputs.view(-1))
            y = torch.Tensor.cpu(labels.view(-1))

            for pred in enumerate(y_hat):
                if pred[1] < coef[0]:
                    y_hat[1] = 0
                elif pred[1] >= coef[0] and pred[1] < coef[1]:
                    y_hat[1] = 1
                elif pred[1] >= coef[1] and pred[1] < coef[2]:
                    y_hat[1] = 2
                elif pred[1] >= coef[2] and pred[1] < coef[3]:
                    y_hat[1] = 3
                else:
                    y_hat[1] = 4
                    
            tmp_eval_loss = criterion(outputs, labels)
            tmp_eval_kappa = quadratic_kappa(y_hat, y)            
            eval_loss += tmp_eval_loss.mean().item()
            eval_kappa += tmp_eval_kappa.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_kappa = eval_kappa / nb_eval_steps

        print('Validation Loss: {:.4f}'.format(eval_loss))
        print('Validation Kappa: {:.4f}'.format(eval_kappa))
        
        #eval_loss = 1 - eval_loss
        early_stopping(eval_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))

    return model

# Model

model = EfficientNet.from_name('efficientnet-b5')
model.load_state_dict(torch.load('../input/efficientnetb5/efficientnet-b5-586e6cc6.pth'))

device = torch.device("cuda:0")

for param in model.parameters():
    param.requires_grad = False

model._fc = torch.nn.Linear(2048, 1)
model.to(device)

criterion = torch.nn.MSELoss()
plist = [{'params': model.parameters(), 'lr': lr}]
optimizer = optim.Adam(plist, lr=lr)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

# Data Loaders

transform_train = Compose([
    Resize(img_size, img_size),
    VerticalFlip(p=0.2),
    Rotate(limit=365, p=0.2),
    HorizontalFlip(p=0.2),
    ToTensor()
])

transform_valid = Compose([
    Resize(img_size, img_size),
    ToTensor()
])

train_dataset = RetinopathyDatasetTrain(
    csv_file='../input/aptos2019-blindness-detection/train.csv', transform=transform_train)

valid_dataset = RetinopathyDatasetTrain(
    csv_file='../input/aptos2019-blindness-detection/train.csv', transform=transform_valid)

test_dataset = RetinopathyDatasetTest(
    csv_file='../input/aptos2019-blindness-detection/sample_submission.csv', transform=transform_valid)
    
# CV Split (1 fold currently)

num_train = len(train_dataset)
indices = list(range(num_train))
kf = KFold(n_splits=5, random_state=1, shuffle=True)

train_idx = []
valid_idx = []

for t, v in kf.split(indices):
    train_idx.append(t)
    valid_idx.append(v)

fold = 0
    
train_sampler = SubsetRandomSampler(train_idx[fold])
valid_sampler = SubsetRandomSampler(valid_idx[fold])
    
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, sampler=valid_sampler)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = train_model(model, patience, n_epochs)

# Inference

model.eval()
    
for param in model.parameters():
    param.requires_grad = False

test_preds = np.zeros((len(test_dataset), 1))   

for i, x_batch in enumerate(data_loader_test):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds[i * batch_size:(i + 1) * batch_size] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
                       
for i, pred in enumerate(test_preds):
    if pred < coef[0]:
        test_preds[i] = 0
    elif pred >= coef[0] and pred < coef[1]:
        test_preds[i] = 1
    elif pred >= coef[1] and pred < coef[2]:
        test_preds[i] = 2
    elif pred >= coef[2] and pred < coef[3]:
        test_preds[i] = 3
    else:
        test_preds[i] = 4

# Export

sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = test_preds.astype(int)
sample.to_csv("submission.csv", index=False)

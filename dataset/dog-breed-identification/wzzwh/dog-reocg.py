from __future__ import print_function, division
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from PIL import Image
import matplotlib.pyplot as plt
import time
import os

plt.ion()   # interactive mode
multiGPU = False
TRAIN_IMG_PATH = "../input/train"
TEST_IMG_PATH = "../input/test"
LABELS_CSV_PATH = "../input/labels.csv"
SAMPLE_SUB_PATH = "../input/sample_submission.csv"
class DogsDataset(Dataset):
    """Dog breed identification dataset."""

    def __init__(self, img_dir, dataframe, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.        
            dataframe (pandas.core.frame.DataFrame): Pandas dataframe obtained
                by read_csv().
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_frame.id[idx]) + ".jpg"
        image = Image.open(img_name)
        label = self.labels_frame.target[idx]

        if self.transform:
            image = self.transform(image)

        return [image, label] 
dframe = pd.read_csv(LABELS_CSV_PATH)
labelnames = pd.read_csv(SAMPLE_SUB_PATH).keys()[1:]
codes = range(len(labelnames))
breed_to_code = dict(zip(labelnames, codes))
code_to_breed = dict(zip(codes, labelnames))
dframe['target'] =  [breed_to_code[x] for x in dframe.breed]

cut = int(len(dframe)*0.8)
train, test = np.split(dframe, [cut], axis=0)
test = test.reset_index(drop=True)

train_ds = DogsDataset(TRAIN_IMG_PATH, train)
test_ds = DogsDataset(TRAIN_IMG_PATH, test)
idx = 29
plt.imshow(train_ds[idx][0])
print(code_to_breed[train_ds[idx][1]])
print("Shape of the image is: ", train_ds[idx][0].size)
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
train_ds = DogsDataset(TRAIN_IMG_PATH, train, data_transform)
test_ds = DogsDataset(TRAIN_IMG_PATH, test, data_transform)
datasets = {"train": train_ds, "val": test_ds}

idx = 29
print(code_to_breed[train_ds[idx][1]])
print("Shape of the image is: ", train_ds[idx][0].shape)
trainloader = DataLoader(train_ds, batch_size=4,
                        shuffle=True, num_workers=4)

testloader = DataLoader(test_ds, batch_size=4,
                        shuffle=True, num_workers=4)

dataloaders = {"train": trainloader, "val": testloader}
if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:     
            since_epoch = time.time()
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
    
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                inputs = Variable(inputs.type(Tensor))
                labels = Variable(labels.type(LongTensor))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects / len(datasets[phase])

            time_elapsed_epoch = time.time() - since_epoch
            print('{} Loss: {:.4f} Acc: {:.4f} in {:.0f}m {:.0f}s'.format(
                phase, epoch_loss, epoch_acc, time_elapsed_epoch // 60, time_elapsed_epoch % 60))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
model_ft = models.resnet152()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 120)

if torch.cuda.device_count() > 1 and multiGPU:
  print("Using", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model_ft = nn.DataParallel(model_ft)

if use_gpu:
   model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)
submission_df = pd.read_csv(SAMPLE_SUB_PATH)
output_df = pd.DataFrame(index=submission_df.index, columns=submission_df.keys() )
output_df['id'] = submission_df['id']
submission_df['target'] =  [0] * len(submission_df)

tdata_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

submission_ds = DogsDataset(TEST_IMG_PATH, submission_df, tdata_transform)

sub_loader = DataLoader(submission_ds, batch_size=4,
                        shuffle=False, num_workers=4)


def test_sumission(model):
    since = time.time()
    sub_outputs = []
    model.train(False)  # Set model to evaluate mode
    # Iterate over data.
    for data in sub_loader:
        # get the inputs
        inputs, labels = data

        inputs = Variable(inputs.type(Tensor))
        labels = Variable(labels.type(LongTensor))

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        sub_outputs.append(outputs.data.cpu().numpy())

    sub_outputs = np.concatenate(sub_outputs)
    for idx,row in enumerate(sub_outputs.astype(float)):
        sub_outputs[idx] = np.exp(row)/np.sum(np.exp(row))

    output_df.loc[:,1:] = sub_outputs
        
    print()
    time_elapsed = time.time() - since
    print('Run complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return output_df
odf = test_sumission(model_ft)
odf.to_csv("dogs_id.csv", index=False)
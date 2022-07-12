# Versions

# v1 - Initial Commit
# v6 - Added mixed precision, early stopping, cross validation

# References

# Pre-trained weights: https://github.com/facebookresearch/WSL-Images
# Early stopping: https://github.com/Bjarten/early-stopping-pytorch
# Apex: https://github.com/NVIDIA/apex
# Borrowed a lot from abhishek, so give him an upvote: https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59

# Parameters

lr = 1e-5
img_size = 224
batch_size = 64
n_epochs = 25
n_freeze = 1
n_folds = 4
patience = 2
coef = [0.5, 1.5, 2.5, 3.5]
    
# Libraries

import torch
import os
import gc
import sys
import subprocess
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models.resnet import ResNet, Bottleneck
import torch.optim as optim
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold

package_dir = '../input/earlystoppingpytorch/early-stopping-pytorch/early-stopping-pytorch'
sys.path.append(package_dir)
from pytorchtools import EarlyStopping

# Install Apex for mixed precision

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

class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/train_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return {'image': image, 'labels': label}
    
class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        return {'image': image}
    
def _resnext(path, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    model.load_state_dict(torch.load(path))
    return model

def resnext101_32x8d_wsl(path, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext(path, Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

def train_model(model, patience, n_epochs):
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):
        
        if epoch == n_freeze:
            
            for param in model.parameters():
                param.requires_grad = True

        nb_tr_steps = 0
        tr_loss = 0

        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)
        
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
            nb_tr_examples += inputs.size(0)
            nb_tr_steps += 1
            
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = tr_loss / len(data_loader_train)
        print('Training Loss: {:.4f}'.format(epoch_loss))

        model.eval()
        eval_loss = 0
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

            tmp_eval_loss = quadratic_kappa(y_hat, y)

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        print('Validation Kappa: {:.4f}'.format(eval_loss))

        eval_loss = 1 - eval_loss
        early_stopping(eval_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    return model

# DataLoader

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

transform_valid = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

train_dataset = RetinopathyDatasetTrain(
    csv_file='../input/aptos2019-blindness-detection/train.csv', transform=transform_train)

valid_dataset = RetinopathyDatasetTrain(
    csv_file='../input/aptos2019-blindness-detection/train.csv', transform=transform_valid)

test_dataset = RetinopathyDatasetTest(
    csv_file='../input/aptos2019-blindness-detection/sample_submission.csv', transform=transform_valid)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
# CV Split

num_train = len(train_dataset)
indices = list(range(num_train))
kf = KFold(n_splits=n_folds, random_state=1337, shuffle=True)

train_idx = []
valid_idx = []

for t, v in kf.split(indices):
    train_idx.append(t)
    valid_idx.append(v)

# Training 

fold_predictions = np.zeros((len(test_dataset), n_folds))
                       
for fold in np.arange(n_folds):
    
    print('Fold:',fold)
    
    # Model    

    device = torch.device("cuda:0")
    model = resnext101_32x8d_wsl(path='../input/resnext101-32x8/ig_resnext101_32x8-c38310e5.pth')

    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(2048, 1)
    model.to(device)

    criterion = torch.nn.MSELoss()
    plist = [{'params': model.parameters(), 'lr': 1e-5}]
    optimizer = optim.Adam(plist, lr=1e-5)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    # Train with early stopping
    
    train_sampler = SubsetRandomSampler(train_idx[fold])
    valid_sampler = SubsetRandomSampler(valid_idx[fold])

    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, sampler=valid_sampler)

    model = train_model(model, patience, n_epochs)
    
    # Inference for each CV split

    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    test_preds = np.zeros((len(test_dataset), 1))   
    
    for i, x_batch in enumerate(data_loader_test):
        x_batch = x_batch["image"]
        pred = model(x_batch.to(device))
        test_preds[i * batch_size:(i + 1) * batch_size] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
                       
    fold_predictions[:,fold] = test_preds.reshape(fold_predictions.shape[0])
    
    del(model, data_loader_train, data_loader_valid, test_preds)
    gc.collect()
    torch.cuda.empty_cache() 
    
fold_predictions_avg = np.mean(fold_predictions, axis=1)

for i, pred in enumerate(fold_predictions_avg):
    if pred < coef[0]:
        fold_predictions_avg[i] = 0
    elif pred >= coef[0] and pred < coef[1]:
        fold_predictions_avg[i] = 1
    elif pred >= coef[1] and pred < coef[2]:
        fold_predictions_avg[i] = 2
    elif pred >= coef[2] and pred < coef[3]:
        fold_predictions_avg[i] = 3
    else:
        fold_predictions_avg[i] = 4

# Export

sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = fold_predictions_avg.astype(int)
sample.to_csv("submission.csv", index=False)
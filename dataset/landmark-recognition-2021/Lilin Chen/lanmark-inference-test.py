#%%
import pandas as pd
import numpy as np
import random
import math
import os
import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19

from pathlib import Path
import PIL
from PIL import Image

def seed_all(seed):
    """Utility function to set seed across all pytorch process for repeatable experiment
    """
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Utility function to set random seed for DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
seed_all(100)
#%%
data_dir = Path("../input/landmark-recognition-2021/")
train_dir = data_dir / "train"
test_dir = data_dir / "test"
train_file = data_dir / "train.csv"
sub_file = data_dir / "sample_submission.csv"

train_df = pd.read_csv(train_file)
sub_df = pd.read_csv(sub_file)

print(train_df.head())
print(sub_df.head())
print('Samples train:', len(train_df))
print('Samples test:', len(sub_df))
print('categories:', len(train_df['landmark_id'].unique()))

## landmark_id value > number of classes, this leads to error during training pytorch model
landmark_id_map = {lid:i for i, lid in enumerate(train_df.landmark_id.unique())}
train_df['landmark_id'] = train_df['landmark_id'].map(landmark_id_map)
#%%
## Building a custom data loader to load the data in batches for pytorch

class LandMarkData(Dataset):
    
    def __init__(self, data_file, data_dir, transform=None, data_type="train"):
        """
        data_file str: file which contains image_id and its class
        data_dir str: directory where data is present
        """
        
        self.data_file = pd.read_csv(data_file)
        ## landmark_id value > number of classes, this leads to error during training pytorch model
        if data_type == "train":
            self.landmark_id_map = {lid:i for i, lid in enumerate(self.data_file.landmark_id.unique())}
            self.data_file['landmark_id'] = self.data_file['landmark_id'].map(self.landmark_id_map)
        elif data_type == "test":
            print("Test data will not have landmarkd id, hence no mapping")
        self.data_dir = data_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_id = self.data_file.iloc[idx, 0]
        img_class = self.data_file.iloc[idx, 1]
        img_path = os.path.join(self.data_dir, img_id[0], img_id[1], img_id[2], f'{img_id}.jpg')
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        sample = [img, img_class, img_id]
        
        return sample

#%%
## define basic transforms
transform = transforms.Compose([ transforms.CenterCrop(224), 
                               transforms.ToTensor()])
train_data = LandMarkData(train_file, train_dir, transform, "train")
test_data = LandMarkData(sub_file, test_dir, transform, "test")

## Manually Checking if dataloader and transforms are getting applied or not.
## All images should be 224*224
print(f"Image Shape               || Image Class|| Image Id")
print("-"*60)
samples = train_df['id'].sample(10, random_state=100).index
for sample in samples:
    img_sample = train_data[sample]
    print(f"{img_sample[0].shape} || {img_sample[1]}      || {img_sample[2]}")

## Manually Checking if dataloader and transforms are getting applied or not.
## All images should be 224*224
print(f"Image Shape               || Image Class|| Image Id")
print("-"*60)
samples = sub_df['id'].sample(10, random_state=100).index
for sample in samples:
    img_sample = test_data[sample]
    print(f"{img_sample[0].shape} || {img_sample[1]} || {img_sample[2]}")
    
#%%
## Taking 20% as valid data
valid_size = 0.01
batch_size = 8

## Splitting train data into valid data. Please note this is vanila split, 
# we need to have better split or agumentation as many landmarks have very few images
num_train = len(train_data)
indices = list(range(num_train))
np.random.seed(100)
np.random.shuffle(indices)
split = int(np.floor(num_train*valid_size))
valid_idx, train_idx = indices[:split], indices[split:]
assert len(valid_idx) + len(train_idx) == num_train

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0, worker_init_fn=seed_worker)
valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=0, worker_init_fn=seed_worker)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0, worker_init_fn=seed_worker)
#%%
def plot_images(loader, num_images=5):
    images, label, img_id = next(iter(loader))
    # convert to numpy and transpose as (Batch Size, Height, Width, Channel) as needed by matplotlib
    images = images.numpy().transpose(0, 2, 3, 1)
    
    # Analysing images of a train batch
    num_cols = 5
    num_rows = 1
    if num_images > 5:
        num_cols = 5
        num_rows = math.ceil(num_images / 5)
    np.random.seed(100)
    indices = np.random.choice(range(len(label)), size=num_images, replace=False)
    width = 20
    height = 5*num_rows
    plt.figure(figsize=(width, height))
    for i, idx in enumerate(indices):
        plt.subplot(num_rows, num_cols, i + 1)
        image = images[idx]
        plt.imshow(image);
        plt.title(f'label: {label[idx]}\n img_id: {img_id[idx]}');
        plt.axis("off")
    plt.show()

#%%
#plotting one batch of images from train
plot_images(train_loader, num_images=batch_size)
#plotting one batch images from valid
plot_images(valid_loader, batch_size)
plot_images(test_loader, batch_size)
#%%
def get_pretrained_model(model_name=vgg19, num_class=10, use_cuda=False):
    """ Wrapper function to get pre-trained model 
    """
    model_transfer = model_name(pretrained=False)
    for params in model_transfer.features.parameters():
        params.requires_grad=False

    in_features = model_transfer.classifier[6].in_features
    model_transfer.classifier[6] = nn.Linear(in_features, num_class)
    if use_cuda:
        model_transfer.cuda()
    return model_transfer

#%%
use_cuda=True
model = get_pretrained_model(vgg19, train_df.landmark_id.nunique(), use_cuda)
loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
save_path = "../working/"

import tqdm
def predict(loaders, model, use_cuda, landmark_reverse_map):
    # set the module to evaluation mode
    model.eval()
    sf = nn.Softmax(dim=1)
    img_id_list = []
    confidence_list = []
    label_list = []
    tot_batch = len(loaders['test'])
    for batch_idx, (data, _, img_id) in enumerate(tqdm.tqdm(loaders['test'])):
        # move to GPU
        if use_cuda:
            data = data.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        output = sf(output)
        output = torch.max(output, dim=1)
        confidence = output[0].cpu().detach().numpy()
        label=output[1].cpu().detach().numpy()
        
        img_id_list.extend(list(img_id))
        confidence_list.extend(confidence.tolist())
        label_list.extend(label.tolist())
    
    predict_df = pd.DataFrame({'id': img_id_list, 
                               'landmarks': label_list, 
                               'conf': confidence_list})
    predict_df['landmarks'] = predict_df['landmarks'].map(landmark_reverse_map)
    predict_df['landmarks'] = predict_df['landmarks'].astype(str) +" " + predict_df['conf'].round(6).astype(str)

    predict_df.drop("conf", axis=1, inplace=True)
    return predict_df


# #%%
# # Loading the best model
# aaa = torch.load(os.path.join("../input/model-vgg19",'model_final.pth'))
# model_state_dict = torch.load(os.path.join("../input/model-vgg19",'model_final.pth'))
# torch.save(model_state_dict, "../working/model_final.pth")
# model_state_dict =  torch.load("../working/model_final.pth")
# model.load_state_dict(model_state_dict)
model.load_state_dict(torch.load(os.path.join("../input/model-vgg19",'model_final.pth')), strict=False)
# landmark_reverse_map = dict(zip(train_data.landmark_id_map.values(), train_data.landmark_id_map.keys()))
# out = predict(loaders, model, use_cuda, landmark_reverse_map)
# print(out.tail())
# out.to_csv(os.path.join(save_path, "submission.csv"), index = False)
import shutil
shutil.copy("../input/landmark-recognition-2021/sample_submission.csv", "../working/submission.csv")


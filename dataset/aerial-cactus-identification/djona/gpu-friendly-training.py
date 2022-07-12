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
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils import data
from PIL import Image
import numpy as np
from torch.backends import cudnn
import os, random
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt



SEED = 1234
def seed_everything(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
## dataset generator
class Dataset(data.Dataset):
    def __init__(self, data_dir, list_IDs, labels, transform):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        
        filename = self.list_IDs[index]
        path = self.data_dir + filename
        X = Image.open(path)
        y = self.labels[filename]
        
        if self.transform is not None:
            X = self.transform(X)
            
        return X, y
        

## prepare data: generate the list_IDs and labels
def prepare_data(data_path, random_state=1, prob=0.2):
    
    #read the dataframe
    df = pd.read_csv(data_path)
    
    if df.isnull().values.sum() != 0:
        raise ValueError('Your dataframe must not contain a nan value')
        
    train, validation = train_test_split(df, test_size=prob, random_state=random_state)
    list_IDs = {'train':[], 'validation':[]}
    list_IDs['train'].extend(train.id)
    list_IDs['validation'].extend(validation.id)
    labels = dict(zip(df.id, df.has_cactus))
    
    return list_IDs, labels
    
    
def train_model(model, train_dataloaders, valid_dataloaders, criterion, optimizer, epochs, device):
    print('Training the model ... \n')
    
    model.to(device)
    
    start_time = time.time()
    train_losses, test_losses = [], []
    
    for e in range(epochs):
        model.train()
        
        training_loss = 0
        for images, labels in train_dataloaders:
            labels = labels.type(torch.FloatTensor).view(-1, 1)
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients 
            optimizer.zero_grad()
            
            # get the prediction
            y_hat = model.forward(images)
            
            # get the loss
            loss = criterion(y_hat, labels)
            
            # backpropagate
            loss.backward()
            
            # gradient descent
            optimizer.step()
            
            # keep track of the training loss
            training_loss += loss.item()

        else:
            # keep trck of the validation loss and accuracy
            validation_loss = 0
            accuracy = 0
            
            # set the model to evaluate mode
            model.eval()
            
            with torch.no_grad():
                for images, labels in valid_dataloaders:
                    labels = labels.type(torch.FloatTensor).view(-1, 1)
                    images, labels = images.to(device), labels.to(device)
                    
                    y_hat = model.forward(images)
                    
                    # get the loss
                    loss = criterion(y_hat, labels)
                    
                    output = (y_hat > 0.5).float()
                    correct = (output == labels).float().mean()
                    
                    # keep track of the validation loss
                    validation_loss += loss.item()
                    accuracy += correct.item()
                    
            # Get the total time that has elapsed
            elapsed_time = time.time() - start_time
            
            # update the training and valdiation losses
            train_losses.append(training_loss/len(train_dataloaders))
            test_losses.append(validation_loss/len(valid_dataloaders))
            
            # Print out the statistical information
            print("Training Epoch: {}\n".format(e),
                    "Training Loss: {}\n".format(training_loss/len(train_dataloaders)),
                    "Validation Loss: {}\n".format(validation_loss/len(valid_dataloaders)),
                    "Accuracy: {}\n".format(accuracy/len(valid_dataloaders) * 100),
                    "Total Time: {}\n".format(elapsed_time))  
            
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()
    print("\nDone training the model \n")
         
         
# model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1   = nn.Linear(50*5*5, 120)
        self.dropout = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(120, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2, 2)
        out = out.view(-1, 5*5*50)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
        
if __name__ == '__main__':
    data_path = '../input/train.csv'
    image_path = "../input/train/train/"
    seed_everything(seed=SEED)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True
    
    
    # get the list of ids, labels
    list_IDs, labels = prepare_data(data_path)

# create ids and labels
    train_ids = list_IDs['train']
    validation_ids = list_IDs['validation']
    
    params = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 6 }

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    valid_transforms = transforms.Compose([transforms.Resize(32),
                                        transforms.CenterCrop(32),
                                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# create a datasets
    train_datasets = Dataset(image_path, train_ids, labels, train_transforms)
    validation_datasets = Dataset(image_path, validation_ids, labels, valid_transforms)

    training_generator = data.DataLoader(train_datasets, **params)
    validation_generator = data.DataLoader(validation_datasets, **params)
    
    model = LeNet()
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.BCELoss()
    epochs = 10
    
    train_model(model, train_dataloaders=training_generator, valid_dataloaders=validation_generator, criterion=loss_func, optimizer=optimizer, epochs=10, device=device)
    
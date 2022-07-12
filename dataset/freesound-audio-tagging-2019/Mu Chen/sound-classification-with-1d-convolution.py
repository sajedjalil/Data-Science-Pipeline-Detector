import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import time
import os
import copy
from sklearn.metrics import label_ranking_average_precision_score

class WaveConv1dNet(nn.Module):
    """
    1d convolution network for sound classification
    Modified from Sajjad Abdoli et al. (2019)
    Version without gammatone filterbank initialization
    """

    def __init__(self, num_labels, use_batchnorm=True, use_dropout=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        ## Define Model Weights
        self.conv1 = nn.Conv1d(1, 16, 64, stride=2)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 16, stride=2)
        self.conv4 = nn.Conv1d(64, 128, 8, stride=2)
        self.conv1_bn = nn.BatchNorm1d(16)
        self.conv2_bn = nn.BatchNorm1d(32)
        self.conv3_bn = nn.BatchNorm1d(64)
        self.conv4_bn = nn.BatchNorm1d(128)


        self.fc1 = nn.Linear(128*8 ,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_labels)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc1_dp = nn.Dropout(0.5) # max regularization
        self.fc2_dp = nn.Dropout(0.5)

        self.num_labels = num_labels

    def forward(self, x):
        batch_size = len(x)

        # Number of splitted frames in each sample
        num_frames = [0] + [x[i].size(0) for i in range(batch_size)]

        # Indices breakpoints for the samples
        idx_breaks = np.cumsum(num_frames)

        # Stack all samples together for better parallelism
        x = torch.cat(x, dim=0)

        # Add an extra dimension to fit wwith the shape requirement of Conv1d
        #print(np.prod(x.size()[1:]))
        x = x.view(-1, 1, int(np.prod(x.size()[1:])))


        # Hierachical 1d convolutions
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.conv1_bn(x)
        x = F.max_pool1d(F.relu(x), 8, stride=8)

        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.conv2_bn(x)
        x = F.max_pool1d(F.relu(x), 8, stride=8)

        x = self.conv3(x)
        if self.use_batchnorm:
            x = self.conv3_bn(x)
        x = F.relu(x)

        x = self.conv4(x)
        if self.use_batchnorm:
            x = self.conv4_bn(x)
        x = F.relu(x)

        # Flatten the activation
        x = x.view(-1, np.prod(x.size()[1:]))

        # Affine Layer
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.fc1_bn(x)
        x = self.fc1_dp(F.relu(x))

        x = self.fc2(x)
        if self.use_batchnorm:
            x = self.fc2_bn(x)
        x = self.fc2_dp(F.relu(x))

        x = torch.sigmoid(self.fc3(x))

        # Initialize output
        out = torch.zeros(batch_size, self.num_labels)
        for i in range(1, len(idx_breaks)):
            start_idx = idx_breaks[i-1]
            end_idx = idx_breaks[i]

            # Use the average of probabilities as the final class probabilities
            # An alternative is majority vote
            out[i-1] = torch.mean(x[start_idx:end_idx], dim=0)

        return out


class WaveConv1dNetDeep(nn.Module):
    """
    1d convolution network for sound classification
    Modified from Sajjad Abdoli et al. (2019)
    Version without gammatone filterbank initialization
    Deeper version + batchnorm layers
    """

    def __init__(self, num_labels, use_batchnorm=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        ## Define Model Weights
        self.conv1 = nn.Conv1d(1, 16, 128, stride=2) # 7937, pooled (4) => 1984
        self.conv2 = nn.Conv1d(16, 32, 64, stride=2) # 961, pooled (4) => 240
        self.conv3 = nn.Conv1d(32, 64, 32, stride=2) # 105, pooled (2) => 52
        self.conv4 = nn.Conv1d(64, 128, 32, stride=1) # 21
        self.conv5 = nn.Conv1d(128, 196, 16, stride=1) # 6 
        self.conv1_bn = nn.BatchNorm1d(16)
        self.conv2_bn = nn.BatchNorm1d(32)
        self.conv3_bn = nn.BatchNorm1d(64)
        self.conv4_bn = nn.BatchNorm1d(128)
        self.conv5_bn = nn.BatchNorm1d(196)

  

        self.fc1 = nn.Linear(196*6 ,200)
        self.fc2 = nn.Linear(200, 128)
        self.fc3 = nn.Linear(128, 100)
        self.fc4 = nn.Linear(100, num_labels)
        self.fc1_bn = nn.BatchNorm1d(200)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3_bn = nn.BatchNorm1d(100)

        self.num_labels = num_labels

    def forward(self, x):
        batch_size = len(x)
        
        # Number of splitted frames in each sample
        num_frames = [0] + [x[i].size(0) for i in range(batch_size)]
        
        # Indices breakpoints for the samples
        idx_breaks = np.cumsum(num_frames)
        
        # Stack all samples together for better parallelism
        x = torch.cat(x, dim=0)
        
        # Add an extra dimension to fit with the shape requirement of Conv1d
        #print(np.prod(x.size()[1:]))
        x = x.view(-1, 1, int(np.prod(x.size()[1:])))


        # Hierachical 1d convolutions
        x = F.relu(self.conv1(x))
        if self.use_batchnorm:
            x = self.conv1_bn(x)
        x = F.max_pool1d(x, 4, stride=4)

        x = F.relu(self.conv2(x))
        if self.use_batchnorm:
            x = self.conv2_bn(x)
        x = F.max_pool1d(x, 4, stride=4)

        x = F.relu(self.conv3(x))
        if self.use_batchnorm:
            x = self.conv3_bn(x)
        x = F.max_pool1d(x, 2, stride=2)

        x = F.relu(self.conv4(x))
        if self.use_batchnorm:
            x = self.conv4_bn(x)

        x = F.relu(self.conv5(x))
        if self.use_batchnorm:
            x = self.conv5_bn(x)
       
        # Flatten the activation
        x = x.view(-1, np.prod(x.size()[1:]))
        
        # Affine Layer
        x = F.relu(self.fc1(x))
        if self.use_batchnorm:
            x = self.fc1_bn(x)

        x = F.relu(self.fc2(x))
        if self.use_batchnorm:
            x = self.fc2_bn(x)

        x = F.relu(self.fc3(x))
        if self.use_batchnorm:
            x = self.fc3_bn(x)

        x = torch.sigmoid(self.fc4(x))
        
        # Initialize output
        out = torch.zeros(batch_size, self.num_labels)
        for i in range(1, len(idx_breaks)):
            start_idx = idx_breaks[i-1]
            end_idx = idx_breaks[i]
            
            # Use the average of probabilities as the final class probabilities
            # An alternative is majority vote
            out[i-1] = torch.mean(x[start_idx:end_idx], dim=0)

        return out

"""
Function for training model
Args:
    model: the WaveConv1dNet model for training
    dataloader: the WaveSoundDataLoader instance for data batch generation
    criterion: the loss function
    optimizer: the PyTorch optimizer
    scheduler: the learning rate scheduler
    num_epoch: number of epochs used in training
"""
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time() # Record start time
    
    model = model.to(device)

    # Initialize the best model and the best accuracy
    best_model = copy.deepcopy(model.state_dict())
    best_lrap = 0.0

    for epoch in range(num_epochs):
        print('Epoch {0}/{1}'.format(epoch + 1, num_epochs))
        
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            # Initialize statistic variables
            data_size = 0
            running_loss = 0.0
            running_lrap = 0
            for inputs, labels in dataloaders[phase]:
                
                # Transfer x_train and y_train to the correct device
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].float().to(device)
                labels = torch.from_numpy(labels)
                labels = labels.to(torch.device('cpu')).float()

                data_size += len(inputs)
                
                # Clear accumulated gradients in optimizer
                optimizer.zero_grad()
                
                # Only enable gradient tracking in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    #print(outputs)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * len(inputs)
                
                # Accumulate LRAP
                running_lrap += len(inputs) * label_ranking_average_precision_score(labels, outputs.detach().numpy())


            epoch_loss = running_loss / data_size
            epoch_lrap = running_lrap/ data_size
            print('{} Loss: {:.4f} LRAP: {:.4f}'.format(phase, epoch_loss, epoch_lrap))

            if phase == 'val':
                print('{} Loss: {:.4f} LRAP: {:.4f}'.format(phase, epoch_loss, epoch_lrap))

            if phase == 'val' and epoch_lrap > best_lrap:
                # Update the best model and best accuracy
                best_lrap = epoch_lrap
                best_model = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val LRAP: {:4f}'.format(best_lrap))
    
    # Return the model that is the best model in the training history
    model.load_state_dict(best_model)
    return model

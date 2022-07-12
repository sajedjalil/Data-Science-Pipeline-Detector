import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as torch_init
from sklearn import preprocessing
from torch.utils.data import Dataset

# basic setting
gpus = [0]
seed = 1
batch_size = 256
feature_size = 170

max_epoch = 30
changeLR_list = [25]

lr = 0.0001
weight_decay = 0.0001
momentum = 0.9
dropout = 0.5
inp_dir = '../input/pubg-finish-placement-prediction/'


def weights_init_random(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def feature_engineering(inp_dir, is_train):
    if is_train:
        print("processing train_V2.csv")
        df = pd.read_csv(inp_dir + 'train_V2.csv')

        # Only take the samples with matches that have more than 1 player
        # there are matches with no players or just one player ( those samples could affect our model badly)
        df = df[df['maxPlace'] > 1]

        # When this function is used for the test data, load test_V2.csv :
    else:
        print("processing test_V2.csv")
        df = pd.read_csv(inp_dir + 'test_V2.csv')

        # Make a new feature indecating the total distance a player cut :

    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]

    # Process the 'rankPoints' feature by replacing any value of (-1) to be (0) :
    df['rankPoints'] = np.where(df['rankPoints'] <= 0, 0, df['rankPoints'])

    target = 'winPlacePerc'
    # Get a list of the features to be used
    features = list(df.columns)

    # Remove some features from the features list :
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchDuration")
    features.remove("matchType")

    y = None

    # If we are processing the training data, process the target
    # (group the data by the match and the group then take the mean of the target)
    if is_train:
        y = np.array(df.groupby(['matchId', 'groupId'])[target].agg('mean'), dtype=np.float64)
        # Remove the target from the features list :
        features.remove(target)

    # Make new features indicating the mean of the features ( grouped by match and group ) :
    print("get group mean feature")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('mean')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    # If we are processing the training data let df_out = the grouped  'matchId' and 'groupId'
    if is_train:
        df_out = agg.reset_index()[['matchId', 'groupId']]
    # If we are processing the test data let df_out = 'matchId' and 'groupId' without grouping
    else:
        df_out = df[['matchId', 'groupId']]

    # Merge agg and agg_rank (that we got before) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    # Make new features indicating the max value of the features for each group ( grouped by match )
    print("get group max feature")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('max')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    # Merge the new (agg and agg_rank) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    # Make new features indicating the minimum value of the features for each group ( grouped by match )
    print("get group min feature")
    agg = df.groupby(['matchId', 'groupId'])[features].agg('min')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    # Merge the new (agg and agg_rank) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    # Make new features indicating the number of players in each group ( grouped by match )
    print("get group size feature")
    agg = df.groupby(['matchId', 'groupId']).size().reset_index(name='group_size')

    # Merge the group_size feature with df_out :
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    # Make new features indicating the mean value of each features for each match :
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()

    # Merge the new agg with df_out :
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])

    # Make new features indicating the number of groups in each match :
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')

    # Merge the match_size feature with df_out :
    df_out = df_out.merge(agg, how='left', on=['matchId'])

    # Drop matchId and groupId
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    # X is the output dataset (without the target) and y is the target :
    X = np.array(df_out, dtype=np.float64)

    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y


def Step_decay_lr(epoch):
    lr_list = []
    current_epoch = epoch + 1
    for i in range(0, len(changeLR_list) + 1):
        lr_list.append(lr * (0.1 ** i))

    lr_range = changeLR_list.copy()
    lr_range.insert(0, 0)
    lr_range.append(max_epoch + 1)

    if len(changeLR_list) != 0:
        for i in range(0, len(lr_range) - 1):
            if lr_range[i + 1] >= current_epoch > lr_range[i]:
                lr_step = i
                break

    current_lr = lr_list[lr_step]
    return current_lr


def smooth_l1_loss(input, target, beta=1. / 10, reduction='none'):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    ret = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


class DataLoader(Dataset):
    def __init__(self, x, y):
        self.train_data = x
        self.train_lbl = y

    def __len__(self):
        return int(len(self.train_data))

    def __getitem__(self, idx):
        sample = dict()
        feat = self.train_data[idx]
        labs = self.train_lbl[idx]
        sample['data'] = feat
        sample['labels'] = labs

        return sample


class Model(nn.Module):
    def __init__(self, feature_size, dropout):
        super().__init__()
        self.n_in = feature_size
        self.dropout = dropout

        self.Basic_Modules = nn.ModuleList((
            Basic_Module(self.n_in, 128),
            Basic_Module(128, 128),
            Basic_Module(128, 128),
            Basic_Module(128, 64),
            Basic_Module(64, 64),
            Basic_Module(64, 64),
        ))

        self.post_layer = nn.Linear(64, 1)
        self.data_bn = nn.BatchNorm1d(self.n_in)

        self.apply(weights_init_random)

    def forward(self, x):
        x = self.data_bn(x)
        for layer in self.Basic_Modules:
            x = layer(x)
        x = self.post_layer(x)
        return x


class Basic_Module(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Res_Module(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layer(x) + x
        x = self.relu(x)
        return x


torch.manual_seed(seed)
device = torch.device('cuda:' + str(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 else 'cpu')

# read data
# train data
x_train, y_train = feature_engineering(inp_dir, True)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit(x_train)
scaler.transform(x_train)
y_train = y_train * 2 - 1
# test data
x_test, _ = feature_engineering(inp_dir, False)
scaler.transform(x_test)
np.clip(x_test, out=x_test, a_min=-1, a_max=1)

train_data_loader = torch.utils.data.DataLoader(DataLoader(x_train, y_train),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2,
                                                drop_last=False)

model = Model(feature_size, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.99], weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

model.train(True)
epoch_range = range(max_epoch)
for epoch in epoch_range:
    # Training
    acc, num_sample = 0, 0
    for num, sample in enumerate(train_data_loader):

        current_lr = Step_decay_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        x = sample['data'].numpy()
        y = sample['labels'].numpy()
        # Using GPU
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)

        # Calculating Output
        out = model(x)
        out = torch.squeeze(out, -1)
        loss = smooth_l1_loss(out, y, reduction='mean')

        # BP Method
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: {}/{}, Batch: {}, Lr: {:.6f}, Loss: {:.4f}'.format(epoch + 1,
                                                                         max_epoch,
                                                                         num + 1,
                                                                         current_lr,
                                                                         loss))

# Testing
model.eval()
x_test_torch = torch.from_numpy(x_test).float().to(device)

with torch.no_grad():
    pred = model(x_test_torch)
    pred = np.squeeze(pred.cpu().data.numpy(), axis=-1)

pred = (pred + 1) / 2
df_test = pd.read_csv(inp_dir + 'test_V2.csv')

print("fix winPlacePerc")
for i in range(len(df_test)):
    winPlacePerc = pred[i]
    maxPlace = int(df_test.iloc[i]['maxPlace'])
    if maxPlace == 0:
        winPlacePerc = 0.0
    elif maxPlace == 1:
        winPlacePerc = 1.0
    else:
        gap = 1.0 / (maxPlace - 1)
        winPlacePerc = round(winPlacePerc / gap) * gap

    if winPlacePerc < 0: winPlacePerc = 0.0
    if winPlacePerc > 1: winPlacePerc = 1.0
    pred[i] = winPlacePerc

    if (i + 1) % 100000 == 0:
        print(i, flush=True, end=" ")

df_test['winPlacePerc'] = pred

submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)
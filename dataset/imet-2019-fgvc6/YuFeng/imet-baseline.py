import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

num_epochs = 9
num_classes = 1103
batch_size = 128
learning_rate = 0.002
threshold = 0.1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# labels = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")
train = pd.read_csv("../input/imet-2019-fgvc6/train.csv")
sample = pd.read_csv("../input/imet-2019-fgvc6/sample_submission.csv")
train_path = "../input/imet-2019-fgvc6/train/"
test_path = '../input/imet-2019-fgvc6/test/'
y = train.attribute_ids.map(lambda x: x.split()).values
train['y'] = y
test = sample['id']

train, val = train_test_split(train, test_size=0.1)

class MyDataSet(Dataset):
    def __init__(self, df_data, mode, data_dir = './', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if(self.mode == 'train'):
            img_name = self.df[index][0]
        else:
            img_name = self.df[index]
        img_path = os.path.join(self.data_dir, img_name+'.png')
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        if(self.mode == 'train'):
            label = self.df[index][2]
            label_tensor = np.zeros((1, 1103))
            for i in label:
                label_tensor[0, int(i)] = 1     
            label_tensor = label_tensor.flatten()
            label_tensor = torch.from_numpy(label_tensor).float()
            return image,label_tensor
        else: 
            return image


trans_train = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
trans_valid = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

dataset_train = MyDataSet(df_data=train, mode='train', data_dir=train_path, transform=trans_train)
dataset_valid = MyDataSet(df_data=val, mode='train', data_dir=train_path, transform=trans_valid)
dataset_test = MyDataSet(df_data=test, mode='test', data_dir=test_path, transform=trans_valid)

loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
loader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0)
loader_test = DataLoader(dataset=dataset_test, batch_size=128, shuffle=False, num_workers=0)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(512*1*1, 1103)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        x = self.pool(F.leaky_relu(self.bn5(self.conv5(x))))
        x = self.avg(x)
        x = x.view(-1, 512*1*1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.BCELoss(reduction="mean")
print("**********Loading model.***********")
model.load_state_dict(torch.load("../input/imet-model-simple-a/best_model.pth"))
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

total_step = len(loader_train)
min_val_loss = np.inf
print("*********Strat to train.***********")
for epoch in range(num_epochs):
    avg_loss = 0.0
    for i, (images, labels) in enumerate(loader_train):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    for i, (images, labels) in enumerate(loader_valid):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        avg_loss += loss.item() / len(loader_valid)
    if min_val_loss > avg_loss:
        best_model = model.state_dict()
        min_val_loss = avg_loss
    print('epoch:[{}], valid_loss:[{}]'.format(epoch+1, avg_loss))
print("*************Finish.***************")
torch.save(best_model, "best_model.pth")
model.load_state_dict(best_model)
model.eval()
preds = np.zeros((len(test), 1103))
for batch_i, data in enumerate(loader_test):
    with torch.no_grad():
        data = data.to(device)
        y_pred = model(data).detach()
        preds[batch_i*128: batch_i*128+len(data)] = y_pred.cpu().numpy()
preds = (preds > threshold).astype(int)
prediction = []
for i in range(preds.shape[0]):
    pred1 = np.argwhere(preds[i]==1.0).reshape(-1).tolist()
    pred_str = " ".join(list(map(str, pred1)))
    prediction.append(pred_str)
sample.attribute_ids = prediction
sample.to_csv("submission.csv", index=False)
sample.head()

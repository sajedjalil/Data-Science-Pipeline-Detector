# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, datasets

from PIL import Image
import matplotlib.pyplot as plt

TRAINSET_PATH = "/kaggle/input/Kannada-MNIST/train.csv"
VALSET_PATH = "/kaggle/input/Kannada-MNIST/Dig-MNIST.csv"
TESTSET_PATH = "/kaggle/input/Kannada-MNIST/test.csv"
OUTPUT_PATH = "/kaggle/working/submission.csv"
MODEL_SAVE_PATH = "/kaggle/working/kannada.pth"
BATCH_SIZE = 1024
Commit = True
EPOCHES = 100
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print("CUDA STATUE: {}".format(DEVICE))

train_transform = transforms.Compose([
    transforms.RandomAffine(10, (0.25, 0.25), (0.8, 1.2), 5),
    transforms.ToTensor()
])

predict_transform = transforms.Compose([
    transforms.ToTensor()
])


class Kannada(Dataset):
    def __init__(self, np_array, transform=None):
        self.np_array = np_array
        self.transform = transform

    def __len__(self):
        return self.np_array.shape[0]

    def __getitem__(self, item):
        label = self.np_array[item][0]
        np_img = self.np_array[item][1:].reshape((28, 28)).astype('uint8')
        img = Image.fromarray(np_img).convert("L")

        if self.transform:
            img = self.transform(img)
        return img, label


class Sq_Ex_Block(nn.Module):
    def __init__(self, in_ch, r):
        super(Sq_Ex_Block, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // r, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        x = x.mul(se_weight)
        return x


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.view(*(x.shape[:-2]), -1).mean(-1)


class KannadaNet(nn.Module):
    def __init__(self):
        super(KannadaNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # 28 x 28 x 1 => 28 x 28 x 64
            nn.BatchNorm2d(64, 1e-3, 1e-2),
            nn.LeakyReLU(0.1, True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # 28 x 28 x 64 => 28 x 28 x 64
            nn.BatchNorm2d(64, 1e-3, 1e-2),
            nn.LeakyReLU(0.1, True)
        )
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # 28 x 28 x 64 => 28 x 28 x 64
            nn.BatchNorm2d(64, 1e-3, 1e-2),
            nn.LeakyReLU(0.1, True)
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),  # 28 x 28 x 64 => 14 x 14 x 64
            nn.Dropout2d(0.4)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # 14 x 14 x 64 => 14 x 14 x 128
            nn.BatchNorm2d(128, 1e-3, 1e-2),
            nn.LeakyReLU(0.1, True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # 14 x 14 x 128 => 14 x 14 x 128
            nn.BatchNorm2d(128, 1e-3, 1e-2),
            nn.LeakyReLU(0.1, True)
        )
        self.layer5_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # 14 x 14 x 128 => 14 x 14 x 128
            nn.BatchNorm2d(128, 1e-3, 1e-2),
            nn.LeakyReLU(0.1, True)
        )
        self.layer6 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),  # 14 x 14 x 128 => 7 x 7 x 128
            nn.Dropout2d(0.4)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # 7 x 7 x 128 => 7 x 7 x 256
            nn.BatchNorm2d(256, 1e-3, 1e-2),
            nn.LeakyReLU(0.1, True)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # 7 x 7 x 256 => 7 x 7 x 256
            nn.BatchNorm2d(256, 1e-3, 1e-2),
            nn.LeakyReLU(0.1, True)
        )
        self.layer9 = nn.Sequential(
            Sq_Ex_Block(in_ch=256, r=8),
            nn.MaxPool2d(2, stride=2),  # 7 x 7 x 256 => 3 x 3 x 256
            nn.Dropout2d(0.4)
        )
        self.dense = nn.Sequential(
            nn.Linear(2304, 256),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(256, 1e-3, 1e-2)
        )
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer5_1(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = x.view(-1, 3 * 3 * 256)
        x = self.dense(x)
        x = self.fc(x)
        return x


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_trainset():
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % labels[j].item() for j in range(4)))


train_csv = pd.read_csv(TRAINSET_PATH)
val_csv = pd.read_csv(VALSET_PATH)
test_csv = pd.read_csv(TESTSET_PATH)

train_np = np.array(train_csv)
val_np = np.array(val_csv)
test_np = np.array(test_csv)

trainset = Kannada(train_np, transform=train_transform)
valset = Kannada(val_np, transform=predict_transform)
testset = Kannada(test_np, transform=predict_transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

kannada_net = KannadaNet()
kannada_net = kannada_net.to(DEVICE)

max_acc = 0
best_model_dict = None

if Commit:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(kannada_net.parameters(), lr=1e-3, alpha=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True)
    for epoch in range(EPOCHES):
        running_loss = 0.0
        kannada_net.train()
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = kannada_net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        kannada_net.eval()
        total = 0
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = kannada_net(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= total
        scheduler.step(val_loss)
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        print('lr = {}'.format(optimizer.param_groups[0]['lr']))
        if optimizer.param_groups[0]['lr'] < 5e-5:
            print("Learning Rate is Smaller than 0.00005, Stoping Trainning")
            break
        if (correct / total) > max_acc:
            max_acc = (correct / total)
            best_model_dict = kannada_net.state_dict()

    print('Trainning Completed')
    # torch.save(best_model_dict, MODEL_SAVE_PATH)
    torch.cuda.empty_cache()

    df = pd.DataFrame()
    df.index.name = 'id'
    kannada_net.load_state_dict(best_model_dict)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = kannada_net(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                df = df.append(pd.Series({"label": predicted[i].item()}, name=labels[i].item()))
    df["label"] = df["label"].astype(int)
    df.to_csv(OUTPUT_PATH)
    print(df.head())
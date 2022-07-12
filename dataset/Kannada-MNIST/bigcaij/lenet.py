#../input/Kannada-MNIST/
import pandas as pd
import numpy as np
import torch.utils.data as data
import torch
import csv
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 11, 5, 1, 2), #padding=2保证输入输出尺寸相同
            nn.ReLU(),      #input_size=(11*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(11*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(11, 22, 5),
            nn.ReLU(),      #input_size=(22*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(22*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(22 * 5 * 5, 440),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(440, 110),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(110, 11)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 超参数设置
EPOCH = 25   #遍历数据集次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.001        #学习率

# 定义数据预处理方式
transform = transforms.ToTensor()

file = '../input/Kannada-MNIST/train.csv'
file1 = '../input/Kannada-MNIST/test.csv'


class MNISTCSVDataset(data.Dataset):

    def __init__(self, csv_file,Train=True,
                 transform=transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,))])
                 ):
        df = pd.read_csv(csv_file)
        self.Train = Train
        if not Train:
            # test data
            self.X = df.values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = torch.from_numpy(df.iloc[:, 0].values)
        else:
            # training data
            self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = torch.from_numpy(df.iloc[:, 0].values)

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])


mydataset = MNISTCSVDataset(file)
myMnist = MNISTCSVDataset(file1)
testloader = torch.utils.data.DataLoader(myMnist, batch_size=BATCH_SIZE, shuffle=True)
trainloader = torch.utils.data.DataLoader(mydataset, batch_size=BATCH_SIZE, shuffle=True)

# 定义损失函数loss function 和优化方式（采用SGD）
net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# 训练
if __name__ == "__main__":

    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                sum_loss = 0.0

    f = open('submission.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'label'])

    # 输出结果文件
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            for i in range(labels.size(0)):
                csv_writer.writerow([i, predicted[i].item()])
    f.close()

sub = pd.read_csv('submission.csv')
sub.to_csv('submission.csv', index=False)

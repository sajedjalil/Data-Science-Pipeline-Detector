import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2
from torch.utils.data import Dataset, DataLoader
import glob

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# 디렉토리 설정들 
input_dir = "/kaggle/input/histopathologic-cancer-detection"
train_dir = input_dir+"/train"
test_dir = input_dir+"/test"
#제출 파일은 submission.csv에 id , labels를 column으로 두면되는 일
# 제출하는 건 csv파일을 직접 사이트에 올려야 되고
# 채점은 Kaggle에서 그 csv 파일 가지고 알아서 채점하게 됨.
submission_file = "/kaggle/working/sub2mission.csv"


# train labels column = {"id", "label"}
train_labels = pd.read_csv(input_dir + "/train_labels.csv")

# 라벨만 따로 리스트, id들만 따로 리스트화
labels = train_labels.label.tolist()
ids = train_labels.id.tolist()

transformation = transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])


#Dataset을 상속하여 자기가 직접 알맞게 만들어서 돌릴 수 있도록 함.
class CustomDataset(Dataset):
    def __init__(self, ids, labels, img_dir, transform=None):
        self.ids = ids
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.ids) 

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_name = img_id+'.tif'
        img = cv2.imread(os.path.join(self.img_dir, img_name)) #OpenCV 라이브러리를 이용하여 이미지를 읽기 "{파일이름}.tif"
        if self.transform != None:
            img = self.transform(img) # transform 설정
        return img, self.labels[idx], img_id # img와 , 라벨값 , img_id 값을 리턴, img_id는 트레이닝 말고 submission으로 활용
    """ 
    주의 img_id는 내가 submission 할 때 이미지 id를 받을 수 있는 방법이 없어서 땜빵처리 한거여서 
    정확도가 loss값에 비해 너무 낮게 나올 경우 이것에 문제가 있는 거일 수 있음
    
    """


trainset = CustomDataset(ids,labels,train_dir,transform = transformation) #
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

#validset = CustomDataset(ids,labels,"{valid_dir}",transform = transformation) #나중에 해보자
#validloader = torch.utils.data.DataLoader(validset, batch_size=4, shuffle=True, num_workers=4)

test_list = glob.glob(os.path.join(test_dir+"/",'*.tif'))
test_ids = list(map(lambda x: x.split('/')[5].split(".")[0], test_list)) #test파일의 id는 test파일 폴더에서 직접 파일이름으로
testset = CustomDataset(test_ids,labels,test_dir,transform = transformation)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)


import torch.nn as nn
import torch.nn.functional as F

# 이전에 정의한 network 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3) # input이 96*96*3 으로 들어옴
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*21*21, 640)
        self.fc2 = nn.Linear(640, 120)
        self.fc3 = nn.Linear(120, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*21*21) #-1이면 다른 차원으로부터 해당 값을 유추
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 


net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 학습을 수행할때는 GPU로 전환
net.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

""" 
주의
시간이 쓸데없이 ㅈㄴ 걸리므로 트레이닝 데이터중 일부만 보는 식으로 해야할듯.
어느정도 이상의 반복을 했을 때 break 한다는 식으로?
"""

for epoch in range(1): 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameters gradients
        optimizer.zero_grad()

        # forward + backward +optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:  # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
print('Finished Training')

device = torch.device("cpu") # test는 cpu로 돌려야함 GPU로 못돌림
net.to(device)

"""
test는 모든 파일 test해야 하는 걸로 알아 
kaggle이 원하는 row의 수는 57458개 (참고)
"""
print("Start Testing")
with torch.no_grad(): # test로 학습할 이유는 없으므로 no_grad로
    submission = pd.DataFrame();
    count = 0
    for data in testloader:
        inputs, labels, img_ids = data[0].to(device),data[1].to(device), data[2]
        outputs = net(inputs)
        _, predict = torch.max(outputs.data,1)
        for i in range(len(img_ids)):  # batch_size  
            test_df = pd.DataFrame({'id': [img_ids[i]],'label':[predict[i].item()]});
            submission = pd.concat([submission,test_df])
            count+=1
            if count%1000 == 0:
                print(str(count) +" files completed")
    print("finished testing " + str(count) +" files")
    print("Start submit")
    submission.to_csv(submission_file,index=False,header=True)    
    print("End submission")
    
    

# Output을 sample_submission.csv에 쓰는 것이 필요함

# Any results you write to the current directory are saved as output. 

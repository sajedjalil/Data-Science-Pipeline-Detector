import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class DatasetMNIST(Dataset):

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be careful for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class SubmissionDataSet(Dataset):

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))

        if self.transform is not None:
            image = self.transform(image)

        return image
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = 0.2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv1_bn = nn.BatchNorm2d(num_features=64)

        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv1_1_bn = nn.BatchNorm2d(num_features=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.d2_1 = nn.Dropout2d(p=self.dropout)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.d2_2 = nn.Dropout2d(p=self.dropout)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.d2_3 = nn.Dropout2d(p=self.dropout)

        # 4608 input features, 256 output features (see sizing flow below)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)  
        self.d1_1 = nn.Dropout(p=self.dropout)
        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.d1_2 = nn.Dropout(p=self.dropout)
        self.fc3 = nn.Linear(in_features=256, out_features=128) 
        self.d1_3 = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(in_features=128, out_features=10) 

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv1_1(x)
        x = self.conv1_1_bn(x)
        x = F.relu(x)

        x = self.d2_1(x)
        x = self.pool1(x)  # Size changes from (18, 28, 28) to (18, 14, 14)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.d2_2(x)
        x = self.pool2(x)  # Size changes from (18, 14, 14) to (18, 7, 7)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.d2_3(x)
        x = self.pool3(x)  # Size changes from (18, 7, 7) to (18, 3, 3)

        x = x.view(-1, 256 * 3 * 3)

        x = F.relu(self.fc1(x))
        x = self.d1_1(x)

        x = F.relu(self.fc2(x))
        x = self.d1_2(x)

        x = F.relu(self.fc3(x))
        x = self.d1_3(x)

        x = self.out(x)
        return F.log_softmax(x, dim=1)

def load_data():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Training datase
    train_loader = torch.utils.data.DataLoader(DatasetMNIST('../input/Kannada-MNIST/train.csv', transform=transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.RandomCrop(28),
                           transforms.RandomAffine(degrees=(-5, 10), translate=(0.05, 0.05), shear=1.5),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=512, pin_memory=False, shuffle=True, num_workers=8)
# Test dataset
    test_loader = torch.utils.data.DataLoader(DatasetMNIST('../input/Kannada-MNIST/Dig-MNIST.csv', transform=transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=512, pin_memory=False, shuffle=False, num_workers=8)

    sub_loader = torch.utils.data.DataLoader(SubmissionDataSet('../input/Kannada-MNIST/test.csv', transform=transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=1, pin_memory=False, shuffle=False, num_workers=0)
    return train_loader, test_loader, sub_loader, device

def implement_the_model():
    train_loader, test_loader, sub_loader, device = load_data()

#     model = Net().to(device)

#     if torch.cuda.is_available():
#         model.cuda()
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)
#     for epoch in range(1, 160 + 1):
#         train(epoch, train_loader, device, model, optimizer)
#         test(test_loader, device, model)
        
    model = torch.load('../input/cnn-with-spatial-transform-network-v4/model_MNIST_Kannada_cnn_spatial_v4.pt', map_location='cpu')
    model = model.to(device)
    model.eval()

    submit = {
        'id': [],
        'label': []
    }

    for batch_idx, data in enumerate(sub_loader):
        data = data.to(device)
        output = model(data)
        pred = torch.argmax(output, dim=1)
        pred = int(pred.cpu().numpy())
        submit['label'].append(pred)
        submit['id'].append(batch_idx)

    submit_df = pd.DataFrame(submit)
    submit_df.head()
    submit_df.to_csv('submission.csv', index=False)


def train(epoch, train_loader, device, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

#         lambda2 = 0.000015

#         all_linear1_params = torch.cat([x.view(-1) for x in model.layer1.parameters()])
#         all_linear2_params = torch.cat([x.view(-1) for x in model.layer2.parameters()])
#         all_linear3_params = torch.cat([x.view(-1) for x in model.layer3.parameters()])
#         all_conv1_params = torch.cat([x.view(-1) for x in model.layer4.parameters()])
#         all_conv2_params = torch.cat([x.view(-1) for x in model.conv1.parameters()])
#         all_conv3_params = torch.cat([x.view(-1) for x in model.fc.parameters()])
#         l2_regularization = lambda2 * (torch.norm(all_linear1_params, 2) + \
#                                        torch.norm(all_linear2_params, 2) + \
#                                        torch.norm(all_linear3_params, 2) + \
#                                        torch.norm(all_conv1_params, 2) + \
#                                        torch.norm(all_conv2_params, 2) + \
#                                        torch.norm(all_conv3_params, 2))

        loss = F.cross_entropy(output, target) #+ l2_regularization
        
        loss.backward()
        optimizer.step()
        if batch_idx % 115 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(test_loader, device, model):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, size_average=False).item()
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

        
implement_the_model()
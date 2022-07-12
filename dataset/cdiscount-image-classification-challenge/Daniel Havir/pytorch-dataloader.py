import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import cv2

transform = transforms.Compose([
	transforms.ToPILImage(),
	### other PyTorch transforms
	transforms.ToTensor()
])

class CDiscountDataset(torch.utils.data.Dataset):
	'''
		Custom Dataset object for the CDiscount competition
		Parameters:
			root_dir - directory including category folders with images

		Example:
		images/
			1000001859/
				26_0.jpg
				26_1.jpg
				...
			1000004141/
				...
			...
	'''
	
	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		self.categories = sorted(os.listdir(root_dir))
		self.cat2idx = dict(zip(self.categories, range(len(self.categories))))
		self.idx2cat = dict(zip(self.cat2idx.values(), self.cat2idx.keys()))
		self.files = []
		for (dirpath, dirnames, filenames) in os.walk(self.root_dir):
			for f in filenames:
				if f.endswith('.jpg'):
					o = {}
					o['img_path'] = dirpath + '/' + f
					o['category'] = self.cat2idx[dirpath[dirpath.find('/')+1:]]
					self.files.append(o)
		self.transform = transform
	
	def __len__(self):
		return len(self.files)
	
	def __getitem__(self, idx):
		img_path = self.files[idx]['img_path']
		category = self.files[idx]['category']
		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.transform:
			image = self.transform(image)
			
		return {'image': image, 'category': category}

dset = CDiscountDataset('images', transform=transform)
print('######### Dataset class created #########')
print('Number of images: ', len(dset))
print('Number of categories: ', len(dset.categories))
print('Sample image shape: ', dset[0]['image'].shape, end='\n\n')


dataloader = torch.utils.data.DataLoader(dset, batch_size=4, shuffle=True, num_workers=4)

### Define your network below
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(180**2 * 3, 84)
        self.fc2 = nn.Linear(84, 36)

    def forward(self, x):
        x = x.view(-1, 180**2 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
print('######### Network created #########')
print('Architecture:\n', net)

### Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):

    running_loss = 0.0
    examples = 0
    for i, data in enumerate(dataloader, 0):
        # Get the inputs
        inputs, labels = data['image'], data['category']

        # Wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.data[0]
        examples += 4
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / examples))

print('Finished Training')

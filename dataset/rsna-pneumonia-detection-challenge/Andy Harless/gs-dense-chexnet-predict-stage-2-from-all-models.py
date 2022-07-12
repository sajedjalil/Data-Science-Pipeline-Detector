weight_files = [
    'loss.best.pth_fold0.tar',
    'loss.best.pth_fold0_1st_run.tar',
    'loss.best.pth_fold0_auc91.tar',
    'loss.best.pth_fold0_for_combined_folds.tar',
    'loss.best.pth_fold1.tar',
    'loss.best.pth_fold2.tar',
    'loss.best.pth_fold3.tar',
    'loss.best.pth_fold4.tar',
]

import os
import PIL
import time
import re
import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as torchDataset
import torchvision as tv
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from subprocess import call

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import shutil
from sklearn.metrics import roc_auc_score
import pydicom
from sklearn.metrics import confusion_matrix

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.catch_warnings()

# KAGGLE kernel paths
datapath_orig = '/kaggle/input/'


# get test set patient IDs
pIds_test = pd.read_csv('/kaggle/input/stage_2_sample_submission.csv').patientId.values



class PneumoniaDataset(torchDataset):
    """
        Pneumonia dataset that contains radiograph lung images as .dcm. 
        Each patient has one image named patientId.dcm.
    """

    def __init__(self, root, subset, pIds, predict, targets, transform=None):
        """
        :param root: it has to be a path to the folder that contains the dataset folders
        :param subset: 'train' or 'test'
        :param pIds: list of patient IDs
        :param predict: boolean, true if network is predicting or false if network is training
        :param targets: a {patientId : target} dictionary (ex: {'pId': 0})
        :param transform: transformations applied to the images 
        """

        # initialize variables
        self.root = os.path.expanduser(root)
        self.subset = subset
        if self.subset not in ['train', 'test']:
            raise RuntimeError('Invalid subset ' + self.subset + ', it must be one of: \'train\' or \'test\'')
        self.pIds = pIds
        self.predict = predict
        self.targets = targets
        self.transform = transform

        self.data_path = self.root + 'stage_2_'+self.subset+'_images/'
        
    def __getitem__(self, index):
        """
        :param index:
        :return: tuple (img, target) with the input data and its target mask
        """
        # get the corresponding pId
        pId = self.pIds[index]
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.data_path, pId+'.dcm')).pixel_array
        # check if image is square
        if (img.shape[0]!=img.shape[1]):
            raise RuntimeError('Image shape {} should be square.'.format(img.shape))
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        # create a 3-channel image (as expected by pretrained resnet) using the original grey scale image for channel 0,
        # the image squared as channel 1, and the square root of the image as channel 2
        img = np.concatenate([img, 
                              (np.rint((img**2.0)/255.)).astype('uint8'), 
                              (np.rint((img**0.5)*(255.**0.5))).astype('uint8')], 
                             axis=2)
        # apply transforms to image
        if self.transform is not None:
            img = self.transform(img)
        
        if not self.predict:
            target = self.targets[pId] 
            target = np.expand_dims(float(target), -1) # use this line in case of binary cross entropy loss
            return img, target # comment above line in case of softmax + nllloss
        else: 
            return img, pId

    def __len__(self):
        return len(self.pIds)
        
        
        
# manual model parameters
image_shape = 256 
batch_size = 20

# define (only) transformation for validation and test set
transform = tv.transforms.Compose([tv.transforms.ToPILImage(), 
                                   tv.transforms.Resize(image_shape), 
                                   tv.transforms.ToTensor()
                                  ])

# create datasets
dataset_test = PneumoniaDataset(root=datapath_orig, subset='test', pIds=pIds_test, predict=True, 
                                targets=None, transform=transform)

# define the dataloaders with the previous dataset
loader_test = DataLoader(dataset=dataset_test,
                         batch_size=batch_size,
                         shuffle=False,
                         pin_memory=True)



# redefine densenet model to allow arbitrary size for input image

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121_m(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'], progress=False)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model






class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        
        # 2D adaptive average pooling to allow any image input shape (ok >224)
        self.aap2d = nn.AdaptiveAvgPool2d(1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.aap2d(out).view(features.size(0), -1) # replacing avgpool2d below
        ### out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1) works only with 224x224 images
        out = self.classifier(out)
        return out
    
# define network architecture based on densenet121
class DenseNet121(nn.Module):
    def __init__(self, n_classes, pretrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = densenet121_m(pretrained=pretrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, n_classes), nn.Sigmoid())
    def forward(self, x):
        x = self.densenet121(x)
        return x
    

# define a running average class
class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
        
        
# define some metrics
def accuracy(outputs, true_labels, threshold=0.5):
    """
    Compute the accuracy, given the CNN outputs and true labels.
    Returns: (float) accuracy in [0,1]
    """
    predicted_probas = outputs.flatten() # apply softmax to tensor then convert to numpy array and get results for class 1
    predicted_labels = np.zeros(predicted_probas.shape) 
    predicted_labels[predicted_probas>threshold] = 1 # collapse probabilities to 0 and 1 according to threshold
    true_labels = true_labels.flatten() # convert tensor to numpy array
    accuracy = np.sum(predicted_labels==true_labels)/float(true_labels.size) # calculate raw accuracy (rate of true positives)
    return accuracy

def auc(outputs, true_labels):
    """
    Compute the roc_auc score, given the CNN outputs and true labels.
    Returns: (float) auc score
    """
    true_labels = true_labels.flatten()
    if sum(true_labels) == 0: # in this case the roc_auc score is not defined
        return np.nan
    else:
        predicted_probas = outputs.flatten()
        return roc_auc_score(true_labels, predicted_probas)
        
        
        
def predict(model, dataloader):
    """Generate predictions from the model.
    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    """

    # set model to evaluation mode
    model.eval()

    predictions = {}

    for i, (test_batch, pIds) in enumerate(dataloader):
        # Convert torch tensor to Variable
        test_batch = Variable(test_batch).cuda()
        # compute output
        output_batch = model(test_batch)
        # send to cpu and numpy array
        predictions_probas = output_batch.data.cpu().numpy().flatten()
        # iterate over individual patient IDs in batch and fill the dictionary
        for pId, pp in zip(pIds, predictions_probas):
            predictions[pId] = pp

    return predictions
    
    
threshold=.5
    
for wfile in weight_files:

    model = DenseNet121(n_classes=1, pretrained=True).cuda()
    
    file_url = 'http://andy.harless.us/rsnaweights/' + wfile
    call( ['wget', '-q', file_url] )
    call( ['md5sum', wfile] )
    
    best_model = torch.nn.DataParallel(model).cuda()
    best_model.load_state_dict(torch.load(wfile)['state_dict'])

    predictions_test = predict(model, loader_test)

    df_out = pd.DataFrame().from_dict(predictions_test, orient='index')
    df_out.columns = ['targetPredProba']
    df_out['patientId'] = df_out.index
    df_out.reset_index(drop=True, inplace=True)
    df_out['targetPred'] = df_out['targetPredProba'].apply(lambda x: 0 if x<threshold else 1)

    df_out[['patientId', 'targetPred', 'targetPredProba']].to_csv('test_preds_'+wfile.split('.')[-2]+'.csv', index=False)


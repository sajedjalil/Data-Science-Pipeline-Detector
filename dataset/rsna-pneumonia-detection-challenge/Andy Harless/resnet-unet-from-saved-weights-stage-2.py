import os
import time
import re
import copy

import skimage 
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import PIL

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as torchDataset
import torchvision as tv
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
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

original_image_shape = 1024



# get test set patient IDs
df_test = pd.read_csv('/kaggle/input/stage_2_sample_submission.csv').drop(['PredictionString'],axis=1)
pIds_test = df_test.patientId.values


# arbitrary value for minimum box area in the CNN model
min_box_area = 10000



def get_boxes_per_patient(df, pId):
    '''
    Given the dataset and one patient ID, 
    return an array of all the bounding boxes and their labels associated with that patient ID.
    Example of return: 
    array([[x1, y1, width1, height1],
           [x2, y2, width2, height2]])
    '''
    
    boxes = df.loc[df['patientId']==pId][['x', 'y', 'width', 'height']].astype('int').values.tolist()
    return boxes
    
# define a MinMaxScaler function for the images
def imgMinMaxScaler(img, scale_range):
    """
    :param img: image to be rescaled
    :param scale_range: (tuple) (min, max) of the desired rescaling
    """
    warnings.filterwarnings("ignore")
    img = img.astype('float64')
    img_std = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_scaled = img_std * float(scale_range[1] - scale_range[0]) + float(scale_range[0])
    # round at closest integer and transform to integer 
    img_scaled = np.rint(img_scaled).astype('uint8')

    return img_scaled
    
# define a "warping" image/mask function 
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       Code adapted from https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """
    assert len(image.shape)==2, 'Image must have 2 dimensions.'

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    image_warped = map_coordinates(image, indices, order=1).reshape(shape)
    
    return image_warped
    
# define the data generator class
class PneumoniaDataset(torchDataset):
    """
        Pneumonia dataset that contains radiograph lung images as .dcm. 
        Each patient has one image named patientId.dcm.
    """

    def __init__(self, root, subset, pIds, predict, boxes, rescale_factor=1, transform=None, rotation_angle=0, warping=False):
        """
        :param root: it has to be a path to the folder that contains the dataset folders
        :param subset: 'train' or 'test'
        :param pIds: list of patient IDs
        :param predict: boolean, if true returns images and target labels, otherwise returns only images
        :param boxes: a {patientId : list of boxes} dictionary (ex: {'pId': [[x1, y1, w1, h1], [x2, y2, w2, h2]]}
        :param rescale_factor: image rescale factor in network (image shape is supposed to be square)
        :param transform: transformation applied to the images and their target masks
        :param rotation_angle: float, defines range of random rotation angles for augmentation (-rotation_angle, +rotation_angle)
        :param warping: boolean, whether applying augmentation warping to image
        """
        
        # initialize variables
        self.root = os.path.expanduser(root)
        self.subset = subset
        if self.subset not in ['train', 'test']:
            raise RuntimeError('Invalid subset ' + self.subset + ', it must be one of: \'train\' or \'test\'')
        self.pIds = pIds
        self.predict = predict
        self.boxes = boxes
        self.rescale_factor = rescale_factor
        self.transform = transform
        self.rotation_angle = rotation_angle
        self.warping = warping

        self.data_path = self.root + 'stage_2_'+self.subset+'_images/'
        
    def __getitem__(self, index):
        # get the corresponding pId
        pId = self.pIds[index]
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.data_path, pId+'.dcm')).pixel_array
        # check if image is square
        if (img.shape[0]!=img.shape[1]):
            raise RuntimeError('Image shape {} should be square.'.format(img.shape))
        original_image_shape = img.shape[0]
        # calculate network image shape
        image_shape = original_image_shape / self.rescale_factor
        # check if image_shape is an integer
        if (image_shape != int(image_shape)):
            raise RuntimeError('Network image shape should be an integer.'.format(image_shape))
        image_shape = int(image_shape)
        # resize image 
        # IMPORTANT: skimage resize function rescales the output from 0 to 1, and pytorch doesn't like this!
        # One solution would be using torchvision rescale function (but need to differentiate img and target transforms)
        # Here I use skimage resize and then rescale the output again from 0 to 255
        img = resize(img, (image_shape, image_shape), mode='reflect')
        # recale image from 0 to 255
        img = imgMinMaxScaler(img, (0,255))
        # image warping augmentation
        if self.warping:
            img = elastic_transform(img, image_shape*2., image_shape*0.1)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        # apply rotation augmentation
        if self.rotation_angle>0:
            angle = self.rotation_angle * (2 * np.random.random_sample() - 1) # generate random angle 
            img = tv.transforms.functional.to_pil_image(img)
            img = tv.transforms.functional.rotate(img, angle, resample=PIL.Image.BILINEAR)
                                            
        # apply transforms to image
        if self.transform is not None:
            img = self.transform(img)
        
        if not self.predict:
            # create target mask
            target = np.zeros((image_shape, image_shape))
            # if patient ID has associated target boxes (=if image contains pneumonia)
            if pId in self.boxes:
                # loop through boxes
                for box in self.boxes[pId]:
                    # extract box coordinates 
                    x, y, w, h = box
                    # rescale box coordinates
                    x = int(round(x/rescale_factor))
                    y = int(round(y/rescale_factor))
                    w = int(round(w/rescale_factor))
                    h = int(round(h/rescale_factor))
                    # create a mask of 1s (255 is used because pytorch will rescale to 0-1) inside the box
                    target[y:y+h, x:x+w] = 255 #
                    target[target>255] = 255 # correct in case of overlapping boxes (shouldn't happen)
            # add trailing channel dimension
            target = np.expand_dims(target, -1)   
            target = target.astype('uint8')
            # apply rotation augmentation
            if self.rotation_angle>0:
                target = tv.transforms.functional.to_pil_image(target)
                target = tv.transforms.functional.rotate(target, angle, resample=PIL.Image.BILINEAR)
            # apply transforms to target
            if self.transform is not None:
                target = self.transform(target)
            return img, target, pId
        else: 
            return img, pId

    def __len__(self):
        return len(self.pIds)
        
        
# manual model parameters
rescale_factor = 2 # resize factor to reduce image size (new_image_shape = original_image_shape / rescale_factor)
batch_size = 6 # I used 25 on GCP

# recalculate minimum box area
min_box_area = int(round(min_box_area / float(rescale_factor**2)))

# TBD add normalization of images into transforms
# define transformation 
transform = tv.transforms.Compose([tv.transforms.ToTensor()])

# create datasets
dataset_test = PneumoniaDataset(root=datapath_orig, subset='test', pIds=pIds_test, predict=True, 
                                boxes=None, rescale_factor=rescale_factor, transform=transform,
                                rotation_angle=0, warping=False)

loader_test = DataLoader(dataset=dataset_test,
                         batch_size=batch_size,
                         shuffle=False) 



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# Define the nn transposed convolutional block
class conv_t_block(nn.Module):
    """
    Define the [convolution_transpose - batch normalization - activation] block 
    """

    def __init__(self, in_channels, out_channels, output_size=None, kernel_size=3, bias=False,
                 bn_momentum=0.9, alpha_leaky=0.03):
        super(conv_t_block, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, 
                                         bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=bn_momentum)
        self.activ = nn.LeakyReLU(negative_slope=alpha_leaky)

    def forward(self, x, output_size):
        return self.activ(self.bn(self.conv_t(x, output_size=output_size))) 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.down_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.down_bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_layer1 = self._make_layer(block, 64, layers[0])
        self.down_layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.down_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.down_layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.down_layer4_t = conv_t_block(in_channels=512, out_channels=256)
        self.up_layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.up_layer3_t = conv_t_block(in_channels=256, out_channels=128)
        self.up_layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.up_layer2_t = conv_t_block(in_channels=128, out_channels=64)
        self.up_layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.up_layer1_t = conv_t_block(in_channels=64, out_channels=32)
        self.up_layer0_t = conv_t_block(in_channels=96, out_channels=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x_orig = x
        x = self.down_conv1(x)
        x = self.down_bn1(x)
        x = self.relu(x)
        down0 = x
        x = self.maxpool(x)

        x = self.down_layer1(x)
        down1 = x
        x = self.down_layer2(x)
        down2 = x
        x = self.down_layer3(x)
        down3 = x
        x = self.down_layer4(x)
        down4 = x
        
        x = self.down_layer4_t(x, output_size=down3.size())
        x = torch.cat([x, down3], 1)
        x = self.up_layer3(x) 
        x = self.up_layer3_t(x, output_size=down2.size())
        x = torch.cat([x, down2], 1)
        x = self.up_layer2(x) 
        x = self.up_layer2_t(x, output_size=down1.size())
        x = torch.cat([x, down1], 1)
        x = self.up_layer1(x) 
        x = self.up_layer1_t(x, output_size=down0.size())
        x = torch.cat([x, down0], 1)
        x = self.up_layer0_t(x, output_size=x_orig.size())

        return x

def Resnet34Unet(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

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
        
# Create the loss function
# Define the 2D Sigmoid + Binary Cross Entropy loss function BCEWithLogitsLoss
# TBD add weights for unbalanced class
# NOTE: Rather than using weights, I also intended to try implement focal loss (see RetinaNet)
class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(weight, size_average)

    def forward(self, scores, targets):
        scores_flat = scores.view(-1)
        targets_flat = targets.view(-1)
        return self.loss(scores_flat, targets_flat)
        
# Define auxiliary metric functions

# define function that creates a square mask for a box from its coordinates 
def box_mask(box, shape=1024):
    """
    :param box: [x, y, w, h] box coordinates
    :param shape: shape of the image (default set to maximum possible value, set to smaller to save memory)
    :returns: (np.array of bool) mask as binary 2D array
    """
    x, y, w, h = box
    mask = np.zeros((shape, shape), dtype=bool)
    mask[y:y+h, x:x+w] = True 
    return mask

# define function that extracts confidence and coordinates of boxes from a prediction mask
def parse_boxes(msk, threshold=0.20, connectivity=None):
    """
    :param msk: (torch.Tensor) CxWxH tensor representing the prediction mask
    :param threshold: threshold in the range 0-1 above which a pixel is considered a positive target
    :param connectivity: connectivity parameter for skimage.measure.label segmentation (can be None, 1, or 2)
                         http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
    :returns: (list, list) predicted_boxes, confidences
    """
    # extract 2d array
    msk = msk[0]
    # select pixels above threshold and mark them as positives (1) in an array of equal size as the input prediction mask
    pos = np.zeros(msk.shape)
    pos[msk>threshold] = 1.
    # label regions
    lbl = skimage.measure.label(pos, connectivity=connectivity)
    
    predicted_boxes = []
    confidences = []
    # iterate over regions and extract box coordinates
    for region in skimage.measure.regionprops(lbl):
        # retrieve x, y, height and width and add to prediction string
        y1, x1, y2, x2 = region.bbox
        h = y2 - y1
        w = x2 - x1
        c = np.nanmean(msk[y1:y2, x1:x2])
        # add control over box size (eliminate if too small)
        if w*h > min_box_area: 
            predicted_boxes.append([x1, y1, w, h])
            confidences.append(c)
    
    return predicted_boxes, confidences

# define function that creates prediction strings as expected in submission
def prediction_string(predicted_boxes, confidences):
    """
    :param predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes coordinates 
    :param confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    :returns: prediction string 'c1 x1 y1 w1 h1 c2 x2 y2 w2 h2 ...'
    """
    prediction_string = ''
    for c, box in zip(confidences, predicted_boxes):
        prediction_string += ' ' + str(c) + ' ' + ' '.join([str(b) for b in box])
    return prediction_string[1:]   

# define iou function
def IoU(pr, gt):
    """
    :param pr: (numpy_array(bool)) prediction array 
    :param gt: (numpy_array(bool)) ground truth array 
    :returns: IoU (pr, gt) = intersection (pr, gt) / union (pr, gt)
    """
    IoU = (pr & gt).sum() / ((pr | gt).sum() + 1.e-9)
    return IoU

# define precision function
def precision(tp, fp, fn):
    """
    :param tp: (int) number of true positives
    :param fp: (int) number of false positives
    :param fn: (int) number of false negatives
    :returns: precision metric for one image at one threshold
    """
    return float(tp) / (tp + fp + fn + 1.e-9)

# define function that calculates the average precision of an image
def average_precision_image(predicted_boxes, confidences, target_boxes, shape=1024):
    """
    :param predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes coordinates 
    :param confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    :param target_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of target boxes coordinates 
    :param shape: shape of the boolean masks (default set to maximum possible value, set to smaller to save memory)
    :returns: average_precision
    """
    
    # if both predicted and target boxes are empty, precision is NaN (and doesn't count towards the batch average)
    if predicted_boxes == [] and target_boxes == []:
        return np.nan
    else:
        # if we have predicted boxes but no target boxes, precision is 0
        if len(predicted_boxes)>0 and target_boxes == []:
            return 0.0
        # if we have target boxes but no predicted boxes, precision is 0
        elif len(target_boxes)>0 and predicted_boxes == []:
            return 0.0
        # if we have both predicted and target boxes, proceed to calculate image average precision
        else:
            # define list of thresholds for IoU [0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75]
            thresholds = np.arange(0.4, 0.8, 0.05) 
            # sort boxes according to their confidence (from largest to smallest)
            predicted_boxes_sorted = list(reversed([b for _, b in sorted(zip(confidences, predicted_boxes), 
                                                                         key=lambda pair: pair[0])]))            
            average_precision = 0.0
            for t in thresholds: # iterate over thresholds
                # with a first loop we measure true and false positives
                tp = 0 # initiate number of true positives
                fp = len(predicted_boxes) # initiate number of false positives 
                for box_p in predicted_boxes_sorted: # iterate over predicted boxes coordinates
                    box_p_msk = box_mask(box_p, shape) # generate boolean mask
                    for box_t in target_boxes: # iterate over ground truth boxes coordinates
                        box_t_msk = box_mask(box_t, shape) # generate boolean mask
                        iou = IoU(box_p_msk, box_t_msk) # calculate IoU
                        if iou>t:
                            tp += 1 # if IoU is above the threshold, we got one more true positive
                            fp -= 1 # and one less false positive
                            break # proceed to the next predicted box
                # with a second loop we measure false negatives
                fn = len(target_boxes) # initiate number of false negatives
                for box_t in target_boxes: # iterate over ground truth boxes coordinates
                    box_t_msk = box_mask(box_t, shape) # generate boolean mask
                    for box_p in predicted_boxes_sorted: # iterate over predicted boxes coordinates
                        box_p_msk = box_mask(box_p, shape) # generate boolean mask
                        iou = IoU(box_p_msk, box_t_msk) # calculate IoU
                        if iou>t:
                            fn -= 1
                            break # proceed to the next ground truth box
                # TBD: this algo must be checked against the official Kaggle evaluation method which is still not clear...
                average_precision += precision(tp, fp, fn) / float(len(thresholds))
            return average_precision

def average_precision_batch(output_batch, pIds, pId_boxes_dict, rescale_factor, shape=1024, return_array=False):
    """
    :param output_batch: cnn model output batch
    :param pIds: (list) list of patient IDs contained in the output batch
    :param rescale_factor: CNN image rescale factor
    :param shape: shape of the boolean masks (default set to maximum possible value, set to smaller to save memory)
    :returns: average_precision
    """
    
    batch_precisions = []
    for msk, pId in zip(output_batch, pIds): # iterate over batch prediction masks and relative patient IDs
        # retrieve target boxes from dictionary (quicker than from mask itself)
        target_boxes = pId_boxes_dict[pId] if pId in pId_boxes_dict else []
        # rescale coordinates of target boxes
        if len(target_boxes)>0:
            target_boxes = [[int(round(c/float(rescale_factor))) for c in box_t] for box_t in target_boxes]
        # extract prediction boxes and confidences
        predicted_boxes, confidences = parse_boxes(msk) 
        batch_precisions.append(average_precision_image(predicted_boxes, confidences, target_boxes, shape=shape))
    if return_array:
        return np.asarray(batch_precisions)
    else:
        return np.nanmean(np.asarray(batch_precisions))

def save_checkpoint(state, is_best, metric):
    """Saves model and training parameters at 'last.pth.tar'. If is_best==True, also saves
    'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
    """
    filename = 'last.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, metric+'.best.pth.tar')
        
def rescale_box_coordinates(box, rescale_factor):
    x, y, w, h = box
    x = int(round(x/rescale_factor))
    y = int(round(y/rescale_factor))
    w = int(round(w/rescale_factor))
    h = int(round(h/rescale_factor))
    return [x, y, w, h]

def predict(model, dataloader): 

    # set model to evaluation mode
    model.eval()
    
    predictions = {}

    for i, (test_batch, pIds) in enumerate(dataloader):
        # Convert torch tensor to Variable
        test_batch = Variable(test_batch).cuda(async=True)
            
        # compute output
        output_batch = model(test_batch)
        sig = nn.Sigmoid().cuda()
        output_batch = sig(output_batch)
        output_batch = output_batch.data.cpu().numpy()
        for pId, output in zip(pIds, output_batch):
            predictions[pId] = output
        
    return predictions



model = Resnet34Unet().cuda()

wfile = 'loss.best.pth_resnet34_unet.tar'
file_url = 'http://andy.harless.us/rsnaweights/' + wfile
call( ['wget', '-q', file_url] )
print( call( ['md5sum', wfile] ) )

best_model = model.cuda()
best_model.load_state_dict(torch.load(wfile)['state_dict'])


# create submission predictions for the test set
predictions_test = predict(best_model, loader_test)

best_threshold = 0.35

df_sub = df_test[['patientId']].copy(deep=True)
def get_prediction_string_per_pId(pId):
    prediction = predictions_test[pId]
    predicted_boxes, confidences = parse_boxes(prediction, threshold=best_threshold, connectivity=None)
    predicted_boxes = [rescale_box_coordinates(box, 1./rescale_factor) for box in predicted_boxes]
    return prediction_string(predicted_boxes, confidences)
df_sub['predictionString'] = df_sub['patientId'].apply(lambda x: get_prediction_string_per_pId(x) if x in pIds_test else '')
print('Number of non null prediction strings: {} ({:05.2f}%)'.format(df_sub.loc[df_sub['predictionString']!=''].shape[0],
                                                    100. * df_sub.loc[df_sub['predictionString']!=''].shape[0]/df_sub.shape[0]))

try:
    df_sub.to_csv(f'submission_resnet34unet_thr{best_threshold}.csv', index=False)
except:
    df_sub.to_csv('submission.csv', index=False)
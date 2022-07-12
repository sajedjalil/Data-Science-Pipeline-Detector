import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict # for ordering the pre-trained model weights
import PIL.Image as Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision import models
import gc
import wandb
import time
import cv2

DATA_PATH = "/kaggle/input/bengaliai-cv19/"

# Path to our trained model weights
TRAINED_MODEL_PATH = "/kaggle/input/pretrainedmodel/rausnaus_resnet50_1584561626.pth"

IMG_HEIGHT = 137
IMG_WIDTH = 236

USE_CPU = False

########## PREPROCESSING CLASSES ##########

class CustomCrop(object):
    def __call__(self, image):
        image_array = np.array(image)
        cropped_image_array = self.crop_image(image_array)
        return cv2.resize(self.remove_low_values(cropped_image_array), (224,224))

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    def remove_low_values(self, img, threshold=20):
        """
        Sets low values to 0 to save memory
        """
        return np.where(img < threshold, 0, img )

    def boundary_box(self, img, original):
        """
        Returns the x and y-values of the top, bottom, left, and right of the first non-zero entries in an array
        Source: https://www.kaggle.com/iafoss/image-preprocessing-128x128
        """ 
        # For any row/column containing >=1 True values, np.any returns True
        rows = np.any(img, axis = 1)
        cols = np.any(img, axis = 0)
        # Select indices of the first and last row to be "True", i.e. have a non-zero element. 
        row_top, row_bottom = np.where(rows)[0][[0,-1]] 
        column_left, column_right = np.where(cols)[0][[0,-1]]   
        return row_top, row_bottom, column_left, column_right

    def crop_image(self, image, threshold = 40):
        image = image[5:-5, 5:-5] 
        row_top, row_bottom, column_left, column_right = self.boundary_box(image > threshold, image)
        image = image[row_top:row_bottom, column_left:column_right]
        n_rows, n_cols = image.shape
        diff = int(abs(n_rows - n_cols)/2)
        # Als we een oneven zijde hebben, introduceren we een afrondingsfout in de padding die we moeten compenseren (2*0.5)
        fix = (n_rows+n_cols) % 2
        if (n_rows > n_cols):           
            padded_image = np.pad(image, [(0,0),(diff, diff+fix)], mode = 'constant')
        else:
            padded_image = np.pad(image, [(diff, diff+fix),(0,0)], mode = 'constant')
        return padded_image
    
class TestBengaliDataset(Dataset):
    def __init__(self, file_nr, transform=None):       
        self.transform = transform
        self.df = pd.read_parquet(f'{DATA_PATH}test_image_data_{file_nr}.parquet')
        self.data = self.df.iloc[:, 1:].values.reshape(-1, IMG_HEIGHT, IMG_WIDTH).astype(np.uint8)
                
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        img_id = self.df.iloc[index,0]
        img = ((255-self.data[index])*(255.0/self.data[index].max())).astype(np.uint8)
        if self.transform:
            img = self.transform(img)
        return img

########## DEFINE TRANFORMATIONS ##########

class ToRGBArray(object):
    """
    Converts a 1D array of shape (W,H) to a shape (3,W,H) by duplicating the original image
    """
    def __call__(self, image):
        rgb_image = np.repeat(np.expand_dims(image, axis=0), repeats=3, axis=0)
        # The shape is now (H, 3, W) for some reason, but we want (3, W, H) so we move the axis
        # Edit ejw: according to PyTorch ResNet50 doc, actually: (3 x H x W), but does not matter for 224x224. 
        return np.moveaxis(rgb_image, 0, -1)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
preprocess = transforms.Compose([
    CustomCrop(),
    ToRGBArray(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

########## LOAD RESNET-50 MODEL ##########

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model     
        fc_in = model.fc.in_features
        
        # Remove last layer of original model
        self.model = nn.Sequential(*list(model.children())[:-1])
        
        # Define layers to be added
        self.bn1  = nn.BatchNorm1d(fc_in)
        self.drop1 = nn.Dropout(0.25)
        self.lin1  = nn.Linear(2048, 512)
        self.relu = nn.ReLU(inplace=False)
        
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.5)
        
        # Final layers
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512,168)
        self.fc3 = nn.Linear(512,7)
        
        self.sm1 = nn.Softmax(dim=1)
        self.sm2 = nn.Softmax(dim=1)
        self.sm3 = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.model(x)
        
        # Turn x into the right shape
        x = x.view(x.size(0), -1)
        
        # Put output x through our self defined layers
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.lin1(x)
        x = self.relu(x)
        
        x = self.bn2(x)
        x = self.drop2(x)
        vowel_preds = self.sm1(self.fc1(x))
        root_preds = self.sm2(self.fc2(x))
        cons_preds = self.sm3(self.fc3(x))
        
        return vowel_preds, root_preds, cons_preds

# Load model structure
model = models.resnet50(pretrained=False)
    
# Change the final layers of our model:
model = Net(model)

# LOAD TRAINED MODEL WEIGHTS
if USE_CPU:
    trained_weights = torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu'))
else:
    trained_weights = torch.load(TRAINED_MODEL_PATH)
    
model.load_state_dict(trained_weights)

# Use GPU if GPU available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tell model to use hardware available (CPU/GPU)
model = model.to(device)

# Set model in eval mode for inference
model.eval()

########## INFERENCE TIME ##########

batch_size = 64
predictions = []    
row_ids = []
row_id = 0

for i in range(4):
    print(f"Predicting for parquet {i}")
    dataset = TestBengaliDataset(i, transform=preprocess)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        start_time_batch = time.time()
        for batch_idx, imgs in enumerate(test_loader):
            #get predictions
            imgs = imgs.to(device, dtype=torch.float32)  
            vowel_pred, root_pred, consonant_pred = model(imgs)
            
            vowel_prob, vowel_diacritic = torch.max(vowel_pred.cpu(), 1)
            grapheme_prob, grapheme_root = torch.max(root_pred.cpu(), 1)
            consonant_prob, consonant_diacritic = torch.max(consonant_pred.cpu(), 1)
            
            # Necessary to loop when using batch_size > 1
            for i in range(len(vowel_diacritic)):
                row_ids.append(f"Test_{row_id}_consonant_diacritic")
                row_ids.append(f"Test_{row_id}_grapheme_root")
                row_ids.append(f"Test_{row_id}_vowel_diacritic")
                row_id += 1

                predictions.append(consonant_diacritic.numpy()[i])
                predictions.append(grapheme_root.numpy()[i])
                predictions.append(vowel_diacritic.numpy()[i])

submission_df = pd.DataFrame({'row_id':row_ids,'target':predictions})
# In case of NA (are there any?)
submission_df.fillna('0',inplace=True)

# Writes to /kaggle/working
submission_df.to_csv('./submission.csv', index=False)
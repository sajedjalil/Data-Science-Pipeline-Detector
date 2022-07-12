# %% [markdown]
# Visualizing LBP & HOG features (I used Gldfish kernel to load images : https://www.kaggle.com/go1dfish/clear-mask-visualization-and-simple-eda) . 

# %% [code]
import numpy as np # linear algebra
import pandas as pd
pd.set_option("display.max_rows", 101)
import os
print(os.listdir("../input"))
import cv2
import json
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 15
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
import skimage
from skimage.feature import  hog
from skimage import  exposure

input_dir = "../input/"

train_df = pd.read_csv("../input/train.csv")
sample_df = pd.read_csv("../input/sample_submission.csv")

class_dict = defaultdict(int)

kind_class_dict = defaultdict(int)

no_defects_num = 0
defects_num = 0

for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError
        
    labels = train_df.iloc[col:col+4, 1]
    if labels.isna().all():
        no_defects_num += 1
    else:
        defects_num += 1
    
    kind_class_dict[sum(labels.isna().values == False)] += 1
        
    for idx, label in enumerate(labels.isna().values.tolist()):
        if label == False:
            class_dict[idx+1] += 1
            

print("the number of images with no defects: {}".format(no_defects_num))
print("the number of images with defects: {}".format(defects_num))


train_size_dict = defaultdict(int)
train_path = Path("../input/train_images/")

for img_name in train_path.iterdir():
    img = Image.open(img_name)
    train_size_dict[img.size] += 1
    
test_size_dict = defaultdict(int)
test_path = Path("../input/test_images/")

for img_name in test_path.iterdir():
    img = Image.open(img_name)
    test_size_dict[img.size] += 1
    
palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]


def name_and_mask(start_idx):
    col = start_idx
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col+4, 1]
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            mask_label = np.zeros(1600*256, dtype=np.uint8)
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                mask_label[pos:(pos+le)] = 1
            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')
    return img_names[0], mask

def Show_LBP_and_HOG(col):
    name, mask = name_and_mask(col)
    img = cv2.imread(str(train_path / name), cv2.IMREAD_GRAYSCALE)
    lbp = skimage.feature.local_binary_pattern(img, 10, 5, method="uniform")
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, multichannel=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 30), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(lbp)
    ax1.set_title('Local Binary Pattern')
    # Rescale histogram for better display
    hog_image_rescaled = skimage.exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    
      

idx_no_defect = []
idx_defect = []

for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError
        
    labels = train_df.iloc[col:col+4, 1]
    if labels.isna().all():
        idx_no_defect.append(col)
    else: 
          idx_defect.append(col)

                       
        
for idx in idx_no_defect[:3]:
    Show_LBP_and_HOG(idx)
    

for idx in idx_defect[:3]:
    Show_LBP_and_HOG(idx)
    
    
    
    
   
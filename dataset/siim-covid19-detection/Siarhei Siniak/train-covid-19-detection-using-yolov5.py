# %% [markdown]
# This is a starter kernel to train a YOLOv5 model on [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection/overview) dataset. Given an input image the task is to find the region of opacity in the chest  using bounding box coordinates. Check out [Visualize Bounding Boxes Interactively](https://www.kaggle.com/ayuraj/visualize-bounding-boxes-interactively) for interactive bounding box EDA. 
# 
# ## 🖼️ What is YOLOv5?
# 
# YOLO an acronym for 'You only look once', is an object detection algorithm that divides images into a grid system. Each cell in the grid is responsible for detecting objects within itself.
# 
# [Ultralytics' YOLOv5](https://ultralytics.com/yolov5) ("You Only Look Once") model family enables real-time object detection with convolutional neural networks. 
# 
# ## 🦄 What is Weights and Biases?
# 
# Weights & Biases (W&B) is a set of machine learning tools that helps you build better models faster. Check out [Experiment Tracking with Weights and Biases](https://www.kaggle.com/ayuraj/experiment-tracking-with-weights-and-biases) to learn more.  
# Weights & Biases is directly integrated into YOLOv5, providing experiment metric tracking, model and dataset versioning, rich model prediction visualization, and more.
# 
# 
# It's a work in progress:
# 
# ✔️ Required folder structure. <br>
# ✔️ Bounding box format required for YOLOv5. <br>
# ✔️ **Train** a small YOLOv5 model. <br>
# ✔️ Experiment tracking with W&B. <br>
# ✔️ Proper documentation <br>
# ✔️ Inference <br>
# 
# ❌ Model prediction visualization. 
# 
# ## Results 
# 
# ### [Check out W&B Run Page $\rightarrow$](https://wandb.ai/ayush-thakur/kaggle-siim-covid/runs/1bk93e3j)
# 
# ![img](https://i.imgur.com/quOYtNN.gif)

# %% [markdown]
# # ☀️ Imports and Setup
# 
# According to the official [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) guide, YOLOv5 requires a certain directory structure. 
# 
# ```
# /parent_folder
#     /dataset
#          /images
#          /labels
#     /yolov5
# ```
# 
# * We thus will create a `/tmp` directory. <br>
# * Download YOLOv5 repository and pip install the required dependencies. <br>
# * Install the latest version of W&B and login with your wandb account. You can create your free W&B account [here](https://wandb.ai/site).

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:27:59.979957Z","iopub.execute_input":"2021-06-29T19:27:59.980286Z","iopub.status.idle":"2021-06-29T19:28:00.618611Z","shell.execute_reply.started":"2021-06-29T19:27:59.980253Z","shell.execute_reply":"2021-06-29T19:28:00.617637Z"}}
%cd ../
!mkdir tmp
%cd tmp

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:28:04.317054Z","iopub.execute_input":"2021-06-29T19:28:04.317402Z","iopub.status.idle":"2021-06-29T19:28:15.089545Z","shell.execute_reply.started":"2021-06-29T19:28:04.317368Z","shell.execute_reply":"2021-06-29T19:28:15.088568Z"}}
# Download YOLOv5
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
# Install dependencies
%pip install -qr requirements.txt  # install dependencies

%cd ../
import torch
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:27:26.451666Z","iopub.execute_input":"2021-06-29T19:27:26.452048Z","iopub.status.idle":"2021-06-29T19:27:54.758753Z","shell.execute_reply.started":"2021-06-29T19:27:26.45197Z","shell.execute_reply":"2021-06-29T19:27:54.757899Z"}}
# Install W&B 
#!pip install -q --upgrade wandb
# Login 
#import wandb
#wandb.login()
!wandb offline

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:28:23.003348Z","iopub.execute_input":"2021-06-29T19:28:23.003702Z","iopub.status.idle":"2021-06-29T19:28:24.175496Z","shell.execute_reply.started":"2021-06-29T19:28:23.003667Z","shell.execute_reply":"2021-06-29T19:28:24.174633Z"}}
# Necessary/extra dependencies. 
import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))

# %% [markdown]
# # 🦆 Hyperparameters

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:28:29.872519Z","iopub.execute_input":"2021-06-29T19:28:29.87286Z","iopub.status.idle":"2021-06-29T19:28:29.876488Z","shell.execute_reply.started":"2021-06-29T19:28:29.872811Z","shell.execute_reply":"2021-06-29T19:28:29.875673Z"}}
TRAIN_PATH = 'input/siim-covid19-resized-to-256px-jpg/train/'
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10

# %% [markdown]
# # 🔨 Prepare Dataset
# 
# This is the most important section when it comes to training an object detector with YOLOv5. The directory structure, bounding box format, etc must be in the correct order. This section builds every piece needed to train a YOLOv5 model.
# 
# I am using [xhlulu's](https://www.kaggle.com/xhlulu) resized dataset. The uploaded 256x256 Kaggle dataset is [here](https://www.kaggle.com/xhlulu/siim-covid19-resized-to-256px-jpg). Find other image resolutions [here](https://www.kaggle.com/c/siim-covid19-detection/discussion/239918).
# 
# * Create train-validation split. <br>
# * Create required `/dataset` folder structure and more the images to that folder. <br>
# * Create `data.yaml` file needed to train the model. <br>
# * Create bounding box coordinates in the required YOLO format. 

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:28:33.227464Z","iopub.execute_input":"2021-06-29T19:28:33.227797Z","iopub.status.idle":"2021-06-29T19:28:33.556206Z","shell.execute_reply.started":"2021-06-29T19:28:33.227765Z","shell.execute_reply":"2021-06-29T19:28:33.555222Z"}}
# Everything is done from /kaggle directory.
%cd ../

# Load image level csv file
df = pd.read_csv('input/siim-covid19-detection/train_image_level.csv')

# Modify values in the id column
df['id'] = df.apply(lambda row: row.id.split('_')[0], axis=1)
# Add absolute path
df['path'] = df.apply(lambda row: TRAIN_PATH+row.id+'.jpg', axis=1)
# Get image level labels
df['image_level'] = df.apply(lambda row: row.label.split(' ')[0], axis=1)

df.head(5)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:28:38.561007Z","iopub.execute_input":"2021-06-29T19:28:38.561353Z","iopub.status.idle":"2021-06-29T19:28:38.601858Z","shell.execute_reply.started":"2021-06-29T19:28:38.561324Z","shell.execute_reply":"2021-06-29T19:28:38.601049Z"}}
# Load meta.csv file
# Original dimensions are required to scale the bounding box coordinates appropriately.
meta_df = pd.read_csv('input/siim-covid19-resized-to-256px-jpg/meta.csv')
train_meta_df = meta_df.loc[meta_df.split == 'train']
train_meta_df = train_meta_df.drop('split', axis=1)
train_meta_df.columns = ['id', 'dim0', 'dim1']

train_meta_df.head(2)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:28:41.636745Z","iopub.execute_input":"2021-06-29T19:28:41.637102Z","iopub.status.idle":"2021-06-29T19:28:41.664323Z","shell.execute_reply.started":"2021-06-29T19:28:41.637067Z","shell.execute_reply":"2021-06-29T19:28:41.663375Z"}}
# Merge both the dataframes
df = df.merge(train_meta_df, on='id',how="left")
df.head(2)

# %% [markdown]
# ## 🍘 Train-validation split

# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true,"execution":{"iopub.status.busy":"2021-06-29T19:28:44.726019Z","iopub.execute_input":"2021-06-29T19:28:44.726341Z","iopub.status.idle":"2021-06-29T19:28:44.762013Z","shell.execute_reply.started":"2021-06-29T19:28:44.726313Z","shell.execute_reply":"2021-06-29T19:28:44.760307Z"}}
# Create train and validation split.
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.image_level.values)

train_df.loc[:, 'split'] = 'train'
valid_df.loc[:, 'split'] = 'valid'

df = pd.concat([train_df, valid_df]).reset_index(drop=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:28:48.607463Z","iopub.execute_input":"2021-06-29T19:28:48.607791Z","iopub.status.idle":"2021-06-29T19:28:48.612711Z","shell.execute_reply.started":"2021-06-29T19:28:48.607759Z","shell.execute_reply":"2021-06-29T19:28:48.611808Z"}}
print(f'Size of dataset: {len(df)}, training images: {len(train_df)}. validation images: {len(valid_df)}')

# %% [markdown]
# ## 🍚 Prepare Required Folder Structure
# 
# The required folder structure for the dataset directory is: 
# 
# ```
# /parent_folder
#     /dataset
#          /images
#              /train
#              /val
#          /labels
#              /train
#              /val
#     /yolov5
# ```
# 
# Note that I have named the directory `covid`.

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:28:52.075565Z","iopub.execute_input":"2021-06-29T19:28:52.075929Z","iopub.status.idle":"2021-06-29T19:28:52.73827Z","shell.execute_reply.started":"2021-06-29T19:28:52.075894Z","shell.execute_reply":"2021-06-29T19:28:52.737305Z"}}
os.makedirs('tmp/covid/images/train', exist_ok=True)
os.makedirs('tmp/covid/images/valid', exist_ok=True)

os.makedirs('tmp/covid/labels/train', exist_ok=True)
os.makedirs('tmp/covid/labels/valid', exist_ok=True)

! ls tmp/covid/images

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:28:52.797711Z","iopub.execute_input":"2021-06-29T19:28:52.798062Z","iopub.status.idle":"2021-06-29T19:29:49.581074Z","shell.execute_reply.started":"2021-06-29T19:28:52.798027Z","shell.execute_reply":"2021-06-29T19:29:49.579312Z"}}
# Move the images to relevant split folder.
for i in tqdm(range(len(df))):
    row = df.loc[i]
    if row.split == 'train':
        copyfile(row.path, f'tmp/covid/images/train/{row.id}.jpg')
    else:
        copyfile(row.path, f'tmp/covid/images/valid/{row.id}.jpg')

# %% [markdown]
# ## 🍜 Create `.YAML` file
# 
# The `data.yaml`, is the dataset configuration file that defines 
# 
# 1. an "optional" download command/URL for auto-downloading, 
# 2. a path to a directory of training images (or path to a *.txt file with a list of training images), 
# 3. a path to a directory of validation images (or path to a *.txt file with a list of validation images), 
# 4. the number of classes, 
# 5. a list of class names.
# 
# > 📍 Important: In this competition, each image can either belong to `opacity` or `none` image-level labels. That's why I have  used the number of classes, `nc` to be 2. YOLOv5 automatically handles the images without any bounding box coordinates. 
# 
# > 📍 Note: The `data.yaml` is created in the `yolov5/data` directory as required. 

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:30:30.208884Z","iopub.execute_input":"2021-06-29T19:30:30.209235Z","iopub.status.idle":"2021-06-29T19:30:30.868549Z","shell.execute_reply.started":"2021-06-29T19:30:30.209202Z","shell.execute_reply":"2021-06-29T19:30:30.867512Z"}}
# Create .yaml file 
import yaml

data_yaml = dict(
    train = '../covid/images/train',
    val = '../covid/images/valid',
    nc = 2,
    names = ['none', 'opacity']
)

# Note that I am creating the file in the yolov5/data/ directory.
with open('tmp/yolov5/data/data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)
    
%cat tmp/yolov5/data/data.yaml

# %% [markdown]
# ## 🍮 Prepare Bounding Box Coordinated for YOLOv5
# 
# For every image with **bounding box(es)** a `.txt` file with the same name as the image will be created in the format shown below:
# 
# * One row per object. <br>
# * Each row is class `x_center y_center width height format`. <br>
# * Box coordinates must be in normalized xywh format (from 0 - 1). We can normalize by the boxes in pixels by dividing `x_center` and `width` by image width, and `y_center` and `height` by image height. <br>
# * Class numbers are zero-indexed (start from 0). <br>
# 
# > 📍 Note: We don't have to remove the images without bounding boxes from the training or validation sets. 

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:30:36.483339Z","iopub.execute_input":"2021-06-29T19:30:36.483765Z","iopub.status.idle":"2021-06-29T19:30:36.496897Z","shell.execute_reply.started":"2021-06-29T19:30:36.483726Z","shell.execute_reply":"2021-06-29T19:30:36.495875Z"}}
# Get the raw bounding box by parsing the row value of the label column.
# Ref: https://www.kaggle.com/yujiariyasu/plot-3positive-classes
def get_bbox(row):
    bboxes = []
    bbox = []
    for i, l in enumerate(row.label.split(' ')):
        if (i % 6 == 0) | (i % 6 == 1):
            continue
        bbox.append(float(l))
        if i % 6 == 5:
            bboxes.append(bbox)
            bbox = []  
            
    return bboxes

# Scale the bounding boxes according to the size of the resized image. 
def scale_bbox(row, bboxes):
    # Get scaling factor
    scale_x = IMG_SIZE/row.dim1
    scale_y = IMG_SIZE/row.dim0
    
    scaled_bboxes = []
    for bbox in bboxes:
        x = int(np.round(bbox[0]*scale_x, 4))
        y = int(np.round(bbox[1]*scale_y, 4))
        x1 = int(np.round(bbox[2]*(scale_x), 4))
        y1= int(np.round(bbox[3]*scale_y, 4))

        scaled_bboxes.append([x, y, x1, y1]) # xmin, ymin, xmax, ymax
        
    return scaled_bboxes

# Convert the bounding boxes in YOLO format.
def get_yolo_format_bbox(img_w, img_h, bboxes):
    yolo_boxes = []
    for bbox in bboxes:
        w = bbox[2] - bbox[0] # xmax - xmin
        h = bbox[3] - bbox[1] # ymax - ymin
        xc = bbox[0] + int(np.round(w/2)) # xmin + width/2
        yc = bbox[1] + int(np.round(h/2)) # ymin + height/2
        
        yolo_boxes.append([xc/img_w, yc/img_h, w/img_w, h/img_h]) # x_center y_center width height
    
    return yolo_boxes

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:30:40.192563Z","iopub.execute_input":"2021-06-29T19:30:40.192899Z","iopub.status.idle":"2021-06-29T19:30:42.810617Z","shell.execute_reply.started":"2021-06-29T19:30:40.192865Z","shell.execute_reply":"2021-06-29T19:30:42.809614Z"}}
# Prepare the txt files for bounding box
for i in tqdm(range(len(df))):
    row = df.loc[i]
    # Get image id
    img_id = row.id
    # Get split
    split = row.split
    # Get image-level label
    label = row.image_level
    
    if row.split=='train':
        file_name = f'tmp/covid/labels/train/{row.id}.txt'
    else:
        file_name = f'tmp/covid/labels/valid/{row.id}.txt'
        
    
    if label=='opacity':
        # Get bboxes
        bboxes = get_bbox(row)
        # Scale bounding boxes
        scale_bboxes = scale_bbox(row, bboxes)
        # Format for YOLOv5
        yolo_bboxes = get_yolo_format_bbox(IMG_SIZE, IMG_SIZE, scale_bboxes)
        
        with open(file_name, 'w') as f:
            for bbox in yolo_bboxes:
                bbox = [1]+bbox
                bbox = [str(i) for i in bbox]
                bbox = ' '.join(bbox)
                f.write(bbox)
                f.write('\n')

# %% [markdown]
# # 🚅 Train with W&B
# 
# 

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:30:49.777249Z","iopub.execute_input":"2021-06-29T19:30:49.777588Z","iopub.status.idle":"2021-06-29T19:30:49.783469Z","shell.execute_reply.started":"2021-06-29T19:30:49.777557Z","shell.execute_reply":"2021-06-29T19:30:49.782385Z"}}
%cd tmp/yolov5/

# %% [markdown]
# ```
# --img {IMG_SIZE} \ # Input image size.
# --batch {BATCH_SIZE} \ # Batch size
# --epochs {EPOCHS} \ # Number of epochs
# --data data.yaml \ # Configuration file
# --weights yolov5s.pt \ # Model name
# --save_period 1\ # Save model after interval
# --project kaggle-siim-covid # W&B project name
# ```

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:30:54.321995Z","iopub.execute_input":"2021-06-29T19:30:54.322333Z","iopub.status.idle":"2021-06-29T19:48:59.935845Z","shell.execute_reply.started":"2021-06-29T19:30:54.322305Z","shell.execute_reply":"2021-06-29T19:48:59.934703Z"}}
!python train.py --img {IMG_SIZE} \
                 --batch {BATCH_SIZE} \
                 --epochs {EPOCHS} \
                 --data data.yaml \
                 --weights yolov5s.pt \
                 --save_period 1
#                 --save_period 1\
#                 --project kaggle-siim-covid

# %% [markdown]
# ## Model Saved Automatically as Artifact
# 
# Since it's a kernel based competition, you can easily download the best model from the W&B Artifacts UI and upload as a Kaggle dataset that you can load in your inference kernel (internel disabled).
# 
# ### [Path to saved model $\rightarrow$](https://wandb.ai/ayush-thakur/kaggle-siim-covid/artifacts/model/run_jbt74n7q_model/4c3ca5752dba99bd227e)
# 
# ![img](https://i.imgur.com/KhRLQvR.png)
# 
# > 📍 Download the model with the `best` alias tagged to it. 

# %% [markdown]
# # Inference
# 
# You will probably use a `Submission.ipynb` kernel to run all the predictions. After training a YOLOv5 based object detector -> head to the artifacts page and download the best model -> upload the model as a Kaggle dataset -> Use it with the submission folder. 
# 
# > 📍 Note that you might have to clone the YOLOv5 repository in a Kaggle dataset as well. 
# 
# In this section, I will show you how you can do the inference and modify the predicted bounding box coordinates.

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:48:59.937806Z","iopub.execute_input":"2021-06-29T19:48:59.938192Z","iopub.status.idle":"2021-06-29T19:48:59.944866Z","shell.execute_reply.started":"2021-06-29T19:48:59.938148Z","shell.execute_reply":"2021-06-29T19:48:59.944049Z"}}
TEST_PATH = '/kaggle/input/siim-covid19-resized-to-256px-jpg/test/' # absolute path

# %% [markdown]
# Since I am training the model in this kernel itself, I will not be using the method that I have described above. The best model is saved in the directory `project_name/exp*/weights/best.pt`. In `exp*`, * can be 1, 2, etc. 

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:50:46.801325Z","iopub.execute_input":"2021-06-29T19:50:46.801686Z","iopub.status.idle":"2021-06-29T19:50:47.680384Z","shell.execute_reply.started":"2021-06-29T19:50:46.801652Z","shell.execute_reply":"2021-06-29T19:50:47.67952Z"}}
!wget -O best.pt 'https://api.wandb.ai/artifactsV2/gcp-us/ayush-thakur/QXJ0aWZhY3Q6NjY0MTM4Ng==/e1a56acb077a10cb732a1079e98f5c43'
#!wget 'https://storage.googleapis.com/wandb-artifacts-prod/ayush-thakur/kaggle-siim-covid/wandb_artifacts/761745/6641386/e1a56acb077a10cb732a1079e98f5c43?Expires=1624995645&GoogleAccessId=wandb-production%40appspot.gserviceaccount.com&Signature=Gn3cTiK4roLmxkfMsD51nkLbXy%2Fy8L1c%2FFSkcJ6sI0Ib1ltajnWCIuCcNWTTgXzEMiNASNEcwNFY5zqt2K1Xon0uHVDt1h0CarGw%2FZDfYymXv7AVavEroZDG87kLxMgjsjuQMhkpUlI6yomXxxIPBcIpvxPe9cfaIOA9%2FcMpCZYDww9yK5h9r60EkTFEH%2FcHxV1d%2BS6l%2F4dbnWSlr0m8QGRQUDsgPz6wFV28XO30v3NHDFqICZ6%2BQrTB3IqcwYI1IxJIDV7F2hsPgX9AS%2Bu7isFEQCJ4GgUH0lMeIBHtOgZ9NxUTWjAofzHgT22qGmVMWEAi1agFyHYTYMFzrB2xuA%3D%3D'

# %% [code] {"execution":{"iopub.status.busy":"2021-06-29T19:51:23.899347Z","iopub.execute_input":"2021-06-29T19:51:23.899678Z","iopub.status.idle":"2021-06-29T19:51:23.903556Z","shell.execute_reply.started":"2021-06-29T19:51:23.899646Z","shell.execute_reply":"2021-06-29T19:51:23.902437Z"}}
#MODEL_PATH = 'kaggle-siim-covid/exp/weights/best.pt'
MODEL_PATH = '/kaggle/tmp/yolov5/best.pt'

# %% [markdown]
# ```
# --weights {MODEL_PATH} \ # path to the best model.
# --source {TEST_PATH} \ # absolute path to the test images.
# --img {IMG_SIZE} \ # Size of image
# --conf 0.281 \ # Confidence threshold (default is 0.25)
# --iou-thres 0.5 \ # IOU threshold (default is 0.45)
# --max-det 3 \ # Number of detections per image (default is 1000) 
# --save-txt \ # Save predicted bounding box coordinates as txt files
# --save-conf # Save the confidence of prediction for each bounding box
# ```

# %% [code] {"scrolled":true,"_kg_hide-output":true,"execution":{"iopub.status.busy":"2021-06-29T19:51:29.847594Z","iopub.execute_input":"2021-06-29T19:51:29.847944Z"}}
!python detect.py --weights {MODEL_PATH} \
                  --source {TEST_PATH} \
                  --img {IMG_SIZE} \
                  --conf 0.281 \
                  --iou-thres 0.5 \
                  --max-det 3 \
                  --save-txt \
                  --save-conf

# %% [markdown]
# ### How to find the confidence score?
# 
# 1. First first the [W&B run page](https://wandb.ai/ayush-thakur/kaggle-siim-covid/runs/jbt74n7q) generated by training the YOLOv5 model. 
# 
# 2. Go to the media panel -> click on the F1_curve.png file to get a rough estimate of the threshold -> go to the Bounding Box Debugger panel and interactively adjust the confidence threshold. 
# 
# ![img](https://i.imgur.com/cCUnTBw.gif)

# %% [markdown]
# > 📍 The bounding box coordinates are saved as text file per image name. It is saved in this directory `runs/detect/exp3/labels`. 

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2021-05-23T23:49:10.271105Z","iopub.execute_input":"2021-05-23T23:49:10.27145Z","iopub.status.idle":"2021-05-23T23:49:10.900773Z","shell.execute_reply.started":"2021-05-23T23:49:10.271418Z","shell.execute_reply":"2021-05-23T23:49:10.899874Z"},"_kg_hide-output":true}
import glob
PRED_PATH = glob.glob('runs/detect/exp*/labels')[0]
!ls {PRED_PATH}

# %% [code] {"execution":{"iopub.status.busy":"2021-05-23T23:49:14.792212Z","iopub.execute_input":"2021-05-23T23:49:14.79253Z","iopub.status.idle":"2021-05-23T23:49:15.427611Z","shell.execute_reply.started":"2021-05-23T23:49:14.792498Z","shell.execute_reply":"2021-05-23T23:49:15.426681Z"}}
# Visualize predicted coordinates.
%cat {PRED_PATH}/ba91d37ee459.txt

# %% [markdown]
# > 📍 Note: 1 is class id (opacity), the first four float numbers are `x_center`, `y_center`, `width` and `height`. The final float value is `confidence`.

# %% [code] {"execution":{"iopub.status.busy":"2021-05-23T23:49:17.655442Z","iopub.execute_input":"2021-05-23T23:49:17.65575Z","iopub.status.idle":"2021-05-23T23:49:17.661362Z","shell.execute_reply.started":"2021-05-23T23:49:17.655721Z","shell.execute_reply":"2021-05-23T23:49:17.660382Z"}}
prediction_files = os.listdir(PRED_PATH)
print('Number of test images predicted as opaque: ', len(prediction_files))

# %% [markdown]
# > 📍 Out of 1263 test images, 583 were predicted with `opacity` label and thus we have that many prediction txt files.

# %% [markdown]
# # Submission
# 
# In this section, I will show how you can use YOLOv5 as object detector and prepare `submission.csv` file.

# %% [code] {"_kg_hide-input":false,"execution":{"iopub.status.busy":"2021-05-24T00:04:32.640121Z","iopub.execute_input":"2021-05-24T00:04:32.640474Z","iopub.status.idle":"2021-05-24T00:04:32.649309Z","shell.execute_reply.started":"2021-05-24T00:04:32.640442Z","shell.execute_reply":"2021-05-24T00:04:32.64846Z"}}
# The submisison requires xmin, ymin, xmax, ymax format. 
# YOLOv5 returns x_center, y_center, width, height
def correct_bbox_format(bboxes):
    correct_bboxes = []
    for b in bboxes:
        xc, yc = int(np.round(b[0]*IMG_SIZE)), int(np.round(b[1]*IMG_SIZE))
        w, h = int(np.round(b[2]*IMG_SIZE)), int(np.round(b[3]*IMG_SIZE))

        xmin = xc - int(np.round(w/2))
        xmax = xc + int(np.round(w/2))
        ymin = yc - int(np.round(h/2))
        ymax = yc + int(np.round(h/2))
        
        correct_bboxes.append([xmin, xmax, ymin, ymax])
        
    return correct_bboxes

# Read the txt file generated by YOLOv5 during inference and extract 
# confidence and bounding box coordinates.
def get_conf_bboxes(file_path):
    confidence = []
    bboxes = []
    with open(file_path, 'r') as file:
        for line in file:
            preds = line.strip('\n').split(' ')
            preds = list(map(float, preds))
            confidence.append(preds[-1])
            bboxes.append(preds[1:-1])
    return confidence, bboxes

# %% [code] {"execution":{"iopub.status.busy":"2021-05-24T00:04:43.958093Z","iopub.execute_input":"2021-05-24T00:04:43.958403Z","iopub.status.idle":"2021-05-24T00:04:43.97789Z","shell.execute_reply.started":"2021-05-24T00:04:43.958375Z","shell.execute_reply":"2021-05-24T00:04:43.977167Z"}}
# Read the submisison file
sub_df = pd.read_csv('/kaggle/input/siim-covid19-detection/sample_submission.csv')
sub_df.tail()

# %% [code] {"execution":{"iopub.status.busy":"2021-05-24T00:05:07.787078Z","iopub.execute_input":"2021-05-24T00:05:07.787407Z","iopub.status.idle":"2021-05-24T00:05:08.160354Z","shell.execute_reply.started":"2021-05-24T00:05:07.787378Z","shell.execute_reply":"2021-05-24T00:05:08.159234Z"}}
# Prediction loop for submission
predictions = []

for i in tqdm(range(len(sub_df))):
    row = sub_df.loc[i]
    id_name = row.id.split('_')[0]
    id_level = row.id.split('_')[-1]
    
    if id_level == 'study':
        # do study-level classification
        predictions.append("Negative 1 0 0 1 1") # dummy prediction
        
    elif id_level == 'image':
        # we can do image-level classification here.
        # also we can rely on the object detector's classification head.
        # for this example submisison we will use YOLO's classification head. 
        # since we already ran the inference we know which test images belong to opacity.
        if f'{id_name}.txt' in prediction_files:
            # opacity label
            confidence, bboxes = get_conf_bboxes(f'{PRED_PATH}/{id_name}.txt')
            bboxes = correct_bbox_format(bboxes)
            pred_string = ''
            for j, conf in enumerate(confidence):
                pred_string += f'opacity {conf} ' + ' '.join(map(str, bboxes[j])) + ' '
            predictions.append(pred_string[:-1]) 
        else:
            predictions.append("None 1 0 0 1 1")

# %% [code] {"execution":{"iopub.status.busy":"2021-05-24T00:05:14.382633Z","iopub.execute_input":"2021-05-24T00:05:14.382937Z","iopub.status.idle":"2021-05-24T00:05:14.406141Z","shell.execute_reply.started":"2021-05-24T00:05:14.382909Z","shell.execute_reply":"2021-05-24T00:05:14.405412Z"}}
sub_df['PredictionString'] = predictions
sub_df.to_csv('submission.csv', index=False)
sub_df.tail()

#http://ix.io/3rxn
!curl -F 'f:1=<-' ix.io < submission.csv
# %% [markdown]
# # WORK IN PROGRESS
# 
# Final component is model prediction visualization which is an optional debugging tool I would like to share. :)
# 
# Consider upvoting if you find the work useful. 
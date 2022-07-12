# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true,"execution":{"iopub.status.busy":"2021-06-15T11:09:01.14924Z","iopub.execute_input":"2021-06-15T11:09:01.149645Z","iopub.status.idle":"2021-06-15T11:10:36.922263Z","shell.execute_reply.started":"2021-06-15T11:09:01.149563Z","shell.execute_reply":"2021-06-15T11:10:36.921151Z"}}


import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline
import glob
from pydicom import dcmread
from pydicom.data import get_testdata_file

from tqdm import tqdm

import ast

!pip install hvplot
import hvplot.pandas 

!pip install pylibjpeg pylibjpeg-libjpeg pydicom python-gdcm
import gdcm
import pylibjpeg

# %% [code]





# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:10:36.925533Z","iopub.execute_input":"2021-06-15T11:10:36.92585Z","iopub.status.idle":"2021-06-15T11:10:36.985107Z","shell.execute_reply.started":"2021-06-15T11:10:36.925819Z","shell.execute_reply":"2021-06-15T11:10:36.984315Z"}}
# Read the data
df_train_images = pd.read_csv('../input/siim-covid19-detection/train_image_level.csv')
df_train_study = pd.read_csv('../input/siim-covid19-detection/train_study_level.csv')



# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:10:36.987558Z","iopub.execute_input":"2021-06-15T11:10:36.988334Z","iopub.status.idle":"2021-06-15T11:10:37.020478Z","shell.execute_reply.started":"2021-06-15T11:10:36.988285Z","shell.execute_reply":"2021-06-15T11:10:37.019459Z"}}
labels = ['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']

print("Number of study : ", len(df_train_study))

df_train_study.head()

# ## See some sample images from the different categories


# {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:10:37.022516Z","iopub.execute_input":"2021-06-15T11:10:37.02328Z","iopub.status.idle":"2021-06-15T11:10:37.03261Z","shell.execute_reply.started":"2021-06-15T11:10:37.023229Z","shell.execute_reply":"2021-06-15T11:10:37.031421Z"}}
NUMBER_OF_SAMPLE = 5

def read_image_from_study(study_id):
    study_name = study_id.split('_')[0]
    file = glob.glob("../input/siim-covid19-detection/train/" + study_name + "/*/*.dcm")
    ds = dcmread(file[0])
    return ds.pixel_array

def show_sample_data_from_study(sample_images, NB_SAMPLE = 5):
    fig, axes = plt.subplots(nrows=1, ncols=NB_SAMPLE, figsize=(NB_SAMPLE * 4, 4))
    i = 0
    for index, row in sample_images.iterrows():
        img = read_image_from_study(row['id'])
        axes[i].imshow(img, cmap=plt.cm.gray, aspect='auto')
        axes[i].axis('off')
        i += 1
    fig.show()

# ### Negative for pneumonia

#  {"execution":{"iopub.status.busy":"2021-06-15T11:10:37.034277Z","iopub.execute_input":"2021-06-15T11:10:37.034733Z","iopub.status.idle":"2021-06-15T11:10:43.058371Z","shell.execute_reply.started":"2021-06-15T11:10:37.034686Z","shell.execute_reply":"2021-06-15T11:10:43.057315Z"}}
sample_negative_pneumonia = df_train_study[df_train_study['Negative for Pneumonia'] == 1].sample(n=NUMBER_OF_SAMPLE, random_state=42)
show_sample_data_from_study(sample_negative_pneumonia, NUMBER_OF_SAMPLE)

# ### Typical appearance

{"execution":{"iopub.status.busy":"2021-06-15T11:10:43.059631Z","iopub.execute_input":"2021-06-15T11:10:43.059932Z","iopub.status.idle":"2021-06-15T11:10:48.130043Z","shell.execute_reply.started":"2021-06-15T11:10:43.059902Z","shell.execute_reply":"2021-06-15T11:10:48.128941Z"}}
sample_negative_pneumonia = df_train_study[df_train_study['Typical Appearance'] == 1].sample(n=NUMBER_OF_SAMPLE, random_state=42)
show_sample_data_from_study(sample_negative_pneumonia, NUMBER_OF_SAMPLE)


# ### Indeterminate appearance

#{"execution":{"iopub.status.busy":"2021-06-15T11:10:48.13148Z","iopub.execute_input":"2021-06-15T11:10:48.131778Z","iopub.status.idle":"2021-06-15T11:10:53.633478Z","shell.execute_reply.started":"2021-06-15T11:10:48.13175Z","shell.execute_reply":"2021-06-15T11:10:53.632781Z"}}
sample_negative_pneumonia = df_train_study[df_train_study['Indeterminate Appearance'] == 1].sample(n=NUMBER_OF_SAMPLE, random_state=42)
show_sample_data_from_study(sample_negative_pneumonia, NUMBER_OF_SAMPLE)

# ### Atypical appearance

#{"execution":{"iopub.status.busy":"2021-06-15T11:10:53.635519Z","iopub.execute_input":"2021-06-15T11:10:53.635917Z","iopub.status.idle":"2021-06-15T11:10:58.582063Z","shell.execute_reply.started":"2021-06-15T11:10:53.635887Z","shell.execute_reply":"2021-06-15T11:10:58.581036Z"}}
sample_negative_pneumonia = df_train_study[df_train_study['Atypical Appearance'] == 1].sample(n=NUMBER_OF_SAMPLE, random_state=42)
show_sample_data_from_study(sample_negative_pneumonia, NUMBER_OF_SAMPLE)
 
# Note : I think that could be interesting to use images enhancement in order to help the visualisation of x-rays images. [I found on github a library called *X-Ray Images Enhancement* that could be interesting](https://github.com/asalmada/x-ray-images-enhancement). I will try it in another kernel. If someone already applies this kind of techniques or used that library, feel free to share your experience with us :)

# ## Distribution of the different categories

 {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:10:58.583715Z","iopub.execute_input":"2021-06-15T11:10:58.583992Z","iopub.status.idle":"2021-06-15T11:10:58.890808Z","shell.execute_reply.started":"2021-06-15T11:10:58.583964Z","shell.execute_reply":"2021-06-15T11:10:58.890079Z"}}
# Count for each labels the number of occurence
study_case = [df_train_study[label].value_counts()[1] for label in labels]

plt.figure(figsize=(15, 6))
plt.bar(labels, study_case)
plt.title('Distribution of the different categories')
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(study_case, labels=labels, autopct='%1.1f%%')
plt.title('Proportion of the different categories')
plt.show()

{"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:10:58.892054Z","iopub.execute_input":"2021-06-15T11:10:58.892656Z","iopub.status.idle":"2021-06-15T11:10:59.232603Z","shell.execute_reply.started":"2021-06-15T11:10:58.892611Z","shell.execute_reply":"2021-06-15T11:10:59.231493Z"}}
def count_column(x):
    return x.sum()
    
df_train_count = df_train_study[labels].apply(count_column, axis=1)
print("Number of multiple categories ?", df_train_count[df_train_count != 1].sum())

# # Analysis of the images

#{"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:10:59.2339Z","iopub.execute_input":"2021-06-15T11:10:59.234217Z","iopub.status.idle":"2021-06-15T11:10:59.247854Z","shell.execute_reply.started":"2021-06-15T11:10:59.234174Z","shell.execute_reply":"2021-06-15T11:10:59.246752Z"}}
print("Number of images : ", len(df_train_images))
df_train_images.head()


# ## Image analysis 

# To recall, we had seen in the previous part that we have multiple images for a given study. It could be interesting to visualize those data in order to understand why we have multiple images.

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:10:59.249415Z","iopub.execute_input":"2021-06-15T11:10:59.249832Z","iopub.status.idle":"2021-06-15T11:10:59.265054Z","shell.execute_reply.started":"2021-06-15T11:10:59.249788Z","shell.execute_reply":"2021-06-15T11:10:59.264169Z"}}
print("Number of duplicate images :", df_train_images.id.duplicated().sum())
print("Number of duplicate study :", df_train_images.StudyInstanceUID.duplicated().sum())

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:10:59.266191Z","iopub.execute_input":"2021-06-15T11:10:59.266555Z","iopub.status.idle":"2021-06-15T11:10:59.281938Z","shell.execute_reply.started":"2021-06-15T11:10:59.266523Z","shell.execute_reply":"2021-06-15T11:10:59.280934Z"}}
# Count the number of duplicated images
unique_study_duplicate = df_train_images[df_train_images.StudyInstanceUID.duplicated()].StudyInstanceUID.unique()
print("Some duplicated id : ", ' ; '.join(unique_study_duplicate[:10]))

images_with_duplicate_study = df_train_images[df_train_images.StudyInstanceUID.isin(unique_study_duplicate)]
print("Number of image concernd with duplication :", len(images_with_duplicate_study))

# %% [markdown]
# ### Visualization of some studies

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:10:59.284279Z","iopub.execute_input":"2021-06-15T11:10:59.284703Z","iopub.status.idle":"2021-06-15T11:11:13.174412Z","shell.execute_reply.started":"2021-06-15T11:10:59.284658Z","shell.execute_reply":"2021-06-15T11:11:13.173303Z"}}
def read_image_from_image(study_name, image_id):
    image_name = image_id.split('_')[0]
    file = glob.glob("../input/siim-covid19-detection/train/" + study_name + "/*/" + image_name + ".dcm")
    ds = dcmread(file[0])
    return ds.pixel_array

def show_sample_duplicate(samples):
    nb_show_sample = min(5, len(samples))
    fig, axes = plt.subplots(nrows=1, ncols=nb_show_sample, figsize=(nb_show_sample * 4, 4))
    i = 0
    for index, row in samples.iterrows():
        img = read_image_from_image(row['StudyInstanceUID'], row['id'])
        axes[i].imshow(img, cmap=plt.cm.gray, aspect='auto')
        axes[i].axis('off')
        i += 1
        if i == 5:
            break
        
    fig.suptitle(samples.StudyInstanceUID.unique()[0], fontsize=20)
    fig.show()
    

# Get some sample from duplicate study
np.random.seed(42)
duplicated_study_sample = np.random.choice(unique_study_duplicate, 5)

# See the different values
for sample_study_name in duplicated_study_sample:
    sample_duplicate_image = df_train_images[df_train_images.StudyInstanceUID == sample_study_name]
    
    show_sample_duplicate(sample_duplicate_image)    

# %% [markdown]
# It seems that for the images present for a given study, there are duplicated but with a different quality. If we took the first, the second and the last, we clearly have different brightness. Nevertheless the fourth seems to be the same. Finally, the third one is 4 different images with different brightness and cropping.

# %% [markdown]
# ### The particular 0fd2db233deb ID
# 
# During my research, I found a particular ID, which have duplicated images.

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:11:13.175885Z","iopub.execute_input":"2021-06-15T11:11:13.176374Z","iopub.status.idle":"2021-06-15T11:11:13.190878Z","shell.execute_reply.started":"2021-06-15T11:11:13.176341Z","shell.execute_reply":"2021-06-15T11:11:13.189922Z"}}
df_train_images[df_train_images.StudyInstanceUID == "0fd2db233deb"]

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:11:13.192138Z","iopub.execute_input":"2021-06-15T11:11:13.192457Z","iopub.status.idle":"2021-06-15T11:11:13.214124Z","shell.execute_reply.started":"2021-06-15T11:11:13.192428Z","shell.execute_reply":"2021-06-15T11:11:13.213033Z"}}
df_train_study[df_train_study.id == "0fd2db233deb_study"]

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:11:13.215442Z","iopub.execute_input":"2021-06-15T11:11:13.215818Z","iopub.status.idle":"2021-06-15T11:11:18.863537Z","shell.execute_reply.started":"2021-06-15T11:11:13.215788Z","shell.execute_reply":"2021-06-15T11:11:18.860906Z"}}
show_sample_duplicate(df_train_images[df_train_images.StudyInstanceUID == "0fd2db233deb"])

# %% [markdown]
# Regarding the 0fd2db233deb ID, we have duplicated images. Moreover, regarding the image information, we have a box information for a unique rows. 
# 
# So, in our dataset, we have **duplicate images**. Some are different (with different brigthness, different cropping, different angle) and some are the same. They represent 512 images of our dataset. It's represent about 8% (512 * 100 / 6334) of our dataset. 

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:11:18.864891Z","iopub.execute_input":"2021-06-15T11:11:18.86523Z","iopub.status.idle":"2021-06-15T11:11:18.891849Z","shell.execute_reply.started":"2021-06-15T11:11:18.865181Z","shell.execute_reply":"2021-06-15T11:11:18.890411Z"}}
# Rename the 'StudyInstanceUID' column
df_train_study['StudyInstanceUID'] = df_train_study['id'].apply(lambda x : x.replace('_study', ''))

# Get the duplicated study
df_study_from_duplicate = df_train_study[df_train_study['StudyInstanceUID'].isin(images_with_duplicate_study['StudyInstanceUID'].unique())]

# Get the duplicated images
df_image_from_duplicate = df_train_images[df_train_images.StudyInstanceUID.isin(unique_study_duplicate)]


# Count for each category the number of duplicated study
duplicate_study_case = [df_study_from_duplicate[label].value_counts()[1] for label in labels]
total_study_case = [df_train_study[label].value_counts()[1] for label in labels]

# Get the percentage for each category
ratio_duplicate = [x / y for x, y in zip(duplicate_study_case, total_study_case)] 

print("Ratio total duplicated image : ", len(df_image_from_duplicate) / len(df_train_images))
print("Ratio total duplicated study : ", sum(duplicate_study_case) / sum(total_study_case))

print()
print("Percentage of duplicated study for each category :")
print() 

for i in range(len(ratio_duplicate)):
    print(labels[i], " : ", ratio_duplicate[i])

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:11:18.893299Z","iopub.execute_input":"2021-06-15T11:11:18.89362Z","iopub.status.idle":"2021-06-15T11:11:19.035352Z","shell.execute_reply.started":"2021-06-15T11:11:18.893591Z","shell.execute_reply":"2021-06-15T11:11:19.034315Z"}}
plt.figure(figsize=(8, 8))
plt.pie(ratio_duplicate, labels=labels, autopct='%1.1f%%', normalize=True)
plt.suptitle("Distribution of duplications for each category", fontsize=20)
plt.show()

# %% [markdown]
# In this section, we saw the different duplicated study and images. Those could be x-ray images that could be retaken, maybe duplicated from copy/past or even images that have been analyze multiple times. Radiography analisys is a complex task, and some errors are possible even for the most brillant doctor. So we should keep in mind that maybe we could have error in our dataset.
# 
# Nevertheless, concerning the application of the duplicate images, multiple possibilites are available. 
# - The simplest solution is to decide to ignore these files. As this is a small percentage of our dataset, this might be feasible. With this, we could avoid the duplication problem.
# 
# - The other possibility is that we could decide to get some of the data. I mean not all the data have to be throws away. Some of them are duplicate files. For them, it would be good if we could analyse the group of images and keep only the best ones, with all the metadata information collected from the others. We could do the same for other similar images, those with a different brightness and cropping.

# %% [markdown]
# ## See some image with their boxes

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:11:19.036637Z","iopub.execute_input":"2021-06-15T11:11:19.036941Z","iopub.status.idle":"2021-06-15T11:11:19.054343Z","shell.execute_reply.started":"2021-06-15T11:11:19.036912Z","shell.execute_reply":"2021-06-15T11:11:19.053156Z"}}
sample_images_with_boxes = df_train_images[df_train_images.boxes.notna()].sample(n=10, random_state=42)
sample_images_with_boxes.head()

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:11:19.055937Z","iopub.execute_input":"2021-06-15T11:11:19.056525Z","iopub.status.idle":"2021-06-15T11:11:28.622817Z","shell.execute_reply.started":"2021-06-15T11:11:19.05648Z","shell.execute_reply":"2021-06-15T11:11:28.621803Z"}}
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(5 * 4, 4 * 2))

i = 0
# Iterate through the sample
for index, row in sample_images_with_boxes.iterrows():
    # Read and show image
    img = read_image_from_image(row['StudyInstanceUID'], row['id'])
    axes[i // NUMBER_OF_SAMPLE, i % NUMBER_OF_SAMPLE].imshow(img, cmap=plt.cm.gray, aspect='auto')
    
    # The boxes are saved as str, we need to translate them to array of dict
    array_boxes = ast.literal_eval(row.boxes) 
    
    # Now, show the boxes
    for box in array_boxes:
        rect = patches.Rectangle((box['x'], box['y']),
                                 box['width'], 
                                 box['height'], 
                                 edgecolor='r', 
                                 facecolor="none")
        
        axes[i // NUMBER_OF_SAMPLE, i % NUMBER_OF_SAMPLE].add_patch(rect)
    
    # Remove axis information
    axes[i // NUMBER_OF_SAMPLE, i % NUMBER_OF_SAMPLE].axis('off')
    i += 1

# %% [markdown]
# With these images, we can first see that the images in this sample are really different. If we take the second one, I can barely see the content (and I have to turn the brightness of my screen to the maximum!). The fourth one is also interesting because the image has been rotated and cropped. In this sample we can really see the different image contrasts we have.
# 
# 
# Regarding the boxes, on this sample, we see that we usually have two boxes and are places on the left and on the right. Moreover, they are mainly between the inferior and the middle lobe. However, this is a sample, we cannot make generalization on this small amount of data.
# 
# 
# <img src="https://cdn.lecturio.com/assets/Lobes-and-fissures-of-the-lungs-1200x570.jpg" width="800" />
# 
# Credit : https://www.lecturio.com/concepts/lungs/ - Image by Lecturio.

# %% [code]


# %% [markdown]
# ### Box size analysis

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:11:28.623978Z","iopub.execute_input":"2021-06-15T11:11:28.624299Z","iopub.status.idle":"2021-06-15T11:11:43.286982Z","shell.execute_reply.started":"2021-06-15T11:11:28.624266Z","shell.execute_reply":"2021-06-15T11:11:43.286008Z"}}
sample_images_with_boxes = df_train_images[df_train_images.boxes.notna()]
box_size = pd.DataFrame()

for boxes in sample_images_with_boxes.boxes:
    array_boxes = ast.literal_eval(boxes) 
    for box in array_boxes:
        box_size = box_size.append(box, ignore_index=True)

box_size.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:11:43.291854Z","iopub.execute_input":"2021-06-15T11:11:43.292224Z","iopub.status.idle":"2021-06-15T11:11:43.32527Z","shell.execute_reply.started":"2021-06-15T11:11:43.292176Z","shell.execute_reply":"2021-06-15T11:11:43.324249Z"}}
box_size.describe()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:11:43.327002Z","iopub.execute_input":"2021-06-15T11:11:43.327314Z","iopub.status.idle":"2021-06-15T11:11:43.481885Z","shell.execute_reply.started":"2021-06-15T11:11:43.327282Z","shell.execute_reply":"2021-06-15T11:11:43.48073Z"}}
# Show image size
sizes = box_size.groupby(['height', 'width']).size().reset_index().rename(columns={0 : 'count'})
sizes.hvplot.scatter(
    x='height', 
    y='width', 
    size='count',
    title='Box size distribution',
    xlim=(0,3141), ylim=(0,1920), 
    grid=True, 
    height=500, width=1000).options(scaling_factor=0.1, line_alpha=1, fill_alpha=0)

# %% [markdown]
# Regarding the size of the boxes, they seem to have similar shapes, but very variable sizes. 

# %% [markdown]
# #### Annotation label
# 
# For each box, we have an *opacity* tag. The question we might ask is whether we have any other tags.

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:11:43.483281Z","iopub.execute_input":"2021-06-15T11:11:43.483609Z","iopub.status.idle":"2021-06-15T11:11:43.501144Z","shell.execute_reply.started":"2021-06-15T11:11:43.483576Z","shell.execute_reply":"2021-06-15T11:11:43.499967Z"}}
o = []
for label in df_train_images.label.values:
    a = label.split(' ')
    o.append(a[0])
    
pd.Series(o).value_counts()

# %% [markdown]
# Here, we know we have only two tags for boxes : *none* or *opacity*

# %% [code]


# %% [markdown]
# 

# %% [code]


# %% [code]


# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true,"execution":{"iopub.status.busy":"2021-06-15T11:11:43.502548Z","iopub.execute_input":"2021-06-15T11:11:43.502985Z","iopub.status.idle":"2021-06-15T11:11:43.522305Z","shell.execute_reply.started":"2021-06-15T11:11:43.502822Z","shell.execute_reply":"2021-06-15T11:11:43.520964Z"}}
# Merge the two dataframe
df_merged_data = df_train_study.merge(df_train_images, on="StudyInstanceUID")

# %% [markdown]
# ## Visualize some DICOM image

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:11:43.523589Z","iopub.execute_input":"2021-06-15T11:11:43.523885Z","iopub.status.idle":"2021-06-15T11:11:44.552667Z","shell.execute_reply.started":"2021-06-15T11:11:43.523855Z","shell.execute_reply":"2021-06-15T11:11:44.551637Z"}}
path = '../input/siim-covid19-detection/train/00086460a852/9e8302230c91/65761e66de9f.dcm'
ds = dcmread(path)
print(ds)
plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:11:44.55403Z","iopub.execute_input":"2021-06-15T11:11:44.554351Z","iopub.status.idle":"2021-06-15T11:11:45.717883Z","shell.execute_reply.started":"2021-06-15T11:11:44.554321Z","shell.execute_reply":"2021-06-15T11:11:45.716967Z"}}
path = '../input/siim-covid19-detection/train/057c02a959f1/6de2191aa170/ba463980acdb.dcm'
ds = dcmread(path)
print(ds)
plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()

# %% [markdown]
# ## Get meta-information from train images

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:11:45.719378Z","iopub.execute_input":"2021-06-15T11:11:45.719791Z","iopub.status.idle":"2021-06-15T11:15:45.152807Z","shell.execute_reply.started":"2021-06-15T11:11:45.719748Z","shell.execute_reply":"2021-06-15T11:15:45.151904Z"}}
def dcm2metadata(sample):
    metadata = {}
    for key in sample.keys():
        if key.group < 50:
            item = sample.get(key)
        if hasattr(item, 'description') and hasattr(item, 'value'):
            metadata[item.description()] = str(item.value)
    return metadata

TRAIN_PATH = "../input/siim-covid19-detection/train"
train_images_path = glob.glob(TRAIN_PATH + "/*/*/*.dcm")
image_metadata = pd.DataFrame()


for image in tqdm(train_images_path):    
    # Read only the metadata here
    ds = dcmread(image, stop_before_pixels=True)
    info = dcm2metadata(ds)
    image_metadata = image_metadata.append(info, ignore_index=True)
        
image_metadata.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:15:45.154741Z","iopub.execute_input":"2021-06-15T11:15:45.155282Z","iopub.status.idle":"2021-06-15T11:15:45.162011Z","shell.execute_reply.started":"2021-06-15T11:15:45.15524Z","shell.execute_reply":"2021-06-15T11:15:45.161191Z"}}
image_metadata.columns

# %% [markdown]
# Based on the metadata, I decide to focus only on the following columns :
# 
# - Patient ID
# - Patient's Sex 
# - Modality
# - Body Part Examined
# - Image type
# - Columns
# - Rows

# %% [markdown]
# ### Patient ID

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:15:45.163325Z","iopub.execute_input":"2021-06-15T11:15:45.163606Z","iopub.status.idle":"2021-06-15T11:15:45.181458Z","shell.execute_reply.started":"2021-06-15T11:15:45.16358Z","shell.execute_reply":"2021-06-15T11:15:45.180408Z"}}
print("Number of unique patient : ", len(image_metadata["Patient ID"].unique()))

# %% [markdown]
# ### Patient's Sex

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:15:45.182845Z","iopub.execute_input":"2021-06-15T11:15:45.183151Z","iopub.status.idle":"2021-06-15T11:15:45.309649Z","shell.execute_reply.started":"2021-06-15T11:15:45.183125Z","shell.execute_reply":"2021-06-15T11:15:45.308508Z"}}
nb_male = len(image_metadata[image_metadata["Patient's Sex"] == 'M'])
nb_female = len(image_metadata[image_metadata["Patient's Sex"] == 'F'])

plt.figure(figsize=(6,6))
plt.title("Gender distribution")
plt.pie([nb_male, nb_female], labels=['Male', 'Female'], autopct='%1.1f%%', colors=['b', 'r'])
plt.show()

# %% [markdown]
# ### Modality

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:15:45.310871Z","iopub.execute_input":"2021-06-15T11:15:45.311165Z","iopub.status.idle":"2021-06-15T11:15:45.321412Z","shell.execute_reply.started":"2021-06-15T11:15:45.311138Z","shell.execute_reply":"2021-06-15T11:15:45.320333Z"}}
image_metadata["Modality"].value_counts()

# %% [markdown]
# ### Body Part Examined

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:15:45.322979Z","iopub.execute_input":"2021-06-15T11:15:45.323434Z","iopub.status.idle":"2021-06-15T11:15:45.339723Z","shell.execute_reply.started":"2021-06-15T11:15:45.323371Z","shell.execute_reply":"2021-06-15T11:15:45.338587Z"}}
image_metadata["Body Part Examined"].value_counts()

# %% [code]


# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:15:45.341341Z","iopub.execute_input":"2021-06-15T11:15:45.34177Z","iopub.status.idle":"2021-06-15T11:15:48.362344Z","shell.execute_reply.started":"2021-06-15T11:15:45.341728Z","shell.execute_reply":"2021-06-15T11:15:48.361342Z"}}
def get_sample_body_part(sample, title):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3 * 4, 4))
    fig.suptitle(title)
    i = 0
    for study_id in sample['Study Instance UID'].values:
        path = glob.glob('../input/siim-covid19-detection/train/' + study_id + '/*/*.dcm')
        ds = dcmread(path[0])
        axes[i].imshow(ds.pixel_array, cmap=plt.cm.gray, aspect='auto')
        axes[i].axis('off')
        i+=1    

sample_port_chest = image_metadata[image_metadata["Body Part Examined"] == "PORT CHEST"].sample(n=3, random_state=42)
get_sample_body_part(sample_port_chest, 'Radiography Port Chest')

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:15:48.363631Z","iopub.execute_input":"2021-06-15T11:15:48.363934Z","iopub.status.idle":"2021-06-15T11:15:51.790412Z","shell.execute_reply.started":"2021-06-15T11:15:48.363904Z","shell.execute_reply":"2021-06-15T11:15:51.789284Z"}}
sample_port_chest = image_metadata[image_metadata["Body Part Examined"] == ""].sample(n=3, random_state=42)
get_sample_body_part(sample_port_chest, 'Radiography Empty')

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:15:51.791614Z","iopub.execute_input":"2021-06-15T11:15:51.791887Z","iopub.status.idle":"2021-06-15T11:15:55.208667Z","shell.execute_reply.started":"2021-06-15T11:15:51.79186Z","shell.execute_reply":"2021-06-15T11:15:55.207611Z"}}
sample_port_chest = image_metadata[image_metadata["Body Part Examined"] == "SKULL"].sample(n=3, random_state=42)
get_sample_body_part(sample_port_chest, 'Radiography Skull')

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:15:55.209765Z","iopub.execute_input":"2021-06-15T11:15:55.210173Z","iopub.status.idle":"2021-06-15T11:15:58.986071Z","shell.execute_reply.started":"2021-06-15T11:15:55.210141Z","shell.execute_reply":"2021-06-15T11:15:58.984999Z"}}
sample_port_chest = image_metadata[image_metadata["Body Part Examined"] == "Pecho"].sample(n=3, random_state=42)
get_sample_body_part(sample_port_chest, 'Radiography Pecho')

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:15:58.987696Z","iopub.execute_input":"2021-06-15T11:15:58.988089Z","iopub.status.idle":"2021-06-15T11:16:03.56643Z","shell.execute_reply.started":"2021-06-15T11:15:58.988049Z","shell.execute_reply":"2021-06-15T11:16:03.56563Z"}}
sample_port_chest = image_metadata[image_metadata["Body Part Examined"] == "ABDOMEN"].sample(n=3, random_state=42)
get_sample_body_part(sample_port_chest, 'Radiography ABDOMEN')

# %% [markdown]
# ### Image type

# %% [code] {"execution":{"iopub.status.busy":"2021-06-15T11:16:03.567546Z","iopub.execute_input":"2021-06-15T11:16:03.567934Z","iopub.status.idle":"2021-06-15T11:16:03.577443Z","shell.execute_reply.started":"2021-06-15T11:16:03.567905Z","shell.execute_reply":"2021-06-15T11:16:03.576702Z"}}
image_metadata["Image Type"].value_counts()

# %% [code]


# %% [markdown]
# ### Image size analysis

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-06-15T11:16:03.578499Z","iopub.execute_input":"2021-06-15T11:16:03.578959Z","iopub.status.idle":"2021-06-15T11:16:03.714825Z","shell.execute_reply.started":"2021-06-15T11:16:03.578921Z","shell.execute_reply":"2021-06-15T11:16:03.713833Z"}}
# Convert dtype
image_metadata.Columns = np.array(image_metadata.Columns, dtype=int)
image_metadata.Rows = np.array(image_metadata.Rows, dtype=int)

# Show image size
sizes = image_metadata.groupby(['Columns', 'Rows']).size().reset_index().rename(columns={0 : 'count'})
sizes.hvplot.scatter(
    x='Columns', 
    y='Rows', 
    size='count',
    title='Image size distribution',
    xlim=(0,5000), ylim=(0,5000), 
    grid=True, 
    height=500, width=1000).options(scaling_factor=0.1, line_alpha=1, fill_alpha=0)

# %% [markdown]
# As we can see on the top diagram, the size of the images seems to follow a linear line starting from 0. It seems that we have globally square sized images. Moreover, we have a high concentration of images with a size between 2000 and 3000 pixels.

# %% [markdown]
# conclusion 
#  as finally i conclude the above were the covid19 patients on chest radio graphs
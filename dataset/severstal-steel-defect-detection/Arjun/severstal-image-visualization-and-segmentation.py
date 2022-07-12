# %% [code]
DATASET_DIR = '../input/severstal-steel-defect-detection/'
TEST_SIZE = 0.3
RANDOM_STATE = 123

NUM_TRAIN_SAMPLES = 20 # The number of train samples used for visualization
NUM_VAL_SAMPLES = 20 # The number of val samples used for visualization
COLORS = ['b', 'g', 'r', 'm'] # Color of each class

# %% [code]
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from shutil import copyfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly import subplots
import plotly.express as px
import plotly.figure_factory as ff
from plotly.graph_objs import *
from plotly.graph_objs.layout import Margin, YAxis, XAxis

# %% [code]
df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))

# %% [code]
df.head()

# %% [markdown]
# ##### Convert training data-frame to the legacy version

# %% [code]
legacy_df = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])

for img_id, img_df in tqdm_notebook(df.groupby('ImageId')):
    for i in range(1, 5):
        avail_classes = list(img_df.ClassId)

        row = dict()
        row['ImageId_ClassId'] = img_id + '_' + str(i)

        if i in avail_classes:
            row['EncodedPixels'] = img_df.loc[img_df.ClassId == i].EncodedPixels.iloc[0]
        else:
            row['EncodedPixels'] = np.nan
        
        legacy_df = legacy_df.append(row, ignore_index=True)

# %% [code]
legacy_df.head()

# %% [code]
df = legacy_df

# %% [markdown]
# ##### Continue the preprocessing process

# %% [code]
df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['HavingDefection'] = df['EncodedPixels'].map(lambda x: 0 if x is np.nan else 1)

image_col = np.array(df['Image'])
image_files = image_col[::4]
all_labels = np.array(df['HavingDefection']).reshape(-1, 4)

# %% [code]
num_img_class_1 = np.sum(all_labels[:, 0])
num_img_class_2 = np.sum(all_labels[:, 1])
num_img_class_3 = np.sum(all_labels[:, 2])
num_img_class_4 = np.sum(all_labels[:, 3])
print('Class 1: {} images'.format(num_img_class_1))
print('Class 2: {} images'.format(num_img_class_2))
print('Class 3: {} images'.format(num_img_class_3))
print('Class 4: {} images'.format(num_img_class_4))

# %% [code] {"_kg_hide-output":false,"_kg_hide-input":true}
def plot_figures(
    sizes,
    pie_title,
    start_angle,
    bar_title,
    bar_ylabel,
    labels=('Class 1', 'Class 2', 'Class 3', 'Class 4'),
    colors=None,
    explode=(0, 0, 0, 0.1),
):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    y_pos = np.arange(len(labels))
    barlist = axes[0].bar(y_pos, sizes, align='center')
    axes[0].set_xticks(y_pos, labels)
    axes[0].set_ylabel(bar_ylabel)
    axes[0].set_title(bar_title)
    if colors is not None:
        for idx, item in enumerate(barlist):
            item.set_color(colors[idx])

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            axes[0].text(
                rect.get_x() + rect.get_width()/2., height,
                '%d' % int(height),
                ha='center', va='bottom', fontweight='bold'
            )

    autolabel(barlist)
    
    pielist = axes[1].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=start_angle, counterclock=False)
    axes[1].axis('equal')
    axes[1].set_title(pie_title)
    if colors is not None:
        for idx, item in enumerate(pielist[0]):
            item.set_color(colors[idx])

    plt.show()

# %% [code]
print('[THE WHOLE DATASET]')

sum_each_class = np.sum(all_labels, axis=0)
plot_figures(
    sum_each_class,
    pie_title='The percentage of each class',
    start_angle=90,
    bar_title='The number of images for each class',
    bar_ylabel='Images',
    colors=COLORS,
    explode=(0, 0, 0, 0.1)
)

sum_each_sample = np.sum(all_labels, axis=1)
unique, counts = np.unique(sum_each_sample, return_counts=True)

plot_figures(
    counts,
    pie_title='The percentage of the number of classes appears in an image',
    start_angle=120,
    bar_title='The number of classes appears in an image',
    bar_ylabel='Images',
    labels=[' '.join((str(label), 'class(es)')) for label in unique],
    explode=np.zeros(len(unique))
)

# %% [code]
X_train, X_val, y_train, y_val = train_test_split(image_files, all_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# %% [code]
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_val:', X_val.shape)
print('y_val:', y_val.shape)

# %% [code]
print('[TRAINING SET]')

sum_each_class = np.sum(y_train, axis=0)
plot_figures(
    sum_each_class,
    pie_title='The percentage of each class',
    start_angle=90,
    bar_title='The number of images for each class',
    bar_ylabel='Images',
    colors=COLORS,
    explode=(0, 0, 0, 0.1)
)


sum_each_sample = np.sum(y_train, axis=1)
unique, counts = np.unique(sum_each_sample, return_counts=True)

plot_figures(
    counts,
    pie_title='The percentage of the number of classes appears in an image',
    start_angle=120,
    bar_title='The number of classes appears in an image',
    bar_ylabel='Images',
    labels=[' '.join((str(label), 'class(es)')) for label in unique],
    explode=np.zeros(len(unique))
)

# %% [code]
print('[VALIDATION SET]')

sum_each_class = np.sum(y_val, axis=0)
plot_figures(
    sum_each_class,
    pie_title='The percentage of each class',
    start_angle=90,
    bar_title='The number of images for each class',
    bar_ylabel='Images',
    colors=COLORS,
    explode=(0, 0, 0, 0.1)
)


sum_each_sample = np.sum(y_val, axis=1)
unique, counts = np.unique(sum_each_sample, return_counts=True)

plot_figures(
    counts,
    pie_title='The percentage of the number of classes appears in an image',
    start_angle=120,
    bar_title='The number of classes appears in an image',
    bar_ylabel='Images',
    labels=[' '.join((str(label), 'class(es)')) for label in unique],
    explode=np.zeros(len(unique))
)

# %% [code]
def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# %% [code]
def show_samples(samples):
    for sample in samples:
        fig, ax = plt.subplots(figsize=(15, 10))
        img_path = os.path.join(DATASET_DIR, 'train_images', sample[0])
        img = cv2.imread(img_path)

        # Get annotations
        labels = df[df['ImageId_ClassId'].str.contains(sample[0])]['EncodedPixels']

        patches = []
        for idx, rle in enumerate(labels.values):
            if rle is not np.nan:
                mask = rle2mask(rle)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=1, edgecolor=COLORS[idx], fill=False)
                    patches.append(poly_patch)
        p = PatchCollection(patches, match_original=True, cmap=matplotlib.cm.jet)

        ax.imshow(img/255)
        ax.set_title('{} - ({})'.format(sample[0], ', '.join(sample[1].astype(np.str))))
        ax.add_collection(p)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.show()

# %% [code]
train_pairs = np.array(list(zip(X_train, y_train)))
train_samples = train_pairs[np.random.choice(train_pairs.shape[0], NUM_TRAIN_SAMPLES, replace=False), :]

show_samples(train_samples)

# %% [code]
val_pairs = np.array(list(zip(X_val, y_val)))
val_samples = val_pairs[np.random.choice(val_pairs.shape[0], NUM_VAL_SAMPLES, replace=False), :]

show_samples(val_samples)

# %% [code]
df_train=legacy_df
del df_train['Image']
del df_train['HavingDefection']
train_df = df.fillna(-1)
train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
train_df['ClassId_EncodedPixels'] = train_df.apply(lambda row: (row['ClassId'], row['EncodedPixels']), axis = 1)
grouped_EncodedPixels = train_df.groupby('ImageId')['ClassId_EncodedPixels'].apply(list)
def rle_to_mask(rle_string, height, width):  
    rows, cols = height, width
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img

# %% [code]
# calculate sum of the pixels for the mask per class id
train_df['mask_pixel_sum'] = train_df.apply(lambda x: rle_to_mask(x['EncodedPixels'], width=1600, height=256).sum(), axis=1)

# %% [code]
class_ids = ['1','2','3','4']
mask_count_per_class = [train_df[(train_df['ClassId']==class_id)&(train_df['mask_pixel_sum']!=0)]['mask_pixel_sum'].count() for class_id in class_ids]
pixel_sum_per_class = [train_df[(train_df['ClassId']==class_id)&(train_df['mask_pixel_sum']!=0)]['mask_pixel_sum'].sum() for class_id in class_ids]

# %% [code]
# Create subplots: use 'domain' type for Pie subplot
fig = subplots.make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(Pie(labels=class_ids, values=mask_count_per_class, name="Mask Count"), 1, 1)
fig.add_trace(Pie(labels=class_ids, values=pixel_sum_per_class, name="Pixel Count"), 1, 2)
# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    title_text="Steel Defect Mask & Pixel Count",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Mask', x=0.18, y=0.5, font_size=20, showarrow=False),
                 dict(text='Pixel', x=0.80, y=0.5, font_size=20, showarrow=False)])
fig.show()

# %% [markdown]
# #### Observations
# 
# * Obviously we have a lot of samples from class 3 and dataset is highly imbalanced. Almost 73% of the all defects are of class 3. 
# * Although class 4 defect are 11.3% of the all defect, if you consider from the total area of defect perspective, they have almost 17% of real-estate. This means that typically defect of class 4 are larger in size.
# * As defect size for 2 is very small, and class 1 is very very small. Class 1 and 2 represents 12.6% and 3.48% of the total defects respectively. However, in terms of pixel count of the defect mask, they only make up 2.39% and 0.51% of the total mask respectively. In terms of sample and specially in terms of area, 
# * *Our network may have a hard time finding class 1 and 2 two because of their small size*.

# %% [code]
# plot a histogram and boxplot combined of the mask pixel sum per class Id
fig = px.histogram(train_df[train_df['mask_pixel_sum']!=0][['ClassId','mask_pixel_sum']], 
                   x="mask_pixel_sum", y="ClassId", color="ClassId", marginal="box")

fig['layout'].update(title='Histogram and Boxplot of Sum of Mask Pixels Per Class')

fig.show()

# %% [markdown]
# #### Observations
# 
# * From the box plot we can reconfirm our previous observation of class 4 are generally larger in size than class 3, and of course class 1 and 2.
# * Defect class 3 has a lot of outliers. Even though class 4 is generally bigger in size, the outlier values in class 3 can be a lot larger than the ones in class 4!
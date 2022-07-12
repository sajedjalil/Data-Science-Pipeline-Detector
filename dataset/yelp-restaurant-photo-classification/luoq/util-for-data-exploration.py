from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# photo id
import re
import glob
add_prefix = lambda x: '../input/'+x
train_images = glob.glob(add_prefix('train_photos/*.jpg'))
test_images = glob.glob(add_prefix('test_photos/*.jpg'))

extract_id = lambda x: int(re.match('.*/([0-9]+).jpg', x).group(1))
train_ids = map(extract_id, train_images)
test_ids = map(extract_id, test_images)

# business and labels

train_labels = pd.read_csv(add_prefix('train.csv')).dropna()
train_labels['labels'] = train_labels['labels'].apply(lambda x: tuple(sorted(int(t) for t in x.split())))
train_labels.set_index('business_id', inplace=True)

train_photo_to_biz_ids = pd.read_csv(add_prefix('train_photo_to_biz_ids.csv'))
photos_in_train_biz = train_photo_to_biz_ids.groupby('business_id')['photo_id'].apply(list)
train_labels['photos'] = photos_in_train_biz
train_labels['n_photo'] = train_labels.photos.apply(len)

test_photo_to_biz_ids = pd.read_csv(add_prefix('test_photo_to_biz.csv'))
photos_in_test_biz = test_photo_to_biz_ids.groupby('business_id')['photo_id'].apply(list).to_dict()
test_biz_ids = list(photos_in_test_biz.keys())

def photos_in_biz(i):
    return train_labels.loc[i].photos

label_desc_dict = dict(map(lambda x: (int(x[0]), x[1]),
                           map(lambda x: x.split(': '), '''0: good_for_lunch
1: good_for_dinner
2: takes_reservations
3: outdoor_seating
4: restaurant_is_expensive
5: has_alcohol
6: has_table_service
7: ambience_is_classy
8: good_for_kids
'''.splitlines())))
n_labels = len(label_desc_dict)

def encode_label(l):
    res = np.zeros(len(label_desc_dict))
    for i in l:
        res[i] = 1
    return res
def decode_label(x):
    return tuple(np.where(x==1)[0])
train_L = np.vstack(train_labels['labels'].apply(encode_label))

# photo display
from types import ModuleType
def get_image(id, test=False):
    if test:
        return add_prefix('test_photos/{}.jpg'.format(id))
    else:
        return add_prefix('train_photos/{}.jpg'.format(id))

def show_image(id, test=False, ax=plt, msg=None):
    ax.imshow(plt.imread(get_image(id, test)))
    ax.grid(False)
    title = str(id)
    if test:
        title = 'test '+title
    if msg is not None:
        title += ': ' + str(msg)
    if isinstance(ax, ModuleType):
        ax.title(title)
    else:
        ax.set_title(title)

def show_photos(photos,m,n, msgs=None):
    with_test = isinstance(photos[0], tuple)
    if msgs is None:
        msgs = [None] * len(photos)

    fig, axes = plt.subplots(m,n, figsize=(15,15))
    for ax, i, msg in zip(axes.ravel(), photos, msgs):
        if with_test:
            show_image(i[0], i[1], ax, msg)
        else:
            show_image(i, ax=ax, msg=msg)
    return fig

def show_photos_in_bussiness(biz_id, m=3,n=3, seed=42):
    max_photo_size = m*n
    photos = train_labels.loc[biz_id].photos
    if len(photos) <= max_photo_size:
        sample_images = photos
    else:
        rng = np.random.RandomState(42)
        sample_images = rng.choice(photos, max_photo_size, replace=False)

    print('labels')
    for l in train_labels.loc[biz_id]['labels']:
        print('{}: {}'.format(l, label_desc_dict[l]))
    fig = show_photos(sample_images, m, n)
    fig.suptitle('{}/{} in business {}'.format(len(sample_images), len(photos), biz_id))
    return fig

# submission
def write_submission(L, fname, biz=test_biz_ids):
    with open(fname, 'w') as f:
        f.write('business_id,labels\n')
        for i, l in zip(biz, L):
            f.write('{},{}\n'.format(i, ' '.join(map(str, np.where(l==1)[0]))))

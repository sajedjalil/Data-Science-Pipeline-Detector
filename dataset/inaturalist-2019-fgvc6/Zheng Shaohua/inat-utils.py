import cv2
import json
import random
import math
import os
import numpy as np
from keras.utils import to_categorical

random.seed(1024)

def center_crop_img(img):
    h = img.shape[0]
    w = img.shape[1]
    short = min(h, w)
    h_center, w_center = h//2, w//2
    half_short = short//2
    croped_img = img[h_center-half_short:h_center+half_short, w_center-half_short:w_center+half_short, :]
    return croped_img

def scale_img(img, scale=299):
    return cv2.resize(img, (scale, scale), interpolation=cv2.INTER_CUBIC)
    
def img2vec(img_path):
    return cv2.imread(img_path)[..., [2, 1, 0]]

def generate_batch(dataset_dir, batch_size, num_samples, kind='train'):
    """生成batch
    Input: 
        dataset_dir: 数据集目录
        batch_size: 
        num_samples: 样本数
        kind: 'train' or 'val'
    Output:
        (X_batch, Y_batch)
    """
    # 读取annotation文件
    annotation_file_name = '/train2019.json' if kind == 'train' else '/val2019.json'
    with open(dataset_dir + annotation_file_name, 'r') as jf:
        raw_annotation = json.load(jf)
    images = raw_annotation['images']
    categories = raw_annotation['categories']
    # print(len(categories))
    annotations = raw_annotation['annotations']
    # 创建字典
    categories_dict = {cate['id']: cate for cate in categories}
    annotations_dict = {annot['id']: annot for annot in annotations}

    # shuffle
    random.shuffle(images)
    
    # 循环生成batch
    batch_start_index = 0
    while True:
        # 确定batch_end_index索引
        if batch_start_index + batch_size > num_samples:
            batch_end_index = num_samples    
        else:
            batch_end_index = batch_start_index + batch_size
        # print(batch_start_index, batch_end_index)
        
        # 生成batch
        raw_images_batch = images[batch_start_index:batch_end_index]
        # images
        images_batch = [image['file_name'] for image in raw_images_batch]
        images_batch = [scale_img(center_crop_img(img2vec(os.path.join(dataset_dir, 'train_val2019', file_name)))) for file_name in images_batch]
        images_batch = np.array(images_batch)
        # labels
        labels_batch = np.array([categories_dict[annotations_dict[image['id']]['category_id']]['id'] for image in raw_images_batch])
        labels_batch = to_categorical(labels_batch, num_classes=1010)
        yield images_batch, labels_batch 
        
        # 重设start_index_of_steps索引
        batch_start_index = 0 if batch_end_index == num_samples else batch_end_index


if __name__ == '__main__':
    DATASET_DIR = '/kaggle/input/inaturalist-2019-fgvc6'
    NUM_SAMPLES = 265213
    NUM_SAMPLES = 5
    generator = generate_batch(DATASET_DIR, 2, NUM_SAMPLES)
    batch = next(generator)
    print(batch[0].shape, batch[1].argmax())
    generator = generate_batch(DATASET_DIR, 2, NUM_SAMPLES, kind='val')
    batch = next(generator)
    print(batch[0].shape, batch[1].argmax())
"""
This code is based on:
-- https://www.kaggle.com/lamdang/fast-shuffle-bson-generator-for-keras
-- https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson

(c) Aleksei Tiulpin, 2017

"""


import os
import pandas as pd
import numpy as np
import bson
import cv2
from tqdm import tqdm
import struct
from PIL import Image
from  torchvision import transforms as transf
import torch.utils.data as data_utils




def read_bson(bson_path, num_records, with_categories):
    """
    Reads BSON
    """
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        records_read = 0 
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break
            
            length = struct.unpack("<i", item_length_bytes)[0]
            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            records_read += 1
            pbar.update()
    pbar.close()
    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df


def make_category_tables(categories_path):
    """
    Converts category name into an index [0, N-1]
    """
    categories_df = pd.read_csv(categories_path, index_col="category_id")
    categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

def get_obs(fname, offset, length):
    fobj = open(fname, 'rb')
    fobj.seek(offset)
    res = bson.BSON.decode(fobj.read(length))
    fobj.close()
    return res

class CdiscountDataset(data_utils.Dataset):
    def __init__(self, dataset, split, transform):
        self.dataset = dataset
        self.metadata = split
        self.transform = transform

    def __getitem__(self, index):
        entry = self.metadata.iloc[index]
        num_imgs, offset, length, target = entry
        obs = get_obs(self.dataset, offset, length)
        keep = np.random.choice(len(obs['imgs']))
        byte_str = obs['imgs'][keep]['picture']
        img = cv2.imdecode(np.fromstring(byte_str, dtype=np.uint8), cv2.IMREAD_COLOR)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)

        return img, target

    def __len__(self):
        return self.metadata.index.values.shape[0]

if __name__ == "__main__":
    TRAIN_BSON_FILE = '../input/train.bson'
    CATEGS = '../input/category_names.csv'
    # Chenge this weh running on your machine
    N_TRAIN = 7069896
    BS = 32
    N_THREADS = 1
    
    # mapping the catigores into 0-5269 range
    cat2idx, idx2cat = make_category_tables(CATEGS)
    #Scanning the metadata
    meta_data = read_bson(TRAIN_BSON_FILE, N_TRAIN, with_categories=True)
    meta_data.category_id = np.array([cat2idx[ind] for ind in meta_data.category_id])
    meta_data = meta_data.iloc[np.arange(500)] # Remove this!!!
    # Dataset and loader
    train_dataset = CdiscountDataset(TRAIN_BSON_FILE, meta_data,transf.ToTensor())
    loader = data_utils.DataLoader(train_dataset, batch_size=BS, num_workers=N_THREADS, shuffle=True)
                                   
    # Let's go fetch some data!
    pbar = tqdm(total=len(loader))
    for i, (batch, target) in enumerate(loader):
        pass
        pbar.update()
    pbar.close()
    
                                
                                   
    
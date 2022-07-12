import os
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import *
import zipfile


PATH = '../input/landmark-retrieval-2020/train/'
IMG_SIZE = 224



def zip_and_remove(path):
    ziph = zipfile.ZipFile(f'{path}.zip', 'w', zipfile.ZIP_DEFLATED)
    
    for root, dirs, files in os.walk(path):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            ziph.write(file_path)
            os.remove(file_path)
    
    ziph.close()


def img_proc(ids):
    path = os.path.join(PATH, ids[0], ids[1], ids[2], ids + '.jpg')
    img = cv2.resize(cv2.imread(path), (IMG_SIZE, IMG_SIZE))
    cv2.imwrite('train_img/' + ids + '.jpg', img)   

def imap_unordered_bar(func, args, n_processes: int = 64):
    p = Pool(n_processes, maxtasksperchild=100)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return None


def main():
    train_df = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
    
    os.makedirs('train_img')
    
    tqdm.pandas('Image processing progress')
    _ = imap_unordered_bar(img_proc, train_df.id.values[10*150_000:])
    zip_and_remove('train_img')

if __name__ == '__main__':
    main()
#-*- uf8-*-
#base on https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality
# add multiprocessing
from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 
from tqdm import tqdm
import zipfile
from IPython.core.display import HTML 
from IPython.display import Image
import multiprocessing
import time
import math
import sys

platform = 'kaggle'

image_zip_path = '../input/sample_avito_images.zip'
limit = 100#1503424//2
lock = multiprocessing.RLock()
feature_name = 'all'#'dullness'
test = 0
if test:
    csv_file = '../input/test.csv'
else:
    csv_file = '../input/train.csv'

if platform == 'kaggle':
    images_path = '../input/train_jpg.zip'
    image_zip_path = '../input/train_jpg.zip'
    if limit:
        df = pd.read_csv(csv_file,usecols=['image','item_id'],nrows=limit+1,skiprows=range(1, limit))
    else:
        df = pd.read_csv(csv_file,usecols=['image','item_id'],nrows=limit)
    if test:
        image_zip_path = '../input/test_jpg.zip'
        
    zip_file = zipfile.ZipFile(image_zip_path)
    
    imgs = df['image'].values.tolist()
    df.to_csv('ids.csv')
    # del df
    # print(imgs)
    def read_img(name):
        lock.acquire()
        if test:
            name1 = 'data/competition_files/test_jpg/{}.jpg'.format(name)
        else:
            name1 = 'data/competition_files/train_jpg/{}.jpg'.format(name)
        with zip_file.open(name1) as im:
            size = zip_file.getinfo(name1)
            res = IMG.open(im),size
            lock.release()
        return res


if platform == 'pc':
    csv_file = '../input/train.csv.zip'
    if limit:
        df = pd.read_csv(csv_file,usecols=['image','item_id'],nrows=limit+1,skiprows=range(1, limit))
    else:
        df = pd.read_csv(csv_file,usecols=['image','item_id'],nrows=limit)
    zip_file = zipfile.ZipFile(image_zip_path)
    images_path = '../input/sample_avito_images.zip'
    imgs = zip_file.namelist()
    def read_img(name):
        with zip_file.open(name) as im:
            size = zip_file.getinfo(name)
            return IMG.open(im),size

features = pd.DataFrame()
features['image'] = imgs

def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

def perform_color_analysis(im, flag):
#     path = images_path + img 
#     im = IMG.open(path) #.convert("RGB")
#     im = read_img(img)
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None

def average_pixel_width(im): 
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

# %%time
def get_dominant_color(im):
#     path = images_path + img 
#     img = cv2.imread(path)
#     im = im.con
    arr = np.float32(im.convert('RGB').getdata())
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
#     quantized = palette[labels.flatten()]
#     quantized = quantized.reshape((im.size[0],im.size[1],3))

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color


def get_average_color(img):
    arr = np.float32(img.convert('RGB').getdata())
#     img = arr[:,:3]
#     print(arr.shape)
    average_color = [arr[:, i].mean() for i in range(arr.shape[-1])]
    return average_color


def getSize(filename):
    filename = images_path + filename
    st = os.stat(filename)
    return st.st_size

def getDimensions(img):
#     filename = images_path + filename
#     img_size = IMG.open(filename).size
    return img.size

def get_blurrness_score(image):
#     image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.convert('RGB')
    open_cv_image = np.array(image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm

def process_img_add_featrue(img_name):
    if pd.isna(img_name):
        return np.nan
    try:
        im,size = read_img(img_name)
    except Exception as e:
        print('error',e)
        # raise e
        return np.nan
    dullness = perform_color_analysis(im, 'black')
    whiteness = perform_color_analysis(im, 'white')
    average_pixel_widthv = average_pixel_width(im)
    color = get_dominant_color(im)
    avg_color = get_average_color(im)
    dim = getDimensions(im)
    blurrness = get_blurrness_score(im)
    res = dullness,whiteness,average_pixel_widthv,color, avg_color, dim, blurrness,size.file_size
    im.close()
    return res

# %%time
import os
pos = 0
plock = multiprocessing.Lock()

def mp_worker(imags):
    global pos
    res = []
    im_ids = imags['image'].values.tolist()
    plock.acquire()
    pos+=1
    t = tqdm(im_ids, position=pos,desc='porc %s'%os.getpid(),file=sys.stdout,mininterval=200)
    
    plock.release()
    for im in t:
        res.append(process_img_add_featrue(im))
    return pd.DataFrame({feature_name:res, 'image':im_ids})

def mp_handler():
    n_theads = 5
    if platform == 'kaggle':
        n_theads = 30
    p = multiprocessing.Pool(n_theads)
    batch_size = math.ceil(features.size/n_theads)
    data = []
    print(features.size,batch_size)
    for i in range(n_theads):
        a = features[i*batch_size:(i+1)*batch_size]
        print(a.shape)
        data.append(a)
        
    d = p.map(mp_worker, data)
    res = pd.DataFrame()
    res = res.append(d,ignore_index=True)
    # res = res.reset_index()
    res = post_process_features(res)
    res.to_csv('train_%s.csv'%feature_name)
    res1=df.merge(res,how='inner',on='image')
    res1.drop_duplicates(inplace=True)
    res1.to_csv('all_train_%s.csv'%feature_name)
    
# def foo(x):
#     print(type(x))
#     print(x)

def post_process_features(df1):
    new_col_list = ['dullness','whiteness','average_pixel_width','dominant_color','avg_color','dimensions','blurrness','size']
    # for n,col in enumerate(new_col_list):
    #     df1[col] = df1[feature_name].apply(lambda x: x[n])
    df1[new_col_list]=df1[feature_name].apply(pd.Series)
    # print(df1.columns)
    df1[['dominant_red','dominant_blue','dominant_green']] = df1['dominant_color'].apply(pd.Series) / 255
    # df1['dominant_green'] = df1['dominant_color'].apply(lambda x: x[1]) / 255
    # df1['dominant_blue'] = df1['dominant_color'].apply(lambda x: x[2]) / 255
    # df1[['dominant_red', 'dominant_green', 'dominant_blue']].head(5)

    df1[['average_red','average_green','average_blue']] = df1['avg_color'].apply(pd.Series) / 255
    # df1['average_green'] = df1['avg_color'].apply(lambda x: x[1]) / 255
    # df1['average_blue'] = df1['avg_color'].apply(lambda x: x[2]) / 255
    # df1[['average_red', 'average_green', 'average_blue']].head(5)
    df1[['width','height']] = df1['dimensions'].apply(pd.Series)
    # df1['height'] = df1['dimensions'].apply(lambda x : x[1])
    df1 = df1.drop(['dimensions', 'avg_color', 'dominant_color',feature_name], axis=1)
    return df1

# mp_worker(features['image'])
if __name__ == '__main__':
    multiprocessing.freeze_support()  # for Windows support
    import time
    now = time.time()
    mp_handler()
    print('time',time.time()-now)
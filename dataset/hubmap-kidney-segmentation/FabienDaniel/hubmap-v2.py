import PIL
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import sys

Image.MAX_IMAGE_PIXELS = None

import json

def make_image_id(mode='train', valid_ids=None):
    train_image_id = {
        0 : '0486052bb',
        1 : '095bf7a1f',
        2 : '1e2425f28',
        3 : '26dc41664',
        4 : '2f6ecfcdf',
        5 : '4ef6695ce',
        6 : '54f2eec69',
        7 : '8242609fa',
        8 : 'aaa6a05cc',
        9 : 'afa5e8098',
        10: 'b2dc8411c',
        11: 'b9a3865fc',
        12: 'c68fe75ea',
        13: 'cb2d976f4',
        14: 'e79de561c',
    }
    test_image_id = {
        0 : '2ec3f1bb9',
        1 : '3589adb90',
        2 : 'd488c759a',
        3 : 'aa05346ff',
        4 : '57512b7f1',
    }

    if mode == 'test-all':
        return list(test_image_id.values())

    elif mode == 'train-all':
        return list(train_image_id.values())

    elif mode == 'valid':
        return [train_image_id[i] for i in valid_ids]

    elif mode == 'train':
        train_ids = [i for i in train_image_id.keys() if i not in valid_ids]
        return [train_image_id[i] for i in train_ids]
    
def image_show_norm(name, image, min=None, max=None, resize=1):
    if max is None: max=image.max()
    if min is None: min=image.min()

    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  #WINDOW_NORMAL
    cv2.imshow(name, (np.clip((image-min)/(max-min),0,1)*255).astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))
    
def get_data_path(server):
    if server == 'local':
        project_repo = '/home/fabien/Kaggle/HuBMAP/HengCherKeng/2020-12-11'
        raw_data_dir = '/home/fabien/Kaggle/HuBMAP/input/'
        data_dir = project_repo + '/data'
    elif server == 'kaggle':
        project_repo = None
        raw_data_dir = '../input/hubmap-kidney-segmentation'
        data_dir = None
    return project_repo, raw_data_dir, data_dir


def read_mask(mask_file):
    mask = np.array(PIL.Image.open(mask_file))
    return mask


def read_json_as_df(json_file):
   with open(json_file) as f:
       j = json.load(f)
   df = pd.json_normalize(j)
   return df



def draw_strcuture(df, height, width, fill=255, structure=[]):
    mask = np.zeros((height, width), np.uint8)
    for row in df.values:
        type  = row[2]   # geometry.type
        coord = row[3]   # geometry.coordinates
        name  = row[4]   # properties.classification.name

        if structure !=[]:
            if not any(s in name for s in structure): continue


        if type=='Polygon':
            pt = np.array(coord).astype(np.int32)
            #cv2.polylines(mask, [coord.reshape((-1, 1, 2))], True, 255, 1)
            cv2.fillPoly(mask, [pt.reshape((-1, 1, 2))], fill)

        if type=='MultiPolygon':
            for pt in coord:
                pt = np.array(pt).astype(np.int32)
                cv2.fillPoly(mask, [pt.reshape((-1, 1, 2))], fill)

    return mask


def draw_strcuture_from_hue(image, fill=255, scale=1/32):

    height, width, _ = image.shape
    vv = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    vv = cv2.cvtColor(vv, cv2.COLOR_RGB2HSV)
    # image_show('v[0]', v[:,:,0])
    # image_show('v[1]', v[:,:,1])
    # image_show('v[2]', v[:,:,2])
    # cv2.waitKey(0)
    mask = (vv[:, :, 1] > 32).astype(np.uint8)
    mask = mask*fill
    mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    return mask


# --- rle ---------------------------------
def rle_decode(rle, height, width , fill=255):
    s = rle.split()
    start, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    start -= 1
    mask = np.zeros(height*width, dtype=np.uint8)
    for i, l in zip(start, length):
        mask[i:i+l] = fill
    mask = mask.reshape(width,height).T
    mask = np.ascontiguousarray(mask)
    return mask

def rle_encode_less_memory(img):
    
    print("Enters RLE encoder")
    
    # the image should be transposed
    pixels = img.T.flatten()
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    
    print("creates encoding")
    
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    print("encoding performed")
    
    return ' '.join(str(x) for x in runs)

def rle_encode(mask):
    m = mask.T.flatten()
    m = np.concatenate([[0], m, [0]])
    run = np.where(m[1:] != m[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle =  ' '.join(str(r) for r in run)
    return rle


def to_mask(tile, coord, height, width, scale, size, step, min_score, aggregate='mean'):
    """
    Aggrège les différentes images. Si elles se recouvrent, les pixels sont pondérés /
     à la position du centre de l'image.
    """

    half = size // 2
    mask  = np.zeros((height, width), np.float32)

    # print(sys.getsizeof(tile) * 1e-6, 'Mb')

    print(f'\nCreating mask w={width} x h={height} from tiles')

    if 'mean' in aggregate:
        w = np.ones((size, size), np.float32)

        #if 'sq' in aggregate:
        if 1:
            #https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
            y, x = np.mgrid[-half:half, -half:half]
            y = half - abs(y)
            x = half - abs(x)
            w = np.minimum(x, y)
            w = w / w.max()
            w = np.minimum(w, 1)

        #--------------
        count = np.zeros((height, width), np.float32)

#         print(len(tile))

        for t, (cx, cy) in enumerate(coord):

            # print(cx, cy, half, tile[t].shape, w.shape)

            mask [cy - half:cy + half, cx - half:cx + half] += tile[t] * w
            count[cy - half:cy + half, cx - half:cx + half] += w

#             print(f"{t} / {len(coord)}",
#                   sys.getsizeof(count) * 1e-6, 'Mb',
#                   sys.getsizeof(mask) * 1e-6, 'Mb', end='\r')


            # see unet paper for "Overlap-tile strategy for seamless segmentation of arbitrary large images"

        # m = (count != 0)
        # mask[m] /= count[m]
#         print(sys.getsizeof(mask) * 1e-6, 'Mb', sys.getsizeof(count) * 1e-6, 'Mb')

        length = 100
        for i in range(mask.shape[0] // length + 1):
            i1 = i * length
            i2 = min((i+1) * length, mask.shape[0])
            # print(i1, i2)
            mask[i1:i2] = np.divide(
                mask[i1:i2],
                count[i1:i2],
                out=np.zeros_like(mask[i1:i2]),
                where=count[i1:i2] != 0
            )

#             print(i1, i2, mask.shape[0] // length + 1, sys.getsizeof(mask) * 1e-6, 'Mb', sys.getsizeof(count) * 1e-6, 'Mb')


    if aggregate == 'max':
        for t, (cx, cy) in enumerate(coord):
            mask[cy - half:cy + half, cx - half:cx + half] = np.maximum(
                mask[cy - half:cy + half, cx - half:cx + half],
                tile[t]
            )

    print("tile probability created")
    
    return mask


# --draw ------------------------------------------
def mask_to_inner_contour(mask):
    mask = mask>0.5
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def draw_contour_overlay(image, mask, color=(0, 0, 255), thickness=1):
    contour =  mask_to_inner_contour(mask)
    if thickness == 1:
        image[contour] = color
    else:
        r = max(1, thickness//2)
        for y, x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x, y), r, color, lineType=cv2.LINE_4)
    return image


#-- tools ---
##################################################################
def run_check_tile():
    tile_scale = 0.25
    tile_size = 320 #640  #
    tile_average_step = 160 #640-160  #
    tile_min_score = 0.25

    if 1:
        id = '2f6ecfcdf'
        #id = 'aaa6a05cc'
        #id = 'cb2d976f4'
        #id = '0486052bb'
        #id = 'e79de561c'
        #id = '095bf7a1f'# mislabel
        #id = '54f2eec69'
        #id = '1e2425f28'# mislabel

        #load image
        image_file = data_dir + '/train/%s.tiff' % id
        image = read_tiff(image_file)
        height, width = image.shape[:2]

        #load mask
        df = pd.read_csv(data_dir + '/train.csv', index_col='id')
        encoding = df.loc[id, 'encoding']
        mask = rle_decode(encoding, height, width, 255)

        #load structure
        # json_file = data_dir + '/train/%s-anatomical-structure.json' % id
        # df = read_json_as_df(json_file)
        # structure = draw_strcuture(df, height, width) #, structure=['Cortex'])

        structure = draw_strcuture_from_hue(image, fill=255, scale=tile_scale/32)

        if 0:
            image_show('image', image)
            image_show('mask', mask)
            image_show('structure', structure)
            cv2.waitKey(0)

        tile = to_tile(image, mask, structure, tile_scale, tile_size, tile_average_step, tile_min_score)

        if 1: #debug
            overlay = tile['image_small'].copy()
            for cx, cy, cv in tile['coord']:
                cv = int(255 * cv)
                cv2.circle(overlay, (cx, cy), 64, [cv,cv,cv], -1)
                cv2.circle(overlay, (cx, cy), 64, [0, 0, 255], 16)
            for cx, cy, cv in tile['reject']:
                cv = int(255 * cv)
                cv2.circle(overlay, (cx, cy), 64, [cv,cv,cv], -1)
                cv2.circle(overlay, (cx, cy), 64, [255, 0, 0], 16)

            #---
            num = len(tile['coord'])
            cx, cy, cv = tile['coord'][num//2]
            cv2.rectangle(overlay,
                          (cx-tile_size//2, cy-tile_size//2),
                          (cx+tile_size//2, cy+tile_size//2),
                          (0, 0, 255),
                          16)

            image_show('image_small', tile['image_small'], resize=0.1)
            image_show('overlay', overlay, resize=0.1)
            cv2.waitKey(1)

        # make prediction for tile
        # e.g. predict = model(tile['tile_image'])
        tile_predict = tile['tile_mask'] # dummy: set predict as ground truth

        # make mask from tile
        height, width = tile['image_small'].shape[:2]
        predict = to_mask(tile_predict, tile['coord'],  height, width,
                          tile_scale, tile_size, tile_average_step, tile_min_score,
                          aggregate = 'mean')

        truth = tile['mask_small']#.astype(np.float32)/255
        diff = np.abs(truth-predict)
        print('diff', diff.max(), diff.mean())

        if 1:
            image_show_norm('diff', diff, min=0, max=1, resize=0.2)
            image_show_norm('predict', predict, min=0, max=1, resize=0.2)
            cv2.waitKey(0)

    


# Based on "Export pixel-wise mask" by visoft:
# https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask/

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from shapely.wkt import loads as wkt_loads
import tifffile as tiff

def _get_image_names(base_path, imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': path.join(base_path,'three_band/{}.tif'.format(imageId)),             # (3, 3348, 3403)
         'A': path.join(base_path,'sixteen_band/{}_A.tif'.format(imageId)),         # (8, 134, 137)
         'M': path.join(base_path,'sixteen_band/{}_M.tif'.format(imageId)),         # (8, 837, 851)
         'P': path.join(base_path,'sixteen_band/{}_P.tif'.format(imageId)),         # (3348, 3403)
         }
    return d


def _convert_coordinates_to_raster(coords, img_size, xymax):
    Xmax,Ymax = xymax
    H,W = img_size
    W1 = 1.0*W*W/(W+1)
    H1 = 1.0*H*H/(H+1)
    xf = W1/Xmax
    yf = H1/Ymax
    coords[:,1] *= yf
    coords[:,0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
    return (xmax,ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list,interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value = 1):
    img_mask = np.ones(raster_img_size,np.uint8)
    if contours is None:
        return img_mask
    perim_list,interior_list = contours
    cv2.fillPoly(img_mask,perim_list,class_value)
    cv2.fillPoly(img_mask,interior_list,1)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,
                                     wkt_list_pandas):
    xymax = _get_xmax_ymin(grid_sizes_panda,imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas,imageId,class_type)
    contours = _get_and_convert_contours(polygon_list,raster_size,xymax)
    mask = _plot_mask_from_contours(raster_size,contours,0)
    return mask


inDir = '../input'

# read the training data from train_wkt_v4.csv
df = pd.read_csv(inDir + '/train_wkt_v4.csv')

# grid size will also be needed later..
gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

image_id = '6120_2_2'

D,H,W = tiff.TiffFile('../input/three_band/{}.tif'.format(image_id)).asarray().shape

mask = {}
for cat_id in range(1, 11):
    mask[cat_id] = generate_mask_for_image_and_class((H,W),image_id,cat_id,gs,df)
    #cv2.imwrite('mask_{}_cat{}.png'.format(image_id, cat_id), mask[cat_id]*255)

full_mask = np.ones((H, W, D))
for h in range(H):
    for w in range(W):
        color = [1, 1, 1]
        if  mask[10][h, w] == 0: color = [0, 0, 1]
        elif mask[9][h, w] == 0: color = [0, 0, 0.8]
        elif mask[8][h, w] == 0: color = [1, 0, 0]
        elif mask[7][h, w] == 0: color = [0.8, 0, 0]
        elif mask[5][h, w] == 0: color = [0, 1, 0]
        elif mask[1][h, w] == 0: color = [1, 0.5, 0.5]
        elif mask[2][h, w] == 0: color = [0.7, 0, 0.7]
        elif mask[3][h, w] == 0: color = [0, 0, 0]
        elif mask[4][h, w] == 0: color = [0.5, 0.5, 0.5]
        elif mask[6][h, w] == 0: color = [0.8, 0.8, 0.8]
        full_mask[h, w, :] = color

cv2.imwrite('full_mask.png', full_mask*255)

import cv2
from shapely.wkt import loads as wkt_loads
import rasterio
import numpy as np
import os
import shapely
from rasterio import features
import shapely.geometry
import shapely.affinity
import pandas as pd
import tifffile as tiff


data_path = '../input'
train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


def _convert_coordinates_to_raster(coords, img_size, xymax):
    x_max, y_max = xymax
    height, width = img_size
    W1 = 1.0 * width * width / (width + 1)
    H1 = 1.0 * height * height / (height + 1)
    xf = W1 / x_max
    yf = H1 / y_max
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return xmax, ymin


def _get_polygon_list(wkt_list_pandas, imageId, class_type):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == class_type].MultipolygonWKT
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
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    img_mask = np.zeros(raster_img_size, np.int8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def mask_to_polygons_layer(mask):
    all_polygons = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask == 1), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):

        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda, wkt_list_pandas):
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


def calculate_score_mask(y_true, y_pred):
    """
    Function calculate jaccard score for real vs mask

    :param y_true:
    :param y_pred:
    :return:
    """
    num_mask_channels = y_true.shape[0]

    result = np.ones(num_mask_channels)

    for mask_channel in range(num_mask_channels):
        intersection = np.dot(y_true[mask_channel, ...].flatten(), y_pred[mask_channel, ...].flatten())
        _sum = y_true[mask_channel, ...].sum() + y_pred[mask_channel, ...].sum()
        if _sum - intersection != 0:
            result[mask_channel] = intersection / (_sum - intersection)
    return np.mean(result)


def mask2polygons(mask, image_id):
    """

    :param mask:
    :return: list of the type: [Polygons_class_1, Polygons_class_2]
    """
    W = mask.shape[1]
    H = mask.shape[2]

    num_mask_channels = mask.shape[0]

    x_max = gs.loc[gs['ImageId'] == image_id, 'Xmax'].values[0]
    y_min = gs.loc[gs['ImageId'] == image_id, 'Ymin'].values[0]

    W_ = W * (W / (W + 1))
    H_ = H * (H / (H + 1))

    x_scaler = W_ / x_max
    y_scaler = H_ / y_min

    x_scaler = 1 / x_scaler
    y_scaler = 1 / y_scaler

    result = []

    for mask_channel in range(num_mask_channels):
        polygons = mask_to_polygons_layer(mask[mask_channel])

        polygons = shapely.affinity.scale(polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

        if not polygons.is_valid:
            polygons = polygons.buffer(0)
            # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
            # need to keep it a Multi throughout
            polygons = shapely.geometry.MultiPolygon([polygons])

        result += [str(polygons)]

    return result


def polygons2mask(polygons, width, height, image_id):
    xymax = _get_xmax_ymin(gs, image_id)

    mask = np.zeros((len(polygons), width, height))

    for i, p in enumerate(polygons):
        polygon_list = wkt_loads(p)
        if polygon_list.length == 0:
            continue
        contours = _get_and_convert_contours(polygon_list, (width, height), xymax)
        mask[i] = _plot_mask_from_contours((width, height), contours, 1)
    return mask


def generate_mask(image_id, width, height, num_mask_channels=10):
    mask = np.zeros((num_mask_channels, width, height))

    for mask_channel in range(num_mask_channels):
        mask[mask_channel, :, :] = generate_mask_for_image_and_class((width, height), image_id, mask_channel + 1, gs, train_wkt)
    return mask


def calculate_polygon_match(image_id, num_mask_channels=10):
    """
    calculates jaccard index between before poly and after poly

    Ideally should be 1

    :param image_id:
    :param num_mask_channels:
    :return:
    """

    image = tiff.imread(os.path.join(data_path, 'three_band', image_id + '.tif'))

    width = image.shape[1]
    height = image.shape[2]

    mask_before = generate_mask(image_id, width, height, num_mask_channels=num_mask_channels)

    polygons = mask2polygons(mask_before, image_id)
    predicted_mask = polygons2mask(polygons, width, height, image_id)

    mask = generate_mask(image_id, width, height)

    return calculate_score_mask(mask, predicted_mask)


if __name__ == '__main__':
    for image_id in train_wkt['ImageId'].unique():
        print(image_id, calculate_polygon_match(image_id))

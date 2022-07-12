# ----------------------------------------------------------------------
#   Imports required libs.
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from tqdm import tqdm
import shutil


# ----------------------------------------------------------------------
#   Defines utility functions for visualizing dataset.
# ----------------------------------------------------------------------
def showImagesTile(titles, images, num_cols=4, fig_size=(6.4, 4.8)):
    num_rows = len(images) // num_cols + 1
    fig = plt.figure(figsize=(fig_size[0] * num_cols, fig_size[1] * num_rows))
    
    for i, (title, image) in enumerate(zip(titles, images)):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.set_title(title)
        plt.imshow(image)
        
    plt.show()
    plt.clf()
    plt.close()
    
def getImages(path_to_images):
    return [Image.open(path_to_image) for path_to_image \
            in path_to_images]
    
# Draws annotations for several images.
def drawAnnotations(images, annotations_xyxy_for_images=[], \
                    texts_for_images=[], line_color="red", \
                    line_width=3, text_color="red"):
    for image, annotations_xyxy, texts in \
        zip(images, annotations_xyxy_for_images, texts_for_images):
        _drawAnnotations(image, annotations_xyxy, texts, \
                         line_color=line_color, line_width=line_width, \
                         text_color=text_color)

# Draws annotations for "a image".
def _drawAnnotations(image, annotations_xyxy=[], texts=[], \
                     line_color="red", line_width=3, text_color="red"):
    draw = ImageDraw.Draw(image)
    for annotation_xyxy, text in zip(annotations_xyxy, texts):
        x_min = annotation_xyxy["x_min"]
        y_min = annotation_xyxy["y_min"]
        x_max = annotation_xyxy["x_max"]
        y_max = annotation_xyxy["y_max"]
        draw.rectangle(xy=[(x_min, y_min), (x_max, y_max)], \
                       outline=line_color, width=line_width)
        draw.text(xy=(x_min, y_max + 2), text=text, fill=text_color)
        
def copyImages(path_to_sources, path_to_targets):
    for path_to_source, path_to_target in \
        tqdm(zip(path_to_sources, path_to_targets)):
        shutil.copy2(path_to_source, path_to_target)

def _pathToImage(path_to_dir, file_name):
    return "%s/%s" % (path_to_dir, file_name)


# ----------------------------------------------------------------------
#   Defines utility functions for processing annotation data.
# ----------------------------------------------------------------------
def _annotationXyxy(class_id, x_min, y_min, x_max, y_max, confidence):
    annotation_xyxy = { \
        "class_id": class_id, \
        "x_min": x_min, \
        "y_min": y_min, \
        "x_max": x_max, \
        "y_max": y_max, \
        "confidence": confidence \
    }
    return annotation_xyxy
    
def _annotationXywh(class_id, x, y, width, height, confidence):
    annotation_xywh = { \
        "class_id": class_id, \
        "x": x, \
        "y": y, \
        "width": width, \
        "height": height, \
        "confidence": confidence \
    }
    return annotation_xywh
    
def _annotationYolov5(class_id, rx_center, ry_center, rwidth, rheight, confidence):
    annotation_yolov5 = { \
        "class_id": class_id, \
        "rx_center": rx_center, \
        "ry_center": ry_center, \
        "rwidth": rwidth, \
        "rheight": rheight, \
        "confidence": confidence \
    }
    return annotation_yolov5

def _xywh2xyxy(x, y, width, height):
    x_min, y_min = x, y
    x_max, y_max = x_min + width, y_min + height
    xyxy = (x_min, y_min, x_max, y_max)
    return xyxy

def _edge2xyxy(edges, margin_x, margin_y, image_width, image_height):
    # Margins between object and x/y edges of image can be set. 
    # Sometimes bounding box is crossing whale fluke, so it is 
    # better to have margin.
    xs = np.array([int(x) for x in edges[::2]])
    ys = np.array([int(y) for y in edges[1::2]])
    
    x_min = max(xs.min() - margin_x, 0)
    y_min = max(ys.min() - margin_y, 0)
    x_max = min(xs.max() + margin_x, image_width)
    y_max = min(ys.max() + margin_y, image_height)
    xyxy = (x_min, y_min, x_max, y_max)
    return xyxy

def _xyxy2xywh(x_min, y_min, x_max, y_max):
    width, height = x_max - x_min, y_max - y_min
    xywh = (x_min, y_min, width, height)
    return xywh

def _xywh2yolov5(x, y, width, height, image_width, image_height):
    x_center = x + width / 2
    y_center = y + height / 2
    yolov5 = [ \
        x_center / image_width, \
        y_center / image_height, \
        width / image_width, \
        height / image_height \
    ]
    return yolov5

def _yolov52label(class_id, rx_center, ry_center, rwidth, rheight):
    label = "%d %.06f %.06f %.06f %.06f" % \
            (class_id, rx_center, ry_center, rwidth, rheight)
    return label

def _readLabel(path_to_label):
    with open(path_to_label) as fin:
        all_records = [record for record in csv.reader(fin)]
        return all_records
    
def _label2yolov5(label):
    class_id = int(label[0])
    rx_center, ry_center, rwidth, rheight, confidence = \
        [float(item) for item in label[1:]]
    yolov5 = (class_id, rx_center, ry_center, rwidth, \
              rheight, confidence)
    return yolov5

def _yolov52xyxy(rx_center, ry_center, rwidth, rheight, \
                 image_width, image_height):
    x_center = rx_center * image_width
    y_center = ry_center * image_height
    width = rwidth * image_width
    height = rheight * image_height
    x_min = max(int(x_center - width / 2), 0)
    y_min = max(int(y_center - height / 2), 0)
    x_max = min(int(x_center + width / 2), image_width)
    y_max = min(int(y_center + height / 2), image_height)
    xyxy = (x_min, y_min, x_max, y_max)
    return xyxy
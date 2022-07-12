# %% [code]
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage import exposure

"""
    Documentation for this Script:
    -----------------------------------
    read_xray:
                This function reads an xray from given dicom images and converts it into numpy array. Depending on parameter hist_normalize
                the output image is histogram normalized.
                PARAMETERS:
                    path= path to the dicom image
                    voi_lut = VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view ( boolean True or False)
                    fix_monochrome = depending on this value, X-ray may look inverted - fix that ( boolean True or False)
                    hist_normalize = apply historgram normalization or not( boolean True or False)
    plot_boxes:
                This function draws bounding boxes on image.
                PARAMETERS:
                    path = path to the dicom image
                    axes = matplotlib axes where boxes are to be plotted
                    train = dataframe containing image data ( for example the train.csv file)
    showXray:
                Function to display Xray image with or without bounding boxes
                PARAMETERS:
                    path = path to dicom image
                    train = dataframe containing image data ( for example the train.csv file)
                    with_boxes = boolean True or False; whether to display bounding boxes
                    legends = boolean True or False; whether to display legends
                    **arguments for read_xray
"""

def read_xray(path, voi_lut = True, fix_monochrome = True, hist_normalize=True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    
    data = data - np.min(data)
    if hist_normalize:
        data = exposure.equalize_hist(data)
        
    return data

def plot_boxes(path=None,axes=None,train=None):
    color_dict = dict(zip(['No finding','Cardiomegaly', 'Aortic enlargement',
       'Pleural thickening', 'ILD', 'Nodule/Mass', 'Pulmonary fibrosis',
       'Lung Opacity', 'Atelectasis', 'Other lesion', 'Infiltration',
       'Pleural effusion', 'Calcification', 'Consolidation',
       'Pneumothorax'],['black','r','g','b','y','c','m','k','w','olive','midnightblue','deeppink','lawngreen','goldenrod','tomato']))
    image = os.path.basename(path).split('.dicom')[0]
    df = train.query("image_id==@image")
    if len(df)>0:
        for row in df.apply(lambda cols: (cols[1],cols[4],cols[5],cols[6],cols[7]),axis=1):
            rect = patches.Rectangle((row[1],row[2]),row[3]-row[1],row[4]-row[2],edgecolor=color_dict[row[0]],facecolor="None",label=row[0])
            axes.add_patch(rect)
            
def showXray(path, train=None,with_boxes=True,legends=True,voi_lut = True, fix_monochrome = True, hist_normalize=True):
    fig,axes = plt.subplots(1,figsize=(10,10))
    axes.imshow(read_xray(path),cmap=plt.cm.bone)
    if with_boxes:
        plot_boxes(path,axes,train)
    if legends:
        plt.legend()
    plt.show()
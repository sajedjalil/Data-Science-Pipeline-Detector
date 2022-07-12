import os
import math
from pathlib import Path
import glob
import math
import threading

import numpy as np
import pandas as pd
import cv2
import pydicom

from tqdm.notebook import tqdm

from IPython.display import IFrame
from IPython.core.display import display, HTML


def get_image_plane(data):
    x1, y1, _, x2, y2, _ = [round(j) for j in data.ImageOrientationPatient]
    cords = [x1, y1, x2, y2]

    if cords == [1, 0, 0, 0]:
        return 'Coronal'
    elif cords == [1, 0, 0, 1]:
        return 'Axial'
    elif cords == [0, 1, 0, 0]:
        return 'Sagittal'
    else:
#         return 'Unknown'
        return 'Axial'

    
def get_voxel(patient_id, scan_type, data_root):
    imgs = []
    dcm_dir = data_root.joinpath(scan_type)
    dcm_paths = sorted(dcm_dir.glob("*.dcm"), key=lambda x: int(x.stem.split("-")[-1]))
    positions = []
    
    for dcm_path in dcm_paths:
        img = pydicom.dcmread(str(dcm_path))
        imgs.append(img.pixel_array)
        positions.append(img.ImagePositionPatient)
        
    plane = get_image_plane(img)
    voxel = np.stack(imgs)
    
    # reorder planes if needed and rotate voxel
    if plane == "Coronal":
        if positions[0][1] < positions[-1][1]:
            voxel = voxel[::-1]
        voxel = voxel.transpose((1, 0, 2))
    elif plane == "Sagittal":
        if positions[0][0] < positions[-1][0]:
            voxel = voxel[::-1]
        voxel = voxel.transpose((1, 2, 0))
        voxel = np.rot90(voxel, 2, axes=(1, 2))
    elif plane == "Axial":
        if positions[0][2] > positions[-1][2]:
            voxel = voxel[::-1]
        voxel = np.rot90(voxel, 2)
    else:
        pass
#         raise ValueError(f"Unknown plane {plane}")
    return voxel, plane


def normalize_contrast(voxel):
    if voxel.sum() == 0:
        return voxel
    voxel = voxel - np.min(voxel)
    voxel = voxel / np.max(voxel)
    voxel = (voxel * 255).astype(np.uint8)
    return voxel


def crop_voxel(voxel):
    if voxel.sum() == 0:
        return voxel
    keep = (voxel.mean(axis=(0, 1)) > 0)
    voxel = voxel[:, :, keep]
    keep = (voxel.mean(axis=(0, 2)) > 0)
    voxel = voxel[:, keep, :]
    keep = (voxel.mean(axis=(1, 2)) > 0)
    voxel = voxel[keep, :, :]
    return voxel


def filter_voxel(voxel, filter_thr):
    voxel_mean = voxel.mean(axis=(1, 2))
    keep = (voxel_mean > voxel_mean.std()*filter_thr)
    voxel = voxel[keep, :, :]
    return voxel


def resize_voxel(voxel, sz=(64, 256, 256)):
    output = np.zeros((sz[0], sz[1], sz[2]), dtype=np.uint8)
    if np.argmax(voxel.shape) == 0:
        for i, s in enumerate(np.linspace(0, voxel.shape[0] - 1, num=sz[0])):
            sampled = voxel[int(s), :, :]
            output[i, :, :] = cv2.resize(sampled, (sz[2], sz[1]), cv2.INTER_CUBIC)
    elif np.argmax(voxel.shape) == 1:
        for i, s in enumerate(np.linspace(0, voxel.shape[1] - 1, num=sz[1])):
            sampled = voxel[:, int(s), :]
            output[:, i, :] = cv2.resize(sampled, (sz[2], sz[0]), cv2.INTER_CUBIC)
    elif np.argmax(voxel.shape) == 2:
        for i, s in enumerate(np.linspace(0, voxel.shape[2] - 1, num=sz[2])):
            sampled = voxel[:, :, int(s)]
            output[:, :, i] = cv2.resize(sampled, (sz[1], sz[0]), cv2.INTER_CUBIC)
    return output


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def prepare_data(patients_paths, scan_types, filter_thr, voxel_dir):
    for study_path in tqdm(patients_paths):
        study_id = study_path.name

#         if study_id in ['00109', '00123', '00709']:
#             continue

        for i, scan_type in enumerate(scan_types):
            voxel, plane = get_voxel(study_id, scan_type, study_path)
            voxel = normalize_contrast(voxel)
            voxel = crop_voxel(voxel)
            voxel = filter_voxel(voxel, filter_thr)
            voxel = resize_voxel(voxel, sz=(64, 256, 256))

            os.makedirs(f'{voxel_dir}/{scan_type}', exist_ok=True)

            with open(f'{voxel_dir}/{scan_type}/{study_id}.npy', 'wb') as f:
                np.save(f, voxel)

                
def convet_images(input_folder:str, voxel_dir: str, scan_types: list, filter_thr: float, n_threads: int = 8):
    # get all files in folder
    paths = list(input_folder.glob("*"))
    # compute chunk size
    
    # split paths into lists of equal size
    thread_paths = list(split(paths, n_threads))
    
    # create and run threads
    threads = []
    file_names = []
    for i in range(n_threads):
        t = threading.Thread(target=prepare_data, args=[thread_paths[i], scan_types, filter_thr, voxel_dir], name="get_statistic")
        threads.append(t)
        t.start()
    
    # waiting for all threads to finish
    for t in threads:
        t.join()

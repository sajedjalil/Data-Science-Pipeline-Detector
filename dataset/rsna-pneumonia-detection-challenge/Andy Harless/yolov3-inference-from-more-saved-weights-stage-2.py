files1 = [
    'rsna_yolov3_15300.weights',
    'rsna_yolov3_fast_3200.weights',
    'rsna_yolov3_fast_10000.weights',
    'rsna_yolov3_fast_4100.weights',
    'rsna_yolov3_fast_13700.weights'
    ]
files2 = [
    'rsna_yolov3_fast_5300.weights',
    'rsna_yolov3_fast_1500.weights',
    'rsna_yolov3_fast_6700.weights',
    'rsna_yolov3_fast_2000.weights'
    ]
    
weight_files = files2

URL_STEM = 'http://andy.harless.us/rsnaweights/'

import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from subprocess import call
import hashlib

finished = False

#sys.stdout = open('log.txt', 'w')  # Log will not print correctly in kernel environment, so redirect output

random_stat = 123
np.random.seed(random_stat)


def cleanup():
    os.chdir( '/kaggle/working' )
    for d in [ 'darknet', 'images', 'labels', 'metadata', 'backup', 'cfg' ]:
        if os.path.exists(d):
            call([ 'rm', '-rf', d ])
    for d in [ 'train_log.txt', 'darknet_gpu', ' __pycache__', '.ipynb_checkpoints' ]:
        if os.path.exists(d):
            call([ 'rm', '-rf', d ])
    finished = True


try:

    call(['git', 'clone', 'https://github.com/pjreddie/darknet.git'])
    
    # Build gpu version darknet
    os.chdir('darknet')
    call(['sed',  '1 s/^.*$/GPU=1/; 2 s/^.*$/CUDNN=1/',  '-i', 'Makefile'])
    os.chdir('..')
    
    # -j <The # of cpu cores to use>. Chang 999 to fit your environment. Actually i used '-j 50'.
    os.chdir('darknet')
    call(['make',  '-j', '4',  '-s'])
    os.chdir('..')
    call(['cp', 'darknet/darknet', 'darknet_gpu'])
    
    
    DATA_DIR = "../input"
    
    test_dcm_dir = os.path.join(DATA_DIR, "stage_2_test_images")
    
    img_dir = os.path.join(os.getcwd(), "images")  # .jpg
    label_dir = os.path.join(os.getcwd(), "labels")  # .txt
    metadata_dir = os.path.join(os.getcwd(), "metadata") # .txt
    
    # YOLOv3 config file directory
    cfg_dir = os.path.join(os.getcwd(), "cfg")
    # YOLOv3 training checkpoints will be saved here
    backup_dir = os.path.join(os.getcwd(), "backup")

    for directory in [img_dir, label_dir, metadata_dir, cfg_dir, backup_dir]:
        if os.path.isdir(directory):
            continue
        os.mkdir(directory)
    
except:
    
    print( sys.exc_info() )
    cleanup()
    
    
    
    
def save_img_from_dcm(dcm_dir, img_dir, patient_id):
    img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
    if os.path.exists(img_fp):
        return
    dcm_fp = os.path.join(dcm_dir, "{}.dcm".format(patient_id))
    img_1ch = pydicom.read_file(dcm_fp).pixel_array
    img_3ch = np.stack([img_1ch]*3, -1)

    img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
    cv2.imwrite(img_fp, img_3ch)
    
def save_label_from_dcm(label_dir, patient_id, row=None):
    # rsna defualt image size
    img_size = 1024
    label_fp = os.path.join(label_dir, "{}.txt".format(patient_id))
    
    f = open(label_fp, "a")
    if row is None:
        f.close()
        return

    top_left_x = row[1]
    top_left_y = row[2]
    w = row[3]
    h = row[4]
    
    # 'r' means relative. 'c' means center.
    rx = top_left_x/img_size
    ry = top_left_y/img_size
    rw = w/img_size
    rh = h/img_size
    rcx = rx+rw/2
    rcy = ry+rh/2
    
    line = "{} {} {} {} {}\n".format(0, rcx, rcy, rw, rh)
    
    f.write(line)
    f.close()
        
def save_yolov3_test_data(test_dcm_dir, img_dir, metadata_dir, name, series):
    list_fp = os.path.join(metadata_dir, name)
    with open(list_fp, "w") as f:
        for patient_id in series:
            save_img_from_dcm(test_dcm_dir, img_dir, patient_id)
            line = "{}\n".format(os.path.join(img_dir, "{}.jpg".format(patient_id)))
            f.write(line)
            

if not finished:
  try:    
    test_dcm_fps = list(set(glob.glob(os.path.join(test_dcm_dir, '*.dcm'))))
    test_dcm_fps = pd.Series(test_dcm_fps).apply(lambda dcm_fp: dcm_fp.strip().split("/")[-1].replace(".dcm",""))
    
    save_yolov3_test_data(test_dcm_dir, img_dir, metadata_dir, "te_list.txt", test_dcm_fps)
    
    
    data_extention_file_path = os.path.join(cfg_dir, 'rsna.data')
    with open(data_extention_file_path, 'w') as f:
        contents = """classes= 1
    names  = {}
    backup = {}
        """.format(os.path.join(cfg_dir, 'rsna.names'),
                   backup_dir)
        f.write(contents)
        
    with open('cfg/rsna.names', 'w') as f:
        f.write('pneumonia')

    call([ 'wget',  '--no-check-certificate',  '-q',  
           'https://docs.google.com/uc?export=download&id=10Yk6ZMAKGz5LeBbikciALy82aK3lX-57',
           '-O', 'cfg/rsna_yolov3.cfg_test' ])

  except:

    print( sys.exc_info() )
    cleanup()


##################################################################################
######################  DARKNET PYTHON WRAPPER BEGINS HERE #######################
##################################################################################

from ctypes import *
import math
import random


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

try:
    # ==============================================================================
    #lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
    darknet_lib_path = os.path.join(os.getcwd(), "darknet", "libdarknet.so")
    lib = CDLL(darknet_lib_path, RTLD_GLOBAL)
    # ==============================================================================
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int
    
    predict = lib.network_predict
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)
    
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]
    
    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE
    
    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
    get_network_boxes.restype = POINTER(DETECTION)
    
    make_network_boxes = lib.make_network_boxes
    make_network_boxes.argtypes = [c_void_p]
    make_network_boxes.restype = POINTER(DETECTION)
    
    free_detections = lib.free_detections
    free_detections.argtypes = [POINTER(DETECTION), c_int]
    
    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]
    
    network_predict = lib.network_predict
    network_predict.argtypes = [c_void_p, POINTER(c_float)]
    
    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]
    
    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p
    
    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
    
    do_nms_sort = lib.do_nms_sort
    do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
    
    free_image = lib.free_image
    free_image.argtypes = [IMAGE]
    
    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE
    
    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA
    
    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE
    
    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]
    
    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

except:
    print( sys.exc_info() )
    cleanup()

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
    
##################################################################################
#######################  DARKNET PYTHON WRAPPER ENDS HERE ########################
##################################################################################

threshold = 0.2


if not finished:
    try:
    
      with open('md5sums.log', 'w') as md5file:
    
        for wf in weight_files:
            
            print( os.getcwd() )
    
            file_url = URL_STEM + wf
            call( ['wget', '-q', file_url] )
            md5file.write( hashlib.md5(open(wf,'rb').read()).hexdigest() + ' ' + wf + '\n' )
            weight_path = wf
    
            version = wf.split('.')[-2].split('_')[-1]        
            submit_file_path = "submission_yolo" + version + ".csv"
            cfg_path = os.path.join(cfg_dir, "rsna_yolov3.cfg_test")
    
            test_img_list_path = os.path.join(metadata_dir, "te_list.txt")
                   
            
            gpu_index = 0
            
            
            net = load_net(cfg_path.encode(),
                           weight_path.encode(), 
                           gpu_index)
            meta = load_meta(data_extention_file_path.encode())
            
            submit_dict = {"patientId": [], "PredictionString": []}
            
            with open(test_img_list_path, "r") as test_img_list_f:
                for line in test_img_list_f:
                    patient_id = line.strip().split('/')[-1].strip().split('.')[0]
            
                    infer_result = detect(net, meta, line.strip().encode(), thresh=threshold)
            
                    submit_line = ""
                    for e in infer_result:
                        confi = e[1]
                        w = e[2][2]
                        h = e[2][3]
                        x = e[2][0]-w/2
                        y = e[2][1]-h/2
                        submit_line += "{} {} {} {} {} ".format(confi, x, y, w, h)
            
                    submit_dict["patientId"].append(patient_id)
                    submit_dict["PredictionString"].append(submit_line)
                    
            pd.DataFrame(submit_dict).to_csv(submit_file_path, index=False)
            
    except:
      print( sys.exc_info() )



cleanup()
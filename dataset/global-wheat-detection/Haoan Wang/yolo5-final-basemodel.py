import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFont, ImageDraw
import sys
sys.path.insert(0, "../input/weightedboxesfusion")
sys.path.insert(0, "../input/yolov5master/yolov5-master")
import torch
from ensemble_boxes import *
from utils.datasets import *
from utils.utils import *



model_path = '../input/yolo5ensemble/last_yolov5x_fold4_fl.pt'
source = '../input/global-wheat-detection/test/'
model_img_size = (1024,1024)
conf_thres = 0.5
iou_thres = 0.6
augment = False
classes=None

def read_dataset(path, size, rotate=True):
    img_ori = cv2.imread(path)
    org_size = img_ori.shape
    if not rotate:
        img = letterbox(img_ori, new_shape=size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return [img], org_size
    else:
        rotate_lst = []
#         for imgo in [img_ori, np.fliplr(img_ori)]:
        for imgo in [img_ori]:
            for i in range(4):
                img = np.rot90(imgo, i)
                img = letterbox(img, new_shape=size)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                rotate_lst.append(img)
        return rotate_lst, org_size

def detect_image(model, img, org_size):
    boxes = []
    scores = []
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    img = img.unsqueeze(0)
    model_output = model(img, augment=False)[0]
    pred = non_max_suppression(model_output, conf_thres, iou_thres, merge=True, classes=None, agnostic=False)[0]
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], org_size).round()
        for *xyxy, conf, cls in pred:
            xywh = torch.tensor(xyxy).view(-1).numpy()
            boxes.append(xywh)
            scores.append(conf)
    
    return np.array(boxes), np.array(scores)
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)['model'].to(device).float().eval()
    print("model loaded")
                
    image_id_lst = [p.strip('.jpg') for p in os.listdir(source)]
    df = pd.DataFrame(columns=['PredictionString'], index=image_id_lst)
    
    for image_id in image_id_lst:
        image_path = source + image_id + '.jpg'
        img_lst, org_size = read_dataset(image_path, model_img_size, rotate=True)
        
        boxes_lst, socres_lst = [], []
        for i in range(len(img_lst)):
            boxes, scores = detect_image(model, img_lst[i], org_size)
            if len(boxes) != 0:
                if i % 4 == 1:
                    boxes[:,[0,1,2,3]] = boxes[:,[3,0,1,2]]
                    boxes[:,0] = 1024 - boxes[:,0]
                    boxes[:,2] = 1024 - boxes[:,2]
                if i % 4 == 2:
                    boxes[:,[0,1,2,3]] = boxes[:,[2,3,0,1]]
                    boxes[:,0] = 1024 - boxes[:,0]
                    boxes[:,1] = 1024 - boxes[:,1]
                    boxes[:,2] = 1024 - boxes[:,2]
                    boxes[:,3] = 1024 - boxes[:,3]
                if i % 4 == 3:
                    boxes[:,[0,1,2,3]] = boxes[:,[1,2,3,0]]
                    boxes[:,1] = 1024 - boxes[:,1]
                    boxes[:,3] = 1024 - boxes[:,3]
                if i > 3:
                    boxes[:,[0,1,2,3]] = boxes[:,[2,1,0,3]]
                    boxes[:,0] = 1024 - boxes[:,0]
                    boxes[:,2] = 1024 - boxes[:,2]
                boxes_lst.append(boxes)
                socres_lst.append(scores)
        try:   
            boxes_lst = np.concatenate(boxes_lst, axis=0)
            socres_lst= np.concatenate(socres_lst, axis=0)
            label_lst = [np.ones(len([socres_lst][idx])) for idx in range(len([socres_lst]))]
            boxes_lst, socres_lst, label_lst = weighted_boxes_fusion([boxes_lst], [socres_lst], label_lst, weights=None, iou_thr=0.5, skip_box_thr=0.7)
        except:
            pass
        result = []
        for i, box in list(enumerate(boxes_lst)):
            score = float(socres_lst[i])
            box = box.round().astype(np.int32).clip(min=0, max=1023)
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            width = xmax - xmin
            height = ymax - ymin
            result.append(' '.join(list(map(str, [score, xmin, ymin, width, height]))))
        df.loc[image_id, 'PredictionString'] = ' '.join(result)
    df.reset_index(inplace=True, drop=False)
    df.columns = ['image_id','PredictionString']
    df.to_csv('submission.csv', index=False)
    print('done')
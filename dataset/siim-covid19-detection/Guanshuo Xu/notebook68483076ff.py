import subprocess
subprocess.run(["pip", "install", "../input/pycocotools202/pycocotools-2.0.2-cp37-cp37m-linux_x86_64.whl"])
subprocess.run(["tar", "-xvf", "../input/gdcminstall/gdcm.tar"])
subprocess.run(["conda", "install", "../working/gdcm/conda-4.8.4-py37hc8dfbb8_2.tar.bz2"])
subprocess.run(["conda", "install", "../working/gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2"])
subprocess.run(["conda", "install", "../working/gdcm/libjpeg-turbo-2.0.3-h516909a_1.tar.bz2"])


def get_none_results(image_id_list_matched_midrc, midrc_dict):
    import sys
    sys.path.insert(0, '../input/covidtimm/')
    import numpy as np
    import pandas as pd
    import os
    import cv2
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    import torch
    import timm
    import pickle
    import time
    from torch.cuda.amp import autocast, GradScaler
    import glob
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    # https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    def read_xray(path, voi_lut = True, fix_monochrome = True):
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
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        
        return data

    class TestDataset(Dataset):
        def __init__(self, label_dict, image_list):
            self.label_dict=label_dict
            self.image_list=image_list
        def __len__(self):
            return len(self.image_list)
        def __getitem__(self, index):
            image_id = self.image_list[index]
            img = read_xray(self.label_dict[image_id]['img_dir'])
            img = cv2.resize(img, (1280, 1280))
            image = np.zeros((1280,1280,3), dtype=np.uint8)
            image[:,:,0] = img
            image[:,:,1] = img
            image[:,:,2] = img
            image672 = cv2.resize(image, (672, 672))
            image608 = cv2.resize(image, (608, 608))
            image672 = image672.transpose(2, 0, 1)
            image608 = image608.transpose(2, 0, 1)
            return image672, image608

    class TestDataset1(Dataset):
        def __init__(self, label_dict, image_list):
            self.label_dict=label_dict
            self.image_list=image_list
        def __len__(self):
            return len(self.image_list)
        def __getitem__(self, index):
            image_id = self.image_list[index]
            image = cv2.imread(self.label_dict[image_id]['img_dir'])
            image = cv2.resize(image, (1280, 1280))
            image672 = cv2.resize(image, (672, 672))
            image608 = cv2.resize(image, (608, 608))
            image672 = image672.transpose(2, 0, 1)
            image608 = image608.transpose(2, 0, 1)
            return image672, image608
    
    class NoneEffb7(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.net = timm.create_model('tf_efficientnet_b7_ns', features_only=True, out_indices=(3, 4), drop_path_rate=0.5, pretrained=False)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = 640
            self.last_linear = nn.Linear(in_features, 1)
            self.conv_mask = nn.Conv2d(224, 1, kernel_size=1, stride=1, bias=True)
        def forward(self, x):
            x1, x = self.net(x)
            x_mask = self.conv_mask(x1)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.last_linear(x)
            return x, x_mask
    
    class NoneEffb8(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.net = timm.create_model('tf_efficientnet_b8', features_only=True, out_indices=(3, 4), drop_path_rate=0.5, pretrained=False)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = 704
            self.last_linear = nn.Linear(in_features, 1)
            self.conv_mask = nn.Conv2d(248, 1, kernel_size=1, stride=1, bias=True)
        def forward(self, x):
            x1, x = self.net(x)
            x_mask = self.conv_mask(x1)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.last_linear(x)
            return x, x_mask


    start_time = time.time()

    # prepare input
    image_list = sorted(glob.glob('../input/siim-covid19-detection/test/*/*/*.dcm'))
    study_id_list = []
    image_id_list = []
    studyid2imageid = {}
    imageid2studyid = {}
    label_dict = {}
    for i in range(len(image_list)):
        study_id = image_list[i].split('/')[-3]
        image_id = image_list[i].split('/')[-1][:-4]
        image_id_list.append(image_id)
        if study_id not in studyid2imageid:
            study_id_list.append(study_id)
            studyid2imageid[study_id] = [image_id]
        else:
            studyid2imageid[study_id].append(image_id)
        imageid2studyid[image_id] = study_id
        label_dict[image_id] = {
            'img_dir': image_list[i],
        }  

    print(len(image_id_list), len(study_id_list), len(studyid2imageid), len(imageid2studyid))

    image_id_list111 = []
    for image_id in image_id_list:
        if image_id not in image_id_list_matched_midrc:
            image_id_list111.append(image_id)
    image_id_list = image_id_list111
    print(len(image_id_list))

    # hyperparameters
    batch_size = 4

    # build model
    model11_fold0 = NoneEffb8()
    model11_fold0.load_state_dict(torch.load('../input/covidpretrained4/model11_fold0/epoch10_polyak'))
    model11_fold0.cuda()
    model11_fold0.eval()

    model11_fold1 = NoneEffb8()
    model11_fold1.load_state_dict(torch.load('../input/covidpretrained4/model11_fold1/epoch10_polyak'))
    model11_fold1.cuda()
    model11_fold1.eval()

    model11_fold2 = NoneEffb8()
    model11_fold2.load_state_dict(torch.load('../input/covidpretrained4/model11_fold2/epoch10_polyak'))
    model11_fold2.cuda()
    model11_fold2.eval()

    model11_fold3 = NoneEffb8()
    model11_fold3.load_state_dict(torch.load('../input/covidpretrained4/model11_fold3/epoch10_polyak'))
    model11_fold3.cuda()
    model11_fold3.eval()

    model11_fold4 = NoneEffb8()
    model11_fold4.load_state_dict(torch.load('../input/covidpretrained4/model11_fold4/epoch10_polyak'))
    model11_fold4.cuda()
    model11_fold4.eval()
    
    model12_fold0 = NoneEffb7()
    model12_fold0.load_state_dict(torch.load('../input/covidpretrained4/model12_fold0/epoch6_polyak'))
    model12_fold0.cuda()
    model12_fold0.eval()

    model12_fold1 = NoneEffb7()
    model12_fold1.load_state_dict(torch.load('../input/covidpretrained4/model12_fold1/epoch6_polyak'))
    model12_fold1.cuda()
    model12_fold1.eval()

    model12_fold2 = NoneEffb7()
    model12_fold2.load_state_dict(torch.load('../input/covidpretrained4/model12_fold2/epoch6_polyak'))
    model12_fold2.cuda()
    model12_fold2.eval()

    model12_fold3 = NoneEffb7()
    model12_fold3.load_state_dict(torch.load('../input/covidpretrained4/model12_fold3/epoch6_polyak'))
    model12_fold3.cuda()
    model12_fold3.eval()

    model12_fold4 = NoneEffb7()
    model12_fold4.load_state_dict(torch.load('../input/covidpretrained4/model12_fold4/epoch6_polyak'))
    model12_fold4.cuda()
    model12_fold4.eval()
    
    # iterator for testing
    datagen = TestDataset(label_dict=label_dict, image_list=image_id_list)
    generator = DataLoader(dataset=datagen, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    pred_prob = np.zeros((len(image_id_list), ), dtype=np.float32)
    for i, (images672, images608) in enumerate(generator):
        with torch.no_grad():
            start = i*batch_size
            end = start+batch_size
            if i == len(generator)-1:
                end = len(generator.dataset)
            images672 = images672.cuda().float() / 255.0
            images608 = images608.cuda().float() / 255.0
            with autocast():
                logits_model11_fold0, _ = model11_fold0(images608)
                logits_model11_fold1, _ = model11_fold1(images608)
                logits_model11_fold2, _ = model11_fold2(images608)
                logits_model11_fold3, _ = model11_fold3(images608)
                logits_model11_fold4, _ = model11_fold4(images608)

                logits_flip_model11_fold0, _ = model11_fold0(images608.flip(3))
                logits_flip_model11_fold1, _ = model11_fold1(images608.flip(3))
                logits_flip_model11_fold2, _ = model11_fold2(images608.flip(3))
                logits_flip_model11_fold3, _ = model11_fold3(images608.flip(3))
                logits_flip_model11_fold4, _ = model11_fold4(images608.flip(3))
                
                logits_model12_fold0, _ = model12_fold0(images672)
                logits_model12_fold1, _ = model12_fold1(images672)
                logits_model12_fold2, _ = model12_fold2(images672)
                logits_model12_fold3, _ = model12_fold3(images672)
                logits_model12_fold4, _ = model12_fold4(images672)

                logits_flip_model12_fold0, _ = model12_fold0(images672.flip(3))
                logits_flip_model12_fold1, _ = model12_fold1(images672.flip(3))
                logits_flip_model12_fold2, _ = model12_fold2(images672.flip(3))
                logits_flip_model12_fold3, _ = model12_fold3(images672.flip(3))
                logits_flip_model12_fold4, _ = model12_fold4(images672.flip(3))

            pred_prob[start:end] += logits_model11_fold0.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model11_fold1.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model11_fold2.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model11_fold3.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model11_fold4.sigmoid().cpu().data.numpy().squeeze()

            pred_prob[start:end] += logits_flip_model11_fold0.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model11_fold1.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model11_fold2.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model11_fold3.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model11_fold4.sigmoid().cpu().data.numpy().squeeze()
            
            pred_prob[start:end] += logits_model12_fold0.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model12_fold1.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model12_fold2.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model12_fold3.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model12_fold4.sigmoid().cpu().data.numpy().squeeze()

            pred_prob[start:end] += logits_flip_model12_fold0.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model12_fold1.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model12_fold2.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model12_fold3.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model12_fold4.sigmoid().cpu().data.numpy().squeeze()

    pred_prob /= 20.0

                    
    # generate submission
    image_level_dict = {}
    for image_id in image_id_list:
        image_level_dict[image_id] = {"opacity": [], "none": 0., "negative": 0., "typical": 0., "indeterminate": 0., "atypical": 0.}

    for i in range(len(image_id_list)):
        image_level_dict[image_id_list[i]]["none"] = pred_prob[i]


    ### re-prediction ###    
    # iterator for testing
    datagen = TestDataset1(label_dict=midrc_dict, image_list=image_id_list_matched_midrc)
    generator = DataLoader(dataset=datagen, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    pred_prob = np.zeros((len(image_id_list_matched_midrc), ), dtype=np.float32)
    for i, (images672, images608) in enumerate(generator):
        with torch.no_grad():
            start = i*batch_size
            end = start+batch_size
            if i == len(generator)-1:
                end = len(generator.dataset)
            images672 = images672.cuda().float() / 255.0
            images608 = images608.cuda().float() / 255.0
            with autocast():
                logits_model11_fold0, _ = model11_fold0(images608)
                logits_model11_fold1, _ = model11_fold1(images608)
                logits_model11_fold2, _ = model11_fold2(images608)
                logits_model11_fold3, _ = model11_fold3(images608)
                logits_model11_fold4, _ = model11_fold4(images608)

                logits_flip_model11_fold0, _ = model11_fold0(images608.flip(3))
                logits_flip_model11_fold1, _ = model11_fold1(images608.flip(3))
                logits_flip_model11_fold2, _ = model11_fold2(images608.flip(3))
                logits_flip_model11_fold3, _ = model11_fold3(images608.flip(3))
                logits_flip_model11_fold4, _ = model11_fold4(images608.flip(3))
                
                logits_model12_fold0, _ = model12_fold0(images672)
                logits_model12_fold1, _ = model12_fold1(images672)
                logits_model12_fold2, _ = model12_fold2(images672)
                logits_model12_fold3, _ = model12_fold3(images672)
                logits_model12_fold4, _ = model12_fold4(images672)

                logits_flip_model12_fold0, _ = model12_fold0(images672.flip(3))
                logits_flip_model12_fold1, _ = model12_fold1(images672.flip(3))
                logits_flip_model12_fold2, _ = model12_fold2(images672.flip(3))
                logits_flip_model12_fold3, _ = model12_fold3(images672.flip(3))
                logits_flip_model12_fold4, _ = model12_fold4(images672.flip(3))

            pred_prob[start:end] += logits_model11_fold0.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model11_fold1.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model11_fold2.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model11_fold3.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model11_fold4.sigmoid().cpu().data.numpy().squeeze()

            pred_prob[start:end] += logits_flip_model11_fold0.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model11_fold1.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model11_fold2.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model11_fold3.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model11_fold4.sigmoid().cpu().data.numpy().squeeze()
            
            pred_prob[start:end] += logits_model12_fold0.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model12_fold1.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model12_fold2.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model12_fold3.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_model12_fold4.sigmoid().cpu().data.numpy().squeeze()

            pred_prob[start:end] += logits_flip_model12_fold0.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model12_fold1.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model12_fold2.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model12_fold3.sigmoid().cpu().data.numpy().squeeze()
            pred_prob[start:end] += logits_flip_model12_fold4.sigmoid().cpu().data.numpy().squeeze()

    pred_prob /= 20.0        

    for image_id in image_id_list_matched_midrc:
        image_level_dict[image_id] = {"opacity": [], "none": 0., "negative": 0., "typical": 0., "indeterminate": 0., "atypical": 0.}
    for i in range(len(image_id_list_matched_midrc)):
        image_level_dict[image_id_list_matched_midrc[i]]["none"] = pred_prob[i]

    end_time = time.time()
    print(end_time-start_time)

    return image_level_dict


def get_study_level_results():
    import sys
    sys.path.insert(0, '../input/covidtimm/')
    import numpy as np
    import pandas as pd
    import os
    import cv2
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    import torch
    import timm
    import pickle
    import time
    from torch.cuda.amp import autocast, GradScaler
    import glob
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    # https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    def read_xray(path, voi_lut = True, fix_monochrome = True):
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
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        
        return data

    class TestDataset(Dataset):
        def __init__(self, label_dict, image_list):
            self.label_dict=label_dict
            self.image_list=image_list
        def __len__(self):
            return len(self.image_list)
        def __getitem__(self, index):
            image_id = self.image_list[index]
            img = read_xray(self.label_dict[image_id]['img_dir'])
            img = cv2.resize(img, (1280, 1280))
            image = np.zeros((1280,1280,3), dtype=np.uint8)
            image[:,:,0] = img
            image[:,:,1] = img
            image[:,:,2] = img
            image672 = cv2.resize(image, (672, 672))
            image608 = cv2.resize(image, (608, 608))
            image512 = cv2.resize(image, (512, 512))
            image672 = image672.transpose(2, 0, 1)
            image608 = image608.transpose(2, 0, 1)
            image512 = image512.transpose(2, 0, 1)
            return image672, image608, image512

    class TestDataset1(Dataset):
        def __init__(self, label_dict, image_list):
            self.label_dict=label_dict
            self.image_list=image_list
        def __len__(self):
            return len(self.image_list)
        def __getitem__(self, index):
            image_id = self.image_list[index]
            image = cv2.imread(self.label_dict[image_id]['img_dir'])
            image = cv2.resize(image, (1280, 1280))
            image672 = cv2.resize(image, (672, 672))
            image608 = cv2.resize(image, (608, 608))
            image672 = image672.transpose(2, 0, 1)
            image608 = image608.transpose(2, 0, 1)
            return image672, image608

    class StudyLevelEffb7(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.net = timm.create_model('tf_efficientnet_b7_ns', features_only=True, out_indices=(3, 4), drop_path_rate=0.5, pretrained=False)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = 640
            self.last_linear = nn.Linear(in_features, 4)
            self.conv_mask = nn.Conv2d(224, 1, kernel_size=1, stride=1, bias=True)
        def forward(self, x):
            x1, x = self.net(x)
            x_mask = self.conv_mask(x1)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.last_linear(x)
            return x, x_mask

    class StudyLevelEffb8(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.net = timm.create_model('tf_efficientnet_b8', features_only=True, out_indices=(3, 4), drop_path_rate=0.5, pretrained=False)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = 704
            self.last_linear = nn.Linear(in_features, 4)
            self.conv_mask = nn.Conv2d(248, 1, kernel_size=1, stride=1, bias=True)
        def forward(self, x):
            x1, x = self.net(x)
            x_mask = self.conv_mask(x1)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.last_linear(x)
            return x, x_mask

    class XrayNet(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.net = timm.create_model('tf_efficientnet_b3_ns', drop_path_rate=0.3, pretrained=False)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        def forward(self, x):
            x = self.net.forward_features(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            return x


    start_time = time.time()

    # prepare input
    image_list = sorted(glob.glob('../input/siim-covid19-detection/test/*/*/*.dcm'))
    study_id_list = []
    image_id_list = []
    studyid2imageid = {}
    imageid2studyid = {}
    label_dict = {}
    for i in range(len(image_list)):
        study_id = image_list[i].split('/')[-3]
        image_id = image_list[i].split('/')[-1][:-4]
        image_id_list.append(image_id)
        if study_id not in studyid2imageid:
            study_id_list.append(study_id)
            studyid2imageid[study_id] = [image_id]
        else:
            studyid2imageid[study_id].append(image_id)
        imageid2studyid[image_id] = study_id
        label_dict[image_id] = {
            'img_dir': image_list[i],
        }  

    print(len(image_id_list), len(study_id_list), len(studyid2imageid), len(imageid2studyid))

    # hyperparameters
    batch_size = 4

    # build model
    model1_fold0 = StudyLevelEffb8()
    model1_fold0.load_state_dict(torch.load('../input/covidpretrained3/model1_fold0/epoch8_polyak'))
    model1_fold0.cuda()
    model1_fold0.eval()

    model1_fold1 = StudyLevelEffb8()
    model1_fold1.load_state_dict(torch.load('../input/covidpretrained3/model1_fold1/epoch8_polyak'))
    model1_fold1.cuda()
    model1_fold1.eval()

    model1_fold2 = StudyLevelEffb8()
    model1_fold2.load_state_dict(torch.load('../input/covidpretrained3/model1_fold2/epoch8_polyak'))
    model1_fold2.cuda()
    model1_fold2.eval()

    model1_fold3 = StudyLevelEffb8()
    model1_fold3.load_state_dict(torch.load('../input/covidpretrained3/model1_fold3/epoch8_polyak'))
    model1_fold3.cuda()
    model1_fold3.eval()

    model1_fold4 = StudyLevelEffb8()
    model1_fold4.load_state_dict(torch.load('../input/covidpretrained3/model1_fold4/epoch8_polyak'))
    model1_fold4.cuda()
    model1_fold4.eval()
    
    model2_fold0 = StudyLevelEffb7()
    model2_fold0.load_state_dict(torch.load('../input/covidpretrained3/model2_fold0/epoch9_polyak'))
    model2_fold0.cuda()
    model2_fold0.eval()

    model2_fold1 = StudyLevelEffb7()
    model2_fold1.load_state_dict(torch.load('../input/covidpretrained3/model2_fold1/epoch9_polyak'))
    model2_fold1.cuda()
    model2_fold1.eval()

    model2_fold2 = StudyLevelEffb7()
    model2_fold2.load_state_dict(torch.load('../input/covidpretrained3/model2_fold2/epoch9_polyak'))
    model2_fold2.cuda()
    model2_fold2.eval()

    model2_fold3 = StudyLevelEffb7()
    model2_fold3.load_state_dict(torch.load('../input/covidpretrained3/model2_fold3/epoch9_polyak'))
    model2_fold3.cuda()
    model2_fold3.eval()

    model2_fold4 = StudyLevelEffb7()
    model2_fold4.load_state_dict(torch.load('../input/covidpretrained3/model2_fold4/epoch9_polyak'))
    model2_fold4.cuda()
    model2_fold4.eval()

    model_xray = XrayNet()
    model_xray.load_state_dict(torch.load('../input/covidnearestneighbors/epoch79'))
    model_xray = model_xray.cuda()
    model_xray.eval()
    
    # iterator for testing
    datagen = TestDataset(label_dict=label_dict, image_list=image_id_list)
    generator = DataLoader(dataset=datagen, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    pred_prob = np.zeros((len(image_id_list), 4), dtype=np.float32)
    valid_feature = np.zeros((len(image_id_list), 1536), dtype=np.float32)
    for i, (images672, images608, images512) in enumerate(generator):
        with torch.no_grad():
            start = i*batch_size
            end = start+batch_size
            if i == len(generator)-1:
                end = len(generator.dataset)
            images672 = images672.cuda().float() / 255.0
            images608 = images608.cuda().float() / 255.0
            images512 = images512.cuda().float() / 255.0
            with autocast():
                logits_model1_fold0, _ = model1_fold0(images608)
                logits_model1_fold1, _ = model1_fold1(images608)
                logits_model1_fold2, _ = model1_fold2(images608)
                logits_model1_fold3, _ = model1_fold3(images608)
                logits_model1_fold4, _ = model1_fold4(images608)

                logits_flip_model1_fold0, _ = model1_fold0(images608.flip(3))
                logits_flip_model1_fold1, _ = model1_fold1(images608.flip(3))
                logits_flip_model1_fold2, _ = model1_fold2(images608.flip(3))
                logits_flip_model1_fold3, _ = model1_fold3(images608.flip(3))
                logits_flip_model1_fold4, _ = model1_fold4(images608.flip(3))
                
                logits_model2_fold0, _ = model2_fold0(images672)
                logits_model2_fold1, _ = model2_fold1(images672)
                logits_model2_fold2, _ = model2_fold2(images672)
                logits_model2_fold3, _ = model2_fold3(images672)
                logits_model2_fold4, _ = model2_fold4(images672)

                logits_flip_model2_fold0, _ = model2_fold0(images672.flip(3))
                logits_flip_model2_fold1, _ = model2_fold1(images672.flip(3))
                logits_flip_model2_fold2, _ = model2_fold2(images672.flip(3))
                logits_flip_model2_fold3, _ = model2_fold3(images672.flip(3))
                logits_flip_model2_fold4, _ = model2_fold4(images672.flip(3))
                
                features = model_xray(images512)

            pred_prob[start:end] += logits_model1_fold0.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model1_fold1.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model1_fold2.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model1_fold3.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model1_fold4.sigmoid().cpu().data.numpy()

            pred_prob[start:end] += logits_flip_model1_fold0.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model1_fold1.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model1_fold2.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model1_fold3.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model1_fold4.sigmoid().cpu().data.numpy()
            
            pred_prob[start:end] += logits_model2_fold0.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model2_fold1.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model2_fold2.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model2_fold3.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model2_fold4.sigmoid().cpu().data.numpy()

            pred_prob[start:end] += logits_flip_model2_fold0.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model2_fold1.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model2_fold2.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model2_fold3.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model2_fold4.sigmoid().cpu().data.numpy()
            
            valid_feature[start:end] = F.normalize(features).cpu().data.numpy()

    pred_prob /= 20.0

                    
    # generate submission
    image_level_dict = {}
    for image_id in image_id_list:
        image_level_dict[image_id] = {"opacity": [], "none": 0., "negative": 0., "typical": 0., "indeterminate": 0., "atypical": 0.}

    for i in range(len(image_id_list)):
        image_level_dict[image_id_list[i]]["negative"] = pred_prob[i][0]
        image_level_dict[image_id_list[i]]["typical"] = pred_prob[i][1]
        image_level_dict[image_id_list[i]]["indeterminate"] = pred_prob[i][2]
        image_level_dict[image_id_list[i]]["atypical"] = pred_prob[i][3]

    ###
    from sklearn.neighbors import NearestNeighbors
    
    train_feature_bimcv = np.load('../input/covidnearestneighbors/ext_features_bimcv.npy')
    
    image_id_list = np.array(image_id_list)
    
    neigh = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=-1)
    neigh.fit(train_feature_bimcv)
    pred_dist, pred_label = np.squeeze(neigh.kneighbors(X=valid_feature, n_neighbors=1, return_distance=True))
    pred_dist = np.squeeze(pred_dist)
    pred_label = np.squeeze(pred_label)

    num_unmatch_bimcv = 0
    num_matched_bimcv = 0
    idx_unmatch_bimcv = []
    idx_matched_bimcv = []
    for i in range(len(pred_dist)):
        if pred_dist[i] < 0.01:
            num_matched_bimcv += 1
            idx_matched_bimcv.append(i)
        else:
            num_unmatch_bimcv += 1
            idx_unmatch_bimcv.append(i)
    print(num_matched_bimcv, num_unmatch_bimcv)

    pred_dist_matched_bimcv = pred_dist[idx_matched_bimcv]
    pred_label_matched_bimcv = pred_label[idx_matched_bimcv]
    image_id_list_matched_bimcv = list(image_id_list[idx_matched_bimcv])
    print(len(image_id_list_matched_bimcv))

    bimcv_id_list = list(pd.read_csv('../input/covidnearestneighbors/ext_bimcv.csv')['id'].values)
    #bimcv_train_id_list = []
    bimcv_dict = {}
    for i in range(len(image_id_list_matched_bimcv)):
        #bimcv_train_id_list.append(bimcv_id_list[int(pred_label_matched_bimcv[i])])
        bimcv_dict[image_id_list_matched_bimcv[i]] = {'bimcv_id': bimcv_id_list[int(pred_label_matched_bimcv[i])]}
    
    valid_feature_unmatch_bimcv = valid_feature[idx_unmatch_bimcv]
    print(valid_feature_unmatch_bimcv.shape)

    train_feature_midrc = np.load('../input/covidnearestneighbors/ext_features_midrc.npy')
    neigh = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=-1)
    neigh.fit(train_feature_midrc)
    pred_dist, pred_label = np.squeeze(neigh.kneighbors(X=valid_feature_unmatch_bimcv, n_neighbors=1, return_distance=True))
    pred_dist = np.squeeze(pred_dist)
    pred_label = np.squeeze(pred_label)
    
    num_unmatch_midrc = 0
    num_matched_midrc = 0
    idx_unmatch_midrc = []
    idx_matched_midrc = []
    for i in range(len(pred_dist)):
        if pred_dist[i] < 0.5:
            num_matched_midrc += 1
            idx_matched_midrc.append(i)
        else:
            num_unmatch_midrc += 1
            idx_unmatch_midrc.append(i)
    print(num_matched_midrc, num_unmatch_midrc)

    pred_dist_matched_midrc = pred_dist[idx_matched_midrc]
    pred_label_matched_midrc = pred_label[idx_matched_midrc]
    image_id_list_matched_midrc = list(image_id_list[idx_unmatch_bimcv][idx_matched_midrc])
    print(len(image_id_list_matched_midrc))
    
    midrc_id_list = list(pd.read_csv('../input/covidnearestneighbors/ext_midrc.csv')['id'].values)
    #midrc_train_id_list = []
    midrc_dict = {}
    for i in range(len(image_id_list_matched_midrc)):
        #midrc_train_id_list.append(midrc_id_list[int(pred_label_matched_midrc[i])])
        midrc_dict[image_id_list_matched_midrc[i]] = {'img_dir': '../input/ricord-covid19-xray-positive-tests/MIDRC-RICORD/MIDRC-RICORD/' + midrc_id_list[int(pred_label_matched_midrc[i])], 'midrc_id': midrc_id_list[int(pred_label_matched_midrc[i])]}

    ### re-prediction ###    
    # iterator for testing
    datagen = TestDataset1(label_dict=midrc_dict, image_list=image_id_list_matched_midrc)
    generator = DataLoader(dataset=datagen, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    pred_prob = np.zeros((len(image_id_list_matched_midrc), 4), dtype=np.float32)
    for i, (images672, images608) in enumerate(generator):
        with torch.no_grad():
            start = i*batch_size
            end = start+batch_size
            if i == len(generator)-1:
                end = len(generator.dataset)
            images672 = images672.cuda().float() / 255.0
            images608 = images608.cuda().float() / 255.0
            with autocast():
                logits_model1_fold0, _ = model1_fold0(images608)
                logits_model1_fold1, _ = model1_fold1(images608)
                logits_model1_fold2, _ = model1_fold2(images608)
                logits_model1_fold3, _ = model1_fold3(images608)
                logits_model1_fold4, _ = model1_fold4(images608)

                logits_flip_model1_fold0, _ = model1_fold0(images608.flip(3))
                logits_flip_model1_fold1, _ = model1_fold1(images608.flip(3))
                logits_flip_model1_fold2, _ = model1_fold2(images608.flip(3))
                logits_flip_model1_fold3, _ = model1_fold3(images608.flip(3))
                logits_flip_model1_fold4, _ = model1_fold4(images608.flip(3))
                
                logits_model2_fold0, _ = model2_fold0(images672)
                logits_model2_fold1, _ = model2_fold1(images672)
                logits_model2_fold2, _ = model2_fold2(images672)
                logits_model2_fold3, _ = model2_fold3(images672)
                logits_model2_fold4, _ = model2_fold4(images672)

                logits_flip_model2_fold0, _ = model2_fold0(images672.flip(3))
                logits_flip_model2_fold1, _ = model2_fold1(images672.flip(3))
                logits_flip_model2_fold2, _ = model2_fold2(images672.flip(3))
                logits_flip_model2_fold3, _ = model2_fold3(images672.flip(3))
                logits_flip_model2_fold4, _ = model2_fold4(images672.flip(3))

            pred_prob[start:end] += logits_model1_fold0.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model1_fold1.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model1_fold2.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model1_fold3.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model1_fold4.sigmoid().cpu().data.numpy()

            pred_prob[start:end] += logits_flip_model1_fold0.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model1_fold1.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model1_fold2.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model1_fold3.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model1_fold4.sigmoid().cpu().data.numpy()
            
            pred_prob[start:end] += logits_model2_fold0.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model2_fold1.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model2_fold2.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model2_fold3.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_model2_fold4.sigmoid().cpu().data.numpy()

            pred_prob[start:end] += logits_flip_model2_fold0.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model2_fold1.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model2_fold2.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model2_fold3.sigmoid().cpu().data.numpy()
            pred_prob[start:end] += logits_flip_model2_fold4.sigmoid().cpu().data.numpy()

    pred_prob /= 20.0        
    for i in range(len(image_id_list_matched_midrc)):
        image_level_dict[image_id_list_matched_midrc[i]]["negative"] = pred_prob[i][0]
        image_level_dict[image_id_list_matched_midrc[i]]["typical"] = pred_prob[i][1]
        image_level_dict[image_id_list_matched_midrc[i]]["indeterminate"] = pred_prob[i][2]
        image_level_dict[image_id_list_matched_midrc[i]]["atypical"] = pred_prob[i][3]

    end_time = time.time()
    print(end_time-start_time)

    return image_level_dict, image_id_list_matched_midrc, midrc_dict, image_id_list_matched_bimcv, bimcv_dict


def get_detection_results(image_id_list_matched_midrc, midrc_dict):
    import sys
    sys.path.insert(0, '../input/covidalbumentations/')
    sys.path.insert(0, '../input/covideffdet/')
    sys.path.insert(0, '../input/covidensembleboxes/')
    sys.path.insert(0, '../input/covidomegaconf/')
    sys.path.insert(0, '../input/covidtimm/')
    sys.path.insert(0, '../input/covidyolo/yolov5-5.0/')
    import numpy as np
    import pandas as pd
    import os
    import cv2
    from tqdm import tqdm
    import torchvision
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torch
    import pickle
    import albumentations
    from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict, load_pretrained
    from effdet.data.parsers import CocoParserCfg, create_parser
    from ensemble_boxes import weighted_boxes_fusion
    import time
    from torch.cuda.amp import autocast, GradScaler
    import glob
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    from utils.general import xyxy2xywh, xywh2xyxy
    from utils.datasets import letterbox
    from utils.torch_utils import select_device
    from utils.general import scale_coords
    from models.experimental import attempt_load

    def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                            labels=()):
        """Runs Non-Maximum Suppression (NMS) on inference results
    
        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 100  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output

    # https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    def read_xray(path, voi_lut = True, fix_monochrome = True):
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
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        
        return data

    class TestDataset(Dataset):
        def __init__(self, label_dict, image_list):
            self.label_dict=label_dict
            self.image_list=image_list
        def __len__(self):
            return len(self.image_list)
        def __getitem__(self,index):
            image_id = self.image_list[index]
            img = read_xray(self.label_dict[image_id]['img_dir'])
            h, w = img.shape[0], img.shape[1]
            img = cv2.resize(img, (1280, 1280))
            image = np.zeros((1280,1280,3), dtype=np.uint8)
            image[:,:,0] = img
            image[:,:,1] = img
            image[:,:,2] = img
            image1024 = cv2.resize(image, (1024, 1024))
            image896 = cv2.resize(image, (896, 896))
            imageyolo = cv2.resize(image, (1024, 1024))
            imageyolo = imageyolo[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            imageyolo = np.ascontiguousarray(imageyolo)
            return image1024, image896, imageyolo, h, w

    class TestDataset1(Dataset):
        def __init__(self, label_dict, image_list):
            self.label_dict=label_dict
            self.image_list=image_list
        def __len__(self):
            return len(self.image_list)
        def __getitem__(self,index):
            image_id = self.image_list[index]
            image = cv2.imread(self.label_dict[image_id]['img_dir'])
            h, w = image.shape[0], image.shape[1]
            image = cv2.resize(image, (1280, 1280))
            image1024 = cv2.resize(image, (1024, 1024))
            image896 = cv2.resize(image, (896, 896))
            imageyolo = cv2.resize(image, (1024, 1024))
            imageyolo = imageyolo[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            imageyolo = np.ascontiguousarray(imageyolo)
            return image1024, image896, imageyolo, h, w

    def collate_fn(batch):
        image1024_list = []
        image896_list = []
        imageyolo_list = []
        h_list = []
        w_list = []
        for i in range(len(batch)):
            image1024_list.append(batch[i][0])
            image896_list.append(batch[i][1])
            imageyolo_list.append(batch[i][2])
            h_list.append(batch[i][3])
            w_list.append(batch[i][4])
        image1024_list = np.stack(image1024_list)
        image896_list = np.stack(image896_list)
        image1024_list = torch.from_numpy(image1024_list)
        image896_list = torch.from_numpy(image896_list)
        imageyolo_list = np.stack(imageyolo_list)
        imageyolo_list = torch.from_numpy(imageyolo_list)
        targets = {'h': h_list, 'w': w_list}
        return image1024_list, image896_list, imageyolo_list, targets

    def get_effdetd5():
        config = get_efficientdet_config('tf_efficientdet_d5')
        config.image_size = [1024, 1024]
        config.norm_kwargs = dict(eps=0.001, momentum=0.01)
        net = EfficientDet(config, pretrained_backbone=False)
        net.reset_head(num_classes=6)
        return DetBenchPredict(net)

    def get_effdetd6():
        config = get_efficientdet_config('tf_efficientdet_d6')
        config.image_size = [896, 896]
        config.norm_kwargs = dict(eps=0.001, momentum=0.01)
        net = EfficientDet(config, pretrained_backbone=False)
        net.reset_head(num_classes=6)
        return DetBenchPredict(net)



    start_time = time.time()

    # prepare input
    image_list = sorted(glob.glob('../input/siim-covid19-detection/test/*/*/*.dcm'))
    study_id_list = []
    image_id_list = []
    studyid2imageid = {}
    imageid2studyid = {}
    label_dict = {}
    for i in range(len(image_list)):
        study_id = image_list[i].split('/')[-3]
        image_id = image_list[i].split('/')[-1][:-4]
        image_id_list.append(image_id)
        if study_id not in studyid2imageid:
            study_id_list.append(study_id)
            studyid2imageid[study_id] = [image_id]
        else:
            studyid2imageid[study_id].append(image_id)
        imageid2studyid[image_id] = study_id
        label_dict[image_id] = {
            'img_dir': image_list[i],
        }  

    print(len(image_id_list), len(study_id_list), len(studyid2imageid), len(imageid2studyid))

    image_id_list111 = []
    for image_id in image_id_list:
        if image_id not in image_id_list_matched_midrc:
            image_id_list111.append(image_id)
    image_id_list = image_id_list111
    print(len(image_id_list))

    # hyperparameters
    batch_size = 4

    # build model
    model0_fold0 = get_effdetd5()
    model0_fold0.load_state_dict(torch.load('../input/covidpretrained/effdetd5_fold0/epoch50_polyak'))
    model0_fold0.cuda()
    model0_fold0.eval()

    model0_fold1 = get_effdetd5()
    model0_fold1.load_state_dict(torch.load('../input/covidpretrained/effdetd5_fold1/epoch50_polyak'))
    model0_fold1.cuda()
    model0_fold1.eval()

    model0_fold2 = get_effdetd5()
    model0_fold2.load_state_dict(torch.load('../input/covidpretrained/effdetd5_fold2/epoch50_polyak'))
    model0_fold2.cuda()
    model0_fold2.eval()

    model0_fold3 = get_effdetd5()
    model0_fold3.load_state_dict(torch.load('../input/covidpretrained/effdetd5_fold3/epoch50_polyak'))
    model0_fold3.cuda()
    model0_fold3.eval()

    model0_fold4 = get_effdetd5()
    model0_fold4.load_state_dict(torch.load('../input/covidpretrained/effdetd5_fold4/epoch50_polyak'))
    model0_fold4.cuda()
    model0_fold4.eval()
    
    model1_fold0 = get_effdetd6()
    model1_fold0.load_state_dict(torch.load('../input/covidpretrained/effdetd6_fold0/epoch44_polyak'))
    model1_fold0.cuda()
    model1_fold0.eval()

    model1_fold1 = get_effdetd6()
    model1_fold1.load_state_dict(torch.load('../input/covidpretrained/effdetd6_fold1/epoch44_polyak'))
    model1_fold1.cuda()
    model1_fold1.eval()

    model1_fold2 = get_effdetd6()
    model1_fold2.load_state_dict(torch.load('../input/covidpretrained/effdetd6_fold2/epoch44_polyak'))
    model1_fold2.cuda()
    model1_fold2.eval()

    model1_fold3 = get_effdetd6()
    model1_fold3.load_state_dict(torch.load('../input/covidpretrained/effdetd6_fold3/epoch44_polyak'))
    model1_fold3.cuda()
    model1_fold3.eval()

    model1_fold4 = get_effdetd6()
    model1_fold4.load_state_dict(torch.load('../input/covidpretrained/effdetd6_fold4/epoch44_polyak'))
    model1_fold4.cuda()
    model1_fold4.eval()
    
    device = select_device('0')
    modelyolo1_fold0 = attempt_load('../input/covidpretrained/yolo1_fold0/best.pt', map_location=device)
    modelyolo1_fold0.eval()
    modelyolo1_fold1 = attempt_load('../input/covidpretrained/yolo1_fold1/best.pt', map_location=device)
    modelyolo1_fold1.eval()
    modelyolo1_fold2 = attempt_load('../input/covidpretrained/yolo1_fold2/best.pt', map_location=device)
    modelyolo1_fold2.eval()
    modelyolo1_fold3 = attempt_load('../input/covidpretrained/yolo1_fold3/best.pt', map_location=device)
    modelyolo1_fold3.eval()
    modelyolo2_fold0 = attempt_load('../input/covidpretrained/yolo2_fold0/best.pt', map_location=device)
    modelyolo2_fold0.eval()
    modelyolo2_fold1 = attempt_load('../input/covidpretrained/yolo2_fold1/best.pt', map_location=device)
    modelyolo2_fold1.eval()
    modelyolo2_fold2 = attempt_load('../input/covidpretrained/yolo2_fold2/best.pt', map_location=device)
    modelyolo2_fold2.eval()
    modelyolo2_fold3 = attempt_load('../input/covidpretrained/yolo2_fold3/best.pt', map_location=device)
    modelyolo2_fold3.eval()

    # iterator for testing
    datagen = TestDataset(label_dict=label_dict, image_list=image_id_list)
    generator = DataLoader(dataset=datagen, 
                           collate_fn=collate_fn, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    detections_all = []
    for i, (images1024, images896, imagesyolo, targets) in enumerate(generator):
        with torch.no_grad():
            start = i*batch_size
            end = start+batch_size
            if i == len(generator)-1:
                end = len(generator.dataset)
            images1024 = images1024.cuda()
            images1024 = images1024.permute(0,3,1,2).float().div(255)
            images896 = images896.cuda()
            images896 = images896.permute(0,3,1,2).float().div(255)
            imagesyolo = imagesyolo.cuda()
            imagesyolo = imagesyolo.float().div(255)
            with autocast():
                detections_effdetd5_fold0 = model0_fold0(images1024).cpu().numpy()
                detections_effdetd5_fold1 = model0_fold1(images1024).cpu().numpy()
                detections_effdetd5_fold2 = model0_fold2(images1024).cpu().numpy()
                detections_effdetd5_fold3 = model0_fold3(images1024).cpu().numpy()
                detections_effdetd5_fold4 = model0_fold4(images1024).cpu().numpy()
                detections_flip_effdetd5_fold0 = model0_fold0(images1024.flip(3)).cpu().numpy()
                detections_flip_effdetd5_fold1 = model0_fold1(images1024.flip(3)).cpu().numpy()
                detections_flip_effdetd5_fold2 = model0_fold2(images1024.flip(3)).cpu().numpy()
                detections_flip_effdetd5_fold3 = model0_fold3(images1024.flip(3)).cpu().numpy()
                detections_flip_effdetd5_fold4 = model0_fold4(images1024.flip(3)).cpu().numpy()
                
                detections_effdetd6_fold0 = model1_fold0(images896).cpu().numpy()
                detections_effdetd6_fold1 = model1_fold1(images896).cpu().numpy()
                detections_effdetd6_fold2 = model1_fold2(images896).cpu().numpy()
                detections_effdetd6_fold3 = model1_fold3(images896).cpu().numpy()
                detections_effdetd6_fold4 = model1_fold4(images896).cpu().numpy()
                detections_flip_effdetd6_fold0 = model1_fold0(images896.flip(3)).cpu().numpy()
                detections_flip_effdetd6_fold1 = model1_fold1(images896.flip(3)).cpu().numpy()
                detections_flip_effdetd6_fold2 = model1_fold2(images896.flip(3)).cpu().numpy()
                detections_flip_effdetd6_fold3 = model1_fold3(images896.flip(3)).cpu().numpy()
                detections_flip_effdetd6_fold4 = model1_fold4(images896.flip(3)).cpu().numpy()
                
                detections_yolo1_fold0 = modelyolo1_fold0(imagesyolo, augment=False)[0]
                detections_yolo1_fold0 = non_max_suppression(detections_yolo1_fold0, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_hflip_yolo1_fold1 = modelyolo1_fold1(imagesyolo.flip(3), augment=False)[0]
                detections_hflip_yolo1_fold1 = non_max_suppression(detections_hflip_yolo1_fold1, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_vflip_yolo1_fold2 = modelyolo1_fold2(imagesyolo.flip(2), augment=False)[0]
                detections_vflip_yolo1_fold2 = non_max_suppression(detections_vflip_yolo1_fold2, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_hvflip_yolo1_fold3 = modelyolo1_fold3(imagesyolo.flip(3).flip(2), augment=False)[0]
                detections_hvflip_yolo1_fold3 = non_max_suppression(detections_hvflip_yolo1_fold3, conf_thres=0.001, iou_thres=0.5, multi_label=True)

                detections_yolo2_fold0 = modelyolo2_fold0(imagesyolo, augment=False)[0]
                detections_yolo2_fold0 = non_max_suppression(detections_yolo2_fold0, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_hflip_yolo2_fold1 = modelyolo2_fold1(imagesyolo.flip(3), augment=False)[0]
                detections_hflip_yolo2_fold1 = non_max_suppression(detections_hflip_yolo2_fold1, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_vflip_yolo2_fold2 = modelyolo2_fold2(imagesyolo.flip(2), augment=False)[0]
                detections_vflip_yolo2_fold2 = non_max_suppression(detections_vflip_yolo2_fold2, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_hvflip_yolo2_fold3 = modelyolo2_fold3(imagesyolo.flip(3).flip(2), augment=False)[0]
                detections_hvflip_yolo2_fold3 = non_max_suppression(detections_hvflip_yolo2_fold3, conf_thres=0.001, iou_thres=0.5, multi_label=True)

            detections_flip_effdetd5_fold0[:,:,[0,2]] = 1024 - detections_flip_effdetd5_fold0[:,:,[2,0]]
            detections_flip_effdetd5_fold1[:,:,[0,2]] = 1024 - detections_flip_effdetd5_fold1[:,:,[2,0]]
            detections_flip_effdetd5_fold2[:,:,[0,2]] = 1024 - detections_flip_effdetd5_fold2[:,:,[2,0]]
            detections_flip_effdetd5_fold3[:,:,[0,2]] = 1024 - detections_flip_effdetd5_fold3[:,:,[2,0]]
            detections_flip_effdetd5_fold4[:,:,[0,2]] = 1024 - detections_flip_effdetd5_fold4[:,:,[2,0]]
            
            detections_flip_effdetd6_fold0[:,:,[0,2]] = 896 - detections_flip_effdetd6_fold0[:,:,[2,0]]
            detections_flip_effdetd6_fold1[:,:,[0,2]] = 896 - detections_flip_effdetd6_fold1[:,:,[2,0]]
            detections_flip_effdetd6_fold2[:,:,[0,2]] = 896 - detections_flip_effdetd6_fold2[:,:,[2,0]]
            detections_flip_effdetd6_fold3[:,:,[0,2]] = 896 - detections_flip_effdetd6_fold3[:,:,[2,0]]
            detections_flip_effdetd6_fold4[:,:,[0,2]] = 896 - detections_flip_effdetd6_fold4[:,:,[2,0]]

            detections_effdetd5_fold0[:,:,0] /= 1024
            detections_effdetd5_fold0[:,:,2] /= 1024
            detections_effdetd5_fold0[:,:,1] /= 1024
            detections_effdetd5_fold0[:,:,3] /= 1024
            detections_flip_effdetd5_fold0[:,:,0] /= 1024
            detections_flip_effdetd5_fold0[:,:,2] /= 1024
            detections_flip_effdetd5_fold0[:,:,1] /= 1024
            detections_flip_effdetd5_fold0[:,:,3] /= 1024
            detections_effdetd5_fold1[:,:,0] /= 1024
            detections_effdetd5_fold1[:,:,2] /= 1024
            detections_effdetd5_fold1[:,:,1] /= 1024
            detections_effdetd5_fold1[:,:,3] /= 1024
            detections_flip_effdetd5_fold1[:,:,0] /= 1024
            detections_flip_effdetd5_fold1[:,:,2] /= 1024
            detections_flip_effdetd5_fold1[:,:,1] /= 1024
            detections_flip_effdetd5_fold1[:,:,3] /= 1024
            detections_effdetd5_fold2[:,:,0] /= 1024
            detections_effdetd5_fold2[:,:,2] /= 1024
            detections_effdetd5_fold2[:,:,1] /= 1024
            detections_effdetd5_fold2[:,:,3] /= 1024
            detections_flip_effdetd5_fold2[:,:,0] /= 1024
            detections_flip_effdetd5_fold2[:,:,2] /= 1024
            detections_flip_effdetd5_fold2[:,:,1] /= 1024
            detections_flip_effdetd5_fold2[:,:,3] /= 1024
            detections_effdetd5_fold3[:,:,0] /= 1024
            detections_effdetd5_fold3[:,:,2] /= 1024
            detections_effdetd5_fold3[:,:,1] /= 1024
            detections_effdetd5_fold3[:,:,3] /= 1024
            detections_flip_effdetd5_fold3[:,:,0] /= 1024
            detections_flip_effdetd5_fold3[:,:,2] /= 1024
            detections_flip_effdetd5_fold3[:,:,1] /= 1024
            detections_flip_effdetd5_fold3[:,:,3] /= 1024
            detections_effdetd5_fold4[:,:,0] /= 1024
            detections_effdetd5_fold4[:,:,2] /= 1024
            detections_effdetd5_fold4[:,:,1] /= 1024
            detections_effdetd5_fold4[:,:,3] /= 1024
            detections_flip_effdetd5_fold4[:,:,0] /= 1024
            detections_flip_effdetd5_fold4[:,:,2] /= 1024
            detections_flip_effdetd5_fold4[:,:,1] /= 1024
            detections_flip_effdetd5_fold4[:,:,3] /= 1024
            
            detections_effdetd6_fold0[:,:,0] /= 896
            detections_effdetd6_fold0[:,:,2] /= 896
            detections_effdetd6_fold0[:,:,1] /= 896
            detections_effdetd6_fold0[:,:,3] /= 896
            detections_flip_effdetd6_fold0[:,:,0] /= 896
            detections_flip_effdetd6_fold0[:,:,2] /= 896
            detections_flip_effdetd6_fold0[:,:,1] /= 896
            detections_flip_effdetd6_fold0[:,:,3] /= 896
            detections_effdetd6_fold1[:,:,0] /= 896
            detections_effdetd6_fold1[:,:,2] /= 896
            detections_effdetd6_fold1[:,:,1] /= 896
            detections_effdetd6_fold1[:,:,3] /= 896
            detections_flip_effdetd6_fold1[:,:,0] /= 896
            detections_flip_effdetd6_fold1[:,:,2] /= 896
            detections_flip_effdetd6_fold1[:,:,1] /= 896
            detections_flip_effdetd6_fold1[:,:,3] /= 896
            detections_effdetd6_fold2[:,:,0] /= 896
            detections_effdetd6_fold2[:,:,2] /= 896
            detections_effdetd6_fold2[:,:,1] /= 896
            detections_effdetd6_fold2[:,:,3] /= 896
            detections_flip_effdetd6_fold2[:,:,0] /= 896
            detections_flip_effdetd6_fold2[:,:,2] /= 896
            detections_flip_effdetd6_fold2[:,:,1] /= 896
            detections_flip_effdetd6_fold2[:,:,3] /= 896
            detections_effdetd6_fold3[:,:,0] /= 896
            detections_effdetd6_fold3[:,:,2] /= 896
            detections_effdetd6_fold3[:,:,1] /= 896
            detections_effdetd6_fold3[:,:,3] /= 896
            detections_flip_effdetd6_fold3[:,:,0] /= 896
            detections_flip_effdetd6_fold3[:,:,2] /= 896
            detections_flip_effdetd6_fold3[:,:,1] /= 896
            detections_flip_effdetd6_fold3[:,:,3] /= 896
            detections_effdetd6_fold4[:,:,0] /= 896
            detections_effdetd6_fold4[:,:,2] /= 896
            detections_effdetd6_fold4[:,:,1] /= 896
            detections_effdetd6_fold4[:,:,3] /= 896
            detections_flip_effdetd6_fold4[:,:,0] /= 896
            detections_flip_effdetd6_fold4[:,:,2] /= 896
            detections_flip_effdetd6_fold4[:,:,1] /= 896
            detections_flip_effdetd6_fold4[:,:,3] /= 896
            
            for n in range(len(detections_yolo1_fold0)):
                detections_yolo1_fold0[n] = detections_yolo1_fold0[n].cpu().numpy()
                detections_yolo1_fold0[n][:,0] /= 1024
                detections_yolo1_fold0[n][:,2] /= 1024
                detections_yolo1_fold0[n][:,1] /= 1024
                detections_yolo1_fold0[n][:,3] /= 1024
                detections_yolo1_fold0[n][:,5] += 1  # effdet labels start from 1

                detections_hflip_yolo1_fold1[n] = detections_hflip_yolo1_fold1[n].cpu().numpy()
                detections_hflip_yolo1_fold1[n][:,[0,2]] = 1024 - detections_hflip_yolo1_fold1[n][:,[2,0]]
                detections_hflip_yolo1_fold1[n][:,0] /= 1024
                detections_hflip_yolo1_fold1[n][:,2] /= 1024
                detections_hflip_yolo1_fold1[n][:,1] /= 1024
                detections_hflip_yolo1_fold1[n][:,3] /= 1024
                detections_hflip_yolo1_fold1[n][:,5] += 1  # effdet labels start from 1

                detections_vflip_yolo1_fold2[n] = detections_vflip_yolo1_fold2[n].cpu().numpy()
                detections_vflip_yolo1_fold2[n][:,[1,3]] = 1024 - detections_vflip_yolo1_fold2[n][:,[3,1]]
                detections_vflip_yolo1_fold2[n][:,0] /= 1024
                detections_vflip_yolo1_fold2[n][:,2] /= 1024
                detections_vflip_yolo1_fold2[n][:,1] /= 1024
                detections_vflip_yolo1_fold2[n][:,3] /= 1024
                detections_vflip_yolo1_fold2[n][:,5] += 1  # effdet labels start from 1

                detections_hvflip_yolo1_fold3[n] = detections_hvflip_yolo1_fold3[n].cpu().numpy()
                detections_hvflip_yolo1_fold3[n][:,[0,2]] = 1024 - detections_hvflip_yolo1_fold3[n][:,[2,0]]
                detections_hvflip_yolo1_fold3[n][:,[1,3]] = 1024 - detections_hvflip_yolo1_fold3[n][:,[3,1]]
                detections_hvflip_yolo1_fold3[n][:,0] /= 1024
                detections_hvflip_yolo1_fold3[n][:,2] /= 1024
                detections_hvflip_yolo1_fold3[n][:,1] /= 1024
                detections_hvflip_yolo1_fold3[n][:,3] /= 1024
                detections_hvflip_yolo1_fold3[n][:,5] += 1  # effdet labels start from 1

                detections_yolo2_fold0[n] = detections_yolo2_fold0[n].cpu().numpy()
                detections_yolo2_fold0[n][:,0] /= 1024
                detections_yolo2_fold0[n][:,2] /= 1024
                detections_yolo2_fold0[n][:,1] /= 1024
                detections_yolo2_fold0[n][:,3] /= 1024
                detections_yolo2_fold0[n][:,5] += 1  # effdet labels start from 1

                detections_hflip_yolo2_fold1[n] = detections_hflip_yolo2_fold1[n].cpu().numpy()
                detections_hflip_yolo2_fold1[n][:,[0,2]] = 1024 - detections_hflip_yolo2_fold1[n][:,[2,0]]
                detections_hflip_yolo2_fold1[n][:,0] /= 1024
                detections_hflip_yolo2_fold1[n][:,2] /= 1024
                detections_hflip_yolo2_fold1[n][:,1] /= 1024
                detections_hflip_yolo2_fold1[n][:,3] /= 1024
                detections_hflip_yolo2_fold1[n][:,5] += 1  # effdet labels start from 1

                detections_vflip_yolo2_fold2[n] = detections_vflip_yolo2_fold2[n].cpu().numpy()
                detections_vflip_yolo2_fold2[n][:,[1,3]] = 1024 - detections_vflip_yolo2_fold2[n][:,[3,1]]
                detections_vflip_yolo2_fold2[n][:,0] /= 1024
                detections_vflip_yolo2_fold2[n][:,2] /= 1024
                detections_vflip_yolo2_fold2[n][:,1] /= 1024
                detections_vflip_yolo2_fold2[n][:,3] /= 1024
                detections_vflip_yolo2_fold2[n][:,5] += 1  # effdet labels start from 1

                detections_hvflip_yolo2_fold3[n] = detections_hvflip_yolo2_fold3[n].cpu().numpy()
                detections_hvflip_yolo2_fold3[n][:,[0,2]] = 1024 - detections_hvflip_yolo2_fold3[n][:,[2,0]]
                detections_hvflip_yolo2_fold3[n][:,[1,3]] = 1024 - detections_hvflip_yolo2_fold3[n][:,[3,1]]
                detections_hvflip_yolo2_fold3[n][:,0] /= 1024
                detections_hvflip_yolo2_fold3[n][:,2] /= 1024
                detections_hvflip_yolo2_fold3[n][:,1] /= 1024
                detections_hvflip_yolo2_fold3[n][:,3] /= 1024
                detections_hvflip_yolo2_fold3[n][:,5] += 1  # effdet labels start from 1

            detections_ensemble = np.zeros(detections_effdetd5_fold0.shape, dtype=np.float32)
            for n in range(detections_effdetd5_fold0.shape[0]):
                boxes = [
                         detections_effdetd5_fold0[n,:,:4].tolist(),
                         detections_effdetd5_fold1[n,:,:4].tolist(),
                         detections_effdetd5_fold2[n,:,:4].tolist(),
                         detections_effdetd5_fold3[n,:,:4].tolist(),
                         detections_effdetd5_fold4[n,:,:4].tolist(),
                         detections_flip_effdetd5_fold0[n,:,:4].tolist(),
                         detections_flip_effdetd5_fold1[n,:,:4].tolist(),
                         detections_flip_effdetd5_fold2[n,:,:4].tolist(),
                         detections_flip_effdetd5_fold3[n,:,:4].tolist(),
                         detections_flip_effdetd5_fold4[n,:,:4].tolist(),
                         detections_effdetd6_fold0[n,:,:4].tolist(),
                         detections_effdetd6_fold1[n,:,:4].tolist(),
                         detections_effdetd6_fold2[n,:,:4].tolist(),
                         detections_effdetd6_fold3[n,:,:4].tolist(),
                         detections_effdetd6_fold4[n,:,:4].tolist(),
                         detections_flip_effdetd6_fold0[n,:,:4].tolist(),
                         detections_flip_effdetd6_fold1[n,:,:4].tolist(),
                         detections_flip_effdetd6_fold2[n,:,:4].tolist(),
                         detections_flip_effdetd6_fold3[n,:,:4].tolist(),
                         detections_flip_effdetd6_fold4[n,:,:4].tolist(),
                         detections_yolo1_fold0[n][:,:4].tolist(),  
                         detections_hflip_yolo1_fold1[n][:,:4].tolist(), 
                         detections_vflip_yolo1_fold2[n][:,:4].tolist(), 
                         detections_hvflip_yolo1_fold3[n][:,:4].tolist(),
                         detections_yolo2_fold0[n][:,:4].tolist(), 
                         detections_hflip_yolo2_fold1[n][:,:4].tolist(), 
                         detections_vflip_yolo2_fold2[n][:,:4].tolist(),  
                         detections_hvflip_yolo2_fold3[n][:,:4].tolist(),
                        ]
                scores = [
                          detections_effdetd5_fold0[n,:,4].tolist(),
                          detections_effdetd5_fold1[n,:,4].tolist(),
                          detections_effdetd5_fold2[n,:,4].tolist(),
                          detections_effdetd5_fold3[n,:,4].tolist(),
                          detections_effdetd5_fold4[n,:,4].tolist(),
                          detections_flip_effdetd5_fold0[n,:,4].tolist(),
                          detections_flip_effdetd5_fold1[n,:,4].tolist(),
                          detections_flip_effdetd5_fold2[n,:,4].tolist(),
                          detections_flip_effdetd5_fold3[n,:,4].tolist(),
                          detections_flip_effdetd5_fold4[n,:,4].tolist(),
                          detections_effdetd6_fold0[n,:,4].tolist(),
                          detections_effdetd6_fold1[n,:,4].tolist(),
                          detections_effdetd6_fold2[n,:,4].tolist(),
                          detections_effdetd6_fold3[n,:,4].tolist(),
                          detections_effdetd6_fold4[n,:,4].tolist(),
                          detections_flip_effdetd6_fold0[n,:,4].tolist(),
                          detections_flip_effdetd6_fold1[n,:,4].tolist(),
                          detections_flip_effdetd6_fold2[n,:,4].tolist(),
                          detections_flip_effdetd6_fold3[n,:,4].tolist(),
                          detections_flip_effdetd6_fold4[n,:,4].tolist(),
                          detections_yolo1_fold0[n][:,4].tolist(),  
                          detections_hflip_yolo1_fold1[n][:,4].tolist(), 
                          detections_vflip_yolo1_fold2[n][:,4].tolist(), 
                          detections_hvflip_yolo1_fold3[n][:,4].tolist(),
                          detections_yolo2_fold0[n][:,4].tolist(), 
                          detections_hflip_yolo2_fold1[n][:,4].tolist(), 
                          detections_vflip_yolo2_fold2[n][:,4].tolist(),  
                          detections_hvflip_yolo2_fold3[n][:,4].tolist(),
                         ]
                labels = [
                          detections_effdetd5_fold0[n,:,5].tolist(),
                          detections_effdetd5_fold1[n,:,5].tolist(),
                          detections_effdetd5_fold2[n,:,5].tolist(),
                          detections_effdetd5_fold3[n,:,5].tolist(),
                          detections_effdetd5_fold4[n,:,5].tolist(),
                          detections_flip_effdetd5_fold0[n,:,5].tolist(),
                          detections_flip_effdetd5_fold1[n,:,5].tolist(),
                          detections_flip_effdetd5_fold2[n,:,5].tolist(),
                          detections_flip_effdetd5_fold3[n,:,5].tolist(),
                          detections_flip_effdetd5_fold4[n,:,5].tolist(),
                          detections_effdetd6_fold0[n,:,5].tolist(),
                          detections_effdetd6_fold1[n,:,5].tolist(),
                          detections_effdetd6_fold2[n,:,5].tolist(),
                          detections_effdetd6_fold3[n,:,5].tolist(),
                          detections_effdetd6_fold4[n,:,5].tolist(),
                          detections_flip_effdetd6_fold0[n,:,5].tolist(),
                          detections_flip_effdetd6_fold1[n,:,5].tolist(),
                          detections_flip_effdetd6_fold2[n,:,5].tolist(),
                          detections_flip_effdetd6_fold3[n,:,5].tolist(),
                          detections_flip_effdetd6_fold4[n,:,5].tolist(),
                          detections_yolo1_fold0[n][:,5].tolist(),  
                          detections_hflip_yolo1_fold1[n][:,5].tolist(), 
                          detections_vflip_yolo1_fold2[n][:,5].tolist(), 
                          detections_hvflip_yolo1_fold3[n][:,5].tolist(),
                          detections_yolo2_fold0[n][:,5].tolist(), 
                          detections_hflip_yolo2_fold1[n][:,5].tolist(), 
                          detections_vflip_yolo2_fold2[n][:,5].tolist(),  
                          detections_hvflip_yolo2_fold3[n][:,5].tolist(),
                         ]
                boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.6)

                boxes[:,0] *= targets['w'][n]
                boxes[:,2] *= targets['w'][n]
                boxes[:,1] *= targets['h'][n]
                boxes[:,3] *= targets['h'][n]

                if len(boxes)>=99:
                    detections_ensemble[n,:99,:4] = boxes[:99,:]
                    detections_ensemble[n,:99,4] = scores[:99]
                    detections_ensemble[n,:99,5] = labels[:99]
                else:
                    detections_ensemble[n,:len(boxes),:4] = boxes
                    detections_ensemble[n,:len(boxes),4] = scores
                    detections_ensemble[n,:len(boxes),5] = labels

                # Estimate the none class using the topK (K=3) opacity probs.
                detections_ensemble[n,-1,:4] = np.array([0,0,1,1], dtype=np.float32)
                non_prob = 1.0
                count = 0
                for bb in range(detections_ensemble.shape[1]):
                    if detections_ensemble[n,bb,5]==1.0:
                        non_prob *= (1.0-detections_ensemble[n,bb,4])
                        count += 1
                        if count>=3:
                            break
                detections_ensemble[n,-1,4] = non_prob
                detections_ensemble[n,-1,5] = 2

                detections_all.append(detections_ensemble[n])
                    
    # generate submission
    image_level_dict = {}
    for image_id in image_id_list:
        image_level_dict[image_id] = {"opacity": [], "none": 0., "negative": 0., "typical": 0., "indeterminate": 0., "atypical": 0.}

    for i in range(len(image_id_list)):
        image_detections = detections_all[i]
        for j in range(len(image_detections)):
            if image_detections[j,5]==1.0:
                image_level_dict[image_id_list[i]]["opacity"].append([image_detections[j,4]]+list(image_detections[j,:4]))
        image_level_dict[image_id_list[i]]["none"] += image_detections[-1,4]
        for j in range(len(image_detections)):
            if image_detections[j,5]==3.0:
                image_level_dict[image_id_list[i]]["negative"] += image_detections[j,4]
                break
        for j in range(len(image_detections)):
            if image_detections[j,5]==4.0:
                image_level_dict[image_id_list[i]]["typical"] += image_detections[j,4]
                break
        for j in range(len(image_detections)):
            if image_detections[j,5]==5.0:
                image_level_dict[image_id_list[i]]["indeterminate"] += image_detections[j,4]
                break
        for j in range(len(image_detections)):
            if image_detections[j,5]==6.0:
                image_level_dict[image_id_list[i]]["atypical"] += image_detections[j,4]
                break

    # reprediction
    # iterator for testing
    datagen = TestDataset1(label_dict=midrc_dict, image_list=image_id_list_matched_midrc)
    generator = DataLoader(dataset=datagen, 
                           collate_fn=collate_fn, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    detections_all = []
    for i, (images1024, images896, imagesyolo, targets) in enumerate(generator):
        with torch.no_grad():
            start = i*batch_size
            end = start+batch_size
            if i == len(generator)-1:
                end = len(generator.dataset)
            images1024 = images1024.cuda()
            images1024 = images1024.permute(0,3,1,2).float().div(255)
            images896 = images896.cuda()
            images896 = images896.permute(0,3,1,2).float().div(255)
            imagesyolo = imagesyolo.cuda()
            imagesyolo = imagesyolo.float().div(255)
            with autocast():
                detections_effdetd5_fold0 = model0_fold0(images1024).cpu().numpy()
                detections_effdetd5_fold1 = model0_fold1(images1024).cpu().numpy()
                detections_effdetd5_fold2 = model0_fold2(images1024).cpu().numpy()
                detections_effdetd5_fold3 = model0_fold3(images1024).cpu().numpy()
                detections_effdetd5_fold4 = model0_fold4(images1024).cpu().numpy()
                detections_flip_effdetd5_fold0 = model0_fold0(images1024.flip(3)).cpu().numpy()
                detections_flip_effdetd5_fold1 = model0_fold1(images1024.flip(3)).cpu().numpy()
                detections_flip_effdetd5_fold2 = model0_fold2(images1024.flip(3)).cpu().numpy()
                detections_flip_effdetd5_fold3 = model0_fold3(images1024.flip(3)).cpu().numpy()
                detections_flip_effdetd5_fold4 = model0_fold4(images1024.flip(3)).cpu().numpy()
                
                detections_effdetd6_fold0 = model1_fold0(images896).cpu().numpy()
                detections_effdetd6_fold1 = model1_fold1(images896).cpu().numpy()
                detections_effdetd6_fold2 = model1_fold2(images896).cpu().numpy()
                detections_effdetd6_fold3 = model1_fold3(images896).cpu().numpy()
                detections_effdetd6_fold4 = model1_fold4(images896).cpu().numpy()
                detections_flip_effdetd6_fold0 = model1_fold0(images896.flip(3)).cpu().numpy()
                detections_flip_effdetd6_fold1 = model1_fold1(images896.flip(3)).cpu().numpy()
                detections_flip_effdetd6_fold2 = model1_fold2(images896.flip(3)).cpu().numpy()
                detections_flip_effdetd6_fold3 = model1_fold3(images896.flip(3)).cpu().numpy()
                detections_flip_effdetd6_fold4 = model1_fold4(images896.flip(3)).cpu().numpy()
                
                detections_yolo1_fold0 = modelyolo1_fold0(imagesyolo, augment=False)[0]
                detections_yolo1_fold0 = non_max_suppression(detections_yolo1_fold0, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_hflip_yolo1_fold1 = modelyolo1_fold1(imagesyolo.flip(3), augment=False)[0]
                detections_hflip_yolo1_fold1 = non_max_suppression(detections_hflip_yolo1_fold1, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_vflip_yolo1_fold2 = modelyolo1_fold2(imagesyolo.flip(2), augment=False)[0]
                detections_vflip_yolo1_fold2 = non_max_suppression(detections_vflip_yolo1_fold2, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_hvflip_yolo1_fold3 = modelyolo1_fold3(imagesyolo.flip(3).flip(2), augment=False)[0]
                detections_hvflip_yolo1_fold3 = non_max_suppression(detections_hvflip_yolo1_fold3, conf_thres=0.001, iou_thres=0.5, multi_label=True)

                detections_yolo2_fold0 = modelyolo2_fold0(imagesyolo, augment=False)[0]
                detections_yolo2_fold0 = non_max_suppression(detections_yolo2_fold0, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_hflip_yolo2_fold1 = modelyolo2_fold1(imagesyolo.flip(3), augment=False)[0]
                detections_hflip_yolo2_fold1 = non_max_suppression(detections_hflip_yolo2_fold1, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_vflip_yolo2_fold2 = modelyolo2_fold2(imagesyolo.flip(2), augment=False)[0]
                detections_vflip_yolo2_fold2 = non_max_suppression(detections_vflip_yolo2_fold2, conf_thres=0.001, iou_thres=0.5, multi_label=True)
                detections_hvflip_yolo2_fold3 = modelyolo2_fold3(imagesyolo.flip(3).flip(2), augment=False)[0]
                detections_hvflip_yolo2_fold3 = non_max_suppression(detections_hvflip_yolo2_fold3, conf_thres=0.001, iou_thres=0.5, multi_label=True)

            detections_flip_effdetd5_fold0[:,:,[0,2]] = 1024 - detections_flip_effdetd5_fold0[:,:,[2,0]]
            detections_flip_effdetd5_fold1[:,:,[0,2]] = 1024 - detections_flip_effdetd5_fold1[:,:,[2,0]]
            detections_flip_effdetd5_fold2[:,:,[0,2]] = 1024 - detections_flip_effdetd5_fold2[:,:,[2,0]]
            detections_flip_effdetd5_fold3[:,:,[0,2]] = 1024 - detections_flip_effdetd5_fold3[:,:,[2,0]]
            detections_flip_effdetd5_fold4[:,:,[0,2]] = 1024 - detections_flip_effdetd5_fold4[:,:,[2,0]]
            
            detections_flip_effdetd6_fold0[:,:,[0,2]] = 896 - detections_flip_effdetd6_fold0[:,:,[2,0]]
            detections_flip_effdetd6_fold1[:,:,[0,2]] = 896 - detections_flip_effdetd6_fold1[:,:,[2,0]]
            detections_flip_effdetd6_fold2[:,:,[0,2]] = 896 - detections_flip_effdetd6_fold2[:,:,[2,0]]
            detections_flip_effdetd6_fold3[:,:,[0,2]] = 896 - detections_flip_effdetd6_fold3[:,:,[2,0]]
            detections_flip_effdetd6_fold4[:,:,[0,2]] = 896 - detections_flip_effdetd6_fold4[:,:,[2,0]]

            detections_effdetd5_fold0[:,:,0] /= 1024
            detections_effdetd5_fold0[:,:,2] /= 1024
            detections_effdetd5_fold0[:,:,1] /= 1024
            detections_effdetd5_fold0[:,:,3] /= 1024
            detections_flip_effdetd5_fold0[:,:,0] /= 1024
            detections_flip_effdetd5_fold0[:,:,2] /= 1024
            detections_flip_effdetd5_fold0[:,:,1] /= 1024
            detections_flip_effdetd5_fold0[:,:,3] /= 1024
            detections_effdetd5_fold1[:,:,0] /= 1024
            detections_effdetd5_fold1[:,:,2] /= 1024
            detections_effdetd5_fold1[:,:,1] /= 1024
            detections_effdetd5_fold1[:,:,3] /= 1024
            detections_flip_effdetd5_fold1[:,:,0] /= 1024
            detections_flip_effdetd5_fold1[:,:,2] /= 1024
            detections_flip_effdetd5_fold1[:,:,1] /= 1024
            detections_flip_effdetd5_fold1[:,:,3] /= 1024
            detections_effdetd5_fold2[:,:,0] /= 1024
            detections_effdetd5_fold2[:,:,2] /= 1024
            detections_effdetd5_fold2[:,:,1] /= 1024
            detections_effdetd5_fold2[:,:,3] /= 1024
            detections_flip_effdetd5_fold2[:,:,0] /= 1024
            detections_flip_effdetd5_fold2[:,:,2] /= 1024
            detections_flip_effdetd5_fold2[:,:,1] /= 1024
            detections_flip_effdetd5_fold2[:,:,3] /= 1024
            detections_effdetd5_fold3[:,:,0] /= 1024
            detections_effdetd5_fold3[:,:,2] /= 1024
            detections_effdetd5_fold3[:,:,1] /= 1024
            detections_effdetd5_fold3[:,:,3] /= 1024
            detections_flip_effdetd5_fold3[:,:,0] /= 1024
            detections_flip_effdetd5_fold3[:,:,2] /= 1024
            detections_flip_effdetd5_fold3[:,:,1] /= 1024
            detections_flip_effdetd5_fold3[:,:,3] /= 1024
            detections_effdetd5_fold4[:,:,0] /= 1024
            detections_effdetd5_fold4[:,:,2] /= 1024
            detections_effdetd5_fold4[:,:,1] /= 1024
            detections_effdetd5_fold4[:,:,3] /= 1024
            detections_flip_effdetd5_fold4[:,:,0] /= 1024
            detections_flip_effdetd5_fold4[:,:,2] /= 1024
            detections_flip_effdetd5_fold4[:,:,1] /= 1024
            detections_flip_effdetd5_fold4[:,:,3] /= 1024
            
            detections_effdetd6_fold0[:,:,0] /= 896
            detections_effdetd6_fold0[:,:,2] /= 896
            detections_effdetd6_fold0[:,:,1] /= 896
            detections_effdetd6_fold0[:,:,3] /= 896
            detections_flip_effdetd6_fold0[:,:,0] /= 896
            detections_flip_effdetd6_fold0[:,:,2] /= 896
            detections_flip_effdetd6_fold0[:,:,1] /= 896
            detections_flip_effdetd6_fold0[:,:,3] /= 896
            detections_effdetd6_fold1[:,:,0] /= 896
            detections_effdetd6_fold1[:,:,2] /= 896
            detections_effdetd6_fold1[:,:,1] /= 896
            detections_effdetd6_fold1[:,:,3] /= 896
            detections_flip_effdetd6_fold1[:,:,0] /= 896
            detections_flip_effdetd6_fold1[:,:,2] /= 896
            detections_flip_effdetd6_fold1[:,:,1] /= 896
            detections_flip_effdetd6_fold1[:,:,3] /= 896
            detections_effdetd6_fold2[:,:,0] /= 896
            detections_effdetd6_fold2[:,:,2] /= 896
            detections_effdetd6_fold2[:,:,1] /= 896
            detections_effdetd6_fold2[:,:,3] /= 896
            detections_flip_effdetd6_fold2[:,:,0] /= 896
            detections_flip_effdetd6_fold2[:,:,2] /= 896
            detections_flip_effdetd6_fold2[:,:,1] /= 896
            detections_flip_effdetd6_fold2[:,:,3] /= 896
            detections_effdetd6_fold3[:,:,0] /= 896
            detections_effdetd6_fold3[:,:,2] /= 896
            detections_effdetd6_fold3[:,:,1] /= 896
            detections_effdetd6_fold3[:,:,3] /= 896
            detections_flip_effdetd6_fold3[:,:,0] /= 896
            detections_flip_effdetd6_fold3[:,:,2] /= 896
            detections_flip_effdetd6_fold3[:,:,1] /= 896
            detections_flip_effdetd6_fold3[:,:,3] /= 896
            detections_effdetd6_fold4[:,:,0] /= 896
            detections_effdetd6_fold4[:,:,2] /= 896
            detections_effdetd6_fold4[:,:,1] /= 896
            detections_effdetd6_fold4[:,:,3] /= 896
            detections_flip_effdetd6_fold4[:,:,0] /= 896
            detections_flip_effdetd6_fold4[:,:,2] /= 896
            detections_flip_effdetd6_fold4[:,:,1] /= 896
            detections_flip_effdetd6_fold4[:,:,3] /= 896
            
            for n in range(len(detections_yolo1_fold0)):
                detections_yolo1_fold0[n] = detections_yolo1_fold0[n].cpu().numpy()
                detections_yolo1_fold0[n][:,0] /= 1024
                detections_yolo1_fold0[n][:,2] /= 1024
                detections_yolo1_fold0[n][:,1] /= 1024
                detections_yolo1_fold0[n][:,3] /= 1024
                detections_yolo1_fold0[n][:,5] += 1  # effdet labels start from 1

                detections_hflip_yolo1_fold1[n] = detections_hflip_yolo1_fold1[n].cpu().numpy()
                detections_hflip_yolo1_fold1[n][:,[0,2]] = 1024 - detections_hflip_yolo1_fold1[n][:,[2,0]]
                detections_hflip_yolo1_fold1[n][:,0] /= 1024
                detections_hflip_yolo1_fold1[n][:,2] /= 1024
                detections_hflip_yolo1_fold1[n][:,1] /= 1024
                detections_hflip_yolo1_fold1[n][:,3] /= 1024
                detections_hflip_yolo1_fold1[n][:,5] += 1  # effdet labels start from 1

                detections_vflip_yolo1_fold2[n] = detections_vflip_yolo1_fold2[n].cpu().numpy()
                detections_vflip_yolo1_fold2[n][:,[1,3]] = 1024 - detections_vflip_yolo1_fold2[n][:,[3,1]]
                detections_vflip_yolo1_fold2[n][:,0] /= 1024
                detections_vflip_yolo1_fold2[n][:,2] /= 1024
                detections_vflip_yolo1_fold2[n][:,1] /= 1024
                detections_vflip_yolo1_fold2[n][:,3] /= 1024
                detections_vflip_yolo1_fold2[n][:,5] += 1  # effdet labels start from 1

                detections_hvflip_yolo1_fold3[n] = detections_hvflip_yolo1_fold3[n].cpu().numpy()
                detections_hvflip_yolo1_fold3[n][:,[0,2]] = 1024 - detections_hvflip_yolo1_fold3[n][:,[2,0]]
                detections_hvflip_yolo1_fold3[n][:,[1,3]] = 1024 - detections_hvflip_yolo1_fold3[n][:,[3,1]]
                detections_hvflip_yolo1_fold3[n][:,0] /= 1024
                detections_hvflip_yolo1_fold3[n][:,2] /= 1024
                detections_hvflip_yolo1_fold3[n][:,1] /= 1024
                detections_hvflip_yolo1_fold3[n][:,3] /= 1024
                detections_hvflip_yolo1_fold3[n][:,5] += 1  # effdet labels start from 1

                detections_yolo2_fold0[n] = detections_yolo2_fold0[n].cpu().numpy()
                detections_yolo2_fold0[n][:,0] /= 1024
                detections_yolo2_fold0[n][:,2] /= 1024
                detections_yolo2_fold0[n][:,1] /= 1024
                detections_yolo2_fold0[n][:,3] /= 1024
                detections_yolo2_fold0[n][:,5] += 1  # effdet labels start from 1

                detections_hflip_yolo2_fold1[n] = detections_hflip_yolo2_fold1[n].cpu().numpy()
                detections_hflip_yolo2_fold1[n][:,[0,2]] = 1024 - detections_hflip_yolo2_fold1[n][:,[2,0]]
                detections_hflip_yolo2_fold1[n][:,0] /= 1024
                detections_hflip_yolo2_fold1[n][:,2] /= 1024
                detections_hflip_yolo2_fold1[n][:,1] /= 1024
                detections_hflip_yolo2_fold1[n][:,3] /= 1024
                detections_hflip_yolo2_fold1[n][:,5] += 1  # effdet labels start from 1

                detections_vflip_yolo2_fold2[n] = detections_vflip_yolo2_fold2[n].cpu().numpy()
                detections_vflip_yolo2_fold2[n][:,[1,3]] = 1024 - detections_vflip_yolo2_fold2[n][:,[3,1]]
                detections_vflip_yolo2_fold2[n][:,0] /= 1024
                detections_vflip_yolo2_fold2[n][:,2] /= 1024
                detections_vflip_yolo2_fold2[n][:,1] /= 1024
                detections_vflip_yolo2_fold2[n][:,3] /= 1024
                detections_vflip_yolo2_fold2[n][:,5] += 1  # effdet labels start from 1

                detections_hvflip_yolo2_fold3[n] = detections_hvflip_yolo2_fold3[n].cpu().numpy()
                detections_hvflip_yolo2_fold3[n][:,[0,2]] = 1024 - detections_hvflip_yolo2_fold3[n][:,[2,0]]
                detections_hvflip_yolo2_fold3[n][:,[1,3]] = 1024 - detections_hvflip_yolo2_fold3[n][:,[3,1]]
                detections_hvflip_yolo2_fold3[n][:,0] /= 1024
                detections_hvflip_yolo2_fold3[n][:,2] /= 1024
                detections_hvflip_yolo2_fold3[n][:,1] /= 1024
                detections_hvflip_yolo2_fold3[n][:,3] /= 1024
                detections_hvflip_yolo2_fold3[n][:,5] += 1  # effdet labels start from 1

            detections_ensemble = np.zeros(detections_effdetd5_fold0.shape, dtype=np.float32)
            for n in range(detections_effdetd5_fold0.shape[0]):
                boxes = [
                         detections_effdetd5_fold0[n,:,:4].tolist(),
                         detections_effdetd5_fold1[n,:,:4].tolist(),
                         detections_effdetd5_fold2[n,:,:4].tolist(),
                         detections_effdetd5_fold3[n,:,:4].tolist(),
                         detections_effdetd5_fold4[n,:,:4].tolist(),
                         detections_flip_effdetd5_fold0[n,:,:4].tolist(),
                         detections_flip_effdetd5_fold1[n,:,:4].tolist(),
                         detections_flip_effdetd5_fold2[n,:,:4].tolist(),
                         detections_flip_effdetd5_fold3[n,:,:4].tolist(),
                         detections_flip_effdetd5_fold4[n,:,:4].tolist(),
                         detections_effdetd6_fold0[n,:,:4].tolist(),
                         detections_effdetd6_fold1[n,:,:4].tolist(),
                         detections_effdetd6_fold2[n,:,:4].tolist(),
                         detections_effdetd6_fold3[n,:,:4].tolist(),
                         detections_effdetd6_fold4[n,:,:4].tolist(),
                         detections_flip_effdetd6_fold0[n,:,:4].tolist(),
                         detections_flip_effdetd6_fold1[n,:,:4].tolist(),
                         detections_flip_effdetd6_fold2[n,:,:4].tolist(),
                         detections_flip_effdetd6_fold3[n,:,:4].tolist(),
                         detections_flip_effdetd6_fold4[n,:,:4].tolist(),
                         detections_yolo1_fold0[n][:,:4].tolist(),  
                         detections_hflip_yolo1_fold1[n][:,:4].tolist(), 
                         detections_vflip_yolo1_fold2[n][:,:4].tolist(), 
                         detections_hvflip_yolo1_fold3[n][:,:4].tolist(),
                         detections_yolo2_fold0[n][:,:4].tolist(), 
                         detections_hflip_yolo2_fold1[n][:,:4].tolist(), 
                         detections_vflip_yolo2_fold2[n][:,:4].tolist(),  
                         detections_hvflip_yolo2_fold3[n][:,:4].tolist(),
                        ]
                scores = [
                          detections_effdetd5_fold0[n,:,4].tolist(),
                          detections_effdetd5_fold1[n,:,4].tolist(),
                          detections_effdetd5_fold2[n,:,4].tolist(),
                          detections_effdetd5_fold3[n,:,4].tolist(),
                          detections_effdetd5_fold4[n,:,4].tolist(),
                          detections_flip_effdetd5_fold0[n,:,4].tolist(),
                          detections_flip_effdetd5_fold1[n,:,4].tolist(),
                          detections_flip_effdetd5_fold2[n,:,4].tolist(),
                          detections_flip_effdetd5_fold3[n,:,4].tolist(),
                          detections_flip_effdetd5_fold4[n,:,4].tolist(),
                          detections_effdetd6_fold0[n,:,4].tolist(),
                          detections_effdetd6_fold1[n,:,4].tolist(),
                          detections_effdetd6_fold2[n,:,4].tolist(),
                          detections_effdetd6_fold3[n,:,4].tolist(),
                          detections_effdetd6_fold4[n,:,4].tolist(),
                          detections_flip_effdetd6_fold0[n,:,4].tolist(),
                          detections_flip_effdetd6_fold1[n,:,4].tolist(),
                          detections_flip_effdetd6_fold2[n,:,4].tolist(),
                          detections_flip_effdetd6_fold3[n,:,4].tolist(),
                          detections_flip_effdetd6_fold4[n,:,4].tolist(),
                          detections_yolo1_fold0[n][:,4].tolist(),  
                          detections_hflip_yolo1_fold1[n][:,4].tolist(), 
                          detections_vflip_yolo1_fold2[n][:,4].tolist(), 
                          detections_hvflip_yolo1_fold3[n][:,4].tolist(),
                          detections_yolo2_fold0[n][:,4].tolist(), 
                          detections_hflip_yolo2_fold1[n][:,4].tolist(), 
                          detections_vflip_yolo2_fold2[n][:,4].tolist(),  
                          detections_hvflip_yolo2_fold3[n][:,4].tolist(),
                         ]
                labels = [
                          detections_effdetd5_fold0[n,:,5].tolist(),
                          detections_effdetd5_fold1[n,:,5].tolist(),
                          detections_effdetd5_fold2[n,:,5].tolist(),
                          detections_effdetd5_fold3[n,:,5].tolist(),
                          detections_effdetd5_fold4[n,:,5].tolist(),
                          detections_flip_effdetd5_fold0[n,:,5].tolist(),
                          detections_flip_effdetd5_fold1[n,:,5].tolist(),
                          detections_flip_effdetd5_fold2[n,:,5].tolist(),
                          detections_flip_effdetd5_fold3[n,:,5].tolist(),
                          detections_flip_effdetd5_fold4[n,:,5].tolist(),
                          detections_effdetd6_fold0[n,:,5].tolist(),
                          detections_effdetd6_fold1[n,:,5].tolist(),
                          detections_effdetd6_fold2[n,:,5].tolist(),
                          detections_effdetd6_fold3[n,:,5].tolist(),
                          detections_effdetd6_fold4[n,:,5].tolist(),
                          detections_flip_effdetd6_fold0[n,:,5].tolist(),
                          detections_flip_effdetd6_fold1[n,:,5].tolist(),
                          detections_flip_effdetd6_fold2[n,:,5].tolist(),
                          detections_flip_effdetd6_fold3[n,:,5].tolist(),
                          detections_flip_effdetd6_fold4[n,:,5].tolist(),
                          detections_yolo1_fold0[n][:,5].tolist(),  
                          detections_hflip_yolo1_fold1[n][:,5].tolist(), 
                          detections_vflip_yolo1_fold2[n][:,5].tolist(), 
                          detections_hvflip_yolo1_fold3[n][:,5].tolist(),
                          detections_yolo2_fold0[n][:,5].tolist(), 
                          detections_hflip_yolo2_fold1[n][:,5].tolist(), 
                          detections_vflip_yolo2_fold2[n][:,5].tolist(),  
                          detections_hvflip_yolo2_fold3[n][:,5].tolist(),
                         ]
                boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.6)

                boxes[:,0] *= targets['w'][n]
                boxes[:,2] *= targets['w'][n]
                boxes[:,1] *= targets['h'][n]
                boxes[:,3] *= targets['h'][n]

                if len(boxes)>=99:
                    detections_ensemble[n,:99,:4] = boxes[:99,:]
                    detections_ensemble[n,:99,4] = scores[:99]
                    detections_ensemble[n,:99,5] = labels[:99]
                else:
                    detections_ensemble[n,:len(boxes),:4] = boxes
                    detections_ensemble[n,:len(boxes),4] = scores
                    detections_ensemble[n,:len(boxes),5] = labels

                # Estimate the none class using the topK (K=3) opacity probs.
                detections_ensemble[n,-1,:4] = np.array([0,0,1,1], dtype=np.float32)
                non_prob = 1.0
                count = 0
                for bb in range(detections_ensemble.shape[1]):
                    if detections_ensemble[n,bb,5]==1.0:
                        non_prob *= (1.0-detections_ensemble[n,bb,4])
                        count += 1
                        if count>=3:
                            break
                detections_ensemble[n,-1,4] = non_prob
                detections_ensemble[n,-1,5] = 2

                detections_all.append(detections_ensemble[n])
                    
    # generate submission
    for image_id in image_id_list_matched_midrc:
        image_level_dict[image_id] = {"opacity": [], "none": 0., "negative": 0., "typical": 0., "indeterminate": 0., "atypical": 0.}

    for i in range(len(image_id_list_matched_midrc)):
        image_detections = detections_all[i]
        for j in range(len(image_detections)):
            if image_detections[j,5]==1.0:
                image_level_dict[image_id_list_matched_midrc[i]]["opacity"].append([image_detections[j,4]]+list(image_detections[j,:4]))
        image_level_dict[image_id_list_matched_midrc[i]]["none"] += image_detections[-1,4]
        for j in range(len(image_detections)):
            if image_detections[j,5]==3.0:
                image_level_dict[image_id_list_matched_midrc[i]]["negative"] += image_detections[j,4]
                break
        for j in range(len(image_detections)):
            if image_detections[j,5]==4.0:
                image_level_dict[image_id_list_matched_midrc[i]]["typical"] += image_detections[j,4]
                break
        for j in range(len(image_detections)):
            if image_detections[j,5]==5.0:
                image_level_dict[image_id_list_matched_midrc[i]]["indeterminate"] += image_detections[j,4]
                break
        for j in range(len(image_detections)):
            if image_detections[j,5]==6.0:
                image_level_dict[image_id_list_matched_midrc[i]]["atypical"] += image_detections[j,4]
                break

    end_time = time.time()
    print(end_time-start_time)

    return image_level_dict


def main():

    import numpy as np
    import pandas as pd
    import os
    import cv2
    from tqdm import tqdm
    import pickle
    import time
    import glob

    # prepare input
    image_list = sorted(glob.glob('../input/siim-covid19-detection/test/*/*/*.dcm'))
    study_id_list = []
    image_id_list = []
    studyid2imageid = {}
    imageid2studyid = {}
    label_dict = {}
    for i in range(len(image_list)):
        study_id = image_list[i].split('/')[-3]
        image_id = image_list[i].split('/')[-1][:-4]
        image_id_list.append(image_id)
        if study_id not in studyid2imageid:
            study_id_list.append(study_id)
            studyid2imageid[study_id] = [image_id]
        else:
            studyid2imageid[study_id].append(image_id)
        imageid2studyid[image_id] = study_id
        label_dict[image_id] = {
            'img_dir': image_list[i],
        }  

    print(len(image_id_list), len(study_id_list), len(studyid2imageid), len(imageid2studyid))

    image_level_dict1, image_id_list_matched_midrc, midrc_dict, image_id_list_matched_bimcv, bimcv_dict = get_study_level_results()
    image_level_dict2 = get_none_results(image_id_list_matched_midrc, midrc_dict)
    image_level_dict3 = get_detection_results(image_id_list_matched_midrc, midrc_dict)

    pred_none2 = np.zeros((len(image_id_list), ), dtype=np.float32)
    pred_none3 = np.zeros((len(image_id_list), ), dtype=np.float32)
    pred_negative1 = np.zeros((len(image_id_list), ), dtype=np.float32)
    pred_negative3 = np.zeros((len(image_id_list), ), dtype=np.float32)
    pred_typical1 = np.zeros((len(image_id_list), ), dtype=np.float32)
    pred_typical3 = np.zeros((len(image_id_list), ), dtype=np.float32)
    pred_indeterminate1 = np.zeros((len(image_id_list), ), dtype=np.float32)
    pred_indeterminate3 = np.zeros((len(image_id_list), ), dtype=np.float32)
    pred_atypical1 = np.zeros((len(image_id_list), ), dtype=np.float32)
    pred_atypical3 = np.zeros((len(image_id_list), ), dtype=np.float32)
    for i in range(len(image_id_list)):
        pred_none2[i] = image_level_dict2[image_id_list[i]]["none"]
        pred_none3[i] = image_level_dict3[image_id_list[i]]["none"]
        pred_negative1[i] = image_level_dict1[image_id_list[i]]["negative"]
        pred_negative3[i] = image_level_dict3[image_id_list[i]]["negative"]
        pred_typical1[i] = image_level_dict1[image_id_list[i]]["typical"]
        pred_typical3[i] = image_level_dict3[image_id_list[i]]["typical"]
        pred_indeterminate1[i] = image_level_dict1[image_id_list[i]]["indeterminate"]
        pred_indeterminate3[i] = image_level_dict3[image_id_list[i]]["indeterminate"]
        pred_atypical1[i] = image_level_dict1[image_id_list[i]]["atypical"]
        pred_atypical3[i] = image_level_dict3[image_id_list[i]]["atypical"]
    pred_none2 = pred_none2.argsort().argsort() / len(image_id_list)
    pred_none3 = pred_none3.argsort().argsort() / len(image_id_list)
    pred_negative1 = pred_negative1.argsort().argsort() / len(image_id_list)
    pred_negative3 = pred_negative3.argsort().argsort() / len(image_id_list)
    pred_typical1 = pred_typical1.argsort().argsort() / len(image_id_list)
    pred_typical3 = pred_typical3.argsort().argsort() / len(image_id_list)
    pred_indeterminate1 = pred_indeterminate1.argsort().argsort() / len(image_id_list)
    pred_indeterminate3 = pred_indeterminate3.argsort().argsort() / len(image_id_list)
    pred_atypical1 = pred_atypical1.argsort().argsort() / len(image_id_list)
    pred_atypical3 = pred_atypical3.argsort().argsort() / len(image_id_list)
    

    ##########
    df_pred_midrc = pd.read_csv('../input/covidnearestneighbors/df_pred_midrc.csv')
    id_list_pred_midrc = df_pred_midrc['id'].values
    negative_list_pred_midrc = df_pred_midrc['negative'].values
    typical_list_pred_midrc = df_pred_midrc['typical'].values
    indeterminate_list_pred_midrc = df_pred_midrc['indeterminate'].values
    atypical_list_pred_midrc = df_pred_midrc['atypical'].values
    none_list_pred_midrc = df_pred_midrc['none'].values
    pred_midrc_dict = {}
    for i in range(len(id_list_pred_midrc)):
        pred_midrc_dict[id_list_pred_midrc[i]] = {"none": none_list_pred_midrc[i], "negative": negative_list_pred_midrc[i], "typical": typical_list_pred_midrc[i], "indeterminate": indeterminate_list_pred_midrc[i], "atypical": atypical_list_pred_midrc[i]}

    df_pred_bimcv = pd.read_csv('../input/covidnearestneighbors/df_pred_bimcv.csv')
    id_list_pred_bimcv = df_pred_bimcv['id'].values
    negative_list_pred_bimcv = df_pred_bimcv['negative'].values
    typical_list_pred_bimcv = df_pred_bimcv['typical'].values
    indeterminate_list_pred_bimcv = df_pred_bimcv['indeterminate'].values
    atypical_list_pred_bimcv = df_pred_bimcv['atypical'].values
    none_list_pred_bimcv = df_pred_bimcv['none'].values
    pred_bimcv_dict = {}
    for i in range(len(id_list_pred_bimcv)):
        pred_bimcv_dict[id_list_pred_bimcv[i]] = {"none": none_list_pred_bimcv[i], "negative": negative_list_pred_bimcv[i], "typical": typical_list_pred_bimcv[i], "indeterminate": indeterminate_list_pred_bimcv[i], "atypical": atypical_list_pred_bimcv[i]}
    ##########


    image_level_dict = {}
    for image_id in image_id_list:
        image_level_dict[image_id] = {"opacity": [], "none": 0., "negative": 0., "typical": 0., "indeterminate": 0., "atypical": 0.}

    for i in range(len(image_id_list)):
        image_level_dict[image_id_list[i]]["opacity"] = image_level_dict3[image_id_list[i]]["opacity"]
        if image_id_list[i] in image_id_list_matched_midrc:
            image_level_dict[image_id_list[i]]["none"] = 0.65*(0.5*pred_none2[i] + 0.5*pred_none3[i])+0.35*(pred_midrc_dict[midrc_dict[image_id_list[i]]['midrc_id']]['none'])
            image_level_dict[image_id_list[i]]["negative"] = 0.65*(0.67*pred_negative1[i] + 0.33*pred_negative3[i])+0.35*(pred_midrc_dict[midrc_dict[image_id_list[i]]['midrc_id']]['negative'])
            image_level_dict[image_id_list[i]]["typical"] = 0.65*(0.67*pred_typical1[i] + 0.33*pred_typical3[i])+0.35*(pred_midrc_dict[midrc_dict[image_id_list[i]]['midrc_id']]['typical'])
            image_level_dict[image_id_list[i]]["indeterminate"] = 0.65*(0.67*pred_indeterminate1[i] + 0.33*pred_indeterminate3[i])+0.35*(pred_midrc_dict[midrc_dict[image_id_list[i]]['midrc_id']]['indeterminate'])
            image_level_dict[image_id_list[i]]["atypical"] = 0.65*(0.67*pred_atypical1[i] + 0.33*pred_atypical3[i])+0.35*(pred_midrc_dict[midrc_dict[image_id_list[i]]['midrc_id']]['atypical'])
        elif image_id_list[i] in image_id_list_matched_bimcv:
            image_level_dict[image_id_list[i]]["none"] = 0.65*(0.5*pred_none2[i] + 0.5*pred_none3[i])+0.35*(pred_bimcv_dict[bimcv_dict[image_id_list[i]]['bimcv_id']]['none'])
            image_level_dict[image_id_list[i]]["negative"] = 0.65*(0.67*pred_negative1[i] + 0.33*pred_negative3[i])+0.35*(pred_bimcv_dict[bimcv_dict[image_id_list[i]]['bimcv_id']]['negative'])
            image_level_dict[image_id_list[i]]["typical"] = 0.65*(0.67*pred_typical1[i] + 0.33*pred_typical3[i])+0.35*(pred_bimcv_dict[bimcv_dict[image_id_list[i]]['bimcv_id']]['typical'])
            image_level_dict[image_id_list[i]]["indeterminate"] = 0.65*(0.67*pred_indeterminate1[i] + 0.33*pred_indeterminate3[i])+0.35*(pred_bimcv_dict[bimcv_dict[image_id_list[i]]['bimcv_id']]['indeterminate'])
            image_level_dict[image_id_list[i]]["atypical"] = 0.65*(0.67*pred_atypical1[i] + 0.33*pred_atypical3[i])+0.35*(pred_bimcv_dict[bimcv_dict[image_id_list[i]]['bimcv_id']]['atypical'])
        else:
            image_level_dict[image_id_list[i]]["none"] = 0.5*pred_none2[i] + 0.5*pred_none3[i]
            image_level_dict[image_id_list[i]]["negative"] = 0.67*pred_negative1[i] + 0.33*pred_negative3[i]
            image_level_dict[image_id_list[i]]["typical"] = 0.67*pred_typical1[i] + 0.33*pred_typical3[i]
            image_level_dict[image_id_list[i]]["indeterminate"] = 0.67*pred_indeterminate1[i] + 0.33*pred_indeterminate3[i]
            image_level_dict[image_id_list[i]]["atypical"] = 0.67*pred_atypical1[i] + 0.33*pred_atypical3[i]

    study_level_dict = {}
    for study_id in study_id_list:
        study_level_dict[study_id] = {"negative": 0., "typical": 0., "indeterminate": 0., "atypical": 0.}

    for i in range(len(study_id_list)):    
        for image_id in studyid2imageid[study_id_list[i]]:
            study_level_dict[study_id_list[i]]["negative"] += image_level_dict[image_id]["negative"]
        study_level_dict[study_id_list[i]]["negative"] /= len(studyid2imageid[study_id_list[i]])
        for image_id in studyid2imageid[study_id_list[i]]:
            study_level_dict[study_id_list[i]]["typical"] += image_level_dict[image_id]["typical"]
        study_level_dict[study_id_list[i]]["typical"] /= len(studyid2imageid[study_id_list[i]])
        for image_id in studyid2imageid[study_id_list[i]]:
            study_level_dict[study_id_list[i]]["indeterminate"] += image_level_dict[image_id]["indeterminate"]
        study_level_dict[study_id_list[i]]["indeterminate"] /= len(studyid2imageid[study_id_list[i]])
        for image_id in studyid2imageid[study_id_list[i]]:
            study_level_dict[study_id_list[i]]["atypical"] += image_level_dict[image_id]["atypical"]
        study_level_dict[study_id_list[i]]["atypical"] /= len(studyid2imageid[study_id_list[i]])

    id_list = []
    PredictionString_list = []
    for i in range(len(study_id_list)): 
        id_list.append(study_id_list[i]+'_study')   
        pred_str = ""
        pred_str += "negative"
        pred_str += " "
        pred_str += str(np.float16(study_level_dict[study_id_list[i]]["negative"]))
        pred_str += " "
        pred_str += "0 0 1 1"
        pred_str += " "
        pred_str += "typical"
        pred_str += " "
        pred_str += str(np.float16(study_level_dict[study_id_list[i]]["typical"]))
        pred_str += " "
        pred_str += "0 0 1 1"
        pred_str += " "
        pred_str += "indeterminate"
        pred_str += " "
        pred_str += str(np.float16(study_level_dict[study_id_list[i]]["indeterminate"]))
        pred_str += " "
        pred_str += "0 0 1 1"
        pred_str += " "
        pred_str += "atypical"
        pred_str += " "
        pred_str += str(np.float16(study_level_dict[study_id_list[i]]["atypical"]))
        pred_str += " "
        pred_str += "0 0 1 1"
        pred_str += " "
        PredictionString_list.append(pred_str)
    for i in range(len(image_id_list)): 
        id_list.append(image_id_list[i]+'_image')   
        pred_str = ""
        pred_str += "none"
        pred_str += " "
        pred_str += str(np.float16(image_level_dict[image_id_list[i]]["none"]))
        pred_str += " "
        pred_str += "0 0 1 1"
        pred_str += " "
        for bb in image_level_dict[image_id_list[i]]["opacity"]:
            pred_str += "opacity"
            pred_str += " "
            pred_str += str(np.float16(bb[0]))
            pred_str += " "
            pred_str += str(np.float16(bb[1]))
            pred_str += " "
            pred_str += str(np.float16(bb[2]))
            pred_str += " "
            pred_str += str(np.float16(bb[3]))
            pred_str += " "
            pred_str += str(np.float16(bb[4]))
            pred_str += " "
        PredictionString_list.append(pred_str)

    # generate submission
    sub_df = pd.DataFrame(data={'id': id_list, 'PredictionString': PredictionString_list})
    sub_df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
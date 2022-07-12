import os
import sys
# import gc
import shutil
import typing as tp
from pathlib import Path
from argparse import ArgumentParser

import yaml
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from joblib import Parallel, delayed

import cv2
import albumentations

from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.utils import data
from torchvision import models as torchvision_models

sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import timm

sys.path.append("../input/resnest-package/resnest-0.0.6b20200701/resnest")
from resnest import torch as resnest_torch
# from efficientnet_pytorch import EfficientNet

ROOT = Path.cwd().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
DATA = INPUT / "ranzcr-clip-catheter-line-classification"
TRAIN = DATA / "train"
TEST = DATA / "test"

TRAINED_MODELS = [
    INPUT / "ranzcr-weight-exp078",  # Multi-Head ResNet200D
    INPUT / "ranzcr-weight-exp080",  # Multi-Head EfficientNetB5
    INPUT / "ranzcr-weight-exp081",  # Multi-Head SE-ResNet152D
]
TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)
TEST_640x640 = TMP / "test_640x640"
TEST_640x640.mkdir(exist_ok=True)

USE_TEST_DIRS = [
    TEST_640x640,
    TEST_640x640,
    TEST_640x640,
]

RANDAM_SEED = 1086
N_CLASSES = 11

CLASSES = [
    'ETT - Abnormal',
    'ETT - Borderline',
    'ETT - Normal',
    'NGT - Abnormal',
    'NGT - Borderline',
    'NGT - Incompletely Imaged',
    'NGT - Normal',
    'CVC - Abnormal',
    'CVC - Borderline',
    'CVC - Normal',
    'Swan Ganz Catheter Present'
]

BATCH_SIZE = 32
FOLD_IDS = [
    0, 1, 2, 3, 4
]
N_FOLD = len(FOLD_IDS)
N_TARGETS = N_CLASSES

CONVERT_TO_RANK = False
CONVERT_TO_RANK_2 = False
CONVERT_TO_POWER = False
USE_HFLIP_TTA = True
FAST_COMMIT = True


def load_setting_file(path: str):
    """Load YAML setting file."""
    with open(path) as f:
        settings = yaml.safe_load(f)
    return settings


def resize_images(img_id, input_dir, output_dir, resize_to=(512, 512)):
    img_path = input_dir / (img_id + ".jpg")
    save_path = output_dir / (img_id + ".png")
    img_arr = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_arr = cv2.resize(img_arr, resize_to)
    cv2.imwrite(str(save_path), img_arr)
    
    
class BasicModelCls(nn.Module):
    
    def __init__(
        self, base_name, dims_head, pretrained=False
    ):
        """Initialize"""
        self.base_name = base_name
        super(BasicModelCls, self).__init__()
        
        # # prepare backbone
        if hasattr(resnest_torch, base_name):
            # # # load base model
            base_model = getattr(resnest_torch, base_name)(pretrained=pretrained)
            in_features = base_model.fc.in_features
            # remove head classifier
            del base_model.fc
            base_model.fc = nn.Identity()

        elif hasattr(timm.models, base_name):
            # # # load base model
            base_model = timm.create_model(base_name, pretrained=pretrained)
            in_features = base_model.num_features
            # # remove head classifier
            base_model.reset_classifier(0)

        else:
            raise NotImplementedError

        self.backbone = base_model
        
        # # prepare head clasifier
        if dims_head[0] is None:
            dims_head[0] = in_features

        layers_list = []
        for i in range(len(dims_head) - 2):
            in_dim, out_dim = dims_head[i: i + 2]
            layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(), nn.Dropout(0.5),])
        layers_list.append(
            nn.Linear(dims_head[-2], dims_head[-1]))
        self.head_cls = nn.Sequential(*layers_list)

    def forward(self, x):
        """Forward"""
        h = self.backbone(x)
        h = self.head_cls(h)
        return h
    

def get_activation(activ_name: str="relu"):
    """"""
    act_dict = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity()}
    if activ_name in act_dict:
        return act_dict[activ_name]
    else:
        raise NotImplementedError
        

class Conv2dBNActiv(nn.Module):
    """Conv2d -> (BN ->) -> Activation"""
    
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, stride: int=1, padding: int=0,
        bias: bool=False, use_bn: bool=True, activ: str="relu"
    ):
        """"""
        super(Conv2dBNActiv, self).__init__()
        layers = []
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(get_activation(activ))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward"""
        return self.layers(x)
            

class SpatialAttentionBlock(nn.Module):
    """Spatial Attention for (C, H, W) feature maps"""
    
    def __init__(
        self, in_channels: int,
        out_channels_list: tp.List[int],
    ):
        """Initialize"""
        super(SpatialAttentionBlock, self).__init__()
        self.n_layers = len(out_channels_list)
        channels_list = [in_channels] + out_channels_list
        assert self.n_layers > 0
        assert channels_list[-1] == 1
        
        for i in range(self.n_layers - 1):
            in_chs, out_chs = channels_list[i: i + 2]
            layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="relu")
            setattr(self, f"conv{i + 1}", layer)
            
        in_chs, out_chs = channels_list[-2:]
        layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="sigmoid")
        setattr(self, f"conv{self.n_layers}", layer)
    
    def forward(self, x):
        """Forward"""
        h = x
        for i in range(self.n_layers):
            h = getattr(self, f"conv{i + 1}")(h)
            
        h = h * x
        return h
    
    
class MultiHeadModelCls(nn.Module):
    
    def __init__(
        self, base_name: str='resnext50_32x4d',
        out_dims_head: tp.List[int]=[3, 4, 3, 1],
        spatial_attention_channels: tp.List[int]=[64, 32, 16, 1],
        pretrained=False
    ):
        """"""
        self.base_name = base_name
        self.n_heads = len(out_dims_head)
        super(MultiHeadModelCls, self).__init__()
        
        # # load base model
        if hasattr(timm.models, base_name):
            # # # load base model
            if type(pretrained) == str:
                base_model = timm.create_model(base_name, pretrained=False)
                in_features = base_model.num_features
                # # remove pooling layer and head classifier
                base_model.reset_classifier(0, "")
                base_model.load_state_dict(
                    torch.load(pretrained, map_location=torch.device('cpu')))
                print("load pretrained:", pretrained)
            else:
                base_model = timm.create_model(base_name, pretrained=pretrained)
                in_features = base_model.num_features
                # # remove head classifier
                base_model.reset_classifier(0, "")
                print("load imagenet pretrained:", pretrained)
        else:
            raise NotImplementedError
        
        # # Shared CNN Bacbone
        self.backbone = base_model
        
        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, spatial_attention_channels),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        """"""
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y
    

def get_model(args):
    """"""
    return eval(args["name"])(**args["params"])


# # ------------------- Data Augmentation for Images ------------------# #
class ImageTransformBase:
    """
    Base Image Transform class.

    Args:
        data_augmentations: List of tuple(method: str, params :dict), each elems pass to albumentations
    """

    def __init__(self, data_augmentations: tp.List[tp.Tuple[str, tp.Dict]]):
        """Initialize."""
        augmentations_list = [
            self._get_augmentation(aug_name)(**params)
            for aug_name, params in data_augmentations]
        self.data_aug = albumentations.Compose(augmentations_list)

    def __call__(self, pair: tp.Tuple[np.ndarray]) -> tp.Tuple[np.ndarray]:
        """You have to implement this by task"""
        raise NotImplementedError

    def _get_augmentation(self, aug_name: str) -> tp.Tuple[ImageOnlyTransform, DualTransform]:
        """Get augmentations from albumentations"""
        if hasattr(albumentations, aug_name):
            return getattr(albumentations, aug_name)
        else:
            return eval(aug_name)


class ImageTransformForCls(ImageTransformBase):
    """Data Augmentor for Classification Task."""

    def __init__(self, data_augmentations: tp.List[tp.Tuple[str, tp.Dict]]):
        """Initialize."""
        super(ImageTransformForCls, self).__init__(data_augmentations)

    def __call__(self, in_arrs: tp.Tuple[np.ndarray]) -> tp.Tuple[np.ndarray]:
        """Apply Transform."""
        img, label = in_arrs
        augmented = self.data_aug(image=img)
        img = augmented["image"]

        return img, label
    
    
# # ------------------- Data Augmentation for Images ------------------# #
class LabeledImageDataset(data.Dataset):
    """
    Dataset class for (image, label) pairs

    reads images and applys transforms to them.

    Attributes
    ----------
    file_list : List[Tuple[tp.Union[str, Path], tp.Union[int, float, np.ndarray]]]
        list of (image file, label) pair
    transform_list : List[Dict]
        list of dict representing image transform 
    """

    def __init__(
        self,
        file_list: tp.List[
            tp.Tuple[
                tp.Union[str, Path],
                tp.Union[int, float, np.ndarray]
            ]],
        transform_list: tp.List[tp.Dict],
    ):
        """Initialize"""
        self.file_list = file_list
        self.transform = ImageTransformForCls(transform_list)

    def __len__(self):
        """Return Num of Images."""
        return len(self.file_list)

    def __getitem__(self, index):
        """Return transformed image and mask for given index."""
        img_path, label = self.file_list[index]
        img = self._read_image_as_array(img_path)
        
        img, label = self.transform((img, label))
        return img, label

    def _read_image_as_array(self, path: str):
        """Read image file and convert into numpy.ndarray"""
        img_arr = cv2.imread(str(path))
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        return img_arr


def get_dataloaders_for_inference(
    file_list: tp.List[tp.List], batch_size=128,
):
    """Create DataLoader"""
    dataset = LabeledImageDataset(
        file_list,
        transform_list=[
            ["Normalize", {
                "always_apply": True, "max_pixel_value": 255.0,
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}],
          ["ToTensorV2", {"always_apply": True}],
        ])
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
        drop_last=False)

    return loader


# # ------------------- inference function ------------------# #
def inference_function(settings, model, loader, device, use_hflip_tta=False):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for x, t in tqdm(loader):
            x = x.to(device)
            if use_hflip_tta:
                x = torch.stack([x, x.flip(dims=(-1,))], 1).reshape(-1, *x.shape[-3:])
                y = model(x)
                y = y.reshape(-1, 2, *y.shape[1:]).mean(axis=1)
            else:
                y = model(x)
            pred_list.append(y.sigmoid().detach().cpu().numpy())
        
        pred_arr = np.concatenate(pred_list)
        del pred_list
    return pred_arr


def get_file_list(train_all, use_fold, ext="jpg"):
    """Get file path and target info."""
    
    train_df = train_all[train_all["fold"] != use_fold]
    val_df = train_all[train_all["fold"] == use_fold]
    
    train_data_dir = TRAIN
    print(train_data_dir)

    train_file_list = list(zip(
        [train_data_dir / f"{img_id}.{ext}" for img_id in train_df["StudyInstanceUID"].values],
        train_df[CLASSES].values.astype("f")
    ))
    val_file_list = list(zip(
        [train_data_dir / f"{img_id}.{ext}" for img_id in val_df["StudyInstanceUID"].values],
        val_df[CLASSES].values.astype("f")
    ))

    return train_file_list, val_file_list


def main():
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(device)
    
    train_all = pd.read_csv(TRAINED_MODELS[-1] / "train_mlsgkf_5fold.csv")
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    
    if FAST_COMMIT and len(sample_sub) == 3582:
        sample_sub = sample_sub.iloc[:BATCH_SIZE * 2].reset_index(drop=True)
    
    # resize test images
    _ = Parallel(n_jobs=2, verbose=5)([
        delayed(resize_images)(img_id, TEST, TEST_640x640, (640, 640))
        for img_id in sample_sub.StudyInstanceUID.values
    ])
    test_pred_list = []
    for model_dir, test_dir in zip(
        TRAINED_MODELS,
        USE_TEST_DIRS,
    ):
        print(model_dir.name)
        test_file_list = [
            (test_dir / f"{img_id}.png", [-1] * 11)
            for img_id in sample_sub["StudyInstanceUID"].values]
        
        test_loader = get_dataloaders_for_inference(test_file_list, batch_size=BATCH_SIZE)
        
        test_preds_arr = np.zeros((N_FOLD, len(sample_sub), N_TARGETS))
        
        for fold_id in FOLD_IDS:
            print(f"[fold {fold_id}]")
            # stgs = load_setting_file(model_dir / f"fold{fold_id:0>2}" / "settings.yml")
            stgs = load_setting_file(model_dir / f"fold{fold_id}" / "settings.yml")
            # # prepare 
            stgs["model"]["params"]["pretrained"] = False
            model = get_model(stgs["model"])
            model_path = model_dir / f"best_model_fold{fold_id}.pth"
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("loading model: done")

            # # inference test
            test_pred = inference_function(stgs, model, test_loader, device, USE_HFLIP_TTA)
            if N_FOLD == 1:
                test_preds_arr[0] = test_pred
            else:
                test_preds_arr[fold_id] = test_pred
            print("inference test data: done")
        
        # # create submission
        if CONVERT_TO_RANK:
            # # shape: (fold, n_example, class)
            test_preds_arr = test_preds_arr.argsort(axis=1).argsort(axis=1)
    
        test_pred_list.append(test_preds_arr.mean(axis=0))
    
    sub = sample_sub.copy()
    sub[CLASSES] = 0
    weights = [1. / len(test_pred_list)] * len(test_pred_list)
    for w, arr in zip(weights, test_pred_list):
        if CONVERT_TO_RANK:
            print("argsort rank avg")
            sub[CLASSES] += w * arr.argsort(axis=0).argsort(axis=0)
        
        else:
            print("value avg")
            sub[CLASSES] += w * arr
    
    sub.to_csv("submission.csv", index=False)
    
    
if __name__ == "__main__":
    main()
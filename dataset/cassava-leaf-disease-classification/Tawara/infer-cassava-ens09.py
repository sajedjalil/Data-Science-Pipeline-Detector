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
DATA = INPUT / "cassava-leaf-disease-classification"
TRAIN = DATA / "train_images"
TEST = DATA / "test_images"

TRAINED_MODELS = [
    INPUT / "cassava-weight-exp020",
    INPUT / "cassava-weight-exp023",
    INPUT / "cassava-weight-exp024",
    INPUT / "cassava-weight-exp025",
    INPUT / "cassava-weight-exp026",
    INPUT / "cassava-weight-exp027",
    INPUT / "cassava-weight-exp028",
    INPUT / "cassava-weight-exp029",
    INPUT / "cassava-weight-exp030",
    INPUT / "cassava-weight-exp031",
]
TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)
TEST_384x384 = TMP / "test_384x384"
TEST_384x384.mkdir(exist_ok=True)
TEST_512x512 = TMP / "test_512x512"
TEST_512x512.mkdir(exist_ok=True)

USE_TEST_DIRS = [
    TEST_512x512,
    TEST_512x512,
    TEST_512x512,
    TEST_512x512,
    TEST_512x512,
    TEST_512x512,
    TEST_512x512,
    TEST_512x512,
    TEST_512x512,
    TEST_512x512,
]

N_FOLDS = 5
RANDAM_SEED = 1086
N_CLASSES = 5

CLASSES = [
    'Cassava Bacterial Blight (CBB)',
    'Cassava Brown Streak Disease (CBSD)',
    'Cassava Green Mottle (CGM)',
    'Cassava Mosaic Disease (CMD)',
    'Healthy'
]
BATCH_SIZE = 64
FOLD_IDS = [
    0,
    1, 2, 3, 4
]
N_FOLD = len(FOLD_IDS)
USE_HFLIP_TTA = False
APPLY_SOFTMAX = True


def load_setting_file(path: str):
    """Load YAML setting file."""
    with open(path) as f:
        settings = yaml.safe_load(f)
    return settings


def resize_images(img_id, input_dir, output_dir, resize_to=(512, 512)):
    img_path = input_dir / img_id
    save_path = output_dir / img_id.replace("jpg", "png")
    img_arr = cv2.imread(str(img_path))
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
          # ["Resize", {"always_apply": True, "height": 384, "width": 384}],
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
def inference_function(
    settings, model, loader, device, apply_softmax=True, use_hflip_tta=False):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for x, t in tqdm(loader):
            if use_hflip_tta:
                x = torch.stack([x, x.flip(dims=(-1,))], 1).reshape(-1, *x.shape[-3:])
                y = model(x.to(device))
                if apply_softmax:
                    y = nn.Softmax(dim=1)(y)
                y = y.reshape(-1, 2, *y.shape[1:]).mean(axis=1)
            else:
                y = model(x.to(device))
                if apply_softmax:
                    y = nn.Softmax(dim=1)(y)
            
            pred_list.append(y.detach().cpu().numpy())
        
        pred_arr = np.concatenate(pred_list)
        del pred_list
    return pred_arr


def get_file_list(stgs):
    """Get file path and target info."""
    train_all = pd.read_csv(CFG.DATA / stgs["globals"]["meta_file"])
    use_fold = stgs["globals"]["val_fold"]
    
    train_df = train_all[train_all["fold"] != use_fold]
    val_df = train_all[
        (train_all["fold"] == use_fold) & (train_all["source"] != 2019)]
    
    train_data_dir = CFG.DATA / stgs["globals"]["dataset_name"]
    print(train_data_dir)

    train_file_list = list(zip(
        [train_data_dir / img_id for img_id in train_df["image_id"].values],
        torch.Tensor(train_df["label"].values).long()
    ))
    val_file_list = list(zip(
        [train_data_dir / img_id for img_id in val_df["image_id"].values],
        torch.Tensor(val_df["label"].values).long()
    ))

    return train_file_list, val_file_list


def main():
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(device)
    
    train_all = pd.read_csv(TRAINED_MODELS[0] / "merged_skf_5fold.csv")
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    
    # resize test images
#     _ = Parallel(n_jobs=2, verbose=5)([
#         delayed(resize_images)(img_id, TEST, TEST_384x384, (384, 384))
#         for img_id in sample_sub.StudyInstanceUID.values
#     ])
    _ = Parallel(n_jobs=2, verbose=5)([
        delayed(resize_images)(img_id, TEST, TEST_512x512, (512, 512))
        for img_id in sample_sub.image_id.values
    ])
    test_pred_list = []
    for model_dir, test_dir in zip(
        TRAINED_MODELS,
        USE_TEST_DIRS,
    ):
        test_file_list = [
            (test_dir / img_id.replace("jpg", "png"), [-1] * 11)
            for img_id in sample_sub["image_id"].values]
        test_loader = get_dataloaders_for_inference(test_file_list, batch_size=BATCH_SIZE)
        
        test_preds_arr = np.zeros((N_FOLD, len(sample_sub), N_CLASSES))
        
    
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
            test_pred = inference_function(
                stgs, model, test_loader, device, APPLY_SOFTMAX, USE_HFLIP_TTA)
            if N_FOLD == 1:
                test_preds_arr[0] = test_pred
            else:
                test_preds_arr[fold_id] = test_pred
            print("inference test data: done")
        
        # # create submission
    
        test_pred_list.append(test_preds_arr.mean(axis=0))

    test_pred_ens = np.zeros((len(sample_sub), N_CLASSES))
    # weights = [1. / len(test_pred_list)] * len(test_pred_list)
#     weights = [
#         0.05306612, 0.14452946, 0.04701213, 0.11939491, 0.07729351,
#         0.12685517, 0.10432277, 0.13695289, 0.13647716, 0.05409587
#     ]
    weights = [
        0.13328647, 0.09956352, 0.13933516, 0.14400941, 0.04102943,
        0.10379062, 0.06670362, 0.07771925, 0.15549429, 0.03906825
    ]
    for w, arr in zip(weights, test_pred_list):
        test_pred_ens += w * arr
        
    sub = sample_sub.copy()
    sub["label"] = test_pred_ens.argmax(axis=1)
    
    sub.to_csv("submission.csv", index=False)
    
    
if __name__ == "__main__":
    main()
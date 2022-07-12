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
STACKING_MODELS = [
    INPUT / "cassava-weight-stack01",
    INPUT / "cassava-weight-stack03",
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
APPLY_SOFTMAX = False


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


# # --------------------- MLP ---------------------- # #
def get_activation(activ_name: str="relu"):
    """"""
    act_dict = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity()}
    if activ_name in act_dict:
        return act_dict[activ_name]
    elif re.match(r"^htanh\_\d{4}$", activ_name):
        bound = int(activ_name[-4:]) / 1000
        return nn.Hardtanh(-bound, bound)
    else:
        raise NotImplementedError

class LBAD(nn.Module):
    """Linear (-> BN) -> Activation (-> Dropout)"""
    
    def __init__(
        self, in_features: int, out_features: int, drop_rate: float=0.0,
        use_bn: bool=False, use_wn: bool=False, activ: str="relu"
    ):
        """"""
        super(LBAD, self).__init__()
        layers = [nn.Linear(in_features, out_features)]
        if use_wn:
            layers[0] = nn.utils.weight_norm(layers[0])
        
        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))
        
        layers.append(get_activation(activ))
        
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)
    
    
class BDLA(nn.Module):
    """(BN -> Dropout ->) Linear -> Activation"""
    
    def __init__(
        self, in_features: int, out_features: int, drop_rate: float=0.0,
        use_bn: bool=False, use_wn: bool=False, activ: str="relu"
    ):
        """"""
        super(BDLA, self).__init__()
        layers = []
        if use_bn:
            layers.append(nn.BatchNorm1d(in_features))
            
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        
        layers.append(nn.Linear(in_features, out_features))
        if use_wn:
            layers[-1] = nn.utils.weight_norm(layers[-1])
            
        layers.append(get_activation(activ))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)
    

class LABD(nn.Module):
    """Linear -> Activation (-> BN -> Dropout) """
    
    def __init__(
        self, in_features: int, out_features: int, drop_rate: float=0.0,
        use_bn: bool=False, use_wn: bool=False, activ: str="relu"
    ):
        """"""
        super(LABD, self).__init__()
        layers = [nn.Linear(in_features, out_features), get_activation(activ)]
        
        if use_wn:
            layers[0] = nn.utils.weight_norm(layers[0])
        
        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))
        
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)


class MLP(nn.Module):
    """Stacked Dense layers"""
    
    def __init__(
        self, n_features_list: tp.List[int], use_tail_as_out: bool=False,
        drop_rate: float=0.0, use_bn: bool=False, use_wn: bool=False,
        activ:str="relu", block_name: str="LBAD"
    ):
        """"""
        super(MLP, self).__init__()
        n_layers = len(n_features_list) - 1
        block_class = {
            "LBAD": LBAD, "BDLA": BDLA, "LABD": LABD}[block_name]
        layers = []
        for i in range(n_layers):
            in_feats, out_feats = n_features_list[i: i + 2]
            if i == n_layers - 1 and use_tail_as_out:
                if block_name in ["BDLA"]:
                    layer = block_class(in_feats, out_feats, drop_rate, use_bn,  use_wn, "identity")
                else:
                    layer = nn.Linear(in_feats, out_feats)
                    if use_wn:
                        layer = nn.utils.weight_norm(layer)
            else:
                layer = block_class(in_feats, out_feats, drop_rate, use_bn,  use_wn, activ)
            layers.append(layer)
                
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)


# # --------------------- CNN ---------------------- # #
class CNNStacking1d(nn.Module):
    """1D-CNN for Stacking."""
    
    def __init__(
        self, n_models: int,
        n_channels_list: tp.List[int], use_bias: bool=False,
        kwargs_head: tp.Dict={},
    ):
        """"""
        super(CNNStacking1d, self).__init__()
        self.n_conv_layers = len(n_channels_list) - 1
        for i in range(self.n_conv_layers):
            in_ch = n_channels_list[i]
            out_ch = n_channels_list[i + 1]
            layer = nn.Sequential(
                nn.Conv1d(
                    in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=use_bias),
                # nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True))
            setattr(self, "conv{}".format(i + 1), layer)
        
        kwargs_head["n_features_list"][0] = (n_models - 2 * self.n_conv_layers) * n_channels_list[-1]
        print(kwargs_head["n_features_list"][0])
        self.head = MLP(**kwargs_head)
    
    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        """"""
        bs = x.shape[0]
        h = x  # shape: (bs, n_classes, n_models)
        for i in range(self.n_conv_layers):
            h = getattr(self, "conv{}".format(i + 1))(h)
            
        h = torch.reshape(h, (bs, -1))
        h = self.head(h)
        return h


class CNNStacking2d(nn.Module):
    """2D-CNN for Stacking."""
    
    def __init__(
        self, n_models: int, n_classes: int,
        n_channels_list: tp.List[int], use_bias: bool=False,
        kwargs_head: tp.Dict={},
    ):
        """"""
        super(CNNStacking2d, self).__init__()
        self.n_conv_layers = len(n_channels_list) - 1
        for i in range(self.n_conv_layers):
            in_ch = n_channels_list[i]
            out_ch = n_channels_list[i + 1]
            layer = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=(1, 3), stride=1, padding=0, bias=use_bias),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
            setattr(self, "conv{}".format(i + 1), layer)
        
        kwargs_head["n_features_list"][0] = (n_models - 2 * self.n_conv_layers) * n_classes * n_channels_list[-1]
        self.head = MLP(**kwargs_head)
    
    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        """"""
        bs = x.shape[0]
        h = x  # shape: (bs, 1, n_classes, n_models)
        for i in range(self.n_conv_layers):
            h = getattr(self, "conv{}".format(i + 1))(h)
        
        h = torch.reshape(h, (bs, -1))
        h = self.head(h)
        return h


class StackingDatasetForCNN(torch.utils.data.Dataset):
    
    def __init__(self, feat: np.ndarray, label: np.ndarray = None):
        """"""
        self.feat = feat
        if label is None:
            self.label = np.full((len(feat), 1), -1)
        else:
            self.label = label
        self.reset_model_order()
        
    def reset_model_order(self):
        self.model_order = np.arange(self.feat.shape[-1])
        
    def shuffle_model_order(self, seed):
        np.random.seed(seed)
        self.model_order = np.random.permutation(self.model_order)
        
    def __len__(self):
        """"""
        return len(self.feat)
    
    def __getitem__(self, index: int):
        """"""
        return [
            torch.from_numpy(self.feat[index][..., self.model_order]).float(),
            self.label[index]
        ]


def softmax(x):
    """
    Softmax Function
    
    x: (n_examples, n_classes)
    """
    x_max = x.max(axis=1)  # shape: (n_examples)
    x = x - x_max[:, None]  #  shape: (n_examples, n_classes)
    x = np.exp(x)
    return x / x.sum(axis=1)[:, None]


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

    
    # # weight_opt
    print("[weight optimize]")
    test_pred_wopt = np.zeros((len(sample_sub), N_CLASSES))
    weights = [
        0.05306612, 0.14452946, 0.04701213, 0.11939491, 0.07729351,
        0.12685517, 0.10432277, 0.13695289, 0.13647716, 0.05409587
    ]
    for w, arr in zip(weights, test_pred_list):
        test_pred_wopt += w * arr
        
        
    # # 1D-CNN
    print("[1D-CNN Stacking]")
    X = np.stack(test_pred_list, axis=2)
    model_dir = STACKING_MODELS[0]
    test_dataset = StackingDatasetForCNN(X)
    test_preds_arr = np.zeros((N_FOLD, len(sample_sub), N_CLASSES))
    for fold_id in FOLD_IDS:
        fold_path =  model_dir / f"fold{fold_id}"
        model_path = model_dir / f"best_model_fold{fold_id}.pth"
        stgs = load_setting_file(fold_path / "settings.yml")
        stgs["globals"]["val_fold"] = fold_id 
        stgs["loader"]["val"]["batch_size"] = int(stgs["loader"]["val"]["batch_size"] * 1 / 2) 
        stgs["loader"]["val"]["num_workers"] = 2
        
        test_loader = torch.utils.data.DataLoader(test_dataset, **stgs["loader"]["val"])
        model = eval(stgs["model"]["name"])(**stgs["model"]["params"])
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        test_pred = inference_function(
            stgs, model, test_loader, device, APPLY_SOFTMAX, USE_HFLIP_TTA)
        test_preds_arr[fold_id] = test_pred
    
    test_pred_1dcnn = test_preds_arr.mean(axis=0)
    
    # # 2D-CNN
    print("[2D-CNN Stacking]")
    X = np.stack(test_pred_list, axis=2)[:, None, ...]
    model_dir = STACKING_MODELS[1]
    test_dataset = StackingDatasetForCNN(X)
    test_preds_arr = np.zeros((N_FOLD, len(sample_sub), N_CLASSES))
    
    for fold_id in FOLD_IDS:
        fold_path =  model_dir / f"fold{fold_id}"
        model_path = model_dir / f"best_model_fold{fold_id}.pth"
        stgs = load_setting_file(fold_path / "settings.yml")
        stgs["globals"]["val_fold"] = fold_id 
        stgs["loader"]["val"]["batch_size"] = int(stgs["loader"]["val"]["batch_size"] * 1 / 2) 
        stgs["loader"]["val"]["num_workers"] = 2
        
        test_loader = torch.utils.data.DataLoader(test_dataset, **stgs["loader"]["val"])
        model = eval(stgs["model"]["name"])(**stgs["model"]["params"])
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        test_pred = inference_function(
            stgs, model, test_loader, device, APPLY_SOFTMAX, USE_HFLIP_TTA)
        test_preds_arr[fold_id] = test_pred
    
    test_pred_2dcnn = test_preds_arr.mean(axis=0)
    
        
    # # ENSEMBLE
#     test_pred_ens = (test_pred_wopt + test_pred_1dcnn + test_pred_2dcnn) / 3
    test_pred_ens = (
        softmax(test_pred_wopt) + softmax(test_pred_1dcnn) + softmax(test_pred_2dcnn)) / 3
    
    sub = sample_sub.copy()
    sub["label"] = test_pred_ens.argmax(axis=1)
    
    sub.to_csv("submission.csv", index=False)
    
    
if __name__ == "__main__":
    main()
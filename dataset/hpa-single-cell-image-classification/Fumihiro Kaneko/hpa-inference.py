import os
import sys

on_kaggle_server = os.path.exists("/kaggle")
if on_kaggle_server:
    os.system("pip install ../input/mmdetection-v280/src/mmdet-2.8.0/mmdet-2.8.0/")
    os.system(
        "pip install ../input/mmdetection-v280/src/mmpycocotools-12.0.3/mmpycocotools-12.0.3/"
    )
    os.system(
        "pip install ../input/iterative-stratification/iterative-stratification-master"
    )
    os.system("pip install ../input/hpapytorchzoozip/pytorch_zoo-master")
    os.system("pip install ../input/hpacellsegmentatormaster/HPA-Cell-Segmentation-master")

#     os.system("pip install ../input/smp20210127/pretrained-models.pytorch-master/pretrained-models.pytorch-master")
#     os.system("pip install ../input/smp20210127/pytorch-image-models-master/pytorch-image-models-master")
#     os.system("pip install ../input/smp20210127/EfficientNet-PyTorch-master/EfficientNet-PyTorch-master")
#     os.system("pip install ../input/smp20210127/segmentation_models.pytorch-master/segmentation_models.pytorch-master")

    sys.path = [
        '../input/smp20210127/segmentation_models.pytorch-master/segmentation_models.pytorch-master/',
        '../input/smp20210127/EfficientNet-PyTorch-master/EfficientNet-PyTorch-master',
        '../input/smp20210127/pytorch-image-models-master/pytorch-image-models-master',
        '../input/smp20210127/pretrained-models.pytorch-master/pretrained-models.pytorch-master',
    ] + sys.path
    sys.path.insert(0, "../input/hpa-ws-repo/kaggle-hpa-single-cell-image-classification-main")

import argparse
import base64
import copy
import csv
import functools
import gc
import glob
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Text, Tuple, Union

import cv2
import hpacellseg.cellsegmentator as cellsegmentator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import yaml
from hpacellseg.utils import label_cell, label_nuclei
from pycocotools import _mask as coco_mask
from tqdm import tqdm

from src.dataset.datamodule import HpaDatamodule
from src.modeling.pl_model import (
    LitModel,
    get_cam_dir,
    get_cam_pred_path,
    load_trained_pl_model,
)

data_dir = "../input/hpa-single-cell-image-classification"
batch_size = 4
input_size = 2048
label_find_size = (512, 512)

PRED_THRESH = 0.01
NEGA_CLASS = 18
# hpa cellsegmentator tool weights
NUC_MODEL = "../input/hpacellsegmentatormodelweights/dpn_unet_nuclei_v1.pth"
CELL_MODEL = "../input/hpacellsegmentatormodelweights/dpn_unet_cell_3ch_v1.pth"

cell_area_thresh = 0.2 * 0.5
cell_h_area_thresh = 0.08 * 0.5
nuc_area_thresh = 0.2
nuc_h_area_thresh = 0.08
green_ch_thresh = 1.0
use_same_thresh = True
cam_thresh = 1.0e-6
high_cam_thresh = 0.75
min_mask_ratio = 0.01
default_bkg_score = 0.75

tta_mode = "scale"  # "skip", "flip", "scale", "split"
scales = [1.2, 1.4]


def get_dm_default_args() -> dict:
    dm_args = {
        "val_fold": 0,
        "aug_mode": 0,
        "num_inchannels": 4,
        "round_nb": 0,
        "sub_label_dir": None,
    }
    return dm_args


def load_ckpt_paths(ckpt_paths: List[Tuple[str, float, str]]) -> List[dict]:
    models = []
    for i, ckpt_ in enumerate(ckpt_paths):
        if isinstance(ckpt_, tuple):
            how_, w, path = ckpt_
        else:
            w = 1.0 / len(ckpt_paths)
            how_ = "both"

        if path.find(".ckpt") > -1:
            model, args_hparams = load_trained_pl_model(LitModel, path)
            model.cuda()
            model.eval()
            is_cuda = True
        else:
            print("use cached cam for inference:", path)
            assert os.path.isdir(path)
            psuedo_path = os.path.dirname(path) + "/checkponts/last.ckpt"
            model, args_hparams = load_trained_pl_model(
                LitModel, psuedo_path, only_load_yaml=True
            )
            is_cuda = False

        models.append(
            {
                "path": path,
                "model": model,
                "hparams": args_hparams,
                "how_join": how_,
                "weight": w,
                "is_cuda": is_cuda,
            }
        )
    return models


def create_cell_masks(im: np.ndarray, segmentator) -> tuple:
    """
    im: (batch, 4, H, W), rbgy image
    """
    # For nuclei List[np.ndarray(H, W)] blue
    nuc_input = [rgby_image[..., 2] for rgby_image in im]
    nuc_segmentations = segmentator.pred_nuclei(nuc_input)
    # For full cells
    # List[List[np.ndarray(H, W), r], List[np.ndaray(H,W), y], List[np.ndarray(H,W), b]]
    cell_input = [[rgby_image[..., j] for rgby_image in im] for j in [0, 3, 2]]
    cell_segmentations = segmentator.pred_cells(cell_input)
    batch_n_masks = []
    batch_c_masks = []
    # post-processing
    for i, pred in enumerate(cell_segmentations):
        nuclei_mask, cell_mask = label_cell(nuc_segmentations[i], cell_segmentations[i])
        batch_n_masks.append(nuclei_mask)
        batch_c_masks.append(cell_mask)
    return batch_n_masks, batch_c_masks


def cache_cell_masks(
    input_ids: List[str],
    batch_n_masks: List[np.ndarray],
    batch_c_masks: List[np.ndarray],
    save_dir: Path = Path("../input/hpa-mask"),
    make_resize_data: bool = False,
    im: Optional[np.ndarray] = None,
    image_size: int = 768,
):
    cell_dir = save_dir / "hpa_cell_mask"
    nucl_dir = save_dir / "hpa_nuclei_mask"
    cell_dir.mkdir(parents=True, exist_ok=True)
    nucl_dir.mkdir(parents=True, exist_ok=True)

    for i, input_id in enumerate(input_ids):
        np.savez_compressed(str(nucl_dir / f"{input_id}.npz"), batch_n_masks[i])
        np.savez_compressed(str(cell_dir / f"{input_id}.npz"), batch_c_masks[i])

    if make_resize_data and (im is not None):
        resize_dir = save_dir / "hpa_resize_ext"
        resize_dir.mkdir(exist_ok=True)
        for i, input_id in enumerate(input_ids):
            save_name = str(resize_dir / input_id)
            red = im[i, :, :, 0]
            red = cv2.resize(red, (image_size, image_size))
            cv2.imwrite(f"{save_name}_red.png", red)

            green = im[i, :, :, 1]
            green = cv2.resize(green, (image_size, image_size))
            cv2.imwrite(f"{save_name}_green.png", green)

            blue = im[i, :, :, 2]
            blue = cv2.resize(blue, (image_size, image_size))
            cv2.imwrite(f"{save_name}_blue.png", blue)

            yellow = im[i, :, :, 3]
            yellow = cv2.resize(yellow, (image_size, image_size))
            cv2.imwrite(f"{save_name}_yellow.png", yellow)

    return


def get_cell_masks(
    data: dict, im: np.ndarray, segmentator: Optional[torch.nn.Module], stage="test"
):
    if stage == "test":
        batch_n_masks, batch_c_masks = create_cell_masks(im, segmentator)
    elif stage == "gen_pseudo":
        if np.all(data["is_load"].numpy()):
            batch_n_masks, batch_c_masks = data["nucl_mask"], data["cell_mask"]
            batch_c_masks = [
                mask.squeeze()
                for mask in np.split(batch_c_masks.numpy(), batch_c_masks.shape[0])
            ]
            batch_n_masks = [
                mask.squeeze()
                for mask in np.split(batch_n_masks.numpy(), batch_n_masks.shape[0])
            ]
        else:
            batch_n_masks, batch_c_masks = create_cell_masks(im, segmentator)
            cache_cell_masks(
                input_ids=data["input_id"],
                batch_n_masks=batch_n_masks,
                batch_c_masks=batch_c_masks,
                make_resize_data=True,
                im=im,
            )
    else:
        raise NotImplementedError
    return batch_n_masks, batch_c_masks


def flip_tta(
    model: LitModel,
    data: dict,
    batch_idx: int,
    cam: torch.Tensor,
    pred: torch.Tensor,
):
    transforms = [
        torchvision.transforms.functional.hflip,
        torchvision.transforms.functional.vflip,
    ]
    inverts = [
        torchvision.transforms.functional.hflip,
        torchvision.transforms.functional.vflip,
    ]
    for trans_, invert_ in zip(transforms, inverts):
        tta_data = {"image": trans_(data["image"])}
        cam_f, pred_f = model.test_step(tta_data, batch_idx, save_npy=False)
        cam += invert_(cam_f)
        pred += pred_f

    pred *= 1.0 / (len(transforms) + 1)
    cam *= 1.0 / (len(transforms) + 1)
    return cam, pred


def check_tta_size(
    infer_size: int, scales: List[float], tta_mode: str
) -> Tuple[List[float], str]:
    if tta_mode == "skip":
        return scales, tta_mode

    if infer_size >= 1024:
        return [], "flip"
    if infer_size >= 768:
        if len(scales) > 0:
            return scales[:2], tta_mode
    return scales, tta_mode


def infer(
    model: LitModel,
    data: dict,
    batch_idx: int,
    args_hparams: dict,
    infer_size: int = 512,
    tta_mode: str = "flip",
    scales: List[float] = [1.2],
    scale_with_flip: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # normal infer
    orig_img = data["image"][:, : args_hparams["num_inchannels"]]
    infer_data = {
        "image": F.interpolate(
            orig_img, infer_size, mode="bilinear", align_corners=False
        )
    }

    cam, pred = model.test_step(infer_data, batch_idx, save_npy=False)
    scales, tta_mode = check_tta_size(
        infer_size=infer_size, scales=scales, tta_mode=tta_mode
    )
    if tta_mode == "skip":
        pass
    elif tta_mode == "flip":
        cam, pred = flip_tta(
            model=model, batch_idx=batch_idx, data=infer_data, cam=cam, pred=pred
        )

    elif tta_mode == "scale":
        tta_sizes = [int(infer_size * scale) for scale in scales]
        if scale_with_flip:
            cam, pred = flip_tta(
                model=model, batch_idx=batch_idx, data=infer_data, cam=cam, pred=pred
            )
        cam_preds = [cam]
        for tta_size in tta_sizes:
            tta_data = {
                "image": F.interpolate(
                    orig_img, tta_size, mode="bilinear", align_corners=False
                )
            }
            cam_s, pred_s = model.test_step(tta_data, batch_idx, save_npy=False)
            if scale_with_flip:
                cam_s, pred_s = flip_tta(
                    model=model,
                    batch_idx=batch_idx,
                    data=tta_data,
                    cam=cam_s,
                    pred=pred_s,
                )
            cam_preds.append(cam_s)
            pred += pred_s
        cam_size = np.max([cam_.shape[-1] for cam_ in cam_preds])
        for i, cam_ in enumerate(cam_preds):

            cam_preds[i] = F.interpolate(
                cam_, cam_size, mode="bilinear", align_corners=False
            )

        pred *= 1.0 / (len(tta_sizes) + 1)
        # cam *= 1.0 / (len(tta_sizes) + 1)
        cam = torch.mean(torch.stack(cam_preds), dim=0)

    elif tta_mode == "split":
        pass

    return cam, pred


def get_class_mask(
    data: dict,
    batch_idx: int,
    args_hparams: dict,
    model: LitModel,
    infer_size: int = 512,
    pred_thresh: float = 0.5,
    label_find_size: Optional[tuple] = (512, 512),
    stage: str = "test",
    mode: str = "cam",
    tta_mode: str = "flip",
    scales: List[float] = [1.2],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # adjust input shape for cam
    tta_mode = "flip" if (mode == "segm") & (tta_mode == "scale") else tta_mode
    cam, pred = infer(
        model=model,
        data=data,
        args_hparams=args_hparams,
        batch_idx=batch_idx,
        infer_size=infer_size,
        tta_mode=tta_mode,
        scales=scales,
    )
    if stage == "test":
        if mode == "segm":
            pred = cam.reshape(cam.shape[0], cam.shape[1], -1)
            pred = torch.where(pred[:, :-1] > pred_thresh, 1.0, 0.0).max(dim=-1)[0]

    elif stage == "gen_pseudo":
        pred = data["target"].cuda()

    if mode == "cam":
        cam_pred = model.convert_cam_to_mask(
            cam.clone(), pred >= pred_thresh, orig_img_size=label_find_size
        )
    elif mode == "segm":
        # remove bkg class
        cam_pred = cam[:, :-1]
        cam_pred = F.interpolate(
            cam_pred, label_find_size, mode="bilinear", align_corners=False
        )
    else:
        raise NotImplementedError

    return cam_pred, pred


def process_ensemble(
    results: List[Dict[str, torch.Tensor]],
    how_join: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1.0e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    pred = torch.zeros_like(results[0]["pred"])
    cam_pred = torch.zeros_like(results[0]["cam_pred"])
    if weights is not None:
        assert np.sum(weights) - 1.0 < eps, f"weight = {np.sum(weights)}, {weights}"
        assert weights.shape[0] == len(results)
    else:
        weights = np.ones((len(results),), dtype=np.float32) / len(results)

    assert how_join.shape[0] == len(results)

    unused_w_pred = 0.0
    unused_w_cam = 0.0
    for i, res in enumerate(results):
        w = weights[i]
        how_ = how_join[i]
        if how_ == "both":
            cam_pred += w * res["cam_pred"]
            pred += w * res["pred"]
        elif how_ == "cam":
            cam_pred += w * res["cam_pred"]
            unused_w_pred += w
        elif how_ == "pred":
            pred += w * res["pred"]
            unused_w_cam += w
        else:
            raise NotImplementedError
    cam_pred *= 1.0 / (1.0 - unused_w_cam)
    pred *= 1.0 / (1.0 - unused_w_pred)
    return cam_pred.cpu().numpy(), pred.cpu().numpy()


def vis_masks(im, batch_n_masks, batch_c_masks, on_kaggle_server=False, ind: int = 0):
    # mt :red
    # er : yellow
    # nu : blue
    # images = [mt, er, nu]

    fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    for i in range(3):
        microtubule = im[i][..., 0]
        endoplasmicrec = im[i][..., 3]
        nuclei = im[i][..., 2]
        mask = batch_c_masks[i]
        img = np.dstack((microtubule, endoplasmicrec, nuclei))
        ax[i].imshow(img)
        ax[i].imshow(mask, alpha=0.7)
        ax[i].axis("off")
    if on_kaggle_server:
        plt.savefig(f"./hpa_cell_{ind}.png")
    else:
        plt.show()
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))

    for i in range(3):
        microtubule = im[i][..., 0]
        endoplasmicrec = im[i][..., 3]
        nuclei = im[i][..., 2]
        mask = batch_n_masks[i]
        img = np.dstack((microtubule, endoplasmicrec, nuclei))
        ax[i].imshow(img)
        ax[i].imshow(mask, alpha=0.7)
        ax[i].axis("off")
    if on_kaggle_server:
        plt.savefig(f"./hpa_nuc_{ind}.png")
    else:
        plt.show()
    plt.close()


def encode_binary_mask(mask: np.ndarray) -> Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s"
            % mask.dtype
        )

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" % mask.shape
        )

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)

    return base64_str.decode()


def decode_ascii_mask(
    base64_str: Text, w_size: int = 2048, h_size: int = 2048, is_numpy: bool = False
) -> dict:
    # check input mask --
    if type(base64_str) != str:
        raise ValueError(
            "decode_ascii_mask, expects a str, received dtype == %s" % type(base64_str)
        )
    base64_str = base64_str.encode()

    # base64 decoding and decompress
    binary_str = base64.b64decode(base64_str)
    encoded_mask = zlib.decompress(binary_str)

    # RLE decode mask --
    rle = [{"size": [h_size, w_size], "counts": encoded_mask}]
    mask_to_encode = coco_mask.decode(rle)

    if is_numpy:
        mask_to_encode = np.ascontiguousarray(mask_to_encode)
    else:
        mask_to_encode = np.empty(0)
    return {"mask": mask_to_encode, "rle": rle[0]}


def calc_conf(
    target_cam: np.ndarray, pred: np.ndarray, cam_rate: np.ndarray, how="max"
) -> float:
    if how == "max":
        cnf = pred * np.max(target_cam)
    elif how == "mean":
        cnf = pred * np.sum(target_cam) / np.sum(target_cam > 0)
    elif how == "cam_rate":
        cnf = pred * np.max(target_cam) * cam_rate
    elif how == "cam_rate_mean":
        cnf = pred * np.sum(target_cam) / np.sum(target_cam > 0) * cam_rate
    return cnf


def calc_cam_rate_cond(
    target_cam: np.ndarray,
    target_mask: np.ndarray,
    target_nuclei: np.ndarray,
    mask_area: int,
    nuclei_area: int,
    cell_area_thresh: float = 0.2,
    cell_h_area_thresh: float = 0.08,
    nuc_area_thresh: float = 0.2,
    nuc_h_area_thresh: float = 0.08,
    high_cam_thresh: float = 0.75,
    min_mask_ratio: float = 0.01,
    use_same_thresh: bool = True,
):
    cam_rate = np.sum(target_cam > 0) / mask_area
    cam_h_rate = np.sum(target_cam >= high_cam_thresh) / mask_area

    nuc_cam_rate_cond = False
    if nuclei_area >= (target_cam.shape[1] * min_mask_ratio) ** 2:
        if use_same_thresh:
            cam_rate = max(
                cam_rate,
                np.sum((target_cam * target_nuclei) > 0) / nuclei_area,
            )
            cam_h_rate = max(
                cam_h_rate,
                np.sum((target_cam * target_nuclei) >= high_cam_thresh) / nuclei_area,
            )
        else:
            nuc_cam_rate = np.sum((target_cam * target_nuclei) > 0) / nuclei_area
            nuc_cam_h_rate = (
                np.sum((target_cam * target_nuclei) >= high_cam_thresh) / nuclei_area
            )
            nuc_cam_rate_cond = (nuc_cam_rate >= nuc_area_thresh) or (
                nuc_cam_h_rate >= nuc_h_area_thresh
            )

    cam_rate_cond = (
        (cam_rate >= cell_area_thresh)
        or (cam_h_rate >= cell_h_area_thresh)
        or nuc_cam_rate_cond
    )
    return cam_rate_cond, cam_rate


def reduce_label_size(
    target_mask: np.ndarray,
    target_nuclei: np.ndarray,
    green_ch: Optional[np.ndarray],
    image_size: tuple,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # target_mask = cv2.resize(target_mask.astype(np.uint8), image_size).astype(np.bool)
    target_mask = cv2.resize(
        target_mask.astype(np.uint8), image_size, interpolation=cv2.INTER_NEAREST
    )
    target_nuclei = cv2.resize(
        target_nuclei.astype(np.uint8), image_size, interpolation=cv2.INTER_NEAREST
    ).astype(np.bool)
    if green_ch is not None:
        green_ch = cv2.resize(green_ch.astype(np.uint8), image_size)
    else:
        green_ch = np.empty(0)
    return target_mask, target_nuclei, green_ch


def check_bkg_score(
    pred_ins: List[str],
    is_bkg: bool,
    low_green: bool,
    class_ids: List[int],
    cnfs: List[float],
):
    if is_bkg or low_green:
        pass
    elif NEGA_CLASS in class_ids:
        max_pred = np.argmax(cnfs)
        max_class = class_ids[max_pred]
        if max_class == NEGA_CLASS:
            # use only bkg class
            pred_ins = [pred_ins[-1]]
        else:
            # remove bkg class
            pred_ins = pred_ins[:-1]
    return pred_ins


def find_cell_labels(
    rle: Text,
    cam_pred: np.ndarray,
    target_mask: np.ndarray,
    target_nuclei: np.ndarray,
    pred: np.ndarray,
    green_ch: np.ndarray = np.empty(0),
    green_ch_thresh: float = 2.0,
    skip_bkg_check: bool = True,
    pred_thresh: float = 0.5,
    cell_area_thresh: float = 0.2,
    cell_h_area_thresh: float = 0.08,
    nuc_area_thresh: float = 0.2,
    nuc_h_area_thresh: float = 0.08,
    use_same_thresh: bool = True,
    high_cam_thresh: float = 0.75,
    min_mask_ratio: float = 0.01,
    conf_how: str = "max",
    default_bkg_score: float = 0.8,
) -> List[str]:
    # class prediction
    assert np.all(cam_pred.shape[0:1] == pred.shape)
    assert np.all(cam_pred.shape[1:3] == target_mask.shape[1:3])
    assert np.all(cam_pred.shape[1:3] == target_nuclei.shape[1:3])
    assert np.all(cam_pred.shape[1:3] == green_ch.shape[1:3])
    assert green_ch.dtype == np.uint8

    pred_ins: List[str] = []
    mask_area = np.sum(target_mask)
    nuclei_area = np.sum(target_nuclei)

    if mask_area <= (cam_pred.shape[1] * min_mask_ratio) ** 2:
        return pred_ins

    bkg_score = []
    target_cam = cam_pred * target_mask

    target_green = green_ch * target_mask

    low_green = target_green.sum() / mask_area < green_ch_thresh
    is_bkg = (np.argmax(pred) == NEGA_CLASS) and (pred[NEGA_CLASS] > 0)
    if skip_bkg_check:
        is_bkg = False

    class_ids = []
    cnfs = []

    for class_id in np.where(pred >= pred_thresh)[0]:
        cam_rate_cond, cam_rate = calc_cam_rate_cond(
            target_cam=target_cam[class_id],
            target_mask=target_mask[0],
            target_nuclei=target_nuclei[0],
            mask_area=mask_area,
            nuclei_area=nuclei_area,
            cell_area_thresh=cell_area_thresh,
            cell_h_area_thresh=cell_h_area_thresh,
            nuc_area_thresh=nuc_area_thresh,
            nuc_h_area_thresh=nuc_h_area_thresh,
            high_cam_thresh=high_cam_thresh,
            min_mask_ratio=min_mask_ratio,
            use_same_thresh=use_same_thresh,
        )
        if cam_rate_cond and (not is_bkg) and (not low_green):
            cnf = calc_conf(
                target_cam=target_cam[class_id],
                pred=pred[class_id],
                cam_rate=cam_rate,
                how=conf_how,
            )
            pred_ins.append(f"{class_id} {cnf} {rle}")
            cnfs.append(cnf)
            class_ids.append(class_id)
        else:
            bkg_score.append(1.0 - cam_rate)
    if not skip_bkg_check:
        pred_ins = check_bkg_score(
            pred_ins=pred_ins,
            is_bkg=is_bkg,
            low_green=low_green,
            class_ids=class_ids,
            cnfs=cnfs,
        )
    if len(pred_ins) == 0:
        class_id = NEGA_CLASS
        cnf = default_bkg_score
        if not np.all(bkg_score == 0) and (not is_bkg):
            nweight = (
                np.median(bkg_score) - (1.0 - cell_area_thresh)
            ) / cell_area_thresh
            nweight = max(1.0e-6, nweight)
            cnf *= nweight
        pred_ins.append(f"{class_id} {cnf} {rle}")
    return pred_ins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run infer for hpa ws",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--yaml_path",
        default="../input/hpa-ws-repo/kaggle-hpa-single-cell-image-classification-main/src/config/kaggle_submission.yaml",
        type=str,
        help="run config path",
    )
    parser.add_argument(
        "--sub_path",
        type=str,
        default="./submission.csv",
        help="path for resutl csv",
    )
    parser.add_argument(
        "--num_workers",
        default="4",
        type=int,
        help="number of cpus for DataLoader",
    )
    parser.add_argument(
        "--para_num",
        default=None,
        type=int,
        help="number of parallel processings at psuedo label generation",
    )
    parser.add_argument(
        "--para_ind",
        default="0",
        type=int,
        help=" parallel run index",
    )
    args = parser.parse_args()
    sub_path = args.sub_path
    print("use submission name:", args.sub_path)

    with open(args.yaml_path) as f:
        configs = yaml.load(f)

    stage = configs["stage"]
    conf_how = configs["conf_how"]
    mask_dir = configs["mask_dir"]
    skip_bkg_check = configs["skip_bkg_check"]
    use_ext_data = configs["use_ext_data"]
    ext_data_mode = configs["ext_data_mode"]
    is_debug = configs["is_debug"]
    ckpt_paths = configs["ckpt_paths"]

    df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    if len(df) == 559:
        debug_num = 5
    else:
        is_debug = False

    datamodule = HpaDatamodule(
        data_dir=data_dir,
        batch_size=batch_size,
        is_debug=is_debug,
        input_size=input_size,
        mask_dir=mask_dir,
        num_workers=args.num_workers,
        use_ext_data=use_ext_data,
        ext_data_mode=ext_data_mode,
        para_num=args.para_num,
        para_ind=args.para_ind,
        **get_dm_default_args(),
    )

    datamodule.prepare_data()
    datamodule.setup(stage=stage)

    test_dataloader = datamodule.test_dataloader()
    segmentator = cellsegmentator.CellSegmentator(
        NUC_MODEL,
        CELL_MODEL,
        scale_factor=0.25,
        device="cuda",
        padding=True,
        multi_channel_model=True,
    )

    models = load_ckpt_paths(ckpt_paths=ckpt_paths)
    gc.collect()
    with torch.no_grad():
        pred_rows = [["ID", "ImageWidth", "ImageHeight", "PredictionString"]]
        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            if is_debug:
                if i >= debug_num:
                    break

            im = data["image"].numpy().transpose(0, 2, 3, 1)
            im = im * np.array(datamodule.img_std) + np.array(datamodule.img_mean)
            im = (im * 255).astype(np.uint8)
            green_batch = im[..., 1]

            batch_n_masks, batch_c_masks = get_cell_masks(
                data=data, im=im, segmentator=segmentator, stage=stage
            )

            if is_debug:
                # pass
                # Visualizing the segmentation masks we just predicted above
                vis_masks(
                    im,
                    batch_n_masks,
                    batch_c_masks,
                    on_kaggle_server=on_kaggle_server,
                    ind=i,
                )

            is_any_cuda = np.any([model["is_cuda"] for model in models])
            if is_any_cuda:
                data["image"] = data["image"].cuda()
            results = []
            weights = np.zeros((len(models),), dtype=np.float32)
            how_join = np.zeros((len(models),), dtype=np.object)
            for i, model_dict in enumerate(models):
                mode = (
                    "segm"
                    if model_dict["hparams"]["segm_label_dir"] is not None
                    else "cam"
                )
                if isinstance(model_dict["model"], LitModel):
                    cam_pred, pred = get_class_mask(
                        data,
                        batch_idx=i,
                        args_hparams=model_dict["hparams"],
                        model=model_dict["model"],
                        infer_size=model_dict["hparams"]["input_size"],
                        pred_thresh=PRED_THRESH,
                        label_find_size=label_find_size,
                        stage=stage,
                        mode=mode,
                        tta_mode=tta_mode,
                        scales=scales,
                    )
                else:
                    cam_dir = model_dict["path"]
                    cam_preds = []
                    preds = []
                    for input_id in data["input_id"]:
                        cam_path, pred_path = get_cam_pred_path(cam_dir, input_id)
                        cam_preds.append(np.load(str(cam_path)))
                        preds.append(np.load(str(pred_path)))

                    cam_pred = torch.from_numpy(np.array(cam_preds))
                    pred = torch.from_numpy(np.array(preds))
                    if is_any_cuda:
                        cam_pred = cam_pred.cuda()
                        pred = pred.cuda()
                    cam_pred = F.interpolate(
                        cam_pred, label_find_size, mode="bilinear", align_corners=False
                    )

                results.append({"cam_pred": cam_pred, "pred": pred})
                weights[i] = model_dict["weight"]
                how_join[i] = model_dict["how_join"]

            cam_pred, pred = process_ensemble(
                results=results, how_join=how_join, weights=weights
            )

            if is_debug:
                num_ = 4
                inputs = F.interpolate(
                    torch.Tensor(im.transpose(0, 3, 1, 2)), label_find_size
                )
                inputs = inputs.numpy().transpose(0, 2, 3, 1)[..., :3].astype(np.uint8)

                _ = plt.figure(figsize=(10 * num_, 10))
                out_pred = LitModel.overlay_cam_on_input(
                    inputs=inputs,
                    cam_mask=cam_pred,
                    targets=pred,
                    batch_num=cam_pred.shape[0],
                    stack_axis=1,
                    threshold=PRED_THRESH,
                )
                inputs = np.concatenate(inputs, axis=1)
                out = np.concatenate((inputs, out_pred), axis=0)
                plt.imshow(out)
                plt.axis("off")
                if on_kaggle_server:
                    plt.savefig(f"./cam_{i}.png")
                else:
                    plt.show()
                plt.close()

            w_size_batch = data["w_size"].numpy()
            h_size_batch = data["h_size"].numpy()
            cam_pred = np.where(cam_pred <= cam_thresh, 0.0, cam_pred)
            for ba_ind, cell_mask in enumerate(batch_c_masks):
                input_id = data["input_id"][ba_ind]
                w_size, h_size = w_size_batch[ba_ind], h_size_batch[ba_ind]

                pred_strs: List[str] = []
                green_ch = green_batch[ba_ind]
                cell_mask, nuclei_mask, green_ch = reduce_label_size(
                    cell_mask, batch_n_masks[ba_ind], green_ch, label_find_size
                )
                for ins_id in np.arange(1, cell_mask.max() + 1):
                    # instance mask generation
                    target_mask = cell_mask == ins_id
                    target_nuclei = nuclei_mask * target_mask
                    if np.all(target_mask is False):
                        continue
                    rle = encode_binary_mask(
                        cv2.resize(
                            target_mask.astype(np.uint8), (w_size, h_size)
                        ).astype(np.bool)
                    )
                    # class prediction for a cell
                    pred_ins = find_cell_labels(
                        rle=rle,
                        cam_pred=cam_pred[ba_ind],
                        target_mask=target_mask[np.newaxis],
                        target_nuclei=target_nuclei[np.newaxis],
                        green_ch=green_ch[np.newaxis],
                        green_ch_thresh=green_ch_thresh,
                        skip_bkg_check=skip_bkg_check,
                        pred=pred[ba_ind],
                        pred_thresh=PRED_THRESH,
                        cell_area_thresh=cell_area_thresh,
                        cell_h_area_thresh=cell_h_area_thresh,
                        nuc_area_thresh=nuc_area_thresh,
                        nuc_h_area_thresh=nuc_h_area_thresh,
                        high_cam_thresh=high_cam_thresh,
                        min_mask_ratio=min_mask_ratio,
                        conf_how=conf_how,
                        use_same_thresh=use_same_thresh,
                        default_bkg_score=default_bkg_score,
                    )
                    pred_strs.extend(pred_ins)
                pred_rows.append([input_id, w_size, h_size, " ".join(pred_strs)])

    with open(sub_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(pred_rows)
    print("save submission file at", sub_path)

    if is_debug:
        pred_df = pd.read_csv(sub_path)
        print(pred_df.head())

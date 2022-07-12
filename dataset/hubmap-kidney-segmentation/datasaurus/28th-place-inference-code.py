import torch
import pandas as pd
import numpy as np
import numba
import yaml
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import dice_score
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from torchvision.transforms.functional import resize
from scipy import ndimage
from pathlib import Path
import getpass
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

KERNEL = False if getpass.getuser() == "anjum" else True

if not KERNEL:
    INPUT_PATH = Path("/mnt/storage_dimm2/kaggle_data/hubmap-kidney-segmentation/")
    OUTPUT_PATH = Path("/mnt/storage/kaggle_output/hubmap-kidney-segmentation/upload")
else:
    INPUT_PATH = Path("../input/hubmap-kidney-segmentation")
    OUTPUT_PATH = Path("../input/hubmap-pl-models")
    import subprocess

    whls = [
        "../input/segmentation-models-wheels/efficientnet_pytorch-0.6.3-py3-none-any.whl",
        "../input/segmentation-models-wheels/pretrainedmodels-0.7.4-py3-none-any.whl",
        "../input/timm-pytorch-image-models/pytorch-image-models-master",
        "../input/smp-nightly/segmentation_models.pytorch-master",
        "../input/ttach-pytorch",
    ]

    for w in whls:
        subprocess.call(["pip", "install", w, "--no-deps"])

# import timm
import ttach as tta
import segmentation_models_pytorch as smp

secondary_losses = {
    "dice": smp.losses.DiceLoss("binary", smooth=1),
    "focal": smp.losses.FocalLoss("binary"),
    "lovasz": smp.losses.LovaszLoss("binary"),
    "jaccard": smp.losses.JaccardLoss("binary"),
    # "sym_lovasz": symmetric_lovasz_hinge,
}


class HubmapUNet(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Unet",
        encoder: str = "resnet34",
        lr: float = 0.01,
        smoothing: float = 0,
        weight_decay: float = 0,
        swa: bool = False,
        swa_lr: float = 0.5,
        swa_start: int = 25,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,  # Typically 1
        loss_mix: float = 1.0,
        second_loss: str = "lovasz",
        full_eval: bool = False,
        use_edges: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.img_size_model = kwargs["img_size_model"]
        self.img_size = kwargs["img_size"]
        classes = 2 if use_edges else 1

        seg_model = getattr(smp, arch)
        self.net = seg_model(encoder_name=self.hparams.encoder, classes=classes, encoder_weights=None, interpolation_type="bilinear")

        self.example_input_array = torch.ones(1, 3, self.img_size, self.img_size)
        self.critereon1 = smp.losses.FocalLoss("binary")
        self.critereon2 = secondary_losses[second_loss]

        # For full mask validation
        self.df = pd.read_csv(INPUT_PATH / "train.csv", index_col=0)

    def loss_fn(self, y_pred, y_true):
        bce = self.critereon1(y_pred, y_true)
        second_loss = self.critereon2(y_pred, y_true)
        return self.hparams.loss_mix * bce + (1 - self.hparams.loss_mix) * second_loss

    # https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py#L119-L138
    def mixup_data(self, x, y, alpha=1.0):
        """Returns mixed inputs, pairs of targets, and lambda"""
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixed_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss_fn(pred, y_a) + (1 - lam) * self.loss_fn(pred, y_b)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img = batch["image"].float()
        mask = batch["mask"].float()

        if self.hparams.mixup_alpha > 0:
            img, mask_a, mask_b, lam = self.mixup_data(
                img, mask, self.hparams.mixup_alpha
            )
            logits = self(img)
            loss = self.mixed_criterion(logits, mask_a, mask_b, lam)
        else:
            logits = self(img)
            loss = self.loss_fn(logits, mask)

        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("loss/train", avg_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        img = batch["image"].float()
        mask = batch["mask"].float()

        # mask = torch.nn.functional.interpolate(mask, size=self.hparams.img_size_model)

        out = self.forward(img)
        loss = self.loss_fn(out, mask)

        if self.hparams.use_edges:
            out = out[:, 0]
            mask = mask[:, 0]

        metric = dice_score(torch.squeeze(out), torch.squeeze(mask))

        out = {
            "val_loss": loss,
            "metric": metric,
        }

        if self.hparams.full_eval:
            added_out = {
                "y_pred": out.sigmoid(),
                "coords": batch["coords"],
                "img_num": batch["img_num"],
            }
            out.update(added_out)

        return out

    def validation_epoch_end(self, outputs):

        outputs = self.all_gather(outputs)

        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        metric = torch.stack([x["metric"] for x in outputs]).mean()

        if self.hparams.full_eval:
            coords = torch.cat([x["coords"] for x in outputs], dim=1)
            img_num = torch.cat([x["img_num"] for x in outputs], dim=1)
            y_pred = torch.cat([x["y_pred"] for x in outputs], dim=1)

            # Concat the world_size dim after all_gather
            coords = torch.cat([x for x in coords]).cpu().numpy()
            img_num = torch.cat([x for x in img_num]).cpu().numpy()
            # y_pred = torch.cat([x for x in y_pred]).cpu().float().squeeze().numpy()
            y_pred = torch.cat([x for x in y_pred]).cpu().float()
            img_name = [num2img[x] for x in img_num]

            tile_size = self.hparams.img_size
            pred_masks = {}

            for img in list(set(img_name)):
                pred_masks[img] = np.zeros(img_dims[img], dtype=np.uint8)

            for tile_pred, coord, name in zip(y_pred, coords, img_name):
                x1, x2, y1, y2 = coord
                y_pred_mask = (
                    torch.nn.functional.interpolate(tile_pred[(None,)], size=tile_size)
                    > 0.5
                )
                pred_masks[name][x1:x2, y1:y2] = y_pred_mask.squeeze().numpy()

            scores = []
            for img, prediction in pred_masks.items():
                target = rle_decode(self.df.loc[img, "encoding"], img_dims[img])
                scores.append(dice_score_numpy(prediction, target))

                # Log images
                prediction = cv2.resize(prediction, (512, 512))

            dice = torch.tensor(np.mean(scores)).type_as(metric)
            self.log("metric", dice)

        self.log_dict(
            {"loss/valid": loss_val, "metric_micro": metric},
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return [opt], [sch]

    def test_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        if self.hparams.swa:
            out = self.forward_swa(img)
        else:
            out = self.forward(img)
        loss = self.loss_fn(out, mask.float())
        out_probs = torch.cat([1 - torch.sigmoid(out), torch.sigmoid(out)], 1)
        metric = dice_score(out_probs, mask)
        return {"test_loss": loss, "test_metric": metric}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        metric = torch.stack([x["test_metric"] for x in outputs]).mean()
        log_dict = {"test_loss": loss, "test_metric": metric}

        return {
            "test_loss": log_dict["test_loss"],
            "test_metric": log_dict["test_metric"],
        }
 
    
@numba.njit()
def rle_numba(pixels):
    size = len(pixels)
    points = []
    if pixels[0] == 1:
        points.append(0)
    flag = True
    for i in range(1, size):
        if pixels[i] != pixels[i - 1]:
            if flag:
                points.append(i + 1)
                flag = False
            else:
                points.append(i + 1 - points[-1])
                flag = True
    if pixels[-1] == 1:
        points.append(size - points[-1] + 1)
    return points


def rle_numba_encode(image):
    pixels = image.flatten(order="F")
    points = rle_numba(pixels)
    return " ".join(str(x) for x in points)


def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2 
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


def get_transform(img_size_model):
    trfm = A.Compose(
        [
            A.Resize(img_size_model, img_size_model, interpolation=cv2.INTER_AREA),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return trfm


def infer_single_img(img_path, overlaps=[256], device="cuda"):
    
    # Load models and configs key=checkpoint, value= ensemble weight
    checkpoints = {"20210504-094231": 1.0}
    models, cfgs, ensemble_weights = [], [], []
    for ckpt, weight in checkpoints.items():
        for mpath in (OUTPUT_PATH).glob(f"{ckpt}/*/*/*dice.ckpt"):
            models.append(HubmapUNet.load_from_checkpoint(str(mpath)))
            ensemble_weights.append(weight)
            print(mpath)

        with open(str(mpath.parent / "hparams.yaml"), "r") as ymlfile:
            cfgs.append(yaml.load(ymlfile, Loader=yaml.FullLoader))

    print(len(models), "models")
    
    cfg = cfgs[0]  # TODO what if models have different image sizes?

    dataset = rasterio.open(img_path.as_posix())

    pred_mask = torch.zeros(dataset.shape, dtype=torch.float16).to(device)
    weights = torch.zeros(dataset.shape, dtype=torch.uint8).to(device)
    
    for overlap in overlaps:
        slices = make_grid(dataset.shape, window=cfg["img_size"], min_overlap=overlap)            

        trfm = get_transform(cfg["img_size_model"])

        for (x1, x2, y1, y2) in tqdm(slices, desc=img_path.stem):
            if dataset.count == 3:
                image = dataset.read(
                    [1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2))
                )
                image = np.moveaxis(image, 0, -1)
            elif dataset.count == 1:
                image = []
                for i, subdataset in enumerate(dataset.subdatasets):
                    with rasterio.open(subdataset) as layer:
                        image_channel = layer.read(
                            [1], window=Window.from_slices((x1, x2), (y1, y2))
                        )
                        image.append(image_channel)
                image = np.concatenate(image, 0)
                image = np.moveaxis(image, 0, -1)
            else:
                print("Image is not RGB or Greyscale", dataset.indexes)
                raise Exception

            image = trfm(image=image)["image"]  # Albumentations

            predictions = 0

            for model, weight in zip(models, ensemble_weights):
                model.to(device)
                model.eval()
                model = tta.SegmentationTTAWrapper(
                    model, tta.aliases.flip_transform(), merge_mode="mean"
                )

                with torch.no_grad():
                    img = image.to(device)[None]
                    predictions += model(img) * weight
                    weights[x1:x2, y1:y2] += 1

            predictions = resize(predictions[:, :1], (x2 - x1, y2 - y1))
            pred_mask[x1:x2, y1:y2] += torch.squeeze(predictions)
            
    bool_mask = (pred_mask / weights).sigmoid() > 0.50
    bool_mask = bool_mask.cpu().numpy()
    return bool_mask


def submission_low_memory():   
    # Process 1 image at a time due to RAM limitations
    sub = []
    for impath in (INPUT_PATH / "test").glob("*.tiff"):
        mask = infer_single_img(impath, overlap=32)
        sub.append({"id": impath.stem, "predicted": rle_numba_encode(mask)})
        del mask

    sub = pd.DataFrame(sub)
    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    submission_low_memory()


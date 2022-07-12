# %% [code]
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import getpass

KERNEL = False if getpass.getuser() == "anjum" else True

if not KERNEL:
    INPUT_PATH = Path("/mnt/storage/kaggle_data/osic-pulmonary-fibrosis-progression")
else:
    INPUT_PATH = Path("../input/osic-pulmonary-fibrosis-progression")
    PL_PATH = Path("/kaggle/input/pytorch-lightning")

    import subprocess
    import sys

    subprocess.call(
        [
            "pip",
            "install",
            PL_PATH / "pytorch_lightning-0.9.0-py3-none-any.whl",
            "--no-dependencies",
        ]
    )
    sys.argv = [""]


import pytorch_lightning as pl


class LaplaceLogLikelihood(nn.Module):
    def __init__(self):
        super(LaplaceLogLikelihood, self).__init__()
        self.l1_loss = nn.SmoothL1Loss(reduction="none")
        self.lrelu = nn.RReLU()
        self.root2 = torch.sqrt(torch.tensor(2, dtype=torch.float, requires_grad=False))

    def forward(self, predictions, target, clamp=False):
        delta = self.l1_loss(predictions[:, 0], target)

        if clamp:
            delta = torch.clamp(delta, max=1000)
            sigma = torch.clamp(predictions[:, 1], min=70)
        # clip sigma without destroying gradient
        else:
            sigma = self.lrelu(predictions[:, 1] - 70) + 70

        laplace_ll = -(self.root2 * delta) / sigma - torch.log(self.root2 * sigma)
        laplace_ll_mean = torch.mean(laplace_ll)
        return -laplace_ll_mean


class FullyConnected(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        batch_size: int = 32,
        weight_decay=0.01,
        lr_patience: int = 3,
        lr_factor: float = 0.5,
        fold: str = "fold_1",
        n_folds: int = 4,
        **kwargs,
    ):
        super(FullyConnected, self).__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.fold = fold
        self.n_folds = n_folds

        self.features = [
            "base_FVC",
            "base_Percent",
            "base_Age",
            "Week_passed",
            "Sex",
            "SmokingStatus",
        ]

        self.net = nn.Sequential(
            nn.Linear(len(self.features), 512),
            nn.SELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.SELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

        self.criterion = LaplaceLogLikelihood()

    def forward(self, x):
        return self.net(x)

    def prepare_data(self):
        print("Preparing data...")
        train_df = pd.read_csv(INPUT_PATH / "train.csv")
        train_df.sort_values(by=["Patient", "Weeks"], inplace=True)

        train_df["Patient_Week"] = (
            train_df["Patient"].astype(str) + "_" + train_df["Weeks"].astype(str)
        )

        output = pd.DataFrame()
        patient_gb = train_df.groupby("Patient")

        for patient, usr_df in tqdm(patient_gb, total=len(patient_gb)):
            usr_output = pd.DataFrame()
            for week, tmp in usr_df.groupby("Weeks"):
                rename_cols = {
                    "Weeks": "base_Week",
                    "FVC": "base_FVC",
                    "Percent": "base_Percent",
                    "Age": "base_Age",
                }
                tmp = tmp.drop(columns="Patient_Week").rename(columns=rename_cols)
                drop_cols = ["Age", "Sex", "SmokingStatus", "Percent"]
                _usr_output = (
                    usr_df.drop(columns=drop_cols)
                    .rename(columns={"Weeks": "predict_Week"})
                    .merge(tmp, on="Patient")
                )
                _usr_output["Week_passed"] = (
                    _usr_output["predict_Week"] - _usr_output["base_Week"]
                )
                usr_output = pd.concat([usr_output, _usr_output])
            output = pd.concat([output, usr_output])

        train = output[output["Week_passed"] != 0].reset_index(drop=True)
        train.sort_values(by=["Patient", "predict_Week"], inplace=True)
        train.reset_index(drop=True, inplace=True)

        # Set folds
        train["fold"] = 0
        kfold = GroupKFold(n_splits=self.n_folds)

        for fold_n, (trn_idx, val_idx) in enumerate(
            kfold.split(train, train["FVC"], train["Patient"])
        ):
            train.loc[val_idx, "fold"] = fold_n

        # Flag final 3 FVC assessments
        final3 = train.groupby("Patient").tail(3)  # The final 3 FVC assessments
        train["final3"] = False
        train.loc[final3.index, "final3"] = True

        # Prepare test
        test_df = pd.read_csv(INPUT_PATH / "test.csv").rename(
            columns={
                "Weeks": "base_Week",
                "FVC": "base_FVC",
                "Percent": "base_Percent",
                "Age": "base_Age",
            }
        )
        submission = pd.read_csv(INPUT_PATH / "sample_submission.csv")
        submission["Patient"] = submission["Patient_Week"].apply(
            lambda x: x.split("_")[0]
        )
        submission["predict_Week"] = (
            submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
        )
        test = submission.drop(columns=["FVC", "Confidence"]).merge(
            test_df, on="Patient"
        )
        test["Week_passed"] = test["predict_Week"] - test["base_Week"]

        # Encode and scale
        cat_mappings = {
            "Sex": {"Male": 0, "Female": 1},
            "SmokingStatus": {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2},
        }
        cat_features = ["Sex", "SmokingStatus"]
        num_features = ["base_FVC", "base_Percent", "base_Age", "Week_passed"]

        for f in cat_features:
            train[f] = train[f].map(cat_mappings[f])
            test[f] = test[f].map(cat_mappings[f])

        scaler = RobustScaler()
        scaler.fit(train[num_features])

        train[num_features] = scaler.transform(train[num_features])
        test[num_features] = scaler.transform(test[num_features])
        fold_n = int(self.fold[-1]) - 1
        self.train_df = train.query(f"fold != {fold_n}")
        self.valid_df = train.query(f"fold == {fold_n} & final3 == True")
        self.valid_df_full = train.query(f"fold == {fold_n}")
        self.test_df = test

    def train_dataloader(self):
        dataset = TensorDataset(
            torch.tensor(self.train_df[self.features].values, dtype=torch.float32),
            torch.tensor(self.train_df["FVC"].values, dtype=torch.float32),
        )
        return DataLoader(
            dataset,
            num_workers=4,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        dataset = TensorDataset(
            torch.tensor(self.valid_df[self.features].values, dtype=torch.float32),
            torch.tensor(self.valid_df["FVC"].values, dtype=torch.float32),
        )
        return DataLoader(dataset, num_workers=4, batch_size=self.batch_size)

    def test_dataloader(self):
        dataset = TensorDataset(
            torch.tensor(self.test_df[self.features].values, dtype=torch.float32),
            torch.zeros(len(self.test_df), dtype=torch.float32),
        )
        return DataLoader(dataset, num_workers=4, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        decay_sched = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5, verbose=True,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "reduce_on_plateau": True,
            "frequency": 1,
        }

        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2078
        return [optimizer], [decay_sched]

    def training_step(self, batch, batch_idx):
        x_data, target = batch
        y_pred = self.forward(x_data)
        loss = self.criterion(y_pred, target)

        tensorboard_logs = {"loss/train": loss}
        return {
            "loss": loss,
            "log": tensorboard_logs,
            "y_pred": y_pred,
            "y_true": target,
        }

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        tensorboard_logs = {"loss/train_epoch": avg_loss}
        return {"log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x_data, target = batch
        y_pred = self.forward(x_data)
        loss = self.criterion(y_pred, target)

        return {
            "val_loss": loss,
            "y_pred": y_pred,
            "y_true": target,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu()
        y_true = torch.cat([x["y_true"] for x in outputs]).cpu()

        metric = self.criterion(y_pred, y_true, True)
        rmse = torch.mean(torch.sqrt((y_pred[:, 0] - y_true) ** 2))

        tensorboard_logs = {
            "loss/validation": avg_loss,
            "metric": metric,
            "rmse": rmse,
        }
        return {"val_loss": avg_loss, "log": tensorboard_logs, "metric": metric}

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x_data, _ = batch
        return {"y_pred": self(x_data)}

    def test_epoch_end(self, outputs):
        y_preds = torch.cat([x["y_pred"] for x in outputs]).cpu()
        return {"y_pred": y_preds}


def modified_laplace_ll(y_true, y_pred, y_pred_conf, reduce=True):
    sigma_clipped = np.maximum(y_pred_conf, 70)
    delta = np.minimum(np.abs(y_true - y_pred), 1000)
    metric = -np.sqrt(2) * delta / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)
    if reduce:
        return np.mean(metric)
    else:
        return metric


if __name__ == "__main__":
    folds = 6

    pl.seed_everything(48)

    fvc_oofs, conf_oofs, target_oofs = [], [], []
    fvc_test, conf_test = [], []

    for fold in range(folds):
        model = FullyConnected(fold=f"fold_{fold+1}", n_folds=folds)
        trainer = pl.Trainer(max_epochs=50)

        trainer.fit(model)

        with torch.no_grad():
            model.eval()
            test_preds = [model(batch[0]) for batch in model.test_dataloader()]
            test_preds = torch.cat(test_preds).numpy()
            fvc_test.append(test_preds[:, 0].reshape(-1, 1))
            conf_test.append(test_preds[:, 1].reshape(-1, 1))

            oofs_preds = [model(batch[0]) for batch in model.val_dataloader()]
            oofs_preds = torch.cat(oofs_preds).numpy()
            fvc_oofs.append(oofs_preds[:, 0])
            conf_oofs.append(oofs_preds[:, 1])
            target = [batch[1] for batch in model.val_dataloader()]
            target_oofs.append(torch.cat(target).numpy())

    fvc_oofs = np.concatenate(fvc_oofs)
    conf_oofs = np.concatenate(conf_oofs)
    target_oofs = np.concatenate(target_oofs)

    score = modified_laplace_ll(target_oofs, fvc_oofs, conf_oofs)
    print(f"Overall CV: {score:0.5f}")

    fvc_test = np.mean(np.concatenate(fvc_test, 1), 1)
    conf_test = np.sqrt((np.concatenate(conf_test, 1) ** 2).sum(1) / folds)

    sub = pd.DataFrame(
        {
            "Patient_Week": model.test_df["Patient_Week"].values,
            "FVC": fvc_test,
            "Confidence": conf_test,
        }
    )
    print(sub.head())
    sub.to_csv("submission.csv", index=False)

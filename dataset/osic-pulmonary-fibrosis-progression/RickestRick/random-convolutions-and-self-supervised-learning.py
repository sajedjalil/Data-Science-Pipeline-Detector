#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom
import logging
import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from catboost import CatBoostRegressor, CatBoostClassifier

### The rationale behind this notebook is that 'age' and 'smoking status'
### are less noisy targets than FCV measurements, so it sort of makes sense
### to build models that predict 'age' and 'smoking status' from the CT scans,
### as an intermediate step.
###
### Random convolutions are like random projections. Not as good as the real thing,
### but they conserve some valuable information nonetheless, and enable us to
### readily plug tree-base models right out of the average/max pooling layer.


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

l2_reg = 5
ntrees = 5000

ROOT = "../input/osic-pulmonary-fibrosis-progression"
dicom_train_dir = "%s/train" % ROOT
dicom_test_dir = "%s/test" % ROOT
train_csv = "%s/train.csv" % ROOT
test_csv = "%s/test.csv" % ROOT


def random_conv(patient_id: str, directory: str, n_draws: int) -> np.ndarray:
    kernel_w = 7
    cnn_out_dim = 128
    kernel = nn.Conv3d(1, cnn_out_dim, kernel_size=(kernel_w, kernel_w, 1), bias=False)
    patient_path = os.path.join(directory, patient_id)
    filenames = os.listdir(patient_path)
    paths = [
        os.path.join(patient_path, f) for f in filenames
    ]

    # remove some files to save some memory and speed
    removable = len(paths) // 5
    paths = paths[removable:-max(1, removable)]
    if len(paths) > 64:
        div = len(paths) // 64
        paths = paths[::div]

    # general info
    img = pydicom.dcmread(paths[0])
    rspacing = float(img.PixelSpacing[0])
    cspacing = float(img.PixelSpacing[1])
    realrows = int(img.Rows * rspacing / 2)  # div 4 to make scans smaller
    realcols = int(img.Columns * cspacing / 2)

    # pixel values
    dicoms = [
        pydicom.dcmread(path) for path in paths
    ]
    try:
        arrays = [d.pixel_array.astype(np.float32) for d in dicoms]
    except:
        logging.exception("skipping CT SCANS")
        return None
    images = [
        torch.Tensor(array).unsqueeze(2) for array in arrays
    ]

    # add the channel and batch_size dimensions
    video = torch.cat(images, dim=2).unsqueeze(0).unsqueeze(0)

    # resize to match real dimensions and normalize the grayscale values to (-1, 1) range
    n = len(paths)
    actual = F.interpolate(video, size=(realrows, realcols, n)).squeeze(0)
    actual -= actual.min()  # (0, inf)
    actual /= 0.5 * actual.max()  # (0, 2)
    actual -= 1  # (-1, 1)
    actual = actual.unsqueeze(0)

    # random projection
    if n_draws == 1:
        draw_size = n
    else:
        draw_size = max(2, n // 2)  # k = n/2 maximizes n! / k! (n-k)!
    with torch.no_grad():
        conved = kernel(actual).clamp(-1, 1).sum(dim=2).sum(dim=2).squeeze()
        choices = torch.cat([
            conved[:, np.random.choice(n, size=draw_size, replace=False)].mean(dim=1, keepdim=True)
            for _ in range(n_draws)
        ], dim=1)

    return np.unique(choices.t().data.numpy(), axis=0)


def patient_data(
        df: pd.DataFrame,
        patients: list,
        n_draws: int,
        directory: str,
        conv_mean: np.ndarray=None) -> list:
    # group by patient
    records = []
    for n, patient in enumerate(patients):
        conv = random_conv(patient, directory, n_draws)
        if conv is None:
            if conv_mean is None:
                continue
            conv = conv_mean
        subdf = df[df["Patient"] == patient]
        subdf = subdf.sort_values(by=["Weeks"])
        subdf.reset_index()
        record = {
            "Patient": patient,
            "Sex": subdf["Sex"].iloc[0] == "Male",
            "Age": subdf["Age"].iloc[0],
            "SmokingStatus": subdf["SmokingStatus"].iloc[0],
            "Ex-smoker": subdf["SmokingStatus"].iloc[0] == "Ex-smoker",
            "Never smoked": subdf["SmokingStatus"].iloc[0] == "Never smoked",
            "Smoker": subdf["SmokingStatus"].iloc[0] == "Currently smokes",
            "Percent": subdf["Percent"].iloc[0],
            "fvc0": subdf["FVC"].iloc[0],
            "weeks": subdf["Weeks"].values,
            "week0": subdf["Weeks"].values[0],
            "fvc": subdf["FVC"].values,
            "conv": conv
        }
        records.append(record)
        if (n+1) % 10 == 0:
            print("patient", n+1, "/", len(patients))

    return records


class AgeModel:
    def fit(self, records: list) -> "AgeModel":
        y = sum([[r["Age"]] * r["conv"].shape[0] for r in records], [])
        self._y_std, self._y_mean = np.std(y), np.mean(y)
        y = (np.array(y).astype(np.float32) - self._y_mean) / self._y_std
        X = self._pack(records)
        X, y = shuffle(X, y)
        self._model = CatBoostRegressor(l2_leaf_reg=l2_reg, iterations=ntrees).fit(X, y, silent=True)
        return self

    def predict(self, records: list) -> np.ndarray:
        y_pred = self._model.predict(self._pack(records))
        return y_pred * self._y_std + self._y_mean

    def _pack(self, records: list) -> np.ndarray:
        conved = np.vstack([r["conv"] for r in records])
        features = np.array(sum([
            [[r["Ex-smoker"], r["Never smoked"],
                r["Smoker"], r["Sex"]]] * r["conv"].shape[0]
            for r in records
        ], []))
        return np.concatenate([conved, features], axis=1)


class FVC0Model(AgeModel):
    def fit(self, records: list) -> "FVC0Model":
        y = sum([[r["fvc0"]] * r["conv"].shape[0] for r in records], [])
        self._y_std, self._y_mean = np.std(y), np.mean(y)
        y = (np.array(y).astype(np.float32) - self._y_mean) / self._y_std
        X = self._pack(records)
        X, y = shuffle(X, y)
        self._model = CatBoostRegressor(l2_leaf_reg=l2_reg, iterations=ntrees).fit(X, y, silent=True)
        return self

    def _pack(self, records: list) -> np.ndarray:
        conved = np.vstack([r["conv"] for r in records])
        features = np.array(sum([
            [[r["Ex-smoker"], r["Never smoked"],
                r["Smoker"], r["Sex"], r["Age"]]] * r["conv"].shape[0]
            for r in records
        ], []))
        return np.concatenate([conved, features], axis=1)


class SmokingModel:
    def fit(self, records: list) -> "SmokingModel":
        classes = {
            "Ex-smoker": 0,
            "Never smoked": 1,
            "Currently smokes": 2
        }
        y = sum([[classes[r["SmokingStatus"]]] * r["conv"].shape[0]
                for r in records], [])
        assert len(set(y)) == 3
        X = self._pack(records)
        X, y = shuffle(X, y)
        self._model = CatBoostClassifier(l2_leaf_reg=l2_reg, iterations=ntrees).fit(X, y, silent=True)
        return self

    def predict(self, records: list) -> np.ndarray:
        y_pred = self._model.predict_proba(self._pack(records))
        return y_pred[:, :2]

    def _pack(self, records: list) -> np.ndarray:
        conved = np.vstack([r["conv"] for r in records])
        features = np.array(sum([
            [[r["Sex"], r["Age"]]] * r["conv"].shape[0] for r in records
        ], []))
        return np.concatenate([conved, features], axis=1)


class MainModel:
    def __init__(
            self,
            age_model: AgeModel,
            fvc0_model: FVC0Model,
            smoking_model: SmokingModel) -> None:

        self._age_model = age_model
        self._fvc0_model = fvc0_model
        self._smoking_model = smoking_model

    def _pack_data(self, records: list) -> None:
        extra1 = self._age_model.predict(records) - np.array([r["Age"] for r in records])
        extra2 = self._fvc0_model.predict(records) - np.array([r["fvc0"] for r in records])
        extra3 = self._smoking_model.predict(records)
        extra = np.column_stack([extra1, extra2, extra3])

        rows = []
        for j, r in enumerate(records):
            cst = extra[j, :].tolist() + [
                r["Sex"], r["Age"], r["week0"], r["Percent"], r["fvc0"],
                r["Ex-smoker"], r["Never smoked"], r["Smoker"]]
            for week in r["weeks"]:
                rows.append(cst + [week])

        return np.vstack(rows)

    def fit(self, records: list) -> "MainModel":
        # sample weights
        weights = []
        for r in records:
            visits = len(r["fvc"])
            #w = np.square(np.clip(np.arange(visits), 0, visits-3) + 1)
            w = np.clip(np.arange(visits), 0, visits-3) + 1
            w = w.astype(np.float32) / w.sum()
            weights += w.tolist()

        # training data
        X = self._pack_data(records)
        cut = 4 * len(X) // 5
        X_fvc = X[:cut, :].copy()
        X_confidence = X[cut:, :].copy()

        # FVC model
        y = np.concatenate([r["fvc"] for r in records])[:cut]
        self._y_std, self._y_mean = np.std(y), np.mean(y)
        y = (y.astype(np.float32) - self._y_mean) / self._y_std
        X_fvc, y = shuffle(X_fvc, y)
        self._model = CatBoostRegressor(iterations=ntrees)
        self._model.fit(X_fvc, y, sample_weight=weights[:cut], silent=True)

        # model for the optimal confidence
        fvc_true = np.concatenate([r["fvc"] for r in records])[cut:]
        fvc_pred = self._model.predict(X_confidence) * self._y_std + self._y_mean
        confidence = np.abs(fvc_true - fvc_pred).clip(0, 1000) * np.sqrt(2)  # solved analytically
        self._c_std, self._c_mean = np.std(confidence), np.mean(confidence)
        confidence = (confidence - self._c_mean) / self._c_std
        X_confidence = np.column_stack([X_confidence, fvc_pred])
        X_confidence, confidence = shuffle(X_confidence, confidence)
        self._confidence_model = CatBoostRegressor()
        self._confidence_model.fit(X_confidence, confidence, sample_weight=weights[cut:], silent=True)

        # average convolution results (for imputing)
        self._conv_mean = records[0]["conv"].copy()
        for r in records[1:]:
            self._conv_mean += r["conv"]
        self._conv_mean /= len(records)

        # reporting
        weights = []
        for r in records:
            visits = len(r["fvc"])
            w = np.zeros(visits)
            w[-3:] = 1
            weights += w.tolist()
        weights = np.array(weights)[cut:]
        err = np.abs(fvc_true - fvc_pred).clip(0, 1000)
        confidence = (np.abs(fvc_true - fvc_pred).clip(0, 1000) * np.sqrt(2)).clip(70, np.inf)
        print("mean FVC pred", np.mean(fvc_pred))
        print("MAE", np.mean(err))
        loss = -np.sqrt(2)*err / confidence - np.log(np.sqrt(2) * confidence)
        print("mean objective:", np.sum(loss * weights ) / weights.sum())

        return self

    def predict(self, records: list) -> tuple:
        # fvc
        X = self._pack_data(records)
        fvc = self._model.predict(X) * self._y_std + self._y_mean

        # confidence
        X = np.column_stack([X, fvc])
        confidence = self._confidence_model.predict(X) * self._c_std + self._c_mean
        confidence = confidence.clip(min=70, max=5000)

        return fvc, confidence


def train_main_model() -> MainModel:
    # split training data into 2 sets
    df = pd.read_csv(train_csv, header=0)
    patients = list(df["Patient"].unique())
    print("number of patients", len(patients))
    random.shuffle(patients)
    cut1 = len(patients) // 2
    cut2 = len(patients) // 4
    records1 = patient_data(df, patients[:cut1], directory=dicom_train_dir, n_draws=2048)
    records2 = patient_data(df, patients[cut2:], directory=dicom_train_dir, n_draws=1)

    # self-supervised training on age, fvc0 and smoking status
    model = MainModel(
                age_model=AgeModel().fit(records1),
                fvc0_model=FVC0Model().fit(records1),
                smoking_model=SmokingModel().fit(records1))

    # training of main model
    return model.fit(records2)


def main():
    model = train_main_model()
    df = pd.read_csv(test_csv, header=0)
    patients = list(df["Patient"].unique())
    records = patient_data(df, patients, directory=dicom_test_dir, n_draws=1, conv_mean=model._conv_mean)

    weeks = np.arange(-12, 134)
    for r in records:
        r["weeks"] = weeks

    fvc_pred, confidence_pred = model.predict(records)

    submission = []
    for i, r in enumerate(records):
        visits = len(r["weeks"])
        patient_fvc = fvc_pred[i:i+visits]
        patient_conf = confidence_pred[i:i+visits]
        submission += [
            {"Patient_Week": "%s_%i" % (r["Patient"], week),
            "FVC": fvc, "Confidence": conf}
            for week, fvc, conf in zip(r["weeks"], patient_fvc, patient_conf)
        ]

    pd.DataFrame(submission).to_csv("submission.csv", index=False)


main()

""" A very simple MLP baseline
* This is a very small dataset, so it's very easy to overfit. Tune carefully.
* Check the "Output" tab for training logs.
"""
from subprocess import call
import os

FNULL = open(os.devnull, 'w')
call("pip install https://github.com/ceshine/pytorch_helper_bot/archive/0.0.3.zip".split(" "), stdout=FNULL, stderr=FNULL)

# set SEED
os.environ["SEED"] = "42"

DEVICE = "cpu"

# ======================
#     Dataset Utils
# ======================
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataset(x, y):
    return TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float()
    )


def get_dataloader(x: np.array, y: np.array, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = get_dataset(x, y)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def get_ndarray(embedding_values):
    results = []
    for row in embedding_values:
        arr = np.array(row)
        results.append(
            np.pad(arr, ((10 - arr.shape[0], 0), (0, 0)), 'constant')
        )
    # shape: (examples, emb_dim, seq_length)
    return np.transpose(np.stack(results), (0, 2, 1))


def read_dataset(data_dir=Path("data/")):
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    df_train = pd.read_json(data_dir / 'train.json')
    df_test = pd.read_json(data_dir / 'test.json')
    x_train = get_ndarray(df_train.audio_embedding)
    y_train = df_train.is_turkey.values
    x_test = get_ndarray(df_test.audio_embedding)
    test_id = df_test.vid_id
    return x_train, y_train, x_test, test_id


# ===============================================
#     Model Creation, Training, and Inference
# ==============================================
import logging

from helperbot.bot import BaseBot
from helperbot.lr_scheduler import TriangularLR
from helperbot.weight_decay import WeightDecayOptimizerWrapper
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import torch.nn as nn
import pandas as pd


class TurkeyBot(BaseBot):
    name = "Turkey"

    def __init__(self, model, train_loader, val_loader, *, optimizer,
                 avg_window=20, log_dir="./cache/logs/",
                 log_level=logging.INFO, checkpoint_dir="./cache/model_cache/"):
        super().__init__(
            model, train_loader, val_loader,
            optimizer=optimizer, avg_window=avg_window,
            log_dir=log_dir, log_level=log_level, checkpoint_dir=checkpoint_dir,
            batch_idx=0, echo=False, device=DEVICE
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.loss_format = "%.8f"


class MLPModel(nn.Module):
    def __init__(self, num_features, dropout=0.25, n_hid=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, n_hid),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Dropout(dropout),            
            nn.Linear(n_hid, n_hid // 4),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid // 4),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 4, 1),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)

        
        
def main():
    x_train, y_train, x_test, test_id = read_dataset("../input/")
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    test_loader = get_dataloader(
        x_test, np.zeros(x_test.shape[0]), batch_size=128, shuffle=False)

    test_pred_list, val_losses = [], []
    kf = StratifiedKFold(n_splits=8, random_state=3829, shuffle=True)
    for train_index, valid_index in kf.split(x_train, y_train):
        train_loader = get_dataloader(
            x_train[train_index], y_train[train_index],
            batch_size=128, shuffle=True
        )
        val_loader = get_dataloader(
            x_train[valid_index], y_train[valid_index],
            batch_size=128, shuffle=False
        )

        model = MLPModel(128 * 10, dropout=0.25, n_hid=1024)
        model.to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)
        optimizer = WeightDecayOptimizerWrapper(
            optimizer, weight_decay=5e-3
        )
        batches_per_epoch = len(train_loader)
        bot = TurkeyBot(
            model, train_loader, val_loader,
            optimizer=optimizer, avg_window=batches_per_epoch
        )
        n_steps = batches_per_epoch * 20
        scheduler = TriangularLR(
            optimizer, max_mul=8, ratio=8,
            steps_per_cycle=n_steps
        )
        bot.train(
            n_steps,
            log_interval=batches_per_epoch // 2,
            snapshot_interval=batches_per_epoch,
            early_stopping_cnt=10, scheduler=scheduler)
        val_preds = torch.sigmoid(bot.predict_avg(
            val_loader, k=3, is_test=True).cpu()).numpy().clip(1e-5, 1-1e-5)
        loss = log_loss(y_train[valid_index], val_preds)
        print("AUC: %.6f" % roc_auc_score(y_train[valid_index], val_preds))
        print("Val loss: %.6f" % loss)
        if loss > 0.2:
            print("Skipped...")
            # Ditch folds that perform terribly
            bot.remove_checkpoints(keep=0)
            continue
        val_losses.append(loss)
        test_pred_list.append(torch.sigmoid(bot.predict_avg(
            test_loader, k=3, is_test=True).cpu()).numpy().clip(1e-5, 1-1e-5))
        bot.remove_checkpoints(keep=0)

    print("# of Folds used:", len(val_losses))
    val_loss = np.mean(val_losses)
    test_preds = np.mean(test_pred_list, axis=0)
    print("Validation losses: %.6f +- %.6f" %
          (np.mean(val_losses), np.std(val_losses)))

    df_sub = pd.DataFrame({
        "vid_id": test_id,
        "is_turkey": test_preds
    })
    df_sub.to_csv("submission.csv", index=False)

main()
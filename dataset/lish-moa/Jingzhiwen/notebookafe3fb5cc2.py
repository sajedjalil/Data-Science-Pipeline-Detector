# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install --no-index --find-links /kaggle/input/pytorchtabnet/pytorch_tabnet-2.0.1-py3-none-any.whl pytorch-tabnet
!pip install /kaggle/input/iterativestratification/iterative-stratification-master/
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np
import pandas as pd 

import os
import random
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from tqdm import tqdm
from sklearn.metrics import log_loss

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
seed_everything(42)


data_path = "../input/lish-moa/"
train = pd.read_csv(data_path+'train_features.csv')
train.drop(columns=["sig_id"], inplace=True)

train_targets_scored = pd.read_csv(data_path+'train_targets_scored.csv')
train_targets_scored.drop(columns=["sig_id"], inplace=True)

test = pd.read_csv(data_path+'test_features.csv')
test.drop(columns=["sig_id"], inplace=True)

submission = pd.read_csv(data_path+'sample_submission.csv')

remove_vehicle = False

if remove_vehicle:
    kept_index = train['cp_type']=='trt_cp'
    train = train.loc[kept_index].reset_index(drop=True)
    train_targets_scored = train_targets_scored.loc[kept_index].reset_index(drop=True)

train["cp_type"] = (train["cp_type"]=="trt_cp") + 0
train["cp_dose"] = (train["cp_dose"]=="D1") + 0

test["cp_type"] = (test["cp_type"]=="trt_cp") + 0
test["cp_dose"] = (test["cp_dose"]=="D1") + 0

X_test = test.values

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

MAX_EPOCH=200
tabnet_params = dict(n_d=24, n_a=24, n_steps=1, gamma=1.3,
                     lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type='entmax',
                     scheduler_params=dict(mode="min",
                                           patience=5,
                                           min_lr=1e-5,
                                           factor=0.9,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=10,
                     )


from sklearn.metrics import log_loss
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score, log_loss

class LogitsLogLoss(Metric):
    """
    LogLoss with sigmoid applied
    """

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            LogLoss of predictions vs targets.
        """
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1-y_true)*np.log(1-logits+1e-15) + y_true*np.log(logits+1e-15)
        return np.mean(-aux)
    

    
scores_auc_all= []
test_cv_preds = []

NB_SPLITS = 10
mskf = MultilabelStratifiedKFold(n_splits=NB_SPLITS, random_state=0, shuffle=True)
oof_preds = []
oof_targets = []
scores = []
scores_auc = []
for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train, train_targets_scored)):
    print("FOLDS : ", fold_nb)

    ## model
    X_train, y_train = train.values[train_idx, :], train_targets_scored.values[train_idx, :]
    X_val, y_val = train.values[val_idx, :], train_targets_scored.values[val_idx, :]
    model = TabNetRegressor(**tabnet_params)

    model.fit(X_train=X_train,
              y_train=y_train,
              eval_set=[(X_val, y_val)],
              eval_name = ["val"],
              eval_metric = ["logits_ll"],
              max_epochs=MAX_EPOCH,
              patience=20, batch_size=1024, virtual_batch_size=128,
              num_workers=1, drop_last=False,
              # use binary cross entropy as this is not a regression problem
              loss_fn=torch.nn.functional.binary_cross_entropy_with_logits)

    preds_val = model.predict(X_val)
    # Apply sigmoid to the predictions
    preds =  1 / (1 + np.exp(-preds_val))
    score = np.min(model.history["val_logits_ll"])
#     name = cfg.save_name + f"_fold{fold_nb}"
#     model.save_model(name)
    ## save oof to compute the CV later
    oof_preds.append(preds_val)
    oof_targets.append(y_val)
    scores.append(score)

    # preds on test
    preds_test = model.predict(X_test)
    test_cv_preds.append(1 / (1 + np.exp(-preds_test)))

oof_preds_all = np.concatenate(oof_preds)
oof_targets_all = np.concatenate(oof_targets)
test_preds_all = np.stack(test_cv_preds)

aucs = []
for task_id in range(oof_preds_all.shape[1]):
    aucs.append(roc_auc_score(y_true=oof_targets_all[:, task_id],
                              y_score=oof_preds_all[:, task_id]))
print(f"Overall AUC : {np.mean(aucs)}")
print(f"Average CV : {np.mean(scores)}")


all_feat = [col for col in submission.columns if col not in ["sig_id"]]
submission[all_feat] = test_preds_all.mean(axis=0)
# set control to 0
submission.loc[test['cp_type']==0, submission.columns[1:]] = 0
submission.to_csv("submission.csv", index=None)
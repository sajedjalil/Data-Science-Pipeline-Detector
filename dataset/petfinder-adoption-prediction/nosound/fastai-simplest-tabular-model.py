import numpy as np
import pandas as pd

from fastai import *
from fastai.tabular import *

train_csv = pd.read_csv('../input/train/train.csv', low_memory=False)
test_csv = pd.read_csv('../input/test/test.csv', low_memory=False)

cat_names = ['Type','Breed1','Breed2','Gender','Color1','Color2','State','Color3','FurLength', 'Vaccinated','Dewormed','Sterilized','Health']
cont_names = ['Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']

procs = [FillMissing, Categorify, Normalize]
test_ds = TabularList.from_df(test_csv, path='../input', cat_names=cat_names, cont_names=cont_names)
train_ds = TabularList.from_df(train_csv, path='../input', cat_names=cat_names, cont_names=cont_names, procs=procs)
data = train_ds.no_split().label_from_df(cols='AdoptionSpeed').add_test(test_ds).databunch()

learn = tabular_learner(data, layers=[200,100], path='/tmp')
learn.fit(10, 1e-2)

pred = learn.get_preds(ds_type=DatasetType.Test)
y_pred = [int(np.round(np.sum(np.array(row)*range(5)))) for row in pred[0]]

test_csv['AdoptionSpeed'] = y_pred
test_csv[['PetID', 'AdoptionSpeed']].to_csv('submission.csv', index=False)
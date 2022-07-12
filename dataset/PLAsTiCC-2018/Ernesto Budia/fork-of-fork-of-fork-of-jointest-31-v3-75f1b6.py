import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/nnmodel-v1-wo"))

# Any results you write to the current directory are saved as output.
MT=pd.read_csv('../input/PLAsTiCC-2018/test_set_metadata.csv')

PRED=pd.read_csv('../input/lgbm-31-wo-0-536906/single_subm_0.529858_2018-12-12-21-17.csv')


PRED2=pd.read_csv('../input/lgbm-31-1-w/single_subm_0.477302_2018-12-13-21-04.csv')

obid=MT.loc[MT.hostgal_specz.notnull(),'object_id']
PRED.loc[PRED.object_id.isin(MT.loc[MT.hostgal_specz.notnull(),'object_id'])]

aux=PRED2.loc[PRED2.object_id.isin(obid)]

PRED.shape
PRED.head()
PRED.loc[PRED.object_id.isin(obid)]=PRED2.loc[PRED2.object_id.isin(obid)]
#PRED.to_csv('submission.csv', index=False)

#PRED=pd.read_csv("../input/join-30-v2/submission.csv")

#def GenUnknown(data):
#    return ((((((data["mymedian"]) + (((data["mymean"]) / 2.0)))/2.0)) + (((((1.0) - (((data["mymax"]) * (((data["mymax"]) * (data["mymax"]))))))) / 2.0)))/2.0)

feats = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
         'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
         'class_92', 'class_95']

y = pd.DataFrame()
y['mymean'] = PRED[feats].mean(axis=1)
y['mymedian'] = PRED[feats].median(axis=1)
y['mymax'] = PRED[feats].max(axis=1)

#PRED['class_99'] = GenUnknown(y)

#PRED.to_csv('submission.csv', index=False)

#PRED['class_99'] = -.0625*y['mymax']**8-0.125*y['mymax']**4-.0625*y['mymax']+.25

#PRED.to_csv('submission2.csv', index=False)


#PRED['class_99'] = 0.5 * (1 - y['mymax'])

#PRED.to_csv('submission3.csv', index=False)


PRED['class_99'] =(0.5 + 0.75 * y["mymean"] - 0.5 * y["mymax"] ** 3) / 2





PRED3=pd.read_csv('../input/nnmodel-v1-wo/single_predictions.csv')


feats = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
         'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
         'class_92', 'class_95']

y = pd.DataFrame()
y['mymean'] = PRED3[feats].mean(axis=1)
y['mymedian'] = PRED3[feats].median(axis=1)
y['mymax'] = PRED3[feats].max(axis=1)

PRED3['class_99'] =(0.5 + 0.75 * y["mymean"] - 0.5 * y["mymax"] ** 3) / 2

PREDAUX=0.85*PRED+0.15*PRED3

PREDAUX['object_id']=PRED['object_id']
PREDAUX.to_csv('submission4.csv', index=False)
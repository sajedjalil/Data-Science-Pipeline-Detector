# Fork V8 Mask-RCNN and COCO transfer learning by Henrique Mendonça/Andy Harless
# Fork Lung Opacity Classification Transfer Learning by Kevin Mader

import pandas as pd

sub = pd.read_csv('../input/fork-v8-henrique-s-model-w-randomly-higher-score/submission.csv')
probs = pd.read_csv('../input/lung-opacity-classification-transfer-learning/image_level_class_probs.csv', usecols=[0,1])

sub = pd.merge(sub, probs, on='patientId')
sub.loc[sub['Lung Opacity']<0.16,'PredictionString'] = None
sub.drop(['Lung Opacity'],1,inplace=True)

sub.to_csv('subprobs16.csv', index=False)

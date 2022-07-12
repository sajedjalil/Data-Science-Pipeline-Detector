import pandas as pd

first_sub = '../input/catboost-stackedae-with-mxnet-meta-1-40lb/submission.csv'
second_sub = '../input/pipeline-kernel-xgb-fe-lb1-39/pipeline_kernel_cv1.0.csv'

first_sub, second_sub = pd.read_csv(first_sub), pd.read_csv(second_sub)
first_sub['target'] = (first_sub['target'] + second_sub['target']) / 2.0
first_sub.to_csv('blending_sin.csv', index=None)

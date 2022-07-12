# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
sample_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

# %% [code]
df_train.head(5)

# %% [code]
print('Count of missing values in TRAIN data is', sum(df_train.isnull().sum(axis=1)))
df_train.dropna(inplace=True)
df_train.to_csv('train.csv', index=False)

# %% [code]
df_test.to_csv('test.csv', index=False)

# %% [code]
from fastai.text import *
data_lm = (TextList.from_csv(path='/kaggle/working', csv_name='test.csv', cols='text')
                   .split_by_rand_pct()
                   .label_for_lm()
                   .databunch())
data_lm.show_batch()

# %% [code]
#learn = language_model_learner(data_lm, Transformer)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.7)
learn.unfreeze()


# %% [code]
#learn.lr_find()

# %% [code]
#learn.recorder.plot()

# %% [code]
learn.fit_one_cycle(8, 8e-3)

# %% [code]
learn.save('mini_train_lm')
learn.save_encoder('mini_train_encoder')

# %% [code]
learn.fit_one_cycle(2, 5e-3)

# %% [code]
learn.save('mini_train_lm_2')
learn.save_encoder('mini_train_encoder_2')

# %% [code]
learn.show_results()

# %% [code]
data_clas = (TextList.from_csv(path='/kaggle/working', csv_name='test.csv', cols='text', vocab=data_lm.vocab)
                   .split_by_rand_pct()
                   .label_from_df(cols='sentiment')
                   .databunch(bs=100))

# %% [code]
data_clas.show_batch()

# %% [code]
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.9)
learn.load_encoder('mini_train_encoder_2')
learn.unfreeze()


# %% [code]
#learn.lr_find()

# %% [code]
#learn.recorder.plot()

# %% [code]
learn.fit_one_cycle(10, slice(5e-3,8e-2))

# %% [code]
learn.freeze_to(-2)
lr = 7e-2
learn.fit_one_cycle(10,slice(lr/(2.6**4),lr), moms=(0.8,0.7) )

# %% [code]
learn.save('mini_train_clas_final')

# %% [code]
txt_ci = TextClassificationInterpretation.from_learner(learn)

# %% [code]
from tqdm import tqdm
selected_texts = []

for text in tqdm(df_test['text'], position=0):
    mask = txt_ci.intrinsic_attention(text)[1] > 0.4
    text = text.split()
    selected_text = ' '.join([x for x, y in zip(text, mask) if y == True])
    selected_texts.append(selected_text)

# %% [code]
sample_submission['selected_text'] = selected_texts

# %% [code]
sample_submission.head(25)

# %% [code]
sample_submission.to_csv('/kaggle/working/submission.csv', index=False)




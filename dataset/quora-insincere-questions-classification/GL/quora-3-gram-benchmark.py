# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from functools import reduce
from tqdm import tqdm


def write_file(filename, headers, items):
    print("\nWriting to file:", filename)
    f = open(filename, "w", encoding='utf8')
    concat = lambda x,y: str(x) + ',' + str(y)
    f.write(reduce(concat, headers) + '\n')
    for i in tqdm(range(len(items))):
        f.write(reduce(concat, items[i]) + '\n')
    f.close()


def get_kwords(df, k=3):
    print('\nProcessing k-words... k=' + str(k))
    progress = tqdm(total=df.shape[0])
    count = {} # (w0,w1,w2) -> (count_sincere, count_insencere)
    for index, row in df.iterrows():
        line = row[1].lower()
        target = row[2]
        for c in '.,:;?!/"\'()*&^%$#@':
            line = line.replace(c, ' ')
    
        line_words = word_tokenize(line)
        for i in range(len(line_words)-k+1):
            key = tuple(line_words[i+j] for j in range(k))
            if key not in count:
                count[key] = (0, 0)
            if target == 0:
                count[key] = (count[key][0] + 1, count[key][1])
            else:
                count[key] = (count[key][0], count[key][1] + 1)

        progress.update(1)

    # sort desc by insincere count
    sorted_count = sorted(count.items(), key=lambda item: item[1][1], reverse=True)
    total0 = float(sum([count[key][0] for key in count]))
    total1 = float(sum([count[key][1] for key in count]))
    result = []
    for item in sorted_count:
        r = list(item[0])
        r.extend(list(item[1]))
        r.append(item[1][0]/total0 if total0 > 0 else 0.)
        r.append(item[1][1]/total1 if total1 > 0 else 0.)
        result.append(tuple(r))
    return result


def get_target_kwords(df, k=3):
    count_all = get_kwords(df, k=3)
    headers = ['w0','w1','w2','count0','count1','rate0','rate1']

    print('\nGetting target k-words...')
    result = {}
    print_list = []
    for item in tqdm(count_all):
        if item[-2]*10 < item[-1]:
            kword = tuple(item[j] for j in range(k))
            result[kword] = (item[-2], item[-1])

    return result


def predict(df_data, target_kwords, k=3, training=False):
    result = []
    predicted_correct = 0
    print('\nPredicting target values...')
    progress = tqdm(total=df_data.shape[0])
    for index, row in df_data.iterrows():
        line = row[1].lower()
        target = row[2] if training else None
        for c in '.,:;?!/"\'()*&^%$#@':
            line = line.replace(c, ' ')

        line_words = word_tokenize(line)
        found_kwords = []
        for i in range(len(line_words)-k+1):
            kword = tuple(line_words[i+j] for j in range(k))
            if kword in target_kwords:
                found_kwords.append(kword)
        predicted = 1 if len(found_kwords) > 1 else 0
        
        if training:
            result.append((row[0], row[1], target, found_kwords, predicted))
            predicted_correct += 1 if predicted == target else 0
        else:
            result.append((row[0], predicted))
        progress.update(1)

    accuracy = predicted_correct/df_data.shape[0] if training else None
    return result, accuracy
    

dir_path = os.path.abspath("../input")
df_data = pd.read_csv(os.path.join(dir_path, 'train.csv'))
target_kwords = get_target_kwords(df_data, k=3)

df_test = pd.read_csv(os.path.join(dir_path, 'test.csv'))
predict_data, accuracy = predict(df_test, target_kwords, k=3, training=False)
write_file('submission.csv', ['qid','prediction'], predict_data)



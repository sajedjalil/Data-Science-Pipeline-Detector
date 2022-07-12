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
import pandas as pd
import re


# Replace all numeric with 'n'
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

COMMENT_COL = 'comment_text'
ID_COL = 'id'
input_dir = '../input/'
output_dir = '../input/'

# redundancy words and their right formats
redundancy_rightFormat = {
    'ckckck': 'cock',
    'fuckfuck': 'fuck',
    'lolol': 'lol',
    'lollol': 'lol',
    'pussyfuck':'fuck',
    'gaygay': 'gay',
    'haha': 'ha',
    'sucksuck': 'suck'}

redundancy = set(redundancy_rightFormat.keys())

# all the words below are included in glove dictionary
# combine these toxic indicators with 'CommProcess.revise_triple_and_more_letters'
toxic_indicator_words = [
    'fuck', 'fucking', 'fucked', 'fuckin', 'fucka', 'fucker', 'fucks', 'fuckers',
    'fck', 'fcking', 'fcked', 'fckin', 'fcker', 'fcks',
    'fuk', 'fuking', 'fuked', 'fukin', 'fuker', 'fuks', 'fukers',
    'fk', 'fking', 'fked', 'fkin', 'fker', 'fks',
    'shit', 'shitty', 'shite',
    'stupid', 'stupids',
    'idiot', 'idiots',
    'suck', 'sucker', 'sucks', 'sucka', 'sucked', 'sucking',
    'ass', 'asses', 'asshole', 'assholes', 'ashole', 'asholes',
    'gay', 'gays',
    'niga', 'nigga', 'nigar', 'niggar', 'niger', 'nigger',
    'monster', 'monsters',
    'loser', 'losers',
    'nazi', 'nazis',
    'cock', 'cocks', 'cocker', 'cockers',
    'faggot', 'faggy',
]
toxic_indicator_words_sets = set(toxic_indicator_words)


def _get_toxicIndicator_transformers():
    toxicIndicator_transformers = dict()
    for word in toxic_indicator_words:
        tmp_1 = []
        for c in word:
            if len(tmp_1) > 0:
                tmp_2 = []
                for pre in tmp_1:
                    tmp_2.append(pre + c)
                    tmp_2.append(pre + c + c)
                tmp_1 = tmp_2
            else:
                tmp_1.append(c)
                tmp_1.append(c + c)
        toxicIndicator_transformers[word] = tmp_1
    return toxicIndicator_transformers


toxicIndicator_transformers = _get_toxicIndicator_transformers()

deny_origin = {
    "you're": ['you', 'are'],
    "i'm": ['i', 'am'],
    "he's": ['he', 'is'],
    "she's": ['she', 'is'],
    "it's": ['it', 'is'],
    "they're": ['they', 'are'],
    "can't": ['can', 'not'],
    "couldn't": ['could', 'not'],
    "don't": ['do', 'not'],
    "don;t": ['do', 'not'],
    "didn't": ['did', 'not'],
    "doesn't": ['does', 'not'],
    "isn't": ['is', 'not'],
    "wasn't": ['was', 'not'],
    "aren't": ['are', 'not'],
    "weren't": ['were', 'not'],
    "won't": ['will', 'not'],
    "wouldn't": ['would', 'not'],
    "hasn't": ['has', 'not'],
    "haven't": ['have', 'not'],
    "what's": ['what', 'is'],
    "that's": ['that', 'is'],
}
denies = set(deny_origin.keys())


class CommProcess(object):
    @staticmethod
    def clean_text(t):
        t = re.sub(r"[^A-Za-z0-9,!?*.;’´'\/]", " ", t)
        t = replace_numbers.sub(" ", t)
        t = t.lower()
        t = re.sub(r",", " ", t)
        t = re.sub(r"’", "'", t)
        t = re.sub(r"´", "'", t)
        t = re.sub(r"\.", " ", t)
        t = re.sub(r"!", " ! ", t)
        t = re.sub(r"\?", " ? ", t)
        t = re.sub(r"\/", " ", t)
        return t

    @staticmethod
    def revise_deny(t):
        ret = []
        for word in t.split():
            if word in denies:
                ret.append(deny_origin[word][0])
                ret.append(deny_origin[word][1])
            else:
                ret.append(word)
        ret = ' '.join(ret)
        ret = re.sub("'", " ", ret)
        ret = re.sub(r";", " ", ret)
        return ret

    @staticmethod
    def revise_star(t):
        ret = []
        for word in t.split():
            if ('*' in word) and (re.sub('\*', '', word) in toxic_indicator_words_sets):
                word = re.sub('\*', '', word)
            ret.append(word)
        ret = re.sub('\*', ' ', ' '.join(ret))
        return ret

    @staticmethod
    def revise_triple_and_more_letters(t):
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            reg = letter + "{2,}"
            t = re.sub(reg, letter + letter, t)
        return t

    @staticmethod
    def revise_redundancy_words(t):
        ret = []
        for word in t.split(' '):
            for redu in redundancy:
                if redu in word:
                    word = redundancy_rightFormat[redu]
                    break
            ret.append(word)
        return ' '.join(ret)

    @staticmethod
    def fill_na(t):
        if t.strip() == '':
            return 'NA'
        return t


def execute_comm_process(df):
    comm_process_pipeline = [
        CommProcess.clean_text,
        CommProcess.revise_deny,
        CommProcess.revise_star,
        CommProcess.revise_triple_and_more_letters,
        CommProcess.revise_redundancy_words,
        CommProcess.fill_na,
    ]
    for cp in comm_process_pipeline:
        df[COMMENT_COL] = df[COMMENT_COL].apply(cp)
    return df



# Process whole train data
print('Comm processing whole train data')
df_train = pd.read_csv(input_dir + 'train.csv')
df_train = execute_comm_process(df_train)
df_train.to_csv('train_processed.csv', index=False)
# Process test data
print('Comm processing test data')
df_test = pd.read_csv(input_dir + 'test.csv')
df_test = execute_comm_process(df_test)
df_test.to_csv('test_processed.csv', index=False)
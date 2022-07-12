import csv

import numpy as np
import pandas as pd


df_train = pd.read_csv('../input/train.csv')

df_id1 = df_train[["qid1", "question1"]].drop_duplicates().copy().reset_index(drop=True)
df_id2 = df_train[["qid2", "question2"]].drop_duplicates().copy().reset_index(drop=True)

df_id1.columns = ["qid", "question"]
df_id2.columns = ["qid", "question"]

df_id = pd.concat([df_id1, df_id2]).drop_duplicates().reset_index(drop=True)

dict_questions = df_id.set_index('question').to_dict()
dict_questions = dict_questions["qid"]

new_id = 538000 # df_id["qid"].max() ==> 537933

def get_id(question):

    global dict_questions
    global new_id
    
    if question in dict_questions:
        return dict_questions[question]
    else:
        new_id += 1
        dict_questions[question] = new_id
        return new_id

rows = []

infile = open('../input/test.csv', 'r', encoding="utf8")

reader = csv.reader(infile, delimiter=",")
header = next(reader)
header.append('qid1')
header.append('qid2')

for row in reader:

    question1 = row[1]
    question2 = row[2]

    qid1 = get_id(question1)
    qid2 = get_id(question2)
    row.append(qid1)
    row.append(qid2)

    rows.append(row)

df_out = pd.DataFrame(data=rows, columns=header)

df_out.to_csv('test_with_ids.csv', index=False)
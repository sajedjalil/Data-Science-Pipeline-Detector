"""
Feature based on Tien-Dung Le's post:
https://www.kaggle.com/c/quora-question-pairs/discussion/33371

Slightly different idea here. Instead of making a id-based graph/dataframe, we do it based on lowercase questions.

"""
import networkx as nx
import pandas as pd
from tqdm import tqdm


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

dfs = (df_train, df_test)

questions = []
for df in dfs:
    df['question1'] = df['question1'].str.lower()
    df['question2'] = df['question2'].str.lower()
    questions += df['question1'].tolist()
    questions += df['question2'].tolist()

graph = nx.Graph()
graph.add_nodes_from(questions)

for df in [df_train, df_test]:
    edges = list(df[['question1', 'question2']].to_records(index=False))
    graph.add_edges_from(edges)

graph.remove_edges_from(graph.selfloop_edges())

df = pd.DataFrame(data=graph.nodes(), columns=["question"])
df['kcores'] = 1

n_cores = 30
for k in tqdm(range(2, n_cores + 1)):
    ck = nx.k_core(graph, k=k).nodes()
    df['kcores'][df.question.isin(ck)] = k

print(df['kcores'].value_counts())

df.to_csv("question_kcores.csv", index=None)

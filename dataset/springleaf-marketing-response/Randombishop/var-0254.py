import pandas as pd

train = pd.read_csv("../input/train.csv")

ct=pd.crosstab(train['VAR_0254'],train['target']).apply(lambda r: r/r.sum(), axis=1)

print(ct)

fig=ct.plot().get_figure()

fig.savefig('VAR_0254.png')
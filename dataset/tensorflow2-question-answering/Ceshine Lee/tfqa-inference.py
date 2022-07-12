import os

os.environ['SEED'] = '3159'

from subprocess import run, PIPE
from pathlib import Path

print([x.name for x in Path("../input/").iterdir()])
print([x.name for x in Path("../input/tfqa-codebase/").iterdir()])
print([x.name for x in Path("../input/tfqa-model/").iterdir()])

# Install Depedencies

run(["pip install --no-deps ../input/tfqa-codebase/delegator/delegator.py-master/."], shell=True, check=True)

import delegator
import pathlib

c = delegator.run(
    'pip install --no-deps ../input/tfqa-codebase/pytorch_helper_bot/pytorch-helper-bot-master/.', 
    block=True
)
assert c.return_code == 0

c = delegator.run(
    'pip install --no-deps ../input/tfqa-codebase/transformers/transformers-master/.', 
    block=True
)
assert c.return_code == 0

c = delegator.run(
    'pip install --no-deps ../input/tfqa-codebase/sacremoses/sacremoses-master/.', 
    block=True
)
assert c.return_code == 0

c = delegator.run(
    'pip install --no-deps ../input/tfqa-codebase/python-fire/python-fire-master/.', 
    block=True
)
assert c.return_code == 0

c = delegator.run(
    'pip install --no-deps ../input/tfqa-codebase/fragile-0.0.2-py3-none-any.whl', 
    block=True
)
assert c.return_code == 0

# Link data
c = delegator.run(
    'ln -s ../input/tensorflow2-question-answering/ data', 
    block=True
)
assert c.return_code == 0
print(delegator.run('ls data/', block=True).out)


# from fragile import inference

# inference.main(
#     model_path="../input/tfqa-model/", run_preprocess=True, max_ex_len=400, batch_size=16,
#     file_pattern = 'test_%d.jl'
# )

print("Predicting...")
c = delegator.run(
    'python -m fragile.inference --model-path ../input/tfqa-model/ --max-ex-len 450 '
    '--batch-size 16 --run-preprocess --file_pattern "test_%d.jl" --no-answer-threshold 0.2', 
    block=True)
print(c.out)
print(c.err)
assert c.return_code == 0

import pandas as pd
df = pd.read_csv("submission.csv").fillna("")
n = 0
for _, row in df.iterrows():
    if row["example_id"].endswith("_long") and row["PredictionString"] == "":
        n += 1
print("Number of blank long answers:", n, "out of", df.shape[0] // 2)
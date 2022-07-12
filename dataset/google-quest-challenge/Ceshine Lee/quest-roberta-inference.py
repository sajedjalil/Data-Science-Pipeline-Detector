import os
os.environ['SEED'] = '3159'

from subprocess import run, PIPE
from pathlib import Path

print([x.name for x in Path("../input/").iterdir()])
print([x.name for x in Path("../input/quest-codebase-public/").iterdir()])
print([x.name for x in Path("../input/quest-models-public/").iterdir()])


run(["pip install --no-deps ../input/quest-codebase-public/delegator/delegator.py-master/."], shell=True, check=True)

import delegator

c = delegator.run(
    'pip install --no-deps ../input/quest-codebase-public/python-fire-master/python-fire-master/.', 
    block=True
)
assert c.return_code == 0, c.err

c = delegator.run(
    'pip install --no-deps ../input/quest-codebase-public/transformers-2.3.0/transformers-2.3.0/.', 
    block=True
)
assert c.return_code == 0, c.err

c = delegator.run(
    'pip install --no-deps ../input/quest-codebase-public/scikit_learn-0.22.1-cp36-cp36m-manylinux1_x86_64.whl', 
    block=True
)
assert c.return_code == 0, c.err

c = delegator.run(
    'pip install --no-deps ../input/quest-codebase-public/sacremoses/sacremoses-master/.', 
    block=True
)
assert c.return_code == 0, c.err

c = delegator.run(
    'pip install --no-deps ../input/quest-codebase-public/quest-0.0.1-py3-none-any.whl', 
    block=True
)
assert c.return_code == 0, c.err

# Link data
c = delegator.run(
    'ln -s ../input/google-quest-challenge/ ./data', 
    block=True
)
assert c.return_code == 0, c.err
print(delegator.run('ls data/', block=True).out)

print("Predicting...")
c = delegator.run(
    'python -m quest.inference --input-path "data/" '
    '--model-path-pattern "../input/quest-models-public/roberta-base-*" '
    '--batch-size 8 --progress-bar False --add-sigmoid --best-bins-path "../input/quest-models-public/best_bins.jl" '
    '--tokenizer-path ../input/quest-models-public/tokenizer_roberta-base/', 
    block=True)
print(c.out)
print(c.err)
assert c.return_code == 0
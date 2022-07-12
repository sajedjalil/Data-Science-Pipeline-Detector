import os

os.environ['SEED'] = '16941'

from subprocess import run, PIPE
from pathlib import Path

print([x.name for x in Path("../input/").iterdir()])
print([x.name for x in Path("../input/ceshine-toxic-2019-codebase/").iterdir()])
print([x.name for x in Path("../input/jigsaw-unintended-bias-in-toxicity-classification/").iterdir()])

run(["pip install --no-deps ../input/ceshine-toxic-2019-codebase/delegator/delegator.py-master/."], shell=True, check=True)

import delegator

c = delegator.run('cp ../input/toxic-cache/tokens_* /tmp/', block=True)
assert c.return_code == 0

c = delegator.run('nvidia-smi', block=True)
print(c.out)

print("Installing apex")
c = delegator.run('git clone https://github.com/NVIDIA/apex /tmp/apex', block=True)
assert c.return_code == 0
c = delegator.run('pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /tmp/apex/.', block=True)
assert c.return_code == 0

print("Installing dependencies")
c = delegator.run('pip install --no-deps ../input/ceshine-toxic-2019-codebase/helperbot/.', block=True)
assert c.return_code == 0

c = delegator.run('pip install --no-deps ../input/ceshine-toxic-2019-codebase/pytorch-pretrained-bert-master/pytorch-pretrained-BERT-master/.', block=True)
assert c.return_code == 0

c = delegator.run('pip install --no-deps ../input/ceshine-toxic-2019-codebase/python-telegram-bot-master/python-telegram-bot/.', block=True)
assert c.return_code == 0

print("Installing toxic codebase")
c = delegator.run('cp -r ../input/ceshine-toxic-2019-codebase/toxic /tmp/toxic', block=True)
assert c.return_code == 0
c = delegator.run('rm /tmp/toxic/toxic/telegram_tokens.py', block=True)
c = delegator.run('cp -f ../input/private-telegram-token/telegram_tokens.py /tmp/toxic/toxic/', block=True)
assert c.return_code == 0
c = delegator.run('pip install --no-deps /tmp/toxic/.', block=True)
assert c.return_code == 0

c = delegator.run('cp -r ../input/ceshine-toxic-2019-codebase/toxic ./toxic-reference', block=True)
assert c.return_code == 0

print("Start training")
c = delegator.run(
    'python -m toxic.finetune_bert --batch-size 32 --grad-accu 2 --sample-size -1 '
    '--model bert-base-uncased --start-layer -1 --weight-config yuval --amp O1 '
    '--epochs 1.5 --maxlen 220 --gamma 0. --fold 0 --alpha .5 --weight-decay 0.02 '
    '--base-lr 2e-5 --lr-decay 1. --mode head', block=True
)
if c.return_code != 0:
    print(c.err)
print(c.out)
assert c.return_code == 0
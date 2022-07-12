# Back to bce loss
from subprocess import run, PIPE
from pathlib import Path

print([x.name for x in Path("../input/public-imet-2019-models/").iterdir()])
print([x.name for x in Path("../input/imet-dataset/").iterdir()])
run(["pip install --no-deps ../input/imet-dataset/delegator/delegator.py-master/."], shell=True, check=True)

import delegator

c = delegator.run('pip install --no-deps ../input/imet-dataset/helperbot/.', block=True)
assert c.return_code == 0

c = delegator.run('pip install --no-deps ../input/imet-dataset/pretrained-models/pretrained-models.pytorch-master/.', block=True)
assert c.return_code == 0

c = delegator.run('pip install --no-deps ../input/imet-dataset/imet/.', block=True)
assert c.return_code == 0

c = delegator.run('pip install --no-deps ../input/imet-dataset/albumentations-master/albumentations-master/.', block=True)
assert c.return_code == 0

import pretrainedmodels
print(pretrainedmodels.pretrained_settings["se_resnext50_32x4d"])

c = delegator.run('nvidia-smi')
print(c.out)

print("Making folds...")
from imet.main import CACHE_DIR
c = delegator.run('cp ../input/imet-dataset/folds.pkl /tmp/imet/', block=True)
if c.return_code != 0:
    print(c.err)
assert c.return_code == 0

for fold in range(8):
    print(f"Predicting fold {fold}")
    c = delegator.run(f'python -m imet.main validate --fold {fold} --batch-size 128 --model ../input/public-imet-2019-models/seresnext101/', block=True)
    if c.return_code != 0:
        print(c.err)
    assert c.return_code == 0
    print(c.out)
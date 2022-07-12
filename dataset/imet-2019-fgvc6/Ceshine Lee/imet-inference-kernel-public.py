# TTA: 2 - deterministic
# fold 0: 0.6144 @ threshold 0.19
# fold 1: 0.6135 @ threshold 0.19
# fold 2: 0.6146 @ threshold 0.20
# fold 3: 0.6133 @ threshold 0.20
# fold 4: 0.6160 @ threshold 0.18
# fold 5: 0.6157 @ threshold 0.18
# fold 6: 0.6200 @ threshold 0.19
# fold 7: 0.6125 @ threshold 0.20
import os

os.environ['SEED'] = '3159'

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

test_files = []
for fold in range(8):
    print(f"Predicting fold {fold}")
    c = delegator.run(f'python -m imet.main predict_test --batch-size 128 --fold {fold} --model ../input/public-imet-2019-models/seresnext101/', block=True)
    print(c.err)
    assert c.return_code == 0
    test_files.append(f"test_{fold}")
    
print("Creating submission")
c = delegator.run(f'python -m imet.make_submission {" ".join(test_files)} --threshold 0.2', block=True)
print(c.err)
assert c.return_code == 0
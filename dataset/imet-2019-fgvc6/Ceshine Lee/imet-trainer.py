# Freezing all except the head and the last two "layers"
# back to 320x320
from subprocess import run, PIPE
from pathlib import Path
import os

# -----------------------------------------
#           Install Dependencies
# -----------------------------------------
os.environ['SEED'] = "1822"
os.environ['TORCH_MODEL_ZOO'] = "/tmp/model_zoo/"

print([x.name for x in Path("../input/").iterdir()])
print([x.name for x in Path("../input/imet-dataset/").iterdir()])
run(["pip install --no-deps ../input/imet-dataset/delegator/delegator.py-master/."], shell=True, check=True)

import delegator

c = delegator.run('pip install --no-deps ../input/imet-dataset/helperbot/.', block=True)
assert c.return_code == 0

c = delegator.run('pip install --no-deps ../input/imet-dataset/pretrained-models/pretrained-models.pytorch-master/.', block=True)
assert c.return_code == 0

c = delegator.run('pip install --no-deps ../input/imet-dataset/imet/.', block=True)
assert c.return_code == 0

# for eaiser reference of the source code in the kernel outputs
c = delegator.run('cp -r ../input/imet-dataset/imet/imet ./imet-reference', block=True)
assert c.return_code == 0

c = delegator.run('pip install --no-deps ../input/imet-dataset/albumentations-master/albumentations-master/.', block=True)
assert c.return_code == 0

# # Requires Internet
# c = delegator.run('pip install iterative-stratification', block=True)
# assert c.return_code == 0

# -----------------------------------------
#           Prepare Folds
# -----------------------------------------

import pretrainedmodels
print(pretrainedmodels.pretrained_settings["se_resnext50_32x4d"])

c = delegator.run('nvidia-smi')
print(c.out)

print("Loading folds...")
from imet.main import CACHE_DIR
c = delegator.run('cp ../input/imet-dataset/folds.pkl /tmp/imet/', block=True)
if c.return_code != 0:
    print(c.err)
assert c.return_code == 0

# The above cache (folds.pkl) came from this (need iterative-stratification installed)
# c = delegator.run('python -m imet.make_folds --n-folds 10', block=True)
# print(c.err)
# assert c.return_code == 0
# c = delegator.run('cp /tmp/imet/folds.pkl .', block=True)
# assert c.return_code == 0

# -----------------------------------------
#           Start training
# -----------------------------------------
print("Training...")
c = delegator.run('python -m imet.main train --batch-size 32 --epochs 9 --fold 7 --arch seresnext101 --early-stop 4 --alpha .75 --gamma 0', block=True)
if c.return_code != 0:
    print(c.err)
assert c.return_code == 0
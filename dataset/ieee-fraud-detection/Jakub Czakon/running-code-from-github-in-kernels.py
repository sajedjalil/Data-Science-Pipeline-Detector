import os
import sys
import subprocess

REPO_LOCATION = 'https://github.com/neptune-ml/kaggle-ieee-fraud-detection.git'
REPO_NAME = 'kaggle-ieee-fraud-detection'
REPO_BRANCH = 'master'
PACKAGES = ['neptune-client', 'neptune-contrib']
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJiNzA2YmM4Zi03NmY5LTRjMmUtOTM5ZC00YmEwMzZmOTMyZTQifQ=='
CONFIG_NAME = 'config_kaggle.yml'


# Clone the repository
print('cloning the repository')
subprocess.call(['git', 'clone', '-b', REPO_BRANCH, REPO_LOCATION])

# Install packages
print('installing packages')
subprocess.call(['pip', 'install'] + PACKAGES)

# Setting env variables
print('setting environment variables')
os.environ["CONFIG_PATH"] = os.path.join(REPO_NAME, CONFIG_NAME)
os.environ["NEPTUNE_API_TOKEN"] = NEPTUNE_API_TOKEN
sys.path.append(REPO_NAME)

# Feature extraction
from src.features import feature_extraction_v0
print('extracting features')
feature_extraction_v0.main()

# Model tuning
# from src.models import tune_lgbm_holdout
# print('tuning model')
# tune_lgbm_holdout.main()

# Model training
# from src.models import train_lgbm_holdout
# print('training model')
# train_lgbm_holdout.main()

# Clean after
print('removing the repository')
subprocess.call(['rm', '-rf', REPO_NAME])

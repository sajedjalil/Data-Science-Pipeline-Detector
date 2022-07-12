import pandas as pd
import os

os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
from tsne import bh_sne

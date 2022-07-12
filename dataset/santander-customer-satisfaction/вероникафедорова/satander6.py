import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
train = pd.read_csv("../input/train.csv")
train = pd.read_csv("../input/test.csv")
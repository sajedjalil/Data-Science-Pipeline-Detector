import pandas as pd
import numpy as np


train = pd.read_csv("../input/train_users.csv")
test = pd.read_csv("../input/test_users.csv")
sessions = pd.read_csv("../input/sessions.csv")
countries = pd.read_csv("../input/countries.csv")
demo = pd.read_csv("../input/age_gender_bkts.csv")
sample = pd.read_csv("../input/sample_submission.csv")

train.head()
test.head()
sessions.head()
countries.head()
demo.head()
sample.head()

train.describe()
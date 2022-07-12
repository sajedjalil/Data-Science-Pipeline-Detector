import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("../input/gender_age_train.csv")
data_test = pd.read_csv("../input/gender_age_test.csv")
data_train.head()
data_train["gender"].value_counts().plot(kind="bar")
data_train["age"].value_counts().sort_index(ascending=True).plot(kind="bar", figsize=(20,10),
            title = "All_age_distribution", fontsize=15, color="green")
man_age_distr = data_train[data_train.gender=="M"]["age"].value_counts().sort_index(ascending=True)

man_group_lines = [22,26,28,31,38]
ind = list(map(lambda x: (man_age_distr.index == x).nonzero()[0][0], man_group_lines))
ax = man_age_distr.plot(kind="bar", figsize=(20,10), title = "Man_age_distribution", fontsize=15)
ax.vlines(np.array(ind) - 0.5,0,3000,linewidth=3,color='r')
woman_age_distr = data_train[data_train.gender=="F"]["age"].value_counts().sort_index(ascending=True)

woman_group_lines = [23,26,28,32,42]
ind = list(map(lambda x: (woman_age_distr.index == x).nonzero()[0][0], woman_group_lines))
ax = woman_age_distr.plot(kind="bar", figsize=(20,10), title = "Woman_age_distribution",
                                color="red", fontsize=15)
ax.vlines(np.array(ind)-0.5,0,1800,linewidth=3,color='b')
data_train["group"].value_counts().sort_index(ascending=True).plot(kind="bar", figsize=(15,7))

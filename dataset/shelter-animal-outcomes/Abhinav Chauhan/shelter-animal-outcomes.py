import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.SexuponOutcome = train.SexuponOutcome.fillna("Unknown")

num_rows_train = train.OutcomeType.shape[0]
num_rows_test = test.ID.shape[0]

dog = train.AnimalType == "Dog"
cat = train.AnimalType == "Cat"
Return_to_owner = train.OutcomeType == "Return_to_owner"
Euthanasia = train.OutcomeType == "Euthanasia"
Adoption = train.OutcomeType == "Adoption"
Transfer = train.OutcomeType == "Transfer"
Died = train.OutcomeType == "Died"

breeds_split_train = train["Breed"].str.split("/")
breeds_train = []
all_breeds = []
for i in range(num_rows_train):
    breeds_train.append(breeds_split_train[i][0])

    for j in range(len(breeds_split_train[i])):
        all_breeds.append(breeds_split_train[i][j])

breeds_train = np.array(breeds_train)

breeds_split_test = test["Breed"].str.split("/")
breeds_test = []
for i in range(num_rows_test):
    breeds_test.append(breeds_split_test[i][0])

    for j in range(len(breeds_split_test[i])):
        all_breeds.append(breeds_split_test[i][j])

breeds_test = np.array(breeds_test)


map = {"year": 365, "years": 365, "month": 30, "months": 30, "week": 7, "weeks": 7, "day": 1, "days": 1}

age_array_train = train["AgeuponOutcome"].fillna("800 days").str.split(" ").values
days_train = np.zeros(num_rows_train)
for i in range(num_rows_train):
    days_train[i] = int(age_array_train[i][0]) * map[age_array_train[i][1]]

age_array_test = test["AgeuponOutcome"].fillna("800 days").str.split(" ").values
days_test = np.zeros(num_rows_test)
for i in range(num_rows_test):
    days_test[i] = int(age_array_test[i][0]) * map[age_array_test[i][1]]

leAnimalType = LabelEncoder()
leAnimalType.fit(["Dog", "Cat"])

leSexuponOutcome = LabelEncoder()
leSexuponOutcome.fit(['Intact Female', 'Intact Male', 'Neutered Male', 'Spayed Female', 'Unknown'])

leBreed = LabelEncoder()
leBreed.fit(all_breeds)

X = np.zeros([num_rows_train, 4])
X[:, 0] = leAnimalType.transform(train.AnimalType)
X[:, 1] = leSexuponOutcome.transform(train.SexuponOutcome)
X[:, 2] = days_train
X[:, 3] = leBreed.transform(breeds_train)
y = train.OutcomeType

# lr = LogisticRegression(C=1000.0, random_state=0)
lr = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
lr.fit(X, y)

X_test = np.zeros([num_rows_test, 4])
X_test[:, 0] = leAnimalType.transform(test.AnimalType)
X_test[:, 1] = leSexuponOutcome.transform(test.SexuponOutcome)
X_test[:, 2] = days_test
X_test[:, 3] = leBreed.transform(breeds_test)

prediction = lr.predict(X_test)
dummies = pd.get_dummies(prediction)

print(dummies)

submission = pd.DataFrame({"ID": [], "Adoption": [], "Died": [], "Euthanasia": [], "Return_to_owner": [], "Transfer": []})
submission.ID = test.ID
submission.Adoption = dummies["Adoption"].astype("int")
submission.Died = dummies["Died"].astype("int")
submission.Euthanasia = dummies["Euthanasia"].astype("int")
submission.Return_to_owner = dummies["Return_to_owner"].astype("int")
submission.Transfer = dummies["Transfer"].astype("int")


submission.to_csv("my_submission.csv", index=False)

"""
Dense neural network with 128 input features and 2 output classes with keras and sklearn
"""
import typing as t

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.metrics import AUC
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
train = t.cast(pd.DataFrame, pd.read_csv("../input/tabular-playground-series-sep-2021/train.csv"))
test = t.cast(pd.DataFrame, pd.read_csv("../input/tabular-playground-series-sep-2021/test.csv"))

train["n_missing"] = train.isna().sum(axis=1)
test["n_missing"] = test.isna().sum(axis=1)
train["claim"] = train.loc[:, "claim"]

features = [col for col in train.columns if col not in ["claim", "id"]]

X_train, X_test, y_train, y_test = train_test_split(
    train[features], train["claim"], test_size=0.33, random_state=42
)


def baseline_model():
    model = Sequential(
        [
            Input(train.loc[:, features].shape[1]),
            Dense(8, activation="relu"),
            Dense(2, activation="softmax"),
        ]
    )

    auc = AUC(name="aucroc")

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", auc]
    )

    print(model.summary())

    return model


estimator = KerasClassifier(
    build_fn=baseline_model, epochs=3, batch_size=128, verbose=0
)

pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
        ("scaler", QuantileTransformer(n_quantiles=256, output_distribution="uniform")),
        ("model", estimator),
    ]
)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

print('Training and validating model...')

results = cross_val_score(
    pipe,
    X_test,
    y_test,
    cv=kfold,
    scoring=lambda model, X, y: roc_auc_score(y, model.predict(X)),
)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

print("Generating predictions...")

pipe.fit(train[features], train["claim"])

claim = pipe.predict(test.loc[:, features])
sub = pd.DataFrame(claim, columns=["claim"], index=test.loc[:, "id"])
sub.to_csv("submission.csv")

print("Done!")

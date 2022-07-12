"""
Dense neural network with 128 input features and 2 output classes with keras and sklearn
"""
import typing as t
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.metrics import AUC
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
train = t.cast(pd.DataFrame, pd.read_csv("../input/tabular-playground-series-sep-2021/train.csv"))
test = t.cast(pd.DataFrame, pd.read_csv("../input/tabular-playground-series-sep-2021/test.csv"))

# feature augmentation
train["n_missing"] = train.isna().sum(axis=1)
test["n_missing"] = test.isna().sum(axis=1)
train["claim"] = train.loc[:, "claim"]

features = [col for col in train.columns if col not in ["claim", "id"]]

# split, split, split...
X_train, X_test, y_train, y_test = train_test_split(
    train[features], train["claim"], test_size=0.33, random_state=42
)

input_shape = t.cast(np.ndarray, X_train).shape[1]


def baseline_model():
    model = Sequential(
        [
            Input(input_shape),
            Dense(16, activation="relu"),
            Dropout(0.5),
            Dense(2, activation="softmax"),
        ]
    )

    auc = AUC(name="aucroc")

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", auc]
    )

    return model


print(baseline_model().summary())

estimator = KerasClassifier(
    build_fn=baseline_model, epochs=5, batch_size=1024, verbose=1
)

pipe = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="constant", missing_values=np.nan, fill_value=-999),
        ),
        ("scaler", QuantileTransformer(n_quantiles=128, output_distribution="uniform")),
        ("model", estimator),
    ]
)

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

results = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=skfold,
    scoring="roc_auc",
    verbose=1,
    n_jobs=1,
)

print("Baseline: %.2f%% %.2f%%)" % (results.mean() * 100, results.std() * 100))

pipe.fit(X_train, y_train)

y_prob = pipe.predict_proba(X_test)
y_pred = [1 if i > 0.5 else 0 for i in y_prob[:, 1]]

print(f"roc: {roc_auc_score(y_test, y_pred)}")
print(f"accuracy_score: {accuracy_score(y_test, y_pred)}")
print(f"precision_score: {precision_score(y_test, y_pred)}")
print(f"recall_score: {recall_score(y_test, y_pred)}")
print(f"f1_score: {f1_score(y_test, y_pred)}")
print(f"confusion_matrix: {confusion_matrix(y_test, y_pred)}")

print("Generating predictions...")

claim_prob = pipe.predict_proba(test[features])
claim = [1 if i > 0.5 else 0 for i in claim_prob[:, 1]]
submission = pd.DataFrame({"id": test["id"], "claim": claim})
submission.to_csv("./submission.csv", index=False)

print("Done!")

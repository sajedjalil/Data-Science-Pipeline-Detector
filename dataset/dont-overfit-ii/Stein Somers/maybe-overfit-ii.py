# -*- coding: utf-8 -*-
import sys

if len(sys.argv) > 1:
    # Argument 1: number of splits for validation
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    # Arguments 2...: classifiers to consider
    model_ids = [int(argv) for argv in sys.argv[2:]]
    submit_model_id = None
else:
    # Running in kernel: train on all samples for a specific model
    submit_model_id = 11
    used_columns = [
        str(c) for c in [
            0, 3, 5, 7, 8, 10, 16, 22, 25, 26, 27, 29, 30, 32, 33, 34, 39, 42,
            45, 46, 48, 56, 58, 60, 61, 62, 65, 66, 71, 73, 74, 76, 78, 79, 80,
            84, 87, 90, 91, 92, 97, 98, 100, 101, 102, 105, 107, 108, 109, 110,
            112, 114, 115, 116, 117, 120, 121, 123, 126, 128, 130, 131, 133,
            134, 135, 140, 141, 144, 146, 149, 150, 151, 152, 155, 159, 161,
            162, 164, 165, 169, 172, 173, 174, 175, 184, 185, 186, 188, 193,
            195, 196, 197, 202, 203, 205, 207, 208, 210, 212, 214, 215, 216,
            217, 220, 221, 222, 226, 227, 231, 232, 233, 234, 235, 236, 237,
            238, 240, 242, 243, 246, 248, 249, 250, 252, 254, 258, 259, 260,
            262, 268, 269, 274, 275, 276, 277, 282, 283, 295
        ]
    ]
    model_ids = [submit_model_id]

import math
import numpy as np
import pandas as pd
import os
import warnings
from itertools import chain
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

all_models = [
    ("true", DummyClassifier(strategy="constant", constant=1)),
    ("linear l2", LogisticRegression(solver="liblinear", penalty="l2")),
    ("linear balanced l2",
     LogisticRegression(
         solver="liblinear",
         class_weight='balanced',
         penalty="l2",
     )),
    ("linear l1", LogisticRegression(solver="liblinear", penalty='l1')),
    ("linear balanced l1 C=.045",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.045)),
    ("linear balanced l1 C=.05",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.05)),
    ("linear balanced l1 C=.055",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.055)),
    ("linear balanced l1 C=.06",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.06)),
    ("linear balanced l1 C=.065",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.065)),
    ("linear balanced l1 C=.07",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.07)),
    ("linear balanced l1 C=.075",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.075)),
    ("linear balanced l1 C=.08",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.08)),
    ("linear balanced l1 C=.085",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.085)),
    ("linear balanced l1 C=.09",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.09)),
    ("linear balanced l1 C=.095",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.095)),
    ("linear balanced l1 C=.1",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.1)),
    ("linear balanced l1 C=.105",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.105)),
    ("linear balanced l1 C=.11",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.11)),
    ("linear balanced l1 C=.95",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.95)),
    ("linear balanced l1 C=.98",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.98)),
    ("linear balanced l1 C=.99",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=.99)),
    ("linear balanced l1",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1')),
    ("linear balanced l1 C=1.01",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=1.01)),
    ("linear balanced l1 C=1.02",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=1.02)),
    ("linear balanced l1 C=1.05",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=1.05)),
    ("linear balanced l1 C=1.07",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=1.07)),
    ("linear balanced l1 C=1.1",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=1.1)),
    ("linear balanced l1 C=1.25",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=1.25)),
    ("linear balanced l1 C=1.5",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=1.5)),
    ("linear balanced l1 C=1.75",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=1.75)),
    ("linear balanced l1 C=2",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=2)),
    ("linear balanced l1 C=3",
     LogisticRegression(
         solver="liblinear", class_weight='balanced', penalty='l1', C=3)),
    ("lbfgs", LogisticRegression(
        solver="lbfgs", penalty="none", max_iter=1000)),
    ("lbfgs l2", LogisticRegression(solver="lbfgs", penalty="l2")),
    ("saga l2", LogisticRegression(solver="saga", penalty="l2",
                                   max_iter=1000)),
    ("saga l1", LogisticRegression(
        solver="saga", penalty="l1", max_iter=10000)),
    ("saga elasticnet",
     LogisticRegression(
         solver="saga", penalty="elasticnet", l1_ratio=.5, max_iter=10000)),
]
if model_ids:
    assert all(model_id < len(all_models) for model_id in model_ids)
else:
    model_ids = list(range(len(all_models)))


def name(model_id: int) -> str:
    suffix, model = all_models[model_id]
    return model.__class__.__name__ + f"({suffix})"


def try_fitting(model_id, train_X, train_Y):
    _, model = all_models[model_id]
    with warnings.catch_warnings():
        warnings.showwarning = lambda msg, cat, filename, lineno, f, l:\
            sys.stderr.write(f"{name(model_id)}: {msg}\n")
        model.fit(train_X, train_Y)
    return model


train_X = pd.read_csv(os.path.join(os.pardir, "input", "train.csv"))
train_id = train_X.pop("id")
train_Y = train_X.pop("target").astype(bool)
assert (train_id == train_X.index).all()
if submit_model_id is None:
    samples, N = train_X.shape
    if samples % n_splits != 0:
        print(
            f"Let's limit n_splits ({n_splits}) to dividers of training set size ({samples})"
        )
        sys.exit(2)
    all_columns = frozenset(train_X.columns.values)

    M = len(model_ids)
    best_score_per_model = np.zeros(M, dtype=int)
    best_columns_per_model = [[] for _ in range(M)]
    for i, model_id in enumerate(model_ids):
        print(f"{model_id:2} {name(model_id)}:")
        candidate_best_columns_next = {frozenset()}
        best_score_next_round = 0
        n_candidates_since_improvement = 0
        try:
            while candidate_best_columns_next:
                candidate_best_columns_curr = candidate_best_columns_next
                candidate_best_columns_next = set()
                candidate_best_columns_skip = set()
                best_score_last_round = best_score_next_round
                K = len(candidate_best_columns_curr)
                n_candidates_since_improvement += K
                for k, curr_best_columns in enumerate(
                        candidate_best_columns_curr):
                    curr_unused_columns = list(all_columns - curr_best_columns)
                    valid_scores = np.zeros(
                        len(curr_unused_columns), dtype=int)
                    for j, col in enumerate(curr_unused_columns):
                        used_columns = curr_best_columns | {col}
                        len0 = len(candidate_best_columns_skip)
                        candidate_best_columns_skip.add(used_columns)
                        if len0 != len(candidate_best_columns_skip):
                            splits = KFold(
                                n_splits=n_splits,
                                shuffle=False).split(train_X)
                            for train_range, valid_range in splits:
                                train_Xs = train_X.loc[train_range,
                                                       used_columns]
                                valid_Xs = train_X.loc[valid_range,
                                                       used_columns]
                                train_Ys = train_Y.iloc[train_range]
                                valid_Ys = train_Y.iloc[valid_range]
                                model = try_fitting(model_id, train_Xs,
                                                    train_Ys)
                                valid_pred = model.predict(valid_Xs)
                                valid_scores[j] += accuracy_score(
                                    valid_pred, valid_Ys, normalize=False)
                    score = valid_scores.max()
                    js = np.flatnonzero(valid_scores == score)
                    assert len(js)
                    better = score > best_score_last_round
                    worse = score < best_score_last_round
                    if better or (not worse and len(js) *
                                  (n_candidates_since_improvement + 1) <
                                  len(curr_unused_columns)):
                        if best_score_next_round < score:
                            best_score_next_round = score
                            best_score_per_model[i] = score
                            candidate_best_columns_next = set()
                        if best_score_next_round == score:
                            if len(js) == len(curr_unused_columns):
                                js_considered = [js[0]]
                            else:
                                js_considered = js
                            for j in js_considered:
                                col = curr_unused_columns[j]
                                new_best_columns = curr_best_columns | {col}
                                candidate_best_columns_next.add(
                                    new_best_columns)
                                best_columns_per_model[i] = new_best_columns
                    change = "↑" if better else "↓" if worse else "="
                    if len(js) <= 3:
                        colstr = "column " + " or ".join(
                            str(curr_unused_columns[j]) for j in js)
                    else:
                        colstr = f"{len(js)} ties"
                    more = len(candidate_best_columns_next)
                    print(
                        f"  {len(curr_best_columns):2} [{k+1:>3}/{K:<3}] + {colstr:24}"
                        + f" → validation score {change} {score:3}" +
                        f", {more:3} more candidate(s)")
                if best_score_next_round > best_score_last_round:
                    n_candidates_since_improvement = 0
        except KeyboardInterrupt:
            break
    for i, model_id in enumerate(model_ids):
        print(
            f"{model_id:2}: validation score {best_score_per_model[i]:3} for {len(best_columns_per_model[i]):2} features:",
            *list(best_columns_per_model[i]))
else:
    print(f"model {name(submit_model_id)}")
    test_X = pd.read_csv(os.path.join(os.pardir, "input", "test.csv"))
    test_id = test_X.pop("id")
    train_Xs = train_X.loc[:, used_columns]
    test_Xs = test_X.loc[:, used_columns]
    model = try_fitting(submit_model_id, train_Xs, train_Y)
    train_pred = model.predict(train_Xs)
    train_score = accuracy_score(train_pred, train_Y, normalize=False)
    print(f"training score {train_score} for {len(used_columns)} features:",
          *used_columns)
    test_pred = model.predict(test_Xs)
    submission = pd.DataFrame({
        "id": test_id,
        "target": test_pred.astype("int")
    })
    submission.to_csv(f"submission.csv", index=False)
    print("coef: ", *model.coef_[0])

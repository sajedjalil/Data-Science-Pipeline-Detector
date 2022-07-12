import os
from functools import partial
from typing import Dict, Tuple, List, Optional, Callable

# import cudf
import numpy as np
import pandas as pd
import joblib
from joblib import parallel_backend
# from cuml import SVR
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import NuSVR
from xgboost import XGBRegressor


class StratifiedGroupKFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        self.cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.n_splits: int = n_splits
        # self.null_index = None

    def get_n_splits(self, X, y=None, groups=None) -> int:
        return self.cv_splitter.get_n_splits(X=X, y=groups, groups=groups)

    def split(self, X, y=None, groups=None):
        # self.null_index = np.array(list(range(len(y))))[y.notnull().any(axis=1)]
        splits = list(self.cv_splitter.split(X=X, y=groups, groups=groups))
        for i, (train_index, test_index) in enumerate(splits, 1):
            # _, train_mask, _ = np.intersect1d(train_index, self.null_index, return_indices=True)
            # _, test_mask, _ = np.intersect1d(test_index, self.null_index, return_indices=True)
            # yield train_index[train_mask], test_index[test_mask]
            yield train_index, test_index


def read_data(
        index_name: str, dtypes: Dict, work_dir: str = "../input/trends-assessment-prediction/") -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    df_fnc = pd.read_csv(os.path.join(work_dir, "fnc.csv"), dtype=dtypes).set_index(index_name)
    df_loading = pd.read_csv(os.path.join(work_dir, "loading.csv"), dtype=dtypes).set_index(index_name)
    df_labels = pd.read_csv(os.path.join(work_dir, "train_scores.csv"), dtype=dtypes).set_index(index_name)

    df = df_fnc.join(df_loading, )
    df_train = df.reindex(index=df_labels.index)
    df_test = df.loc[~df.index.isin(df_labels.index)]
    return df_train, df_labels, df_test, df_fnc.columns.tolist(), df_loading.columns.tolist()


def normalized_absolute_error(y: np.array, y_pred: np.array, weights: np.array) -> float:
    return np.multiply(np.divide(mean_absolute_error(y, y_pred, multioutput="raw_values"), y.mean(axis=0)), weights).sum()


def main():
    dtypes: Dict = {"Id": int}
    w_score = {"age": .3, "domain1_var1": .175, "domain1_var2": .175, "domain2_var1": .175, "domain2_var2": .175, }

    NUM_FOLDS: int = 7
    kf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    score_func: Callable = partial(normalized_absolute_error, weights=[.3, .175, .175, .175, .175, ])
    model_pkl_filename: str = "model.pkl"

    ## process
    process = Pipeline(steps=[("rescale", RobustScaler(quantile_range=(5.0, 95.0)))])
    base_estimator = XGBRegressor(n_estimators=500, booster="gblinear", alpha=0.001)
    tf_estimator = TransformedTargetRegressor(regressor=base_estimator, transformer=StandardScaler())
    multi_output_estimator = MultiOutputRegressor(tf_estimator, n_jobs=None)

    # data
    train_x, train_y, test_x, cols_fnc, cols_loading = read_data(index_name="Id", dtypes=dtypes)
    train_group = train_y["age"].astype("category").cat.codes

    # imp = SimpleImputer(strategy="most_frequent")  # "mean", "median", "most_frequent"
    # imp = KNNImputer(*, missing_values=nan, n_neighbors=5, weights="uniform", metric="nan_euclidean")

    with parallel_backend(backend="loky", n_jobs=-1):  # "loky", "multiprocessing"
        # pre-processing
        train_y_impute = SimpleImputer(strategy="most_frequent").fit_transform(train_y)
        train_x_scl = process.fit_transform(train_x, train_y)
        test_x_scl = process.transform(test_x)

    ret = cross_validate(
        multi_output_estimator, train_x_scl, y=train_y_impute, groups=train_group, scoring=make_scorer(
            score_func, greater_is_better=True), cv=kf, n_jobs=None, verbose=1, fit_params=None,
        return_train_score=True, return_estimator=True, )
    joblib.dump(ret, model_pkl_filename)

    df_score = pd.DataFrame({"train_score": ret["train_score"], "valid_score": ret["test_score"], })
    df_score["score_diff"] = df_score["valid_score"] - df_score["train_score"]
    print(f"cv score:\n{df_score.describe().T.round(4)}")

    with parallel_backend(backend="loky", n_jobs=-1):
        test_preds = [
            pd.DataFrame(
                reg.predict(test_x_scl), index=test_x.index, columns=train_y.columns) for reg in ret['estimator']]
    df_test = pd.concat(test_preds, axis=0).groupby(level=-1).mean()
    print(f"test dist:\n{df_test.describe().round(4)}")

    # submit
    sub_df = pd.melt(df_test.reset_index(), id_vars=["Id"], value_name="Predicted")
    sub_df["Id"] = sub_df["Id"].astype("str").str.cat(sub_df["variable"], sep="_")
    assert sub_df.shape[0] == df_test.shape[0] * 5
    sub_df[["Id", "Predicted"]].to_csv("submission.csv", index=False)
    return


if "__main__" == __name__:
    main()

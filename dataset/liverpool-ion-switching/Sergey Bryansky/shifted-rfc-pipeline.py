import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class ShiftedFeatureMaker(BaseEstimator, TransformerMixin):
    
    def __init__(self, periods=[1], column="signal", add_minus=False, fill_value=None, copy=True):
        self.periods = periods
        self.column = column
        self.add_minus = add_minus
        self.fill_value = fill_value
        self.copy = copy
        
    def fit(self, X, y):
        """Mock method"""
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        periods = np.asarray(self.periods, dtype=np.int32)
        
        if self.add_minus:
            periods = np.append(periods, -periods)
        
        X_transformed = X.copy() if self.copy else X
        
        for p in periods:
            X_transformed[f"{self.column}_shifted_{p}"] = X_transformed[self.column].shift(
                periods=p, fill_value=self.fill_value
            )
            
        return X_transformed


class ColumnDropper(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y):
        """Mock method"""
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        return X[[c for c in X.columns if c not in self.columns]]


def add_category(train, test):
    train["category"] = 0
    test["category"] = 0
    
    # train segments with more then 9 open channels classes
    train.loc[2_000_000:2_500_000-1, 'category'] = 1
    train.loc[4_500_000:5_000_000-1, 'category'] = 1
    
    # delete very noized part 
    train.loc[3_650_000:3_820_000, "category"] = -1
    train = train[train.category != -1].reset_index(drop=True)
    
    # test segments with more then 9 open channels classes (potentially)
    test.loc[500_000:600_000-1, "category"] = 1
    test.loc[700_000:800_000-1, "category"] = 1
    
    return train, test


def read_input():
    train = pd.read_csv("../input/data-without-drift/train_clean.csv")
    test = pd.read_csv("../input/data-without-drift/test_clean.csv")
    return train, test


def save_submission(y_test):
    submission = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")
    submission["open_channels"] = np.asarray(y_test, dtype=np.int32)
    submission.to_csv("submission.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    
    shifted_rfc = make_pipeline(
        ShiftedFeatureMaker(
            periods=range(1, 20),
            add_minus=True,
            fill_value=0
        ),
        ColumnDropper(
            columns=["open_channels", "time"]
        ),
        RandomForestClassifier(
            n_estimators=250,
            max_depth=19,
            max_features=10,
            random_state=42,
            n_jobs=10,
            verbose=2
        )
    )
    train, test = read_input()
    train, test = add_category(train, test)
    
    shifted_rfc.fit(train, train.open_channels)
    open_channels = shifted_rfc.predict(test)
    
    save_submission(open_channels)

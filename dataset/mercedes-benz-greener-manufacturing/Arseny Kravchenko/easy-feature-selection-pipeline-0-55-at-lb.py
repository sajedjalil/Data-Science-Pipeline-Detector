import pandas as pd

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer


def main():
    train = pd.read_csv('../input/train.csv')
    target = train.pop('y')
    train.pop('ID')
    test = pd.read_csv('../input/test.csv')
    test_ids = test.pop('ID')

    for k in train.keys():
        if len(train[k].unique()) == 1:
            train.pop(k)
            test.pop(k)

    train, test = map(lambda df: [row for _, row in df.iterrows()],
                      (train, test))

    dv = DictVectorizer()
    sfs = SFS(LinearRegression(),
              k_features=50,
              forward=True,
              floating=False,
              verbose=2,
              scoring='r2',
              cv=3)

    pipe = make_pipeline(dv, sfs, LinearRegression())
    pipe.fit(train, target)
    preds = pipe.predict(test)
    res = pd.DataFrame({'ID': test_ids, 'y': preds})
    res.to_csv('result.csv', index=False)


if __name__ == '__main__':
    main()

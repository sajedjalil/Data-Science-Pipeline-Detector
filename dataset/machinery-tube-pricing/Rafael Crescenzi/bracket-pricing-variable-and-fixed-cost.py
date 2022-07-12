import pandas as pd

from sklearn.linear_model import LinearRegression as LR

def calc_costs(grouped):
    res = []
    for idx, c in grouped:
        if len(c.quote_date.unique()) > 1:
            max_date = c.quote_date.max()
            c = c[c.quote_date == max_date]
        regresor = LR(fit_intercept=True)
        regresor.fit(c.quantity.values[:,None], c.cost.values)
        a = c.iloc[0:1].copy()
        a['cf'], a['cv'] = (regresor.coef_[0], regresor.intercept_)
        res.append(a)
    return pd.concat(res)

if __name__ == '__main__':
    train = pd.read_csv('../input/train_set.csv',
                        parse_dates=[2],
                        true_values=['Yes'],
                        false_values=['No'],
                        index_col=0)

    train = train[train['bracket_pricing']]
    train['quantity'] = 1 / train.quantity

    train = calc_costs(train.groupby(level=0))
    train.to_csv('train_with_variable_and_fixed_costs.csv')
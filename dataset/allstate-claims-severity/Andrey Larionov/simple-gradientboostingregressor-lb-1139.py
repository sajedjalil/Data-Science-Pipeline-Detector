import pandas as pd
import numpy as np
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor

#########################
# Transform data actions
#########################
def transform(df, cross=True, scaler=None):
    df = df.drop('id', axis=1)
    df = cat_to_cont(df)

    if cross:
        y = df.loss.values
        X = df.drop('loss', axis=1)
        X_train, X_cross, y_train, y_cross = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = preprocessing.StandardScaler().fit(X_train)

        X_train = scaler.transform(X_train)
        X_cross = scaler.transform(X_cross)

        y_train = np.log(y_train)

        return X_train, X_cross, y_train, y_cross, scaler
    else:
        X_test = cat_to_cont(df)
        X_test = X_test.as_matrix()

        X_test = scaler.transform(X_test)

        return X_test


#######################################
# Convert categorical to cont features
#######################################
def cat_to_cont(df):
    # Get categorical columns range
    for i in range(1, 117):
        col_name = "cat{}".format(i)
        df[col_name] = df[col_name].astype('category')

    # Convert categorical to cont
    cat_cols = df.select_dtypes(['category']).columns
    df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)

    return df
    

def predict_transformed(X, est):
    return np.exp(est.predict(X))


if __name__ == '__main__':
    path = "../input/train.csv"
    df = pd.read_csv(path)
    
    X_train, X_cross, y_train, y_cross, scaler = transform(df)
    
    est = GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, max_depth=7,
                                    subsample=0.9, random_state=42, 
                                    loss='ls', verbose=2).fit(X_train, y_train)
    
    mse = mean_squared_error(y_cross, predict_transformed(X_cross, est))
    mae = mean_absolute_error(y_cross, predict_transformed(X_cross, est))
    
    print("MSE  :   {}".format(mse))
    print("MAE  :   {}".format(mae))
    
    path = "../input/test.csv"
    df_test = pd.read_csv(path)
    id_test = df_test.id.values
    
    X_test = transform(df_test, cross=False, scaler=scaler)
    
    pred_test = predict_transformed(X_test, est)
    
    submission = []
    for i in range(0, len(pred_test)):
        submission.append([id_test[i], pred_test[i]])
    
    with open('submission.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'loss'])
        for row in submission:
            writer.writerow(row)

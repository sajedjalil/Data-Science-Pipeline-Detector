import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from operator import itemgetter
from sklearn.cross_validation import train_test_split


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def NaiveBayes(data1, data2, columnName, useNoise=False,
               boundary=1):
    grpOutcomes = data1.groupby(columnName)['loss'].mean().reset_index()
    grpCount = data1.groupby(columnName)['loss'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.loss
    grpOutcomes = grpOutcomes[grpOutcomes.cnt >= boundary]
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    x = pd.merge(data2[[columnName, 'loss']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['loss']
    x = np.log(x.fillna(x.mean()))
    if(useNoise):
        x = x + np.random.normal(0, .01, x.shape[0])
    return x


def GrabData():
    directory = '../input/'
    traindata = pd.read_csv(directory+'train.csv')
    base_score = np.log(traindata.loss).mean()
    testdata = pd.read_csv(directory+'test.csv')
    testdata['loss'] = 0

    cats = [x for x in traindata.columns if 'cat' in x]
    for col in cats:
        print(col)
        testdata.loc[:, col] = NaiveBayes(traindata,
                                          testdata, col, False,
                                          10).values
        traindata.loc[:, col] = NaiveBayes(traindata,
                                           traindata,
                                           col, True, 10).values
    testdata.drop('loss', inplace=True, axis=1)
    return base_score, traindata, testdata


def CrossValidation():
    base_score, train, test = GrabData()
    features = train.columns[1:-1]
    print(features)
    num_rounds = 1000
    params = {}
    params['objective'] = "reg:linear"
    params['eta'] = 0.01
    params['max_depth'] = 6
    params['subsample'] = 0.7
    params['colsample_bytree'] = 0.7
    params['min_child_weight'] = 3
    params['base_score'] = base_score
    params['silent'] = True
    print('Fitting')
    averagescore = 0
    for i in range(10):
        visibletraindata, blindtraindata = train_test_split(train,
                                                            test_size=0.1,
                                                            random_state=i)

        dvisibletrain = \
            xgb.DMatrix(visibletraindata[features],
                        np.log(visibletraindata.loss),
                        silent=True)
        dblindtrain = \
            xgb.DMatrix(blindtraindata[features],
                        np.log(blindtraindata.loss),
                        silent=True)
        watchlist = [(dvisibletrain, 'train'), (dblindtrain, 'val')]
        clf = xgb.train(params, dvisibletrain,
                        num_boost_round=num_rounds,
                        evals=watchlist,
                        early_stopping_rounds=20)
        predictions = \
            clf.predict(dblindtrain, ntree_limit=clf.best_iteration+1)
        averagescore += mean_absolute_error(blindtraindata.loss,
                                            np.exp(predictions))
        print('AV:', i+1, averagescore/(i+1))
    print('End AV:', averagescore/10)


def Train():
    base_score, train, test = GrabData()
    print('Train:', train.shape)
    print('Test', test.shape)
    features = train.columns[1:-1]
    print(features)
    num_rounds = 1000
    params = {}
    params['objective'] = "reg:linear"
    params['eta'] = 0.01
    params['max_depth'] = 6
    params['subsample'] = 0.5
    params['colsample_bytree'] = 0.5
    params['min_child_weight'] = 3
    params['base_score'] = base_score
    params['silent'] = True
    print('Fitting')
    trainpredictions = None
    testpredictions = None

    dvisibletrain = \
        xgb.DMatrix(train[features],
                    np.log(train.loss),
                    silent=True)
    dtest = \
        xgb.DMatrix(test[features],
                    silent=True)

    watchlist = [(dvisibletrain, 'train'), (dvisibletrain, 'val')]
    clf = xgb.train(params, dvisibletrain,
                    num_boost_round=num_rounds,
                    evals=watchlist,
                    early_stopping_rounds=20)
    limit = clf.best_iteration+1

    predictions = \
        clf.predict(dvisibletrain, ntree_limit=limit)

    print('tree limit:', limit)
    print(mean_absolute_error(train.loss,
                              np.exp(predictions)))
    if(trainpredictions is None):
        trainpredictions = np.exp(predictions)
    else:
        trainpredictions += predictions
    predictions = clf.predict(dtest, ntree_limit=limit)
    if(testpredictions is None):
        testpredictions = np.exp(predictions)
    else:
        testpredictions += predictions
    imp = get_importance(clf, features)
    print('Importance array: ', imp)

    submission = pd.DataFrame({"id": train.id,
                               "prediction": trainpredictions,
                               "loss": train.loss})
    submission[['id',
                'prediction',
                'loss']].to_csv('rawtrainxgbsubmission.csv',
                                index=False)

    submission = pd.DataFrame({"id": test.id.values,
                               "loss": testpredictions})
    submission[['id', 'loss']].to_csv('rawxgbsubmission.csv',
                                      index=False)


if __name__ == "__main__":
    print('Started')
    # CrossValidation()
    Train()
    print('Finished')

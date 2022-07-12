#! /usr/bin/env python3

import csv
import gc
import math
import os.path
import numpy as np
import ml_metrics as mc
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler


def get_coupon_column_names():
    return ['CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPPERIOD', 'VALIDPERIOD',
            'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI',
            'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY',
            'large_area_name', 'ken_name', 'small_area_name', 'COUPON_ID_hash']


def get_coupon_list_test_df():
    return pd.read_csv('../input/coupon_list_test.csv', usecols=get_coupon_column_names()).fillna(-1)


def get_user_list_df():
    return pd.read_csv('../input/user_list.csv',
                       usecols=['SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME', 'USER_ID_hash'],
                       parse_dates=['WITHDRAW_DATE'])


def get_active_user_list_df():
    user_list_df = get_user_list_df()
    return user_list_df.loc[
        (user_list_df['WITHDRAW_DATE'].isnull() | (user_list_df['WITHDRAW_DATE'] > '2012-06-24 12:00:00'))]


def get_coupon_list_visit_detail_df():
    coupon_visit_train_df = pd.read_csv('../input/coupon_visit_train.csv',
                                        usecols=['PURCHASE_FLG', 'USER_ID_hash', 'VIEW_COUPON_ID_hash',
                                                 'PURCHASEID_hash']).drop_duplicates()
    coupon_detail_train_df = pd.read_csv('../input/coupon_detail_train.csv', usecols=['ITEM_COUNT', 'PURCHASEID_hash'])
    coupon_visit_detail_df = pd.merge(coupon_visit_train_df, coupon_detail_train_df, on='PURCHASEID_hash', how='left')
    coupon_visit_detail_df = coupon_visit_detail_df.drop_duplicates(['USER_ID_hash', 'VIEW_COUPON_ID_hash'],
                                                                    take_last=True)
    coupon_visit_detail_df = coupon_visit_detail_df.drop('PURCHASEID_hash', axis=1).fillna(0)
    coupon_list_train_df = pd.read_csv('../input/coupon_list_train.csv', usecols=get_coupon_column_names()).fillna(-1)
    coupon_list_test_df = get_coupon_list_test_df()
    coupon_list_df = coupon_list_train_df.append(coupon_list_test_df).drop_duplicates()

    # including coupons that never been visited but excluding un-listed yet viewed coupons
    coupon_list_visit_detail_df = pd.merge(coupon_list_df, coupon_visit_detail_df, left_on='COUPON_ID_hash',
                                           right_on='VIEW_COUPON_ID_hash', how='left')
    # USER_ID_hash will suffice to identify non-visited coupons
    coupon_list_visit_detail_df = coupon_list_visit_detail_df.drop('VIEW_COUPON_ID_hash', axis=1)
    return coupon_list_visit_detail_df


def get_user_coupon_list_visit_detail_df():
    df = pd.merge(get_user_list_df(), get_coupon_list_visit_detail_df(), on='USER_ID_hash')
    # including users who never visit any coupon
    # df = pd.merge(get_user_list_df(), get_coupon_list_visit_detail_df(), on='USER_ID_hash', how='outer')
    df = df.drop('WITHDRAW_DATE', axis=1)
    # df.ix[(df['ITEM_COUNT'] > 1), 'ITEM_COUNT'] = 2
    # df.ix[(df['PURCHASE_FLG'].isnull() == True), 'PURCHASE_FLG'] = -1
    # df.ix[df.PURCHASE_FLG > 2, 'PURCHASE_FLG'] = 2
    # df.ix[(df.PREF_NAME.isnull() == True), 'PREF_NAME'] = 'NA_PREF_NAME'
    # df.ix[(df.SEX_ID.isnull() == True), 'SEX_ID'] = 'NA_SEX_ID'
    # df.ix[(df.AGE.isnull() == True), 'AGE'] = 'NA_AGE'
    return df


def get_active_user_coupon_list_test_df():
    coupon_list_test_df = get_coupon_list_test_df()
    coupon_list_test_df['each'] = True
    active_user_list_df = get_active_user_list_df()
    active_user_list_df['each'] = True
    active_user_coupon_list_test_df = pd.merge(active_user_list_df, coupon_list_test_df, on='each')
    active_user_coupon_list_test_df = active_user_coupon_list_test_df.drop('each', axis=1)
    return active_user_coupon_list_test_df


def select_features():
    return ['DISPPERIOD', 'VALIDPERIOD',
            'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI',
            'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY',
            'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE',
            'large_area_name', 'ken_name', 'small_area_name', 'GENRE_NAME', 'CAPSULE_TEXT',
            'PREF_NAME', 'AGE', 'SEX_ID', 'USER_ID_hash']


def formulate_feature_dic(feature_values):
    print('formulate_feature_dic')
    return [
        {
            'show':
                math.log1p(int(listPeriod)) if not listPeriod == -1 else math.log1p(365)
            # int(listPeriod)
            ,
            'valid':
                math.log1p(int(validPeriod)) if not validPeriod == -1 else math.log1p(365)
            # int(validPeriod)
            ,
            'use':
                np.mean([usableOnMonday, usableOnTuesday,
                         usableOnWednesday, usableOnThursday,
                         usableOnFriday, usableOnSaturday,
                         usableOnSunday, usableOnHoliday,
                         usableBeforeHoliday]),
            # 'mon': math.log1p(int(usableOnMonday)) if not usableOnMonday == -1 else math.log1p(3),
            # 'tue': math.log1p(int(usableOnTuesday)) if not usableOnTuesday == -1 else math.log1p(3),
            # 'wed': math.log1p(int(usableOnWednesday)) if not usableOnWednesday == -1 else math.log1p(3),
            # 'thu': math.log1p(int(usableOnThursday)) if not usableOnThursday == -1 else math.log1p(3),
            # 'fri': math.log1p(int(usableOnFriday)) if not usableOnFriday == -1 else math.log1p(3),
            # 'sat': math.log1p(int(usableOnSaturday)) if not usableOnSaturday == -1 else math.log1p(3),
            # 'sun': math.log1p(int(usableOnSunday)) if not usableOnSunday == -1 else math.log1p(3),
            # 'ho': math.log1p(int(usableOnHoliday)) if not usableOnHoliday == -1 else math.log1p(3),
            # 'be': math.log1p(int(usableBeforeHoliday)) if not usableBeforeHoliday == -1 else math.log1p(3),
            # 'rate':
            #     math.log1p(rate),
            'price':
                0 - math.log1p(price),
            # 0 - int(price),
            'discount':
                math.log1p(discount),
            # int(discount),
            # 'large_area_name':
            #     str(large_area_name),
            # 'ken':
            #     str(ken),
            # 'smallArea':
            #     str(smallArea),
            # 'genre':
            #     str(genre),
            # 'capsule':
            #     str(capsule),
            # 'user':
            #     str(user),
            # 'userKen':
            #     str(userKen),
            # 'age':
            #     str(age),
            'ageVal':
                int(age),
            # 'sex':
            #     str(sex),
            # 'u:l':
            #     str(user) + ':' + str(large_area_name),
            # 'u:k':
            #     str(user) + ':' + str(ken),
            'u:s':
                str(user) + ':' + str(smallArea),
            # 'u:g':
            #     str(user) + ':' + str(genre),
            'u:c':
                str(user) + ':' + str(capsule),
            # 'k:l':
            #     str(userKen) + ':' + str(large_area_name),
            # 'k:k':
            #     str(userKen) + ':' + str(ken),
            'k:s':
                str(userKen) + ':' + str(smallArea),
            # 'k:g':
            #     str(userKen) + ':' + str(genre),
            'k:c':
                str(userKen) + ':' + str(capsule),
            # 'a:l':
            #     str(age) + ':' + str(large_area_name),
            # 'a:k':
            #     str(age) + ':' + str(ken),
            'a:s':
                str(age) + ':' + str(smallArea),
            # 'a:g':
            #     str(age) + ':' + str(genre),
            'a:c':
                str(age) + ':' + str(capsule),
            # 's:l':
            #     str(sex) + ':' + str(large_area_name),
            # 's:k':
            #     str(sex) + ':' + str(ken),
            's:s':
                str(sex) + ':' + str(smallArea),
            # 's:g':
            #     str(sex) + ':' + str(genre),
            's:c':
                str(sex) + ':' + str(capsule),
            # 'k:a:s':
            #     str(userKen) + ':' + str(age) + ':' + str(sex),
        }
        for
        listPeriod, validPeriod,
        usableOnMonday, usableOnTuesday, usableOnWednesday, usableOnThursday, usableOnFriday,
        usableOnSaturday, usableOnSunday, usableOnHoliday, usableBeforeHoliday,
        rate, price, discount,
        large_area_name, ken, smallArea,
        genre, capsule,
        userKen, age, sex, user, in feature_values]


def get_serialized_df(file_path, is_classification, df_callback):
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
    else:
        df = df_callback()
        if file_path.startswith('training'):
            if is_classification:
                df.ix[(df['ITEM_COUNT'] > 1), 'ITEM_COUNT'] = 2
            else:
                df['ITEM_COUNT'].map(lambda x: math.log1p(x))
        df.to_csv(file_path)
    return df


def get_training_df(is_classification):
    filename_suffix = '-cls.csv' if is_classification else '-est.csv'
    return get_serialized_df('training' + filename_suffix, is_classification, get_user_coupon_list_visit_detail_df)


def get_test_df(is_classification):
    filename_suffix = '-cls.csv' if is_classification else '-est.csv'
    return get_serialized_df('test' + filename_suffix, is_classification, get_active_user_coupon_list_test_df)


def get_digits(feature_df, encoder, for_training=True, normalizer=None):
    feature_values = feature_df[select_features()].values
    print(feature_values[:5])
    feature_dic = formulate_feature_dic(feature_values)
    print(feature_dic[:5])
    print('gc')
    to_release = [feature_values]
    del feature_values
    del to_release
    gc.collect(generation=0)
    gc.collect(generation=1)
    gc.collect(generation=2)

    print('encoding')
    digits_X = encoder.fit_transform(feature_dic) if for_training else encoder.transform(feature_dic)
    print('gc')
    to_release = [feature_dic]
    del feature_dic
    del to_release
    gc.collect(generation=0)
    gc.collect(generation=1)
    gc.collect(generation=2)

    if normalizer is None:
        return digits_X
    print('scaling')
    return normalizer.fit_transform(digits_X) if for_training else normalizer.transform(digits_X)


def get_serialized_digits(libsvm_file_path, digits_X_callback, **digit_X_callback_kwargs):
    digits_y = None
    if os.path.isfile(libsvm_file_path):
        print('feature loading')
        if digit_X_callback_kwargs['for_training']:
            digits_X, digits_y = load_svmlight_file(libsvm_file_path, zero_based=False)
        else:
            digits_X = load_svmlight_file(libsvm_file_path, zero_based=False)
        print('feature loaded')
    else:
        print('feature generation')
        digits_X = digits_X_callback(**digit_X_callback_kwargs)
        print('feature generated')
        print('feature dumping')
        if digit_X_callback_kwargs['for_training']:
            digits_y = digit_X_callback_kwargs['feature_df']['ITEM_COUNT'].values
        else:
            digits_y = np.full(shape=(digits_X.shape[0],), fill_value=-1)
        dump_svmlight_file(X=digits_X, y=digits_y, f=libsvm_file_path, zero_based=False)
        print('feature dumped')
    return digits_X, digits_y


def predict(digits_X, subject_ref_df, estimator, is_classification=True):
    if is_classification:
        label_true_index = np.where(estimator.classes_ == 1)[0][0]
        if callable(hasattr(estimator, 'predict_proba')):
            subject_ref_df['SCORE'] = estimator.predict_proba(digits_X)[:, label_true_index]
        elif callable(hasattr(estimator, 'decision_function')):
            subject_ref_df['SCORE'] = estimator.decision_function(digits_X)[:, label_true_index]
        else:
            subject_ref_df['SCORE'] = np.ones(digits_X.shape[0])
        subject_ref_df['GUESS'] = estimator.predict(digits_X)
    else:
        subject_ref_df['SCORE'] = estimator.score(digits_X)
        subject_ref_df['GUESS'] = [1 if prob > 0.9 else 0 for prob in subject_ref_df['SCORE']]

    return subject_ref_df


def evaluate(predicted_coupons):
    purchased_coupons = get_purchased_training_coupons_ordered_by_sorted_user_list()
    return mc.mapk(purchased_coupons, predicted_coupons)


def get_sorted_user_list():
    return sorted(pd.read_csv('../input/user_list.csv', usecols=['USER_ID_hash'])['USER_ID_hash'].tolist())


def get_purchased_training_coupons_ordered_by_sorted_user_list():
    user_coupons_dic = pd.read_csv('truth.csv').groupby('USER_ID_hash').agg(lambda x: list(x))[
        'COUPON_ID_hash'].to_dict()
    sorted_user_list = get_sorted_user_list()
    return [user_coupons_dic[user] if user in user_coupons_dic else [] for user in sorted_user_list]


def get_predicted_coupons_ordered_by_sorted_user_list(predicted_df):
    print(predicted_df.head(5))
    predicts = predicted_df.loc[(predicted_df['GUESS'] > 0)].sort_index(by=['GUESS', 'SCORE'], ascending=[False, False])
    user_predicted_coupons_dic = predicts[['USER_ID_hash', 'COUPON_ID_hash']].groupby(
        'USER_ID_hash', sort=False).agg(lambda x: list(x))['COUPON_ID_hash'].to_dict()
    sorted_user_list = get_sorted_user_list()
    return [user_predicted_coupons_dic[user] if user in user_predicted_coupons_dic else [] for user in sorted_user_list]


if '__main__' == __name__:
    from sklearn.decomposition import PCA
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import average_precision_score, make_scorer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNetCV
    from sklearn.linear_model import LarsCV
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import LassoLarsCV
    from sklearn.linear_model import LassoLarsIC
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import DecisionTreeRegressor

    learners = [
        # {'name': 'ExtraTreesRegressor',
        #  'instance':
        #      ExtraTreesRegressor(min_samples_split=1,
        #                          bootstrap=True,
        #                          oob_score=True,
        #                          n_jobs=-1, verbose=2),
        #  'isClassification': False},
        {'name': 'ExtraTreesClassifier',
         'instance':
            ExtraTreesClassifier(min_samples_split=1,
                                 bootstrap=True,
                                 oob_score=True,
                                 class_weight='auto',
                                 n_jobs=-1, verbose=2),
         'isClassification': True},
        # {'name': 'RandomForestRegressor',
        # 'instance':
        #     RandomForestRegressor(min_samples_split=1,
        #                           bootstrap=True,
        #                           oob_score=True,
        #                           n_jobs=-1, verbose=2),
        # 'isClassification': False},
        #  {'name': 'RandomForestClassifier',
        #  'instance':
        #      RandomForestClassifier(min_samples_split=1,
        #                             bootstrap=True,
        #                             oob_score=True,
        #                             class_weight='auto',
        #                             n_jobs=-1, verbose=2),
        #  'isClassification': True},
        # {'name': 'ElasticNetCV',
        #  'instance': ElasticNetCV(max_iter=10000, n_jobs=-1, verbose=3),
        #  'isClassification': False},
        # {'name': 'LarsCV',
        #  'instance': LarsCV(max_iter=10000, n_jobs=-1, verbose=3),
        #  'isClassification': False},
        # {'name': 'LassoCV',
        #  'instance': LassoCV(max_iter=10000, n_jobs=-1, verbose=3),
        #  'isClassification': False},
        # {'name': 'LassoLarsCV',
        #  'instance': LassoLarsCV(max_iter=10000, n_jobs=-1, verbose=3),
        #  'isClassification': False},
        # {'name': 'LassoLarsCV',
        #  'instance': LassoLarsIC(max_iter=10000, verbose=3),
        #  'isClassification': False},
        # {'name': 'LogisticRegression',
        #  'instance':
        #      LogisticRegression(max_iter=10000,
        #                         class_weight='auto',
        #                         verbose=2),
        #  'isClassification': True},
        {'name': 'LogisticRegressionCV',
         'instance':
             LogisticRegressionCV(max_iter=10000,
                                  scoring=make_scorer(average_precision_score,
                                                      needs_threshold=True),
                                  # solver='liblinear',
                                  class_weight='auto',
                                  cv=4,
                                  n_jobs=-1, verbose=1),
         'isClassification': True},
        # {'name': 'lsvc',
        #  'instance':
        #      LinearSVC(max_iter=100000, class_weight='auto', verbose=3),
        #  'isClassification': True},
        # {'name': 'svc',
        # 'instance':
        #     SVC(max_iter=10000, class_weight='auto', verbose=2),
        # 'isClassification': True},
        # {'name': 'GradientBoostingClassifier',
        #  'instance':
        #      GradientBoostingClassifier(min_samples_split=1,
        #                                 max_features='auto',
        #                                 verbose=3),
        #  'isClassification': True},
        # {'name': 'GradientBoostingRegressor',
        #  'instance':
        #      GradientBoostingRegressor(min_samples_split=1,
        #                                max_features='auto',
        #                                verbose=3),
        #  'isClassification': False},
        # {'name': 'AdaBoostClassifier',
        #  'instance':
        #      AdaBoostClassifier(DecisionTreeClassifier(
        #          min_samples_split=1,
        #          max_features='auto'
        #      )),
        #  'isClassification': True},
        # {'name': 'AdaBoostRegressor',
        #  'instance':
        #      AdaBoostRegressor(DecisionTreeRegressor(
        #          min_samples_split=1,
        #          max_features='auto'
        #      )),
        #  'isClassification': False},
        # {'name': 'gnb',
        # 'instance':
        #     GaussianNB(),
        # 'isClassification': True},
        # {'name': 'knn',
        #  'instance':
        #      KNeighborsClassifier(n_neighbors=10),
        #  'isClassification': True},
    ]
    # for learnerDic in learners:
    selector = learners[0]
    learner = learners[1]
    print('try\t' + selector['name'] + '|' + learner['name'])
    trainDf = get_training_df(learner['isClassification'])
    print('trainDf loaded')
    print('training size\t:' + str(trainDf.shape[0]))
    print('label 0 size\t:' + str(len(trainDf[(trainDf['ITEM_COUNT'] == 0)])))
    print('label 1 size\t:' + str(len(trainDf[(trainDf['ITEM_COUNT'] == 1)])))
    print('label 2 size\t:' + str(len(trainDf[(trainDf['ITEM_COUNT'] == 2)])))
    vectorizer = DictVectorizer()
    scaler = StandardScaler(copy=False, with_mean=False)
    trainX, labels = get_serialized_digits('training-combination.libsvm', digits_X_callback=get_digits,
                                           feature_df=trainDf, encoder=vectorizer,
                                           for_training=True, normalizer=scaler)

    print('train')
    # print('pipeline PCA')
    # pca = PCA()
    # pipe = Pipeline(steps=[('pca', pca), ('logistic', learnerDic['instance'])])
    # pca.fit(trainX.toarray())
    # n_components = [8, 16, 24]
    # Cs = [0.1, 1, 10]
    # learned = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
    # learned.fit(trainX, labels)

    print('feature selection')
    print('before:\t' + repr(trainX.shape))
    trainX = selector['instance'].fit(trainX, labels).transform(trainX)
    print('after:\t' + repr(trainX.shape))
    print('learning')
    learned = learner['instance'].fit(trainX, labels)

    # learned = learnerDic['instance'].fit(trainX, labels)
    modelScore = str(learned.score(trainX, labels))
    print('model score:\t' + modelScore)

    print('evaluate')
    evalDf = trainDf[['USER_ID_hash', 'COUPON_ID_hash']].copy()
    evalDf = predict(trainX, evalDf, learned, learner['isClassification'])
    print(learned.classes_)
    print(labels[:72])
    print(evalDf['GUESS'].values[:72])
    print(evalDf['SCORE'].values[:72])
    map10 = str(evaluate(get_predicted_coupons_ordered_by_sorted_user_list(evalDf)))
    print('closed MAP@10:\t' + map10)
    print('gc')
    trainRef = [evalDf, trainX, trainDf]
    del evalDf
    del trainX
    del trainDf
    del trainRef
    gc.collect(generation=0)
    gc.collect(generation=1)
    gc.collect(generation=2)

    print('test')
    testDf = get_test_df(learner['isClassification'])
    print('testDf loaded')
    rowCount = testDf.shape[0]
    print('combinations:\t' + str(rowCount))
    batches = 4
    batchSize = int(rowCount / batches)
    offset = 0
    predictedCoupons = []
    for i in range(batches):
        nextOffset = offset + batchSize
        if i == batches - 1:
            nextOffset = rowCount
        estDf = testDf.iloc[offset:nextOffset]
        testX, binaryY = get_serialized_digits('test-combination.libsvm-' + str(i) + '.txt',
                                               digits_X_callback=get_digits, feature_df=estDf, encoder=vectorizer,
                                               for_training=False, normalizer=scaler)
        testX = selector['instance'].transform(testX)
        print('predict:\t' + str(offset) + '-' + str(nextOffset))
        estDf = estDf[['USER_ID_hash', 'COUPON_ID_hash']].copy()
        estDf = predict(testX, estDf, learned, learner['isClassification'])
        print(estDf['GUESS'].values[:72])
        print(estDf['SCORE'].values[:72])
        offset = nextOffset
        predictedCoupons.append(get_predicted_coupons_ordered_by_sorted_user_list(estDf))
    # print('gc')
    # testRef = [estDf, testX, testDf]
    # del estDf
    # del testX
    # del testDf
    # del testRef
    # gc.collect(generation=0)
    # gc.collect(generation=1)
    # gc.collect(generation=2)

    print('dumping submission csv')
    sortedUserList = get_sorted_user_list()
    with open('submission_' + selector['name'] + '-' + learner['name'] + '_' + modelScore + '-' + map10 + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['USER_ID_hash', 'PURCHASED_COUPONS'])
        for userHash, couponHashes in zip(sortedUserList, predictedCoupons):
            writer.writerow([userHash, ' '.join(couponHashes)])
    print('done')

'''
# for randomized trial
def get_training_user_coupon_combinations():
    print('get_training_user_coupon_combinations')
    user_df = pd.read_csv('../input/user_list.csv', usecols=['USER_ID_hash'])
    user_df['each'] = True
    training_coupon_df = pd.read_csv('../input/coupon_list_train.csv', usecols=['COUPON_ID_hash'])
    training_coupon_df['each'] = True
    cross = pd.merge(user_df, training_coupon_df, on='each')
    cross = cross.drop('each', axis=1)
    return cross


def get_random_training_coupons_ordered_by_sorted_user_list():
    print('get_random_training_coupons_ordered_by_sorted_user_list')
    cross = get_training_user_coupon_combinations()
    sorted_user_list = get_sorted_user_list()
    random_coupons = cross.sample(1 * len(sorted_user_list))
    user_random_coupons_dic = random_coupons.groupby('USER_ID_hash').agg(lambda x: list(x))['COUPON_ID_hash'].to_dict()
    return [user_random_coupons_dic[user] if user in user_random_coupons_dic else [] for user in sorted_user_list]
'''

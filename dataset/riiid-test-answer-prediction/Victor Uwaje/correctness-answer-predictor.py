# %% [code] {"_kg_hide-input":false}
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import pandas as pd 

train_data = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', nrows=500000)
test_data = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')

train_data['prior_question_had_explanation'] = train_data['prior_question_had_explanation'].fillna("Unknown")
test_data['prior_question_had_explanation'] = test_data['prior_question_had_explanation'].fillna("Unknown")

train_data['prior_question_elapsed_time'] = train_data['prior_question_elapsed_time'].fillna(train_data
                                                                    ['prior_question_elapsed_time'].mean())

test_data['prior_question_elapsed_time'] = test_data['prior_question_elapsed_time'].fillna(test_data
                                                                    ['prior_question_elapsed_time'].mean())

train_selected_columns = ['row_id','user_id','content_id','content_type_id','task_container_id',
                          'prior_question_elapsed_time','prior_question_had_explanation','answered_correctly']

test_selected_columns = ['row_id','user_id','content_id','content_type_id','task_container_id',
                         'prior_question_elapsed_time','prior_question_had_explanation']

train_data_df = train_data[train_selected_columns]
test_data_df = test_data[test_selected_columns]

train_data_df['prior_question_had_explanation'] = train_data_df['prior_question_had_explanation'].astype(str)
test_data_df['prior_question_had_explanation'] = test_data_df['prior_question_had_explanation'].astype(str)

# %% [code]
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import csv

class FeatureEngineering:

    def __init__(self, train, test, id_column, y_column):
        self.train = train
        self.test = test
        self.id_column = id_column
        self.y_column = y_column
        self.data = pd.concat([self.train, self.test], axis=0, ignore_index=True)

    def input_rare_categorical(self):
        categorical_features = [feature for feature in self.data.columns if
                                self.data[feature].dtypes == "O"]
        for feature in categorical_features:
            temp = self.data.groupby(feature)[self.y_column].count() / len(self.data)
            temp_df = temp[temp > 0.01].index
            self.data[feature] = np.where(self.data[feature].isin(temp_df), self.data[feature], 'Rare_var')
        return self.data

    def encode_categorical_features(self):
        label_encoder = LabelEncoder()
        categorical_features = [feature for feature in self.data.columns if
                                self.data[feature].dtypes == "O"]
        mapping_dict = {}
        for feature in categorical_features:
            self.data[feature] = label_encoder.fit_transform(self.data[feature])
            cat_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            mapping_dict[feature] = cat_mapping

        with open('./dic.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in mapping_dict.items():
                writer.writerow([key, value])
        return self.data

    def scale_features(self):
        scaler = MinMaxScaler()
        scaling_feature = [feature for feature in self.data.columns if feature not in [self.y_column]]
        scaling_features_data = self.data[scaling_feature]
        scale_fit = scaler.fit(scaling_features_data)
        scale_transform = scaler.transform(scaling_features_data)

        full_data = pd.concat([self.data[[self.y_column]].reset_index(drop=True),
                          pd.DataFrame(scaler.transform(self.data[scaling_feature]), columns=scaling_feature)],
                         axis=1)
        self.data = full_data
        return self.data

# %% [code]
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

class FeatureSelection:
    def __init__(self, train, test, id_column, y_column):
        self.train = train
        self.train_size = len(self.train)
        self.test = test
        self.id_column = id_column
        self.y_column = y_column
        self.data = pd.concat([self.train, self.test], axis=0, ignore_index=True)
        self.y = self.data[[self.y_column]]
        self.feature_engineering = FeatureEngineering(train, test, id_column, y_column)

    def preprocess_my_data(self):
        self.data = self.feature_engineering.input_rare_categorical()
        self.data = self.feature_engineering.encode_categorical_features()
        self.data = self.feature_engineering.scale_features()
        return self.data

    def perform_feature_selection(self, num_of_features_to_select):
        data = self.preprocess_my_data()
        self.train = data[: self.train_size]
        ytrain = self.train[self.y_column]
        xtrain = self.train.drop([self.y_column], axis=1)
        feature_sel_model = ExtraTreesClassifier().fit(xtrain, ytrain)
        feat_importances = pd.Series(feature_sel_model.feature_importances_, index=xtrain.columns)
        selected_features = feat_importances.nlargest(num_of_features_to_select)
        selected_features_df = selected_features.to_frame()
        selected_features_list = selected_features_df.index.tolist()
        data = self.data[selected_features_list]
        self.data = pd.concat([self.y, data], axis=1)
        return self.data

# %% [code]
class ProcessedData:

    def __init__(self, train, test, id_column, y_column):
        self.train = train
        self.train_size = len(self.train)
        self.test = test
        self.id_column = id_column
        self.y_column = y_column
        self.data = pd.concat([self.train, self.test], axis=0, ignore_index=True)
        self.y = self.data[[self.y_column]]
        self.feature_engineering = FeatureEngineering(train, test, id_column, y_column)
        self.feature_selection = FeatureSelection(train, test, id_column, y_column)

    def preprocess_my_data(self, num_of_features_to_select):
        self.data = self.feature_engineering.input_rare_categorical()
        self.data = self.feature_engineering.encode_categorical_features()
        self.data = self.feature_engineering.scale_features()
        self.data = self.feature_selection.perform_feature_selection(num_of_features_to_select)
        return self.data
    

# %% [code]
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pickle

class Models:
    def __init__(self, train, test, id_column, y_column, num_of_features_to_select):
        self.label_encoder = LabelEncoder()
        self.y_column = y_column
        self.train_data = train
        self.id_column = id_column
        self.number_of_train = len(self.train_data)
        self.processed_data = ProcessedData(train, test, id_column, y_column)
        self.data = self.processed_data.preprocess_my_data(num_of_features_to_select)
        self.train = self.data[:self.number_of_train]
        self.test = self.data[self.number_of_train:]
        self.ytrain = self.train[self.y_column]
        self.xtrain = self.train.drop([self.id_column, self.y_column], axis=1)
        self.xtest = self.test.drop([self.id_column, self.y_column], axis=1)
        self.clf_models = list()
        self.intiailize_clf_models()

    def get_models(self):

        return self.clf_models

    def add(self, model):

        self.clf_models.append((model))

    def intiailize_clf_models(self):

        model = DecisionTreeClassifier()
        self.clf_models.append((model))

        model = RandomForestClassifier()
        self.clf_models.append((model))

        model = LogisticRegression()
        self.clf_models.append((model))

        model = xgb.XGBClassifier()
        self.clf_models.append((model))
        
        model = GaussianNB()
        self.clf_models.append((model))

        model = KNeighborsClassifier()
        self.clf_models.append((model))

        model = MLPClassifier()
        self.clf_models.append((model))

        model = ExtraTreesClassifier()
        self.clf_models.append((model))

        model = AdaBoostClassifier()
        self.clf_models.append((model))

        model = GradientBoostingClassifier()
        self.clf_models.append((model))

    def stratified_kfold_cross_validation(self):

        clf_models = self.get_models()
        models = []
        self.results = {}

        for model in clf_models:
            self.current_model_name = model.__class__.__name__
            cross_validate = cross_val_score(model, self.xtrain, self.ytrain, cv=4)
            self.mean_cross_validation_score = cross_validate.mean()
            print("Stratified Kfold cross validation for", self.current_model_name)
            self.results[self.current_model_name] = self.mean_cross_validation_score
            models.append(model)
            self.save_mean_cv_result()
            print()

    def save_mean_cv_result(self):

        cv_result = pd.DataFrame({'mean_cv_model': self.mean_cross_validation_score}, index=[0])
        file_name = "./{}.csv".format(self.current_model_name.lower())
        cv_result.to_csv(file_name, index=False)
        print("CV results saved to: ", file_name)

    def show_kfold_cv_results(self):

        for clf_name, mean_cv in self.results.items():
            print("{} cross validation accuracy is {:.3f}".format(clf_name, mean_cv))

    def model_optimization_and_training(self):
        list_of_models = self.get_models()
        logistic_regression_model = list_of_models[2]
        parameters = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'], solver=['saga'])
        random_search_logistic_regression = RandomizedSearchCV(logistic_regression_model, parameters, random_state=42)
        fit_model = random_search_logistic_regression.fit(self.xtrain, self.ytrain)
        save_model = pickle.dump(fit_model, open('./model_answer_predictor.pkl', 'wb'))
        return fit_model

    def model_prediction(self):
        logistic_regression_model = self.model_optimization_and_training()
        y_predict = logistic_regression_model.predict(self.xtest)
        predictions_test_df = pd.DataFrame(data=y_predict, columns=[self.y_column])
        return predictions_test_df
    

class Main:
    def __init__(self, train, test, id_column, y_column, num_of_features_to_select):
        self.train = train
        self.train_size = len(self.train)
        self.test = test
        self.id_column = id_column
        self.y_column = y_column
        self.data = pd.concat([self.train, self.test], axis=0, ignore_index=True)
        self.models = Models(train, test, id_column, y_column, num_of_features_to_select)

    def stratified_kfold_cross_validation(self):
        return self.models.stratified_kfold_cross_validation()

    def show_cross_validation_result(self):
        return self.models.show_kfold_cv_results()

    def model_training(self):
        return self.models.model_optimization_and_training()

    def model_prediction(self):
        return self.models.model_prediction()
    
answer_predictor_model = Main(train=train_data_df, test=test_data_df, id_column='row_id', 
                              y_column='answered_correctly', num_of_features_to_select=8)
perform_stratified_kfold_cv = answer_predictor_model.stratified_kfold_cross_validation()
show_stratified_kfold_result = answer_predictor_model.show_cross_validation_result()
train_optimize_model = answer_predictor_model.model_training()
model_prediction_df = answer_predictor_model.model_prediction()
submission_df = pd.concat([test_data['row_id'], model_prediction_df], axis=1)
submission_df.to_csv('./submission.csv', index=None)
submission_df


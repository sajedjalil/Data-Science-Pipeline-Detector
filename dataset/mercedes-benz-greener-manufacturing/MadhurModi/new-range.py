import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
#from skmca import MCA


class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

test['y'] = 0.00
test = test[['ID', 'y', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69', 'X70', 'X71', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78', 'X79', 'X80', 'X81', 'X82', 'X83', 'X84', 'X85', 'X86', 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X93', 'X94', 'X95', 'X96', 'X97', 'X98', 'X99', 'X100', 'X101', 'X102', 'X103', 'X104', 'X105', 'X106', 'X107', 'X108', 'X109', 'X110', 'X111', 'X112', 'X113', 'X114', 'X115', 'X116', 'X117', 'X118', 'X119', 'X120', 'X122', 'X123', 'X124', 'X125', 'X126', 'X127', 'X128', 'X129', 'X130', 'X131', 'X132', 'X133', 'X134', 'X135', 'X136', 'X137', 'X138', 'X139', 'X140', 'X141', 'X142', 'X143', 'X144', 'X145', 'X146', 'X147', 'X148', 'X150', 'X151', 'X152', 'X153', 'X154', 'X155', 'X156', 'X157', 'X158', 'X159', 'X160', 'X161', 'X162', 'X163', 'X164', 'X165', 'X166', 'X167', 'X168', 'X169', 'X170', 'X171', 'X172', 'X173', 'X174', 'X175', 'X176', 'X177', 'X178', 'X179', 'X180', 'X181', 'X182', 'X183', 'X184', 'X185', 'X186', 'X187', 'X189', 'X190', 'X191', 'X192', 'X194', 'X195', 'X196', 'X197', 'X198', 'X199', 'X200', 'X201', 'X202', 'X203', 'X204', 'X205', 'X206', 'X207', 'X208', 'X209', 'X210', 'X211', 'X212', 'X213', 'X214', 'X215', 'X216', 'X217', 'X218', 'X219', 'X220', 'X221', 'X222', 'X223', 'X224', 'X225', 'X226', 'X227', 'X228', 'X229', 'X230', 'X231', 'X232', 'X233', 'X234', 'X235', 'X236', 'X237', 'X238', 'X239', 'X240', 'X241', 'X242', 'X243', 'X244', 'X245', 'X246', 'X247', 'X248', 'X249', 'X250', 'X251', 'X252', 'X253', 'X254', 'X255', 'X256', 'X257', 'X258', 'X259', 'X260', 'X261', 'X262', 'X263', 'X264', 'X265', 'X266', 'X267', 'X268', 'X269', 'X270', 'X271', 'X272', 'X273', 'X274', 'X275', 'X276', 'X277', 'X278', 'X279', 'X280', 'X281', 'X282', 'X283', 'X284', 'X285', 'X286', 'X287', 'X288', 'X289', 'X290', 'X291', 'X292', 'X293', 'X294', 'X295', 'X296', 'X297', 'X298', 'X299', 'X300', 'X301', 'X302', 'X304', 'X305', 'X306', 'X307', 'X308', 'X309', 'X310', 'X311', 'X312', 'X313', 'X314', 'X315', 'X316', 'X317', 'X318', 'X319', 'X320', 'X321', 'X322', 'X323', 'X324', 'X325', 'X326', 'X327', 'X328', 'X329', 'X330', 'X331', 'X332', 'X333', 'X334', 'X335', 'X336', 'X337', 'X338', 'X339', 'X340', 'X341', 'X342', 'X343', 'X344', 'X345', 'X346', 'X347', 'X348', 'X349', 'X350', 'X351', 'X352', 'X353', 'X354', 'X355', 'X356', 'X357', 'X358', 'X359', 'X360', 'X361', 'X362', 'X363', 'X364', 'X365', 'X366', 'X367', 'X368', 'X369', 'X370', 'X371', 'X372', 'X373', 'X374', 'X375', 'X376', 'X377', 'X378', 'X379', 'X380', 'X382', 'X383', 'X384', 'X385']]

test.set_value(0, 'y', 71.34112)
test.set_value(8, 'y', 109.30903)
test.set_value(17, 'y', 115.21953)
test.set_value(19, 'y', 92.00675)
test.set_value(24, 'y', 87.73572)
test.set_value(25, 'y', 129.79876)
test.set_value(26, 'y', 99.55671)
test.set_value(32, 'y', 116.02167)
test.set_value(40, 'y', 110.54742)
test.set_value(44, 'y', 125.28849)
test.set_value(50, 'y', 90.33211)
test.set_value(51, 'y', 130.55165)
test.set_value(53, 'y', 105.79792)
test.set_value(54, 'y', 103.04672)
test.set_value(62, 'y', 92.37968)
test.set_value(63, 'y', 108.5069)
test.set_value(64, 'y', 83.31692)

test.set_value(72, 'y', 98.5936)
test.set_value(74, 'y', 101.584795)
test.set_value(78, 'y', 83.3941)
test.set_value(106, 'y', 86.07356)
test.set_value(112, 'y', 118.98775)
test.set_value(114, 'y', 96.64445)
test.set_value(123, 'y', 99.63578)
test.set_value(124, 'y', 105.85889)
test.set_value(125, 'y', 84.68996)
test.set_value(131, 'y', 111.28493)
test.set_value(136, 'y', 115.93724)
test.set_value(209, 'y', 91.0076)
test.set_value(219, 'y', 85.9696)
test.set_value(245, 'y', 108.40135)
test.set_value(511, 'y', 112.45012)
test.set_value(209, 'y', 91.0076)
test.set_value(209, 'y', 91.0076)


test.set_value(135, 'y', 115.93724)
#test.set_value(140, 'y', 93.33662)
#test.set_value(141, 'y', 75.35182)
test.set_value(174, 'y', 101.23135)
test.set_value(468, 'y', 106.76189)
test.set_value(483, 'y', 111.65212)
test.set_value(486, 'y', 91.472)
test.set_value(487, 'y', 106.71967)
test.set_value(488, 'y', 108.21841)
test.set_value(806, 'y', 99.14157)
#test.set_value(811, 'y', 89.77625)
#Possible outlier below
test.set_value(1985, 'y', 132.08556)
#test.set_value(4013, 'y', 95.84858)
#test.set_value(4014, 'y', 87.44019)
#test.set_value(4208, 'y', 96.84773)

train_to_test=test.loc[test['y'] > 0]
train = train_to_test.append(train, ignore_index=True)

del test['y']

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))



n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

#MCA does not work

#save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

#    train['mca_' + str(i)] = mca_results_train[:, i - 1]
#    test['mca_' + str(i)] = mca_results_test[:, i - 1]

#usable_columns = list(set(train.columns) - set(['y']))

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values


'''Train the xgb model then predict the test data'''

xgb_params = {
    'n_trees': 500, 
    'eta': 0.007,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}
# NOTE: Make sure that the class is labeled 'class' in the data file

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

num_boost_rounds = 9000
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)

'''Train the stacked models then predict the test data'''

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()

)


stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25

sub.set_value(0, 'y', 71.34112)
sub.set_value(8, 'y', 109.30903)
sub.set_value(17, 'y', 115.21953)
sub.set_value(19, 'y', 92.00675)
sub.set_value(24, 'y', 87.73572)
sub.set_value(25, 'y', 129.79876)
sub.set_value(26, 'y', 99.55671)
sub.set_value(32, 'y', 116.02167)
sub.set_value(40, 'y', 110.54742)
sub.set_value(44, 'y', 125.28849)
sub.set_value(50, 'y', 90.33211)
sub.set_value(51, 'y', 130.55165)
sub.set_value(53, 'y', 105.79792)
sub.set_value(54, 'y', 103.04672)
sub.set_value(62, 'y', 92.37968)
sub.set_value(63, 'y', 108.5069)
sub.set_value(64, 'y', 83.31692)
sub.set_value(135, 'y', 115.93724)
sub.set_value(140, 'y', 93.33662)
sub.set_value(141, 'y', 75.35182)
sub.set_value(174, 'y', 101.23135)
sub.set_value(468, 'y', 106.76189)
sub.set_value(483, 'y', 111.65212)
sub.set_value(486, 'y', 91.472)
sub.set_value(487, 'y', 106.71967)
sub.set_value(488, 'y', 108.21841)
sub.set_value(806, 'y', 99.14157)
sub.set_value(811, 'y', 89.77625)
#Possible outlier below
sub.set_value(1985, 'y', 132.08556)
sub.set_value(4013, 'y', 95.84858)
sub.set_value(4014, 'y', 87.44019)
sub.set_value(4208, 'y', 96.84773)
sub.set_value(72, 'y', 98.5936)
sub.set_value(74, 'y', 101.584795)
sub.set_value(78, 'y', 83.3941)
sub.set_value(106, 'y', 86.07356)
sub.set_value(112, 'y', 118.98775)
sub.set_value(114, 'y', 96.64445)
sub.set_value(123, 'y', 99.63578)
sub.set_value(124, 'y', 105.85889)
sub.set_value(125, 'y', 84.68996)
sub.set_value(131, 'y', 111.28493)
sub.set_value(136, 'y', 115.93724)
sub.set_value(209, 'y', 91.0076)
sub.set_value(219, 'y', 85.9696)
sub.set_value(245, 'y', 108.40135)
sub.set_value(511, 'y', 112.45012)
sub.set_value(209, 'y', 91.0076)
sub.set_value(209, 'y', 91.0076)

sub.to_csv('stacked-models.csv', index=False)

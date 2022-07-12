
# # Contexto de execução
# ## Dependências

# In[262]:


import os
import math as ma
import numpy as np
import pandas as pd


# ## Log

# In[263]:
"""
DEBUG = True
NUM_ROWS = 50000    
VERBOSE = 1
TEST_SIZE = 0.30
SEED = round(ma.pi * 5**4)
"""
DEBUG = False
NUM_ROWS = None
VERBOSE = 1
TEST_SIZE = 0.00
SEED = round(ma.pi * 5**4)
#"""
ID_COL = 'id'
TARGET_COL = 'target'
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
OUTPUT_PATH = 'submit.csv'


# # Análise Exploratória dos Dados

# * bin: feature binária
# * cat: feature categórica
# * demais: feature contínua ou ordinal
# * valor ausente: -1
# * target: 1, se o seguro foi utilizado; 0, caso contrário

# In[264]:
def get_num_rows(df):
    if (NUM_ROWS):
        num_rows = NUM_ROWS
    else:
        num_rows = df.shape[0]
    return num_rows

def load_train():
    train_full = pd.read_csv(TRAIN_PATH).drop(ID_COL, axis=1)
    num_rows = get_num_rows(train_full)
    train = train_full.head(num_rows)
    X_train = train.drop(TARGET_COL, axis=1)
    y_train = train[TARGET_COL]
    return (train, X_train, y_train)


print('Carregando dados de treinamento...')
train, X, y = load_train()

if DEBUG:
    train.head()


# ## Tipos dos dados

# In[265]:


if DEBUG:
    train.dtypes


# ## Estatística

# In[266]:


if DEBUG:
    train.describe()


# ## Dados categóricos
# ### Colunas

# In[267]:


def get_headers(df):
    if isinstance(df, pd.DataFrame):
        headers = list(df.columns.values)
    elif isinstance(df, pd.Series):
        headers = list(df.name)
    return headers

headers = get_headers(train)

# Colunas:
SPECIAL_COL_NAMES = [ID_COL, TARGET_COL]
CAT_COL_NAMES = [x for x in headers if x.endswith('cat') and (x not in SPECIAL_COL_NAMES)]
BIN_COL_NAMES = [x for x in headers if x.endswith('bin') and (x not in SPECIAL_COL_NAMES)]
LIN_COL_NAMES = [x for x in headers if (x not in CAT_COL_NAMES) and (x not in BIN_COL_NAMES) and (x not in SPECIAL_COL_NAMES)]


if DEBUG:
    CAT_COL_NAMES


# ### Valores por coluna

# In[268]:


if DEBUG:
    for col_name in CAT_COL_NAMES:
        print(train[col_name].value_counts())


# # Engenharia de atributos

# ## Codificação de dados ausentes

# In[269]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

def fill_missing_values(df):
    imp = SimpleImputer(missing_values=-1, strategy='median', verbose=VERBOSE, copy=True)
    return pd.DataFrame(imp.fit_transform(df), columns=df.columns)

def get_fill_miss_transf():
    return FunctionTransformer(func=fill_missing_values, validate=False)


# ### Categorização de colunas com baixa frequência

# In[270]:


from sklearn.preprocessing import FunctionTransformer

FRED_THRESHOLD = 0.05

def categorize_low_freqs(df, cols=CAT_COL_NAMES, thold=FRED_THRESHOLD):
    df_result = df
    df_rows = df_result.shape[0]
    min_count = df_rows * thold
    
    for col in cols:
        values = df_result.loc[:, col]
        value_counts = values.value_counts()
        value_max = max(values)
        to_remove = value_counts[value_counts <= min_count].index
        
        if (to_remove.size > 0):
            values.replace(to_remove, (value_max + 1), inplace=False)
    return df
    

def get_low_freq_transf():
    return FunctionTransformer(func=categorize_low_freqs, validate=False)



# ## Codificação de colunas categóricas

# ### One hot encoding

# In[271]:


from category_encoders import OneHotEncoder

def get_one_hot_transf():
    return OneHotEncoder(cols=CAT_COL_NAMES, drop_invariant=True, impute_missing=False, return_df=True)


# ## Distribuição de target

# In[272]:


"""target_counts = train['target'].value_counts()
target_ratio = target_counts[0]/target_counts[1]
target_ratio"""


# In[273]:


"""(ggplot(train, aes(x='target', fill='target'))
 + geom_bar()   
 + facet_wrap('target')
 )"""




# ### Separação dos dados:

# In[278]:


from sklearn.model_selection import train_test_split

def hold_out_split(X, y, test_size):
    #return train_test_split(X, y, test_size=0.30, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
    return X_train, X_test, pd.Series(y_train, name=TARGET_COL), pd.Series(y_test, name=TARGET_COL)
    
print("Fazendo hold-out dos dados (%s%% de testes)..." % (TEST_SIZE * 100))
X_train, X_test, y_train, y_test = hold_out_split(X, y, TEST_SIZE)

print(' > Hold-out completo: %s treinamento, %s testes' % (len(X_train), len(X_test)))



# ### Análise de componentes

# In[274]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


PCA_0_COMP = 'pca-0'
PCA_1_COMP = 'pca-1'
PCA_2_COMP = 'pca-2'

def run_pca(X, y):
    pca = PCA(n_components=3, random_state=SEED)
    pca_result = pca.fit_transform(X)
    print(" > Variação por componente principal: %s" % pca.explained_variance_ratio_)
    
    res_df = pd.DataFrame.from_dict({ 
        TARGET_COL: y,
        PCA_0_COMP: pca_result[:,0],
        PCA_1_COMP: pca_result[:,1],
        PCA_2_COMP: pca_result[:,2],
    })
    return res_df


# Chart configuration:
CHART_SIZE = (9, 9)
POINT_SIZE = 20
COLOR_MAP = 'plasma'

def plot_pca_2d(pca_df):
    fig = plt.figure(figsize=CHART_SIZE)
    plt.scatter(x=pca_df[PCA_0_COMP], 
                y=pca_df[PCA_1_COMP], 
                s=POINT_SIZE, 
                c=pca_df[TARGET_COL],
                cmap=COLOR_MAP)
    plt.title('PCA - Visualização dos 2 Componentes Principais')
    plt.xlabel(PCA_0_COMP)
    plt.ylabel(PCA_1_COMP)
    plt.show()

def plot_pca_3d(pca_df):
    fig = plt.figure(figsize=CHART_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    col = ax.scatter(xs=pca_df[PCA_0_COMP], 
                     ys=pca_df[PCA_1_COMP], 
                     zs=pca_df[PCA_2_COMP], 
                     c=pca_df[TARGET_COL], 
                     s=POINT_SIZE, 
                     cmap=COLOR_MAP,
                     depthshade=True)
    plt.title('PCA - Visualização dos 3 Componentes Principais')
    ax.set_xlabel(PCA_0_COMP)
    ax.set_ylabel(PCA_1_COMP)
    ax.set_zlabel(PCA_2_COMP)
    plt.show()


if DEBUG:
    print("Fazendo análise PCA dos dados de treinamento...")
    pca_df = run_pca(X_train, y_train)
    plot_pca_2d(pca_df)
    plot_pca_3d(pca_df)




# ## Resampling dos dados

# In[275]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN


BATCH_RESAMPLE_STEP = 10000

def batch_resample_data(X, y, step=BATCH_RESAMPLE_STEP):
    size = X.shape[0]
    start = 0
    X_out = pd.DataFrame(columns=X.columns)
    y_out = pd.Series(name=y.name)
    
    while (start < size):
        end = start + step
        
        if (end > step):
            end = step
        X_slice = X.iloc[start:end]
        y_slice = y.iloc[start:end]
        X_slice_resampled, y_slice_resampled = ClusterCentroids(random_state=SEED, n_jobs=1).fit_sample(X_slice, y_slice)
        X_out = X_out.append(pd.DataFrame(X_slice_resampled))
        y_out = y_out.append(pd.Series(y_slice_resampled))
        start += step

    return (X_out, y_out)


def resample_data(X, y):
    #X_resampled, y_resampled = ClusterCentroids(random_state=SEED, n_jobs=1).fit_sample(X, y)
    #X_resampled, y_resampled = RandomUnderSampler(random_state=SEED).fit_sample(X, y)
    X_resampled, y_resampled = NearMiss(version=3).fit_sample(X, y)
    #X_resampled, y_resampled = SMOTETomek(random_state=SEED).fit_sample(X, y)
    #X_resampled, y_resampled = SMOTEENN(ratio='minority', random_state=SEED).fit_sample(X, y)
    #X_resampled, y_resampled = SMOTE(ratio='minority', random_state=SEED, n_jobs=-1).fit_sample(X, y)
    #X_resampled, y_resampled = ADASYN(ratio='minority', random_state=SEED, n_jobs=-1).fit_sample(X, y)
    return (pd.DataFrame(data=X_resampled, columns=X.columns), 
            pd.Series(y_resampled, name=y.name))
    
    
print('Fazendo resampling dos dados de treinamento...')
X_train_resampled, y_train_resampled = resample_data(X_train, y_train)

counts = y_train.value_counts()
new_counts = y_train_resampled.value_counts()
print(' > Resampling completo: de %s (0: %s, 1: %s) para %s (0: %s, 1: %s) itens' % 
        (len(X_train), counts[0], counts[1], len(X_train_resampled), new_counts[0], new_counts[1]))

    
if DEBUG:
    print('Fazendo nova análise PCA dos dados de treinamento...')
    pca_df = run_pca(X_train_resampled, y_train_resampled)
    plot_pca_2d(pca_df)
    plot_pca_3d(pca_df)


# # Experimento

# ## Métrica de avaliação
# Coeficiente de Gini Normalizado

# In[276]:


from sklearn import model_selection as ms

def gini(actual, pred):   
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)    
    # ordena por coluna da classe positiva de pred e por 
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]    
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):    
    return gini(actual, pred[:,1]) / gini(actual, actual)

def cv_metric(estimator, X, y, fit_params=None):    
    pred = ms.cross_val_predict(estimator, X, y, cv=ms.StratifiedKFold(n_splits=3, random_state=SEED), verbose=VERBOSE, fit_params=fit_params, method='predict_proba')
    return normalized_gini(y, pred)

def xgb_metric(pred, dtrain):
    print('xgb_gini_normalized')
    return 'gini', normalized_gini(dtrain.get_labels(), pred)

xgb_fit_params = {        
    #'clf__silent': 1,
    'clf__eval_metric': xgb_metric,
    #'clf__objective': 'binary:logistic',
}


# ## Pipeline 

# In[277]:


from sklearn import pipeline as pi
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA

steps_xgb = [
    ('miss', get_fill_miss_transf()),
    #('low_freq', get_low_freq_transf()),
    ('one_hot', get_one_hot_transf()),
    #('norm', Normalizer()),
    #('pca', PCA(random_state=SEED)),
    #('kpca', KernelPCA(random_state=SEED, n_jobs=-1)),
    ('xgb', XGBClassifier(random_state=SEED, n_jobs=-1))
]

steps_adaboost = [
    ('miss', get_fill_miss_transf()),
    #('low_freq', get_low_freq_transf()),
    ('one_hot', get_one_hot_transf()),
    #('norm', Normalizer()),
    ('pca', PCA(random_state=SEED)),
    ('adab', AdaBoostClassifier(n_estimators=50, random_state=SEED))
]

steps_adaboost_knn = [
    ('miss', get_fill_miss_transf()),
    #('low_freq', get_low_freq_transf()),
    ('one_hot', get_one_hot_transf()),
    #('norm', Normalizer()),
    #('pca', PCA(random_state=SEED)),
    ('adab', AdaBoostClassifier(n_estimators=50, random_state=SEED))
]

steps_knn = [
    ('miss', get_fill_miss_transf()),
    #('low_freq', get_low_freq_transf()),
    ('one_hot', get_one_hot_transf()),
    #('norm', Normalizer()),
    #('pca', PCA(random_state=SEED, n_components=100)),
    #('kpca', KernelPCA(random_state=SEED, n_jobs=1)),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1))
]

steps_svc = [
    ('miss', get_fill_miss_transf()),
    #('low_freq', get_low_freq_transf()),
    ('one_hot', get_one_hot_transf()),
    #('norm', Normalizer()),
    #('pca', PCA(random_state=SEED)),
    ('svc', SVC(random_state=SEED, probability=True))
]


STEPS = steps_knn

def new_pipeline(steps=STEPS):
    return pi.Pipeline(steps)



# ### Curva ROC

# In[279]:


from sklearn import metrics
import matplotlib.pyplot as plt

def run_roc(X_train, y_train, X_test, y_test):
    pipeline = new_pipeline()
    pipeline.fit(X_train, y_train)
    y_probs = pipeline.predict_proba(X_test)
    y_preds = y_probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_preds)
    roc_auc = metrics.auc(fpr, tpr)
    return (fpr, tpr, roc_auc)

def plot_roc(fpr, tpr, roc_auc):
    fig = plt.figure(figsize=CHART_SIZE)
    plt.title('Curva ROC (Receiver Operating Characteristic)')
    plt.plot(fpr, 
             tpr, 
             'b', 
             label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    
if DEBUG:
    print("Calculando curva ROC...")
    fpr, tpr, roc_auc = run_roc(X_train_resampled, y_train_resampled, X_test, y_test)
    print(" > FPR: %s, TPR: %s, AUC: %s" % (np.average(fpr), np.average(tpr), roc_auc))
    plot_roc(fpr, tpr, roc_auc)


# ### Matriz de Confusão e Recall

# In[280]:


from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

def run_validation(X_train, y_train, X_test, y_test):
    # Validação Hold-Out:
    pipeline = new_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Matriz de confusão
    cf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion matrix: \n%s" % cf_matrix)

    # Recall score:
    recall = recall_score(y_test, y_pred) 
    print('Recall:\n %s' % recall)

    
#if DEBUG:
#    print("Executando validação...")
#    run_validation(X_train_resampled, y_train_resampled, X_test, y_test)


# ## Submissão

# In[281]:


def load_test():
    test_full = pd.read_csv(TEST_PATH)
    num_rows = get_num_rows(test_full)
    test = test_full.head(num_rows)
    X_test = test.drop(ID_COL, axis=1)
    return (test, X_test)


# In[282]:


def prepare_submission_data():
    X_train, y_train = (X_train_resampled, y_train_resampled)
    test, X_test = load_test()
    return (test[ID_COL], X_train, y_train, X_test)
    
def save_submission_data(preds, ids):
    sub = pd.DataFrame()
    sub[ID_COL] = ids
    sub[TARGET_COL] = preds
    sub.to_csv(OUTPUT_PATH, index=False)

def execute_pipeline(X_train, y_train, X_test):
    pipeline = new_pipeline()
     
    print('> Treinando modelo...')
    pipeline.fit(X_train, y_train)
        
    print('> Classificando...')
    y_probs = pipeline.predict_proba(X_test)
    preds = y_probs[:,1]
    
    return preds
    
def submit():
    print('Preparando dados...')
    ids, X_train, y_train, X_test = prepare_submission_data()
    
    print('Executando pipeline...')
    preds = execute_pipeline(X_train, y_train, X_test)
  
    print('Salvando dados...')
    save_submission_data(preds, ids)


if not DEBUG:
    submit()

    
print('Finalizado.')
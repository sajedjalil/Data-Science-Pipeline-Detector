# coding: utf-8

# # インポート

# In[1]:

model_name = 'CNN_03'

import numpy as np
import pandas as pd
import gc
import time
start_time = time.time()

from scipy.sparse import coo_matrix, hstack
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 以下keras用
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import SpatialDropout1D
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate, Activation


# # データ取得

# In[2]:

# pandasのデータフレームを返す
# train_or_testには'train'か'test'を入れる
def load_data(path,train_or_test,brand_threshold = 100,category_threshold = 50,frequent_brands=None,frequent_categories=None):
    data_pd = pd.read_csv(path, error_bad_lines=False, encoding='utf-8', header=0, delimiter='\t')
    ori_pd = pd.read_csv(path, error_bad_lines=False, encoding='utf-8', header=0, delimiter='\t')
    #ブランド名がないものを'NO_BRAND'とする
    data_pd['brand_name'] = data_pd['brand_name'].fillna('NO_BRAND')
    data_pd=data_pd.fillna("")

    if train_or_test == 'train':
        frequent_brands = data_pd['brand_name'].value_counts()[data_pd['brand_name'].value_counts()>brand_threshold].index
        frequent_categories = data_pd['category_name'].value_counts()[data_pd['category_name'].value_counts()>category_threshold].index
    elif train_or_test != 'test':
        print('Error : Please input "train" or "test" in train_or_test')
        return

    if type(frequent_brands)==type(None) or type(frequent_categories)==type(None):
        print('Error : Please load train data first')
        return
    else:
        data_pd.loc[~data_pd['brand_name'].isin(frequent_brands),'brand_name']= 'SOME_BRAND'
        data_pd.loc[~data_pd['category_name'].isin(frequent_categories),'category_name'] = 'SOME_CATEGORY'

    return ori_pd, data_pd, frequent_brands, frequent_categories


# データ取得
csv_train_path = u'../input/mercari-price-suggestion-challenge/train.tsv'
csv_test_path = u'../input/mercari-price-suggestion-challenge/test.tsv'
# csv_train_path = u'../../../../data/train.tsv'
# csv_test_path = u'../../../../data/test.tsv'
ori_train_data_pd, train_data_pd, frequent_brands, frequent_categories = load_data(csv_train_path,'train',brand_threshold=100,category_threshold=50)
ori_submit_data_pd, submit_data_pd, _, _ = load_data(csv_test_path,'test',frequent_brands=frequent_brands,frequent_categories=frequent_categories)
print('[{}]loading data completed'.format(time.time() - start_time))

use_cols = ['item_condition_id','brand_name','shipping','category_name']
# train_num = len(train_data_pd)
# test_num = len(test_data_pd)


# In[3]:

# price_logの正規化
ms = MinMaxScaler()
prices_log = np.log1p(ori_train_data_pd['price'])
prices_log = ms.fit_transform(prices_log.reshape(-1, 1)).reshape(-1)
print('[{}] converted target price'.format(time.time() - start_time))


# In[4]:

def make_onehot(use_cols,data_pd,train_or_test,save_path=None):
    # scipyのsparse matrix(coo_matrix)X_transform と 変数のリストvariables を返す
    # %save_pathに何も指定しない場合ファイルを保存しない 指定した場合指定したディレクトリ内に保存する
    variables = []
    flag = 0
    for use_col in use_cols:
        dummy_pd = pd.get_dummies(data_pd[use_col]).astype(np.uint8)
        if flag==0:
            X_transform = coo_matrix(dummy_pd.values)
            flag=1
        else:
            X_transform = hstack([X_transform,coo_matrix(dummy_pd.values)])

        variables.extend( list( dummy_pd.columns ) )

        if save_path is not None:
            if train_or_test != 'test' and train_or_test != 'train':
                print('Error : Please input "train" or "test" in train_or_test')
                return
            save_path_ = '{}/{}_{}.csv'.format(save_path,use_col,train_or_test)
            dummy_pd.to_csv(save_path_,index=False,encoding="utf8")

    if save_path is not None:
        # sparse matrixの保存
        io.savemat("{}/X_transform_{}".format(save_path,train_or_test), {"X_transform":X_transform})
        print('sparse matrixを保存しました')

    return X_transform,np.array(variables)

X_transform_train,variables = make_onehot(use_cols,train_data_pd,'train',save_path=None)
del train_data_pd
gc.collect()
X_transform_submit,variables_ = make_onehot(use_cols,submit_data_pd,'test',save_path=None)
print('[{}] converting data completed'.format(time.time() - start_time))
del submit_data_pd
gc.collect()


# In[5]:

# def normalize(df):
#     try:
#         df['item_description'] = df['item_description'].apply(lambda x: x.lower())
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace(".", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace(")", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("(", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("*", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace(":", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace(",", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("/", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("#", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("\\", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("1", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("2", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("3", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("4", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("5", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("6", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("7", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("8", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("9", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("0", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("!", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("$", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("%", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("&", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("-", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("+", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace(";", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: x.replace("[rm]", ""))
#         df['item_description'] = df['item_description'].apply(lambda x: str(x))
#     except:
#         print("There is no attribute named 'item_description'.")
#         df['item_description'] = df['item_description'].apply(lambda x: str(x))

#     finally:
#         return df


# def clean_up(df):
#     with open("../data/long_stop_word_list.txt", "r") as file:
#         stop_words = file.readlines()
#         words = list(map((lambda x: x.replace("\n","")), stop_words))
#         stemmer = stem.LancasterStemmer()

#     for index, sentence in df.iteritems():
#         splitted = sentence.split()
#         splitted = list(map(lambda x: stemmer.stem(x), splitted))
#         copy_list = copy.copy(splitted)
#         for splitted_word in splitted:
#             if splitted_word in words:
#                 copy_list.remove(splitted_word)
#             else:
#                 pass

#         df[index] = copy_list

#     return df


# def frequency(df, price1, price2):
#     histogram = dict()
#     source = df['item_description'][df['price']<price1]
#     source = source[df['price']>price2]
#     for splitted in source:
#         for word in splitted:
#             if word in histogram.keys():
#                 histogram[word] += 1
#             else:
#                 histogram[word] = 1

#     return histogram


def cleaning(df):
    try:
        df = df.apply(lambda x: x.lower())
        df = df.apply(lambda x: x.replace(".", ""))
        df = df.apply(lambda x: x.replace(")", ""))
        df = df.apply(lambda x: x.replace("(", ""))
        df = df.apply(lambda x: x.replace("*", ""))
        df = df.apply(lambda x: x.replace(":", ""))
        df = df.apply(lambda x: x.replace(",", ""))
        df = df.apply(lambda x: x.replace("/", ""))
        df = df.apply(lambda x: x.replace("#", ""))
        df = df.apply(lambda x: x.replace("\\", ""))
        df = df.apply(lambda x: x.replace("1", ""))
        df = df.apply(lambda x: x.replace("2", ""))
        df = df.apply(lambda x: x.replace("3", ""))
        df = df.apply(lambda x: x.replace("4", ""))
        df = df.apply(lambda x: x.replace("5", ""))
        df = df.apply(lambda x: x.replace("6", ""))
        df = df.apply(lambda x: x.replace("7", ""))
        df = df.apply(lambda x: x.replace("8", ""))
        df = df.apply(lambda x: x.replace("9", ""))
        df = df.apply(lambda x: x.replace("0", ""))
        df = df.apply(lambda x: x.replace("!", ""))
        df = df.apply(lambda x: x.replace("$", ""))
        df = df.apply(lambda x: x.replace("%", ""))
        df = df.apply(lambda x: x.replace("&", ""))
        df = df.apply(lambda x: x.replace("-", ""))
        df = df.apply(lambda x: x.replace("+", ""))
        df = df.apply(lambda x: x.replace(";", ""))
        df = df.apply(lambda x: x.replace("[rm]", ""))
        df = df.apply(lambda x: x.replace("?", ""))
        df = df.apply(lambda x: x.replace("~", ""))
        df = df.apply(lambda x: x.replace("\"", ""))
        df = df.apply(lambda x: str(x))
    except:
        print("There is no attribute named 'item_description'.")
        df = df.apply(lambda x: str(x))

    finally:
        return df


def get_name_and_descr_feature(sparsed, df):
    """
    使い方：
    引数sparsedにはスパース化した他の特徴量("item_condition_id"のone-hot表現など)を
    入れてください。
    引数dfには、コラム'name'と'item_description'が存在するDataFrameを入れてください。
    戻り値はsparsedと'name'や'item_description'から作った行列を結合したsparse matrix
    です。
    """
    #  必要ライブラリのインポート
    import pandas as pd
    from scipy.sparse import coo_matrix, hstack

    #  item_descriptionの切り出し
    try:
        descr = df['item_description']
    except KeyError:
        print("キー['item_description']は存在しないようです。")
        print("['item_description']をキーとして持つデータフレームを引数に入れてください。")
        return

    #  nameの切り出し
    try:
        name = df['name']
    except KeyError:
        print("キー['name']は存在しないようです。")
        print("['name']をキーとして持つデータフレームを第二引数に入れてください。")
        return

    #  大文字を小文字にし、記号等を消す
    try:
        descr = cleaning(descr)
        name = cleaning(descr)
    except NameError:
        print("関数cleaning()が未定義のようです。")
        print("cleaning()を定義してください。")
        print("参照 => useful_functions.cleaning()")
        return

    #  特徴語が入っている/いないの行列生成(nameの方)
    f_name = open("../input/features-2/name_feature.txt", "r")
    # f_name = open("../../../../arai_data_making/name_feature.txt", "r")
    selected_words = f_name.readlines()
    words = list(map(lambda x: x.replace("\n", ""), selected_words))
    worddict_train_name = dict()
    for word in words:
        num_list = []
        for sentence in name:
            if word in sentence:
                num_list.append(1)
            else:
                num_list.append(0)

        worddict_train_name[word] = num_list
    f_name.close()

    word_feature_train_name = pd.DataFrame.from_dict(worddict_train_name)
    word_mat_train_name = word_feature_train_name.values
    name_sparse = coo_matrix(word_mat_train_name)

    #  特徴語が入っている/いないの行列生成(descrの方)
    #  処理が同じなので今度暇なときにリファクタリングします。

    f_descr = open("../input/features-2/feature_list_unstemed.txt", "r")
    # f_descr = open("../../../../arai_data_making/feature_list_unstemed.txt", "r")
    selected_words = f_descr.readlines()
    words = list(map(lambda x: x.replace("\n", ""), selected_words))
    worddict_train_descr = dict()
    for word in words:
        num_list = []
        for sentence in descr:
            if word in sentence:
                num_list.append(1)
            else:
                num_list.append(0)

        worddict_train_descr[word] = num_list
    f_descr.close()

    word_feature_train_descr = pd.DataFrame.from_dict(worddict_train_descr)
    word_mat_train_descr = word_feature_train_descr.values
    descr_sparse = coo_matrix(word_mat_train_descr)

    #  sparse行列の結合
    sparsed = hstack([sparsed, name_sparse, descr_sparse])

    return sparsed


X_transform_train = get_name_and_descr_feature(X_transform_train, ori_train_data_pd).tocsr()
X_transform_submit = get_name_and_descr_feature(X_transform_submit, ori_submit_data_pd).tocsr()
print('[{}] extracted important features'.format(time.time() - start_time))


# In[ ]:




# In[6]:

train_text_list = [str(df['category_name']).replace(' ','').replace('/',' ')+' '+df['name']+' '+str(df['brand_name'])+' '+str(df['item_description']) for index, df in ori_train_data_pd.iterrows()]
submit_text_list = [str(df['category_name']).replace(' ','').replace('/',' ')+' '+df['name']+' '+str(df['brand_name'])+' '+str(df['item_description']) for index, df in ori_submit_data_pd.iterrows()]
tok = Tokenizer()
tok.fit_on_texts(texts = train_text_list + submit_text_list)
train_seq = tok.texts_to_sequences(train_text_list)
submit_seq = tok.texts_to_sequences(submit_text_list)

# padding
max_length=120
train_seq = sequence.pad_sequences(train_seq, maxlen=max_length)
submit_seq = sequence.pad_sequences(submit_seq, maxlen=max_length)
print('[{}] converted sentence to sequence'.format(time.time() - start_time))


# In[7]:

tok.word_index


# In[ ]:




# In[8]:

train_X, test_X, train_y, test_y, train_seq, test_seq = train_test_split(X_transform_train, prices_log, train_seq, test_size=0.1, random_state=42)
train_X, valid_X, train_y, valid_y, train_seq, valid_seq = train_test_split(train_X, train_y, train_seq, test_size=0.1, random_state=42)
original_originaloriginaloriginaloriginalvalid_y = ms.inverse_transform(valid_y.reshape(-1,1)).reshape(-1)
# train_X = train_X[:max_size] , train_y = train_y[:max_size]
# valid_max_size = max_size // 4
# test_X = test_X[:valid_max_size] , test_y = test_y[:valid_max_size]
onehot_length = train_X.shape[1]
print('[{}] split vaid and test datas'.format(time.time() - start_time))
gc.collect()


# # 学習

# In[9]:

def build(input, *nodes):
    x = input
    for node in nodes:
        if callable(node):
            x = node(x)
        elif isinstance(node, list):
            x = [build(x, branch) for branch in node]
        elif isinstance(node, tuple):
            x = build(x, *node)
        else:
            x = node
    return x


# In[10]:

#パラメータ設定
epoch_num = 3
BATCH_SIZE = 500
embedding_dim = 32
filter_num1 = 10
filter_length1 = 3
filter_num2 = 10
filter_length2 = 3
hidden_dims = 100


# In[11]:

# EarlyStoppingと、epochごとに重みを保存するように設定
es_cb = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
# learning_model_path = "./weight_variables/CNN/"+model_name+'/'
# fpath = learning_model_path + 'weights_{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
# cp_cb = ModelCheckpoint(filepath = fpath , monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


# In[12]:

#kerasの層構成を指定
onehot_input = Input((onehot_length,),name='onehot_input')
seq_input = Input((max_length,),name='seq_input')
output = build(
    seq_input, 
    Embedding(len(tok.word_index)+1, embedding_dim, input_length=max_length), 
    SpatialDropout1D(rate=0.2),
    Conv1D(filters=filter_num1,
                    kernel_size=filter_length1,
                    strides=2,
                    padding='valid',
                    activation='relu'),
    SpatialDropout1D(rate=0.2),
#     Conv1D(filters=filter_num2,
#                     kernel_size=filter_length2,
#                     strides=2,
#                     padding='valid',
#                     activation='relu'),
    Flatten(),
    [onehot_input, lambda x: x],
    Concatenate(),
    Dense(hidden_dims),
    Dropout(0.2),
    Activation('relu'),
    Dense(1),
    )

model = Model(inputs = [onehot_input, seq_input], outputs = output)
# model = Model(seq_input, output)
model.compile(loss="mse", optimizer='adam')

print(model.summary())


# In[ ]:

def batch_iter(data1, data2, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((data1.shape[0] - 1) / batch_size) + 1

    def data_generator():
        data_size = data1.shape[0]
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data1 = data1[shuffle_indices]
                shuffled_data2 = data2[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data1 = data1
                shuffled_data2 = data2
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X1 = shuffled_data1[start_index: end_index]
                X1 = X1.toarray()
                X2 = shuffled_data2[start_index: end_index]
                y = shuffled_labels[start_index: end_index]
                yield [X1,X2], y

    return num_batches_per_epoch, data_generator()

train_steps, train_batches = batch_iter(train_X, train_seq, train_y, BATCH_SIZE)
valid_steps, valid_batches = batch_iter(valid_X, valid_seq, valid_y, BATCH_SIZE)


# In[13]:

history = model.fit_generator(train_batches, train_steps,\
                    epochs=epoch_num, \
                    validation_data=valid_batches,\
                    validation_steps=valid_steps,\
                    callbacks=[es_cb], verbose=0)
# history = model.fit({'onehot_input':train_X.toarray(), 'seq_input':train_seq}, \
#                     train_y, epochs=epoch_num, batch_size=BATCH_SIZE, \
#                     validation_data=({'onehot_input':valid_X.toarray(), 'seq_input':valid_seq}, valid_y),\
#                     callbacks=[es_cb, cp_cb], verbose=1)
print('/n [{}] training completed'.format(time.time() - start_time))

# In[14]:

scores = model.evaluate([test_X.toarray(),test_seq],test_y, verbose=0)
print ("[{}] test cost=".format(time.time() - start_time) + str(scores))


# In[15]:

test_y = ms.inverse_transform(test_y.reshape(-1,1)).reshape(-1)
test_prediction = ms.inverse_transform(model.predict([test_X.toarray(),test_seq]).reshape(-1,1)).reshape(-1)
test_score = np.sqrt( np.mean((test_prediction-test_y)*(test_prediction-test_y)) )
print("[{}] test score=".format(time.time() - start_time)+str(test_score))


# In[ ]:




# In[16]:

submit_prediction = np.expm1(ms.inverse_transform(model.predict([X_transform_submit.toarray(),submit_seq]).reshape(-1,1)).reshape(-1) )

test_id = np.arange(submit_prediction.shape[0])

submission = pd.DataFrame([])
submission['test_id'] = test_id
submission['price'] = submit_prediction
submission.to_csv('submission.csv',index=None)
# submission.to_csv('../../../../submissions/submission'+model_name+'.csv',index=None)
print('[{}] submission completed'.format(time.time() - start_time))
print(submission.iloc[:10])
#encoding=utf-8

import pandas as pd
import numpy as np
import os,random,gc, re
np.random.seed(2016)
random.seed(2016)
import pyximport
pyximport.install()
from scipy.sparse import csr_matrix, hstack, vstack
import math,time
import datetime
import sys,logging
import multiprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelBinarizer
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
import wordbatch
import os
import psutil

debug=False   # you can debug by setting this
NUM_BRANDS = 4500
NUM_CATEGORIES = 1200
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))

def split_cat(text):
    try:
        if text != "missing":
            return text.split("/")
        else:
            return ("catemissing", "catemissing", "catemissing")
    except:
        return ("catemissing", "catemissing", "catemissing")


def handle_missing_inplace(dataset):
    #dataset['general_cat'].fillna(value='missing', inplace=True)
    #dataset['subcat_1'].fillna(value='missing', inplace=True)
    #dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def isin(x,s):
    if x in s:
        return x
    else:
        return "missing"

def cutting(dataset):
    pop_brand = set(dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS].values)
    dataset.loc[:, 'brand_name'] = dataset.loc[:, 'brand_name'].apply(lambda x:isin(x,pop_brand))
    pop_category1 = set(dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES])
    pop_category2 = set(dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES])
    pop_category3 = set(dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES])
    dataset.loc[:, 'general_cat'] = dataset.loc[:, 'general_cat'].apply(lambda x:isin(x,pop_category1))
    dataset.loc[:, 'subcat_1'] = dataset.loc[:, 'subcat_1'].apply(lambda x:isin(x,pop_category2))
    dataset.loc[:, 'subcat_2'] = dataset.loc[:, 'subcat_2'].apply(lambda x:isin(x,pop_category3))



def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
#stopwords = {x: 1 for x in stopwords.words('english')}
stopwords = {x: 1 for x in set(['the','i','of','to',":",'is','are','this','that','it','am','they'])}
blacklist = set(['the','i','of','to',':','is','are','this','that','it','am','they',','," ",""])

def normalize_text(text):
    return u" ".join(text.split(" "))

def normalize_text2(text):
    l = filter(lambda x:x not in blacklist,re.split(spliter1, text.lower()))
    return u" ".join(l)


def bleu(c,r,iwords= set(),iwords2 = set()):  # function to calc the simiarlity of two words
    if len(c) == 0 or len(r) == 0:
        return 0.0
    if len(set(r) & set(c)) < len(set(r)) * 0.7:
        return 0.0
    if len(c) > len(r):
        bp = 0
    else:
        bp = max(min(1-len(r) * 1.0/len(c),0),-0.5)
    sumpn = 0.0
    for i in range(3):
        rcount = {}
        for j in range(len(r) - i):
            w = 1
            rcount[" ".join(r[j:j+1+i])] = rcount.get(" ".join(r[j:j+1+i]),0) + w
        ccount = {}
        ctcount = {}
        for j in range(len(c) - i):
            w = 1
            t1 = " ".join(c[j:j+1+i])
            ccount[t1] = min(ccount.get(t1,0) + w,rcount.get(t1,0))
            ctcount[t1] = ctcount.get(t1, 0) + w
        temp = 0.33 * sum(map(lambda x:x[1],ccount.items()))*1.0/(sum(map(lambda x:x[1],ctcount.items()))+0.0001)
        if temp < 0.25:
            return 0.0
        sumpn += temp
    return bp + sumpn


filter2 = re.compile(r"(\:\))")
filter1 = re.compile(r"([^A-Za-z0-9\,\. \:\!\?\+\;\/\t\'])")
filter4 = re.compile(r"(❤)")
spliter1 = re.compile(r"([\&\,\ \!\:\(\)\*\.\-\"\/\;\#\?\+\~\{\[\]\}])")
digit = re.compile(r"(^|(?<=[ \,\.:\?\+\;\/\t]))([0-9]{2,})($|(?=[ \,\.:\?\+\;\/\t]))")
digit2 = re.compile(r'([0-9]{1,}\.[0-9]{1,})')
token = re.compile(r'([\.]{2,})')
token2 = re.compile(r'([\&]{1,})')
token1 = re.compile(r'([\+]{1,})')
fran1 = ["ā","á","ǎ","à"]
fran1r = re.compile(r"([āáǎà])")
fran2 = ["ê","ē","é","ě","è","ë"]
fran2r = re.compile(r"([êēéěèë])")
fran3 = ["ī","í","ǐ","ì","ï","î"]
fran3r = re.compile(r"([īíǐìïî])")
fran4 = ["ō","ó","ǒ","ò","ö","ô"]
fran4r = re.compile(r"([ōóǒòöô])")
fran5 = ["ū","ú","ǔ","ù","ǖ","ǘ","ǚ","ǜ","ü","û"]
fran5r = re.compile(r"([ūúǔùǖǘǚǜüû])")
fran6 = ["ç","Ç","č"]
fran6r = re.compile(r"([çÇč])")


indexcate1 = {}
indexcate2 = {}
indexcate3 = {}
indexword = {}

index1 = 1
index2 = 1
index3 = 2
index4 = 2

# preprocess

def pre(i):
    if i==0:
        count = 0
        f = open('./train_p.csv','w')
        train_p=[]
        f.write("train_id" +"\t" +"name" +"\t" +"item_condition_id" +"\t" +"category_name" +"\t" +"brand_name" +"\t" +"price" +"\t" +"shipping" +"\t" +"item_description"+"\t" +"adf\n")
        for line in open('../input/train.tsv'):
            if not line:
                continue
            if count ==0:
                count+=1
                continue
            line = re.sub(filter2," smile ",line.strip().replace("|"," "))
            line = re.sub(token1, " plus ", line)
            line = re.sub(token2, " ", line)
            line = line.replace("❤"," xin ")
            countfr = 0
            for i in fran1:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "a")
            for i in fran2:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "e")
            for i in fran3:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "i")
            for i in fran4:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "o")
            for i in fran5:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "u")
            for i in fran6:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "c")
            linet = re.sub(filter1, "|", line)
            array = linet.split('\t', -1)
            namec = 0
            named = 0
            if array[1]:
                namec = round(math.log(len(re.findall("\|", array[1])) + 1),4)
            if len(array) >= 8 and array[7] and array[7] != 'No description yet':
                named = round(math.log(len(re.findall("\|", array[7])) + 1),4)
            lc = round(math.log(countfr + 1),4)
            line = linet.replace("|"," ")
            line = re.sub(token, " ellipsis ", line)
            train_p.append((line + '\t' + str(namec) + "," + str(named) +"," + str(lc)).split('\t'))
        f.close()
        return train_p
    else:
        f = open('./test_p.csv','w')
        f.write("test_id" +"\t" +"name" +"\t" +"item_condition_id" +"\t" +"category_name" +"\t" +"brand_name" +"\t" +"shipping" +"\t" +"item_description"+"\t" +"adf\n")
        count = 0
        test_p = []
        submission_id = []
        for line in open('../input/test.tsv'):
            if not line:
                continue
            if count ==0:
                count+=1
                continue
            line = re.sub(filter2," smile ",line.strip().replace("|"," "))
            line = re.sub(token1, " plus ", line)
            line = re.sub(token2, " ", line)
            line = line.replace("❤"," xin ")
            countfr = 0
            for i in fran1:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "a")
            for i in fran2:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "e")
            for i in fran3:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "i")
            for i in fran4:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "o")
            for i in fran5:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "u")
            for i in fran6:
                line = line.replace(i,"|")
            array = line.split('\t', -1)
            if array[4]:
                countfr += len(re.findall("\|", array[4]))
            line = line.replace("|", "c")
            linet = re.sub(filter1, "|", line)
            array = linet.split('\t', -1)
            namec = 0
            named = 0
            if array[1]:
                namec = round(math.log(len(re.findall("\|", array[1])) + 1),4)
            if len(array) >= 7 and array[6] and array[6] != 'No description yet':
                named = round(math.log(len(re.findall("\|", array[6])) + 1),4)
            lc = round(math.log(countfr + 1),4)
            line = linet.replace("|"," ")
            line = re.sub(token, " ellipsis ", line)
            f.write(line + '\t' + str(namec) + "," + str(named) +"," + str(lc) +'\n')
            submission_id.append(array[0])
            # f.write(line + '\t' + str(namec) + "," + str(named) +"," + str(lc) +'\n')   # uncomment this to debug with the 3.5M data.
            # submission_id.append(array[0])
            # f.write(line + '\t' + str(namec) + "," + str(named) +"," + str(lc) +'\n')
            # submission_id.append(array[0])
            # f.write(line + '\t' + str(namec) + "," + str(named) +"," + str(lc) +'\n')
            # submission_id.append(array[0])
            # f.write(line + '\t' + str(namec) + "," + str(named) +"," + str(lc) +'\n')
            # submission_id.append(array[0])
        f.close()
        return test_p,submission_id

p = multiprocessing.Pool(2)
preds = p.imap(pre, [0,1])
for i, pred in enumerate(preds):
    if i == 0:
        #print(pred)
        train_p = pred
    elif i==1:
        #print(pred)
        (test_p,submission_id) = pred
p.close()
p.join()
del(p)

def fill(x):
    if len(x)==0:
        return "missing"
    else:
        return x

# Model 1 @anttip. based on ftrl-fm. valid score:0.415

start_time = time.time()
#train = pd.read_table('./train_p.csv', engine='c')
train2 = pd.DataFrame(train_p)
train2.columns=["train_id","name","item_condition_id","category_name","brand_name","price","shipping","item_description","adf"]
train = pd.read_table('../input/train.tsv', engine='c')
train.iloc[:,[1,3,4,7]] = train2.iloc[:,[1,3,4,7]].applymap(fill)
del(train2)
#del(train_p)
res2 = []
testshape = len(submission_id)
print('test len:' + str(testshape))


for i in range(0,testshape,700000):        # batch predict to save the memory. it doesn't make any difference on result
    testdf = pd.read_table('./test_p.csv', engine='c')
    start = i
    end = min(testshape,i+700000)
    test = testdf.iloc[start:end,:]
    #print(test)
    del(testdf)
    print(str(start)+":"+str(end))
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    nrow_test = train.shape[0]
    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    train_y = y
    merge = pd.concat([train, test])
    #merge = train
    nrow_valid=nrow_train
    if debug:
        nrow_valid=1400000
    
    del(test)
    gc.collect()
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
            zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    cateindex = merge['general_cat'][:nrow_train]
    merge.drop('category_name', axis=1, inplace=True)
    handle_missing_inplace(merge)
    cutting(merge)
    to_categorical(merge)
    merge['name'] = merge['name'].apply(lambda x:normalize_text2(x))
    #merge['item_description'] = merge['item_description'].apply(lambda x:normalize_text(x))
    print('[{}] Convert categorical completed'.format(time.time() - start_time))
    mindf = int(merge.shape[0] / 400000)
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                      "hash_size": 2**28, "norm": None, "tf": 'binary',
                                                                      "idf": None,
                                                                      }), procs=8)
    wb.dictionary_freeze= True
    X_train_name = wb.fit_transform(merge['name'][:nrow_valid])
    X_test_name = wb.transform(merge['name'][nrow_valid:])
    X_name = vstack((X_train_name,X_test_name))
    
    X_name = X_name[:, np.array(np.clip(X_name[:nrow_train,:].getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    del(X_train_name)
    del(X_test_name)
    del(wb)
    
    print(X_name.shape)
    #X_test_name = X_name[:, np.array(np.clip(X_test_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))
    wb = CountVectorizer()
    X_category1 = wb.fit_transform(merge['general_cat'])
    X_category2 = wb.fit_transform(merge['subcat_1'])
    X_category3 = wb.fit_transform(merge['subcat_2'])

    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))
    gc.collect()
    info = psutil.virtual_memory()
    print(psutil.Process(os.getpid()).memory_info().rss)
    def tfidf(i):
        if i==0:
            wb = TfidfVectorizer(max_features = 1000000,
                                  ngram_range = (1,1),
                                  stop_words = None,
                                  min_df=10,token_pattern =u'(?u)\\b\\w\\w+\\b')
            X_train_description1 = wb.fit_transform(merge['item_description'][:nrow_valid])
            X_test_description1 = wb.transform(merge['item_description'][nrow_valid:])
            X_description1 = vstack((X_train_description1,X_test_description1)).tocsr()
            del(X_test_description1)
            del(X_train_description1)
            print("d1")
            del(wb)
            gc.collect()
            return (X_description1)
        elif i==1:
            wb = TfidfVectorizer(max_features = 450000,
                                  ngram_range = (2,2),
                                  stop_words = None,
                                  min_df=5,token_pattern =u'(?u)\\b\\w+\\b')
            X_train_description2 = wb.fit_transform(merge['item_description'][:nrow_valid])
            X_test_description2 = wb.transform(merge['item_description'][nrow_valid:])
            X_description2 = vstack((X_train_description2,X_test_description2)).tocsr()
            del(X_test_description2)
            del(X_train_description2)
            print("d2")
            del(wb)
            gc.collect()
            return (X_description2)
        elif i==2:
            wb = TfidfVectorizer(max_features = 250000,
                                  ngram_range = (3,3),
                                  stop_words = None,
                                  min_df=5,token_pattern =u'(?u)\\b\\w+\\b')
            X_train_description3 = wb.fit_transform(merge['item_description'][:nrow_valid])
            X_test_description3 = wb.transform(merge['item_description'][nrow_valid:])
            X_description3 = vstack((X_train_description3,X_test_description3)).tocsr()
            del(X_test_description3)
            del(X_train_description3)
            print("d3")
            del(wb)
            gc.collect()
            return (X_description3)
        elif i==3:
            lb = LabelBinarizer(sparse_output=True)
            X_brand = lb.fit_transform(merge['brand_name'])
            X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                              sparse=True).values)
            print("d4")
            del(lb)
            return (X_brand,X_dummies)
    
    p = multiprocessing.Pool(4)
    preds = p.imap(tfidf, [0,1,2,3])
    for i, pred in enumerate(preds):
        if i == 0:
            #print(pred)
            (X_description1) = pred
        elif i==1:
            #print(pred)
            (X_description2) = pred
        elif i==2:
            #print(pred)
            (X_description3) = pred
        elif i==3:
            #print(pred)
            (X_brand,X_dummies) = pred
    p.close()
    p.join()
    del(p)
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
    print(X_dummies.shape, X_description1.shape, X_description2.shape, X_description3.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
              X_name.shape)
    sparse_merge = hstack((X_dummies, X_description1, X_description2, X_description3, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()
    
    print(sparse_merge.shape)
    #mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    #sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    #X_test = sparse_merge[nrow_valid:nrow_train]
    train_X = X[:nrow_valid]
    valid_X = X[nrow_valid:nrow_train]
    train_y = y[:nrow_valid]
    valid_y = y[nrow_valid:nrow_train]
    train_X = train_X[np.where(train_y != 0.0)[0]]
    train_y = train_y[np.where(train_y != 0.0)[0]]
    #del(X_description)
    del(X_brand)
    del(X_category1)
    del(X_category2)
    del(X_category3)
    del(X_name)
    del(merge)
    gc.collect()
    
    del(X_dummies)
    #del(X_description)
    del(X_description1)
    del(X_description2)
    del(X_description3)
    print(train_X.shape)
    print(valid_X.shape)
    print('[{}] addition feature completed.'.format(time.time() - start_time))

    model = FM_FTRL(alpha=0.03, beta=0.01, L1=0.001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.07, L2_fm=0.001, init_fm=0.01,
                        D_fm=400, e_noise=0.0001, iters=1, inv_link="identity", threads=4,weight_fm = 1.0)
    for i in range(4):
        model.fit(train_X, train_y)
        if debug:
            preds = model.predict(X=valid_X)
            print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
        param = model.__getstate__()
        model.__setstate__((param[0],param[1],param[2],param[3],param[4] * 0.8,param[5],param[6],param[7],
            param[8],param[9],param[10],param[11],param[12],param[13],param[14],param[15],param[16],param[17]
            , param[18],param[19]))
    if debug:
        resdefm = preds
    resf = model.predict(X=X_test)
    res2.extend(resf)
    del(param)
    del(model)
    del(sparse_merge)
    del(train_X)
    del(train_y)
    del(valid_X)
    del(valid_y)
    gc.collect()


resfm = np.array(res2)
del train
print(len(resfm))
print('[{}] ftrl fm fin.'.format(time.time() - start_time))


# Model 2. based on nn. valid score 0.401
def pre2(i):
    if i==0:
        count = 0
        typedetail = {}
        branddetail = {}
        train_x = []

        for line in train_p:
            array = line
            type = ""
            txt = ""
            other = ""
            nametxt = ""
            brandtxt = ""
            array[5] = math.log(float(array[5])+1)
            if array[3]:
                #array[3] = re.sub(digit2," numcus ",array[3])
                array[3] = re.sub(digit," numcus ",array[3])
                cate = array[3].split('/')
                type = array[3]
                typedetaile = typedetail.get(cate[1] + "/" + cate[2])
                if typedetaile is None:
                    typedetail[cate[1]+"/"+cate[2]] = [array[5]]
                else:
                    typedetaile.append(array[5])
                txt += "type "
                l = re.split(spliter1, cate[1] + " " + cate[2])
                for word in l:
                    if word == "" or word == " " or word == "/":
                        continue
                    txt += word + " "
                txt += ""
            else:
                type = "0/0/0"
                txt += "type missing "

            if array[2]:
                other += ""+array[2]
            else:
                other += "-1"

            if array[4]:
                brandtxt += "brand " +array[4] + " "
                branddetaile = branddetail.get(array[4])
                if branddetaile is None:
                    branddetail[array[4]] = [array[5]]
                else:
                    branddetaile.append(array[5])
            else:
                brandtxt += "brand missing "

            if array[6]:
                other += ","+array[6]
            else:
                other += ",-1"

            if array[1]:
                array[1] = re.sub(digit," numcus ",array[1])
                txt += "name "
                l = re.split(spliter1, array[1])
                for word in l:
                    if word == "" or word == " ":
                        continue
                    txt += word + " "
                    nametxt += word + " "
                txt += ". "
            else:
                txt += "name missing . "
                nametxt += "name missing "

            if len(array) >= 8 and array[7] and array[7] != 'No description yet':
                array[7] = re.sub(digit," numcus ",array[7])
                txt += "description "
                l = re.split(spliter1, array[7])
                for word in l:
                    if word == "" or word == " " or word == "/":
                        continue
                    txt += word + " "
                txt += "."
            else:
                txt += "description missing ."
            count += 1
            other += "," + array[-1]
            train_x.append((array[0],type,txt, array[5],nametxt+brandtxt,other,array[4]))
        return train_x,typedetail,branddetail
    elif i==1:
        test_x=[]
        count = 0
        for line in open('./test_p.csv'):
            if not line:
                continue
            if count ==0:
                count+=1
                continue
            array = line.strip().split('\t',-1)
            type = ""
            txt = ""
            nametxt = ""
            brandtxt = ""
            other = ""
            if array[3]:
                array[3] = re.sub(digit," numcus ",array[3])
                cate = array[3].split('/')
                type = array[3]
                txt += "type "
                l = re.split(spliter1, cate[1] + " " + cate[2])
                for word in l:
                    if word == "" or word == " " or word == "/":
                        continue
                    txt += word + " "
                txt += ""
            else:
                type = "0/0/0"
                txt += "type missing "

            if array[2]:
                other += array[2]
            else:
                other += "-1"

            if array[4]:
                brandtxt += "brand " +array[4] + " "
            else:
                brandtxt += "brand missing "

            if array[5]:
                other += ","+array[5]
            else:
                other += ",-1"

            if array[1]:
                array[1] = re.sub(digit," numcus ",array[1])
                txt += "name "
                l = re.split(spliter1, array[1])
                for word in l:
                    if word == "" or word == " ":
                        continue
                    txt += word + " "
                    nametxt += word + " "
                txt += ". "
            else:
                txt += "name missing . "
                nametxt += "name missing "

            if len(array) >= 7 and array[6] and array[6] != 'No description yet':
                array[6] = re.sub(digit," numcus ",array[6])
                num = 0
                txt += "description "
                l = re.split(spliter1, array[6])
                for word in l:
                    if word == "" or word == " " or word == "/":
                        continue
                    txt += word + " "
                txt += "."
            else:
                txt += "description missing ."
            other += "," + array[-1]
            count += 1
            test_x.append((array[0],type,txt, 0.0,nametxt+brandtxt,other,array[4]))
        return test_x

p = multiprocessing.Pool(2)
preds = p.imap(pre2, [0,1])
for i, pred in enumerate(preds):
    if i == 0:
        #print(pred)
        (train_x,typedetail,branddetail) = pred
    elif i==1:
        #print(pred)
        (test_x) = pred
p.close()
p.join()
del(p)

typecount = {}
typemean = {}
typestd = {}
allothertype = []
indexcate3["0/0"] = 0
for k,v in typedetail.items():
    if indexcate3.get(k,0) == 0:
        if len(v) > 10:
            indexcate3[k] = index3
            index3 += 1
            k2 = indexcate3[k]
            typecount[k2] = len(v)
        else:
            indexcate3[k] = 1
            allothertype.extend(v)
typecount[1] = len(allothertype)

brandcount = {}
brandmean = {}
unknownb = set()
brandindex = 1
brandcate = {}
for k,v in branddetail.items():
    if brandcate.get(k,0) == 0:
        if len(v) >= 5:
            brandcate[k] = brandindex
            brandindex += 1
            k2 =  brandcate[k]
            brandcount[k2] = len(v)
        else:
            unknownb.add(k)
            brandcate[k]=0

wordcount = {}
for e in train_x:
    for word in e[2].split(' ',-1):
        word = word.lower()
        if len(word) == 0:
            continue
        wordcount[word] = wordcount.get(word,0.0) + 1.0
    for word in e[4].split(' ',-1):
        word = word.lower()
        if len(word) == 0:
            continue
        wordcount[word] = wordcount.get(word,0.0) + 1.0

r = len(train_x)*0.5/len(test_x)
print(r)
wordcount2 = {}        
for e in test_x:
    for word in e[2].split(' ',-1):
        word = word.lower()
        if len(word) == 0:
            continue
        wordcount2[word] = wordcount2.get(word,0.0) + r
    for word in e[4].split(' ',-1):
        word = word.lower()
        if len(word) == 0:
            continue
        wordcount2[word] = wordcount2.get(word,0.0) + r

for k,v in wordcount.items():
    if v >= 1:
        indexword[k] = index4
        index4 += 1

print(len(indexword.items()))
# mapping = {}

validword = {}
index4 = 2
count = 0
abc = set(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
wordindex = {}
sortcount = sorted(wordcount.items(),key=lambda x:-x[1])

print

# process the wrong-spell word

for k,v in sortcount:
    abstart = k[0] in abc
    isfind = False
    k2 = k.replace("'","")
    scoremax = 0.0
    if round(v) > 10 and len(k) > 4:
        taglist = []
        taglist.extend(wordindex.get(k[:3],[]))
        if not abstart:
            taglist.extend(wordindex.get(k[1:4],[]))
        nearword = ""
        for word in taglist:
            score = 0
            if abs(len(k) - len(word)) > 3:
                continue
            score += bleu(k2, word)
            if score > 0.80:
                scoremax = score
                nearword = word
                isfind = True
                break
        if isfind:
            validword[k] = validword[nearword]
    elif (round(v) > 1 or round(wordcount2.get(k,0)) > 1) and len(k) > 4:
        scoremax = 0.0
        taglist = []
        taglist.extend(wordindex.get(k[:3],[]))
        if not abstart:
            taglist.extend(wordindex.get(k[1:4],[]))
        nearword = ""
        for word in taglist:
            if wordcount.get(word) <= 10:
                continue
            if abs(len(k) - len(word)) > 3:
                continue
            score = 0
            
            score += bleu(k2,word)
            if score > max(0.60,scoremax):
                scoremax = score
                #print word,k,score
                nearword = word
                isfind = True
                if score > 0.75:
                    break
        if isfind:
            validword[k] = validword[nearword]
    if not isfind and round(v) >= 7:
        validword[k] = [index4]
        index4 += 1
        wordindex[k[:3]] = wordindex.get(k[:3],[])
        wordindex[k[:3]].append(k)
    elif isfind and scoremax > 0.70:
        wordindex[k[:3]] = wordindex.get(k[:3],[])
        wordindex[k[:3]].append(k)
    count += 1
    if count % 10000 == 0:
        print(count)

del wordindex,sortcount,wordcount
gc.collect()

print(index4)
print(len(validword.items()))
print('[{}] wordcount'.format(time.time() - start_time))



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=5)
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge, Bidirectional, GRU,MaxoutDense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import  MaxPooling2D,GlobalMaxPooling1D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D,Merge
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import Convolution1D
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from keras.optimizers import *
from keras.initializers import *
from keras.callbacks import Callback
from keras.regularizers import l1,l2
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_session(tf.Session(config=session_conf))

from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers.merge import _Merge
from keras.engine.topology import Layer,InputSpec
from keras.utils import conv_utils

class Cropping1D(Layer):
    def __init__(self, cropping=(1, 1), **kwargs):
        super(Cropping1D, self).__init__(**kwargs)
        self.cropping = conv_utils.normalize_tuple(cropping, 2, 'cropping')
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        if input_shape[1] is not None:
            length = self.cropping[1] - self.cropping[0]
        else:
            length = None
        return (input_shape[0],
                input_shape[1],
                length)

    def call(self, inputs):
        if self.cropping[1] == 0:
            return inputs[:, :, self.cropping[0]:]
        else:
            return inputs[:, :, self.cropping[0]: self.cropping[1]]

    def get_config(self):
        config = {'cropping': self.cropping}
        base_config = super(Cropping1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None
    def call(self, x, mask=None):
        input_shape = K.int_shape(x)
        features_dim = self.features_dim
        step_dim = input_shape[1]
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b[:input_shape[1]]
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_w(self, x, mask=None):
        input_shape = K.int_shape(x)
        features_dim = self.features_dim
        step_dim = input_shape[1]
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b[:input_shape[1]]
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return a

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim



wordnum = index4
FEAT_LENGTH2 = 9
FEAT_LENGTH = 96
FIX_LEN = 29
sizecate1 = 1
sizecate2 = 1
sizecate3 = 1
id1 = []
id2 = []
id3 = []
feat = []
feat2 = []
feat3 = []
featlen = []
y = []
for line in train_x:
    txt = []
    pos = 0
    l = math.log(len(line[2].split(' '))+1)
    for word in line[2].split(' '):
        if word =='' or word.lower() in blacklist:
            continue
        res = validword.get(word.lower(), [1])
        txt.extend(res)
        pos += len(res)
        if pos >= FEAT_LENGTH-FIX_LEN:
            break
    if (len(txt) < FEAT_LENGTH-FIX_LEN):
        txt.extend(txt)
    while (len(txt) < FEAT_LENGTH-FIX_LEN):
        txt.extend([0] * 10)
    txt = txt[:FEAT_LENGTH-FIX_LEN]
    txtn = []
    pos = 0
    for word in line[4].split(' '):
        if word =='' or len(word) <= 1:
            continue
        res = validword.get(word.lower(), [1])
        txtn.extend(res)
        pos += len(res)
        if pos >= FEAT_LENGTH2:
            break
    while (len(txtn) < FEAT_LENGTH2):
        txtn.extend([0] * 10)

    txtn = txtn[:FEAT_LENGTH2]
    cate = line[1].split('/')
    cate3 = indexcate3[cate[1]+"/"+cate[2]]
    lenarray = ""
    lenarray += line[5]
    lenarray += "," + str(l)
    lenarray += "," + str(math.log(typecount.get(cate3,0)+1))
    brand = brandcate.get(line[6],0)
    lenarray += "," + str(math.log(brandcount.get(brand,0)+1))
    #lenarray += "," + str(typemean.get(cate3,0))
    y.append(line[3])
    id3.append([cate3])
    feat.append(list(map(int,txt[:FEAT_LENGTH-FIX_LEN])))
    feat2.append(list(map(int,txtn)))
    featlen.append(list(map(float,lenarray.split(','))))



del train_x
gc.collect()



id3 = np.array(id3)
y = np.array(y)
feat = np.array(feat)
feat2 = np.array(feat2)

featlen = np.array(featlen)

print(id3[0])
print(feat[0])
print(feat2[0])

print(featlen[0])
print('[{}] nn train data load'.format(time.time() - start_time))
splitpoint = 1400000
selnotzero = np.where(y[:splitpoint] > 0.0)[0]
train_feat = feat[selnotzero,:]
train_feat2 = feat2[selnotzero,:]
train_featlen = featlen[selnotzero,:]
train_id3 = id3[selnotzero,:]
train_y = y[selnotzero]
test_feat = feat[splitpoint:,:]
test_feat2 = feat2[splitpoint:,:]
test_featlen = featlen[splitpoint:,:]
test_id3 = id3[splitpoint:,:]
test_y = y[splitpoint:]

index = cateindex[cateindex == "Women"].index
index2 = cateindex[cateindex == 'Beauty'].index
sel = list(set(selnotzero) & (set(index) | set(index2)))
train2_feat = feat[sel, :]
train2_feat2 = feat2[sel, :]
train2_featlen = featlen[sel, :]
train2_id3 = id3[sel, :]
train2_y = y[sel]

feat = []
feat2 = []
feat3 = []
featlen = []
y = []
id3 = []
for line in test_x:
    txt = []
    pos = 0
    l = math.log(len(line[2].split(' '))+1)
    for word in line[2].split(' '):
        if word =='' or word.lower() in blacklist:
            continue
        res = validword.get(word.lower(), [1])
        txt.extend(res)
        pos += len(res)
        if pos >= FEAT_LENGTH-FIX_LEN:
            break
    if (len(txt) < FEAT_LENGTH-FIX_LEN):
        txt.extend(txt)
    while (len(txt) < FEAT_LENGTH-FIX_LEN):
        txt.extend([0] * 10)
    txt = txt[:FEAT_LENGTH-FIX_LEN]
    txtn = []
    pos = 0
    for word in line[4].split(' '):
        if word =='' or len(word) <= 1:
            continue
        res = validword.get(word.lower(), [1])
        txtn.extend(res)
        pos += len(res)
        if pos >= FEAT_LENGTH2:
            break
    while (len(txtn) < FEAT_LENGTH2):
        txtn.extend([0] * 10)
    txtn = txtn[:FEAT_LENGTH2]
    cate = line[1].split('/')
    cate3 = indexcate3.get(cate[1]+"/"+cate[2],1)
    lenarray = ""
    lenarray += line[5]
    lenarray += "," + str(l)
    lenarray += "," + str(math.log(typecount.get(cate3,0)+1))
    #lenarray += "," + str(typemean.get(cate3,0))
    brand = brandcate.get(line[6],0)
    lenarray += "," + str(math.log(brandcount.get(brand,0)+1))
    id3.append([cate3])
    feat.append(list(map(int,txt[:FEAT_LENGTH-FIX_LEN])))
    feat2.append(list(map(int,txtn)))
    featlen.append(list(map(float,lenarray.split(','))))

del test_x
gc.collect()
print('[{}] nn sub data load'.format(time.time() - start_time))


print(id3[0])
print(feat[0])
print(feat2[0])
print(featlen[0])
print(index3)

sub_featlen = np.array(featlen)
sub_id3 = np.array(id3)
sub_feat = np.array(feat)
sub_feat2 = np.array(feat2)

del id3
del feat
del feat2
del featlen

testdf = pd.read_table('./test_p.csv', engine='c')
testdf['general_cat'], _, _ = \
            zip(*testdf['category_name'].apply(lambda x: split_cat(x)))
indexcatesub = testdf['general_cat']
del(testdf)



if True:
    input1 = Input(shape=(sub_featlen.shape[1],), dtype='float32')
    input2 = Input(shape=(1,), dtype='int32')
    embedding_layer0 = Embedding(index3,
                    30,init = RandomNormal(mean=0.0, stddev=0.005),
                        input_length=1,
                        trainable=True)
    xc3 = embedding_layer0(input2)
    xc3 = Flatten()(xc3)
    input3 = Input(shape=(FEAT_LENGTH-FIX_LEN,), dtype='int32')
    embedding_layer0 = Embedding(wordnum,
                            40,
                            init = RandomNormal(mean=0.0, stddev=0.005),
                            trainable=True)
    x30 = embedding_layer0(input3)
    x31 = Cropping1D(cropping=(0,40))(x30)
    la = []
    x3 = Conv1D(15,3,activation='sigmoid',padding = 'same',dilation_rate = 1)(x31)
    x3 = MaxPooling1D(FEAT_LENGTH-FIX_LEN)(x3)
    x3 = Flatten()(x3)
    la.append(x3)

    t=Conv1D(50,2,activation='sigmoid',padding = 'same',dilation_rate = 1)
    x3 = t(x31)
    x3 = MaxPooling1D(FEAT_LENGTH-FIX_LEN)(x3)
    x3 = Flatten()(x3)
    la.append(x3)
    
    
    embedding_layer1 = Embedding(wordnum,
                            160,
                            init = RandomNormal(mean=0.0, stddev=0.005),
                            trainable=True)
    x30 = embedding_layer1(input3)
    x30 = Cropping1D(cropping=(0,80))(x30)
    att = Attention(50)
    x1 = att(x30)
    la.append(x1)
    
    input4 = Input(shape=(FEAT_LENGTH2,), dtype='int32')
    embedding_layer0 = Embedding(wordnum,
                            50,
                            init = RandomNormal(mean=0.0, stddev=0.005),
                            trainable=True)
    x40 = embedding_layer0(input4)
    x41 = Cropping1D(cropping=(0,50))(x40)
    t=Conv1D(55,2,activation='sigmoid',padding = 'same',dilation_rate = 1)
    x3 = t(x41)
    x3 = MaxPooling1D(FEAT_LENGTH2)(x3)
    x3 = Flatten()(x3)
    la.append(x3)

    x40 = embedding_layer1(input4)
    att = Attention(50)
    x1 = att(x40)
    la.append(x1)
    
   
    la.append(xc3)
    x1 = merge(la,mode = 'concat')
    x1 = merge([x1,input1],mode = 'concat')
    x1 = BatchNormalization()(x1)
    
    #x1 = Dropout(0.1)(x1)
    x = Dense(30, activation='sigmoid')(x1)
    x3 = Dense(470)(x1)
    x3 = PReLU()(x3)
    x1 = merge([x,x3], mode='concat')
    x1 = Dropout(0.02)(x1)
    x = Dense(256, activation='sigmoid')(x1)
    x2 = Dense(11, activation='linear')(x1)
    x3 = Dense(11)(x1)
    x3 = PReLU()(x3)
    x1 = merge([x , x2, x3], mode='concat')
    x1 = Dropout(0.02)(x1)
    
    out = Dense(1, activation='linear')(x1)
    out2 = Dense(1, activation='relu')(x1)
    out3 = MaxoutDense(1, 30)(x1)
    out = merge([out,out2,out3],mode='sum')
    
    model = Model(input=[input1,input2,input3,input4], output=out)
    
    model.compile(loss='mse',
                  optimizer=Adam(lr=0.0055,epsilon=8e-5))
    model.summary()
    
    # control the batch_size to speed up the training
    model.fit([train_featlen[:600000],train_id3[:600000],train_feat[:600000],train_feat2[:600000]], train_y[:600000],
              batch_size=907,
              epochs=1,
              verbose=1,
              callbacks=[]
              )
    model.fit([train_featlen[800000:],train_id3[800000:],train_feat[800000:],train_feat2[800000:]], train_y[800000:],
              batch_size=907,
              epochs=1,
              verbose=0,
              callbacks=[]
              )
    
    model.fit([train_featlen[:600000],train_id3[:600000],train_feat[:600000],train_feat2[:600000]], train_y[:600000],
              batch_size=1027,
              epochs=1,
              verbose=0,
              callbacks=[]
              )
    model.fit([train_featlen,train_id3,train_feat,train_feat2], train_y,
              batch_size=1127,
              epochs=1,
              verbose=0,
              callbacks=[]
              )
  
    if not debug:
        model.fit([test_featlen,test_id3,test_feat,test_feat2], test_y,  batch_size=1127,  epochs=1, verbose=0, validation_data=None,  callbacks=[])

    
    
                 
    # adagrad to finetune
    for e in range(13,1,-2):
        if e == 7:
            model.compile(loss='mse',
                 optimizer=Adagrad(lr=0.008,decay=0.0001))
        print(e)
        if debug:
            model.fit([train_featlen[-e * 100000:-(e - 2) * 100000 - 1],train_id3[-e * 100000:-(e - 2) * 100000 - 1],train_feat[-e * 100000:-(e - 2) * 100000 - 1],train_feat2[-e * 100000:-(e - 2) * 100000 - 1]], train_y[-e * 100000:-(e - 2) * 100000 - 1],
                batch_size=1424,
                epochs=1,
                verbose=1,
                validation_data=([test_featlen,test_id3,test_feat,test_feat2], test_y),
                callbacks=[]
                )
            resde = model.predict([test_featlen, test_id3, test_feat, test_feat2],batch_size =2048,verbose = 0)
            resdebase = 0.40 * resdefm + 0.60 * resde.copy().reshape((-1))
            print("nn dev RMSLE:", rmsle(np.expm1(test_y.reshape((-1))), np.expm1(resdebase)))
        else:
            model.fit([train_featlen[-e * 100000:-(e - 2) * 100000 - 1],train_id3[-e * 100000:-(e - 2) * 100000 - 1],train_feat[-e * 100000:-(e - 2) * 100000 - 1],train_feat2[-e * 100000:-(e - 2) * 100000 - 1]], train_y[-e * 100000:-(e - 2) * 100000 - 1],
                batch_size=1424,
                epochs=1,
                verbose=1,
                validation_data=None,
                callbacks=[]
                )
    resde = None
    if debug:
        resde = model.predict([test_featlen,test_id3,test_feat,test_feat2],batch_size =2048,verbose = 0)


    #model.fit([test_featlen,test_id3,test_feat,test_feat2], test_y,  batch_size=1524,  epochs=1, verbose=0, validation_data=None,  callbacks=[])
    
    res = model.predict([sub_featlen,sub_id3,sub_feat,sub_feat2],batch_size =2048,verbose = 0)


    selt = range(1400000, cateindex.shape[0])
    selt = list(map(lambda x:x-1400000,list(set(selt) & (set(index) | set(index2)))))
    print(len(sel))
    print(len(selt))
    
    test2_feat = test_feat[selt, :]
    test2_feat2 = test_feat2[selt, :]
    test2_featlen = test_featlen[selt, :]
    test2_id3 = test_id3[selt, :]
    test2_y = test_y[selt]

    del(train_id3)
    del(train_feat)
    del(train_feat2)
    del(train_featlen)
    del(test_id3)
    del(test_feat)
    del(test_feat2)
    del(test_featlen)

    gc.collect()

    # more training on the "women & beauty" cate
    index = indexcatesub[indexcatesub == "Women"].index
    index2 = indexcatesub[indexcatesub == 'Beauty'].index
    selsub = list(set(range(indexcatesub.shape[0])) & (set(index) | set(index2)))
    sub2_featlen = sub_featlen[selsub,:]
    sub2_id3 = sub_id3[selsub,:]
    sub2_feat = sub_feat[selsub,:]
    sub2_feat2 = sub_feat2[selsub,:]
    del(sub_id3)
    del(sub_feat)
    del(sub_feat2)
    del(sub_featlen)
    if debug:
        model.fit([train2_featlen, train2_id3, train2_feat, train2_feat2], train2_y,
                  batch_size=900,
                  epochs=1,
                  verbose=0,
                  validation_data=None,
                  callbacks=[]
                  )
        resde2 = model.predict([test2_featlen, test2_id3, test2_feat, test2_feat2],batch_size =2048,verbose = 0)
        resde[selt] = resde2
    else:
        train2_feat = np.concatenate((train2_feat,test2_feat),axis = 0)
        train2_feat2 = np.concatenate((train2_feat2,test2_feat2),axis = 0)
        train2_featlen = np.concatenate((train2_featlen,test2_featlen),axis = 0)
        train2_id3 = np.concatenate((train2_id3,test2_id3),axis = 0)
        train2_y = np.concatenate((train2_y,test2_y),axis = 0)
        model.fit([train2_featlen, train2_id3, train2_feat, train2_feat2], train2_y,
                  batch_size=900,
                  epochs=1,
                  verbose=0,
                  validation_data=None,
                  callbacks=[]
                  )


    res2 = model.predict([sub2_featlen,sub2_id3,sub2_feat,sub2_feat2],batch_size =2048,verbose = 0)
    res[selsub] = res2
    del(res2)
    del(train2_id3)
    del(train2_feat)
    del(train2_feat2)
    del(train2_featlen)
    del(test2_id3)
    del(test2_feat)
    del(test2_feat2)
    del(test2_featlen)
    del(sub2_id3)
    del(sub2_feat)
    del(sub2_feat2)
    del(sub2_featlen)
    del(model)
    
    #return res,resde,test_y

#res,resde,test_y = fit_nn(0)


gc.collect()
print(res[:10])

print(resfm[:10])



cate = ["Beauty","Electronics","Handmade","Home","Kids","Men","catemissing","Other","Sports   Outdoors","Vintage   Collectibles","Women"]
catew = [0.60,0.60,0.60,0.55,0.60,0.65,0.60,0.50,0.50,0.55,0.60]  # ensemble params. 

if debug:
    resdebase = 0.40 * resdefm + 0.60 * resde.copy().reshape((-1))
    print("nn dev RMSLE:", rmsle(np.expm1(test_y.reshape((-1))), np.expm1(resdebase)))
    for i in range(len(cate)):
        c = cate[i]
        w = catew[i]
        print(c)
        indextrain = cateindex[cateindex == c].index
        selt = range(1400000, cateindex.shape[0])
        selt = list(map(lambda x:x-1400000,list(set(selt) & set(indextrain))))
        print(len(selt))
        resdebase[selt] = resde.reshape((-1))[selt] * w + resdefm[selt] * (1-w)

    print("nn dev RMSLE:", rmsle(np.expm1(test_y.reshape((-1))), np.expm1(resdebase)))

resbase = 0.40 * resfm + 0.60 * res.copy().reshape((-1))
for i in range(len(cate)):
    c = cate[i]
    w = catew[i]
    print(c)
    indexsub = indexcatesub[indexcatesub == c].index
    selt = list(indexsub)
    print(len(selt))
    resbase[selt] = res.reshape((-1))[selt] * w + resfm[selt] * (1-w)

f = open('./res.csv','w')
f.write("test_id,price\n")
for i in range(resbase.shape[0]):
    f.write(submission_id[i]+','+str(math.e**(resbase[i]) - 1)+'\n')


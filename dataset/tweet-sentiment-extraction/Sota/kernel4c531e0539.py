# %% [code]
import numpy as np
import pandas as pd
import pickle
from math import log
import time
import matplotlib.pyplot as plt


#ジャカード関数
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def savelistAstxt(savelist,filename):
    with open(filename,"w") as f:
        for i in savelist:
            f.write(str(i).replace(',','').replace('.','')+"\n")
           

#一意な単語リスト作成
def make_unique_list(sentences_list):
    words_list=[]
    for i in sentences_list:
        try:
            words_list.append(i.split())
        except:
            words_list.append(i)


    unique_words=[]

    for i in words_list:
        try:
            for j in i:
                if j not in unique_words:
                    unique_words.append(j)
        
        except:
            if i not in unique_words:
                unique_words.append(i)
        
    return unique_words

#文ごとに単語を切ってリスト化
def make_words_list(genre):
    words_list=[]
    for i in genre["text"]:
        try:
            words_list.append(i.split())
        except:
            words_list.append(i)
    return words_list

#一意な単語のリストファイルの読み込み
def read_unique_words(txtname):
    unique_words=[]
    with open(txtname,"r") as f:
        for i in f.readlines():
            unique_words.append(i.replace('\n',''))
    return unique_words

#Bag Of Wordsの作成
def make_BOW(words_list,unique_words,filename):
    count=0
    bow_list=[]
    for words in words_list:
        bag_of_words=[]
        for unique_word in unique_words:
            try:
                num=words.count(unique_word)
                bag_of_words.append(num)
            except:
                continue
        
        bow_list.append(bag_of_words)
        count+=1
        print("\r{}".format(count),end="")

    with open(filename, 'wb') as web:
        pickle.dump(bow_list , web)



#BOWのファイルの読み込み関数
def open_BOW(filename):
    with open(filename, 'rb') as web:
        bow = pickle.load(web)
    return bow


#idf作成関数
def make_idf(num_of_sentnces,unique_words,BOW):
    idf=[]

    for i in range(len(unique_words)):
        count=0
        for bow in BOW:
            try:
                if bow[i]>0:
                    count+=1
            except:
                continue
        
        idf.append(log((num_of_sentnces+1)/(count+1)))
        print("\r{}クリア/{}".format(str(i),str(len(unique_words))),end="")

    idf_dict={}
    for i,j in enumerate(idf):
        idf_dict[unique_words[i]]=str(j)
    
    return idf_dict



#正解である可能性のある単語群の作成関数
def possible_words_list(idf,min,max,smin,smax):
    return [k for k, v in idf.items() if (float(v)>min and max>float(v)) or (float(v)>smin and smax>float(v))]



#正解の予想を作成
def make_predict_sentences(words_list,possible_words_list):
    predict_list=[]

    for i in words_list:
        predict=""
        try:
            for j in i:
                if j in possible_words_list:
                    predict=predict+str(j)+" "

        except:
            pass

        predict_list.append(predict)

    return predict_list

#正答率算出関数
def correct_rate(genre,predict_list,filename):
    sum=0
    for textID,collect_words,my_predict in zip(genre["textID"],genre["selected_text"],predict_list):
        rate=jaccard(str(collect_words),str(my_predict))
        with open(filename,"a") as f:
            f.write("textID:"+str(textID)+ "正答率：{} 正解：{} 予測：{} ".format(str(rate),str(collect_words),str(my_predict))+"\n")

        # print("正解：{} 予測：{} 正答率：{}".format(str(collect_answer),str(my_answer),str(rate)))
        # print(jaccard(str(collect_answer),str(my_answer)))
        sum+=rate
        # print("\r{}クリア/{}".format(str(count),len(neutral["selected_text"])),end="")


    print("正答率："+str(100*sum/len(genre["selected_text"]))+"%")

def submit_file(genre,predict_list,filename):
    for textID,my_predict in zip(genre["textID"],predict_list):
        with open(filename,"a") as f:
            f.write(str(textID)+ ",{}".format(str(my_predict))+"\n")


#csv読み込み
test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

with open("submission.csv","a") as f:
    f.write("textID,selected_text"+"\n")


#ジャンル分け

neutral_test=test[test['sentiment']=="neutral"]
positive_test=test[test['sentiment']=="positive"]
negative_test=test[test['sentiment']=="negative"]

#文ごとに言葉で分けリスト化

neutral_words_list_test=make_words_list(neutral_test)
positive_words_list_test=make_words_list(positive_test)
negative_words_list_test=make_words_list(negative_test)

#一意な言葉のリスト作成

neutral_unique_words_test=make_unique_list(neutral_test['text'])
positive_unique_words_test=make_unique_list(positive_test['text'])
negative_unique_words_test=make_unique_list(negative_test['text'])

# BOW作成
# ニュートラルにネガティブを合体
neutral_words_list_test.extend(positive_words_list_test)
make_BOW(neutral_words_list_test,negative_unique_words_test,"Negative_BOW.binaryfile")
negative_BOW=open_BOW("Negative_BOW.binaryfile")

#idf作成
negative_idf=make_idf(len(negative_test["text"]),negative_unique_words_test,negative_BOW)

# 正解に含まれている可能性のある言葉の群

negative_possible_words=possible_words_list(negative_idf,0,3,6,10)

#予想
negative_predict=make_predict_sentences(negative_words_list_test,negative_possible_words)

submit_file(negative_test,negative_predict,"submission.csv")

#BOW作成
#ニュートラルにネガティブを合体
#BOW作成
#ニュートラルにネガティブを合体
positive_words_list_test.extend(negative_words_list_test)
make_BOW(positive_words_list_test,neutral_unique_words_test,"Neutral_BOW.binaryfile")
neutral_BOW=open_BOW("Neutral_BOW.binaryfile")

#idf作成
neutral_idf=make_idf(len(neutral_test["text"]),neutral_unique_words_test,neutral_BOW)

# # 正解に含まれている可能性のある言葉の群

neutral_possible_words=possible_words_list(neutral_idf,0,3,8,10)

#予想
neutral_predict=make_predict_sentences(neutral_words_list_test,neutral_possible_words)

submit_file(neutral_test,neutral_predict,"submission.csv")

#BOW作成
#ニュートラルにネガティブを合体
neutral_words_list_test.clear()
neutral_words_list_test=make_words_list(neutral_test)


neutral_words_list_test.extend(negative_words_list_test)
make_BOW(neutral_words_list_test,positive_unique_words_test,"Positive_BOW.binaryfile")
positive_BOW=open_BOW("Positive_BOW.binaryfile")

#idf作成
positive_idf=make_idf(len(positive_test["text"]),positive_unique_words_test,positive_BOW)

# 正解に含まれている可能性のある言葉の群

positive_possible_words=possible_words_list(positive_idf,0,3,6,10)

#予想
positive_predict=make_predict_sentences(positive_words_list_test,positive_possible_words)

submit_file(positive_test,positive_predict,"submission.csv")





# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import sys
import os

import pandas as pd
import heapq
import os
import threading
import time


def cos_similarity(str1,str2):
    if (str1 is None or str2 == None): return 0
    str1 = set(str1.split())
    str2 = set(str2.split())
    com = str1.intersection(str2)
    return len(com)*1.0/len(str1)/len(str2)

'''
两个row 都是set类型
'''
def distance(row1,  row2):
    score = 0
    ##品牌是否相同,得8分
    if(row1["brand_name"] == row2["brand_name"]):
        score = score + 5
    
    ##目录是否相同,得2分
    if(row1["category_name"] == row2["category_name"]):
        score = score + 2
    
    ##name有20分的相似度
    score = score + 20 * cos_similarity(row1["name"],row2["name"])

    ##描述类似，得2分
    ##需要计算每个类别的描述词语的权重，可以用tfidf
    
    ##先使用简单的相同的词语的个数/总个数 *3  最多占3分
    try:
        score = score + 3 * stringutil.cos_similarity(row1["item_description"],row2["item_description"])
    except Exception as e:
        pass
        
    
    ##shipping相同，得1分
    if(row1["shipping"] == row2["shipping"]):
        score = score + 1
    
    return score


'''
将测试数据集合中的每一个商品，
计算其与其相同类目的商品的k紧邻 和 价格
其次是计算在训练集中不存在的那些商品
train_data：全量的训练数据
test_data： 部分测试数据
outf：输出的文件的集合

'''
def knn(train_data,test_data, outf,k=5):
    #读取训练数据  测试数据
    #train_data = dataset.read_train()
    #test_data = dataset.read_test()
       
    ##预处理，将价格=0的删除掉
    train_data = train_data[train_data['price']>0]
    
    cur_index = None
    if os.path.exists(outf):
        fr = open(outf, "r", encoding="gbk")
        lines = fr.readlines()
        if len(lines)>0:
            last_line = lines [-1]
            if last_line!=None and last_line.strip()=="":
                last_line = lines[-2]
            if last_line!=None:
                cur_index = int(last_line.split(",")[0])
        del lines
        fr.close()
    
    ##写出文件
    fw = open(outf, "a", encoding="gbk")
    
    print("当前的索引=",cur_index)
    for index in test_data.index:
        if cur_index!=None and index <= cur_index: continue
        print("processing ",index)
        row = test_data.loc[index]
        cat = row["category_name"]
        brand = row["brand_name"]
        ##从训练数据集合中获取该cat相关的数据
        sameCatDs = train_data[train_data['category_name']==cat]
        compareDs = sameCatDs
        
        ##目录超过15000个商品，那么只跟本品牌的比较
        if len(compareDs) >=10000:
            print("目录太多了，只看该品牌的",len(compareDs))
            ##从相同的brand里面做推理
            sameBrandDs = compareDs[compareDs['brand_name']==brand]
            print("该品牌有",len(sameBrandDs))
            if len(sameBrandDs)>200:
                compareDs = sameBrandDs
    
        if len(compareDs) <=0:
            ##从相同的brand里面做推理
            sameBrandDs = train_data[train_data['brand_name']==brand]
            compareDs = sameBrandDs
        if len(compareDs) <=0:
            ##都不相同的，就不预测了
            print(index,0)
            fw.write(str(index)+",0\r\n")
        else:
            score = []
            print("cat=",cat,"\tbrand=",brand,len(compareDs))
            for rowi in range(0,len(compareDs)):
                similar = distance(row, compareDs.iloc[rowi])
                score.append(similar)
            
            topk = heapq.nlargest(k,score)
            ##获取这topk的id和price
            topkDs = sameCatDs[sameCatDs['price']>=topk[-1]]
            if len(topkDs) > 0:
                sumprice = sum(topkDs['price'])/len(topkDs)
                print(index,sumprice)
                fw.write(str(index)+","+str(sumprice)+"\r\n")
            else:
                print(index,'0')
                fw.write(str(index)+",0\r\n")
        fw.flush()
    fw.close()       
    pass
    
if __name__ == '__main__':

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train_data = pd.read_table("../input/train.tsv", index_col="train_id")
    test_data = pd.read_table("../input/test.tsv", index_col="test_id") 
    knn(train_data=train_data, test_data=test_data,outf="./knn_result.csv",k=3)
    pass
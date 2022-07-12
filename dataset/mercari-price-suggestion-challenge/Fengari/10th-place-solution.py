import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
import time
import random
import string
import re
import gc
from multiprocessing import Pool
from fastcache import clru_cache as lru_cache
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import KFold
import ahocorasick
import sys

def object_size(obj):
    size_in_bytes = sys.getsizeof(obj)
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    return size_in_mb

stemmer = PorterStemmer()
start_time = time.time() 
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

@lru_cache(1024)
def stem(s):
    return stemmer.stem(s)

    
class AC_machine:
    def __init__(self,str_list=None,store_words = False):
        self.tree = None
        if not str_list is None:
            self.make_tree(str_list,store_words)
    def make_tree(self,str_list,store_words=False):
        assert isinstance(str_list,list),'input must be list of string'
        self.tree = ahocorasick.Automaton()
        for strs in str_list:
            self.tree.add_word(str(strs),0 if not store_words else str(strs))
    def make_automaton(self):
        assert not self.tree is None,'must make AC machine first!'
        self.tree.make_automaton()
    def find_all_match_words(self,strs):
        res = self.tree.iter(strs)
        ret = [ans[1] for ans in res]
        return ret
    def __contains__(self,word):
        assert not self.tree is None,'must make AC machine first!'
        return str(word) in self.tree


replace_categories={'iPad/Tablet/eBook Readers':'iPad Tablet eBook Readers',
                    'Dance/Ballet':'Dance Ballet',
                    'Flight/Bomber':'Flight Bomber',
                    'Varsity/Baseball':'Varsity Baseball',
                    'iPad/Tablet/eBook Access':'iPad Tablet eBook Access',
                    'Car A/V Installation':'Car A V Installation',
                    'Dance/Ballet':'Dance Ballet',
                    'Indoor/Outdoor Games':'Indoor Outdoor Games',
                    'Entertaining/Serving':'Entertaining Serving',
                   }

replace_list=[
u'(',u' ',
u')',u' ',
u'#',u' ',
u'*',u' ',
u'+',u' add ',
u'&',u' and ',

######Latin Capital Letter #######
u'\xc0',u'a',     
u'\xc1',u'a',
u'\xc2',u'a',
u'\xc3',u'a',
u'\xc4',u'a',
u'\xc5',u'a',
u'\xe0',u'a',
u'\xe1',u'a',
u'\xe2',u'a',
u'\xe3',u'a',
u'\xe4',u'a',
u'\xe5',u'a',

u'\xc8',u'e',
u'\xc9',u'e',
u'\xca',u'e',
u'\xcb',u'e',
u'\xe8',u'e',
u'\xe9',u'e',
u'\xea',u'e',
u'\xeb',u'e',

u'\xcc',u'i',
u'\xcd',u'i',
u'\xce',u'i',
u'\xcf',u'i',
u'\xec',u'i',
u'\xed',u'i',
u'\xee',u'i',
u'\xef',u'i',

u'\xd2',u'o',
u'\xd3',u'o',
u'\xd4',u'o',
u'\xd5',u'o',
u'\xd6',u'o',
u'\xf2',u'o',
u'\xf3',u'o',
u'\xf4',u'o',
u'\xf5',u'o',
u'\xf6',u'o',

u'\xd9',u'u',
u'\xda',u'u',
u'\xdb',u'u',
u'\xdc',u'u',
u'\xf9',u'u',
u'\xfa',u'u',
u'\xfb',u'u',
u'\xfc',u'u',

u'\xc7',u'c',     
u'\xe7',u'c',
###############
u'worn',u'wear',
u'shipped',u'ship',
u'ships',u'ship'

u'great',u'good',
u'excellent',u'good',
u'perfect',u'good',

u'purchased',u'buy',
u'purchase',u'buy',
u'bought',u'buy',
u'purchasing',u'buy',
u"desn't",u'not',
u"didn't",u'not',
u"don't",u'not',

u'questions',u'question',
u'retails',u'retail',
u'comments',u'comment',
u'scratches',u'scratch',
u'items',u'item',
u'tags',u'tag',
u'stains',u'stain',
u'pictures',u'picture',
u'colors',u'color',
u'colours',u'color',
u'colour',u'color',
u'included',u'include',
u'includes',u'include',
u'including',u'include',

u'opened',u'open',
u'closed',u'close',
u'fits',u'fit',
u'fitting',u'fit',
u'fitted',u'fit',

u'washed',u'wash',
u'washing',u'wash',
############
u'iphone 4/4s',u'iphones',
u'iphone 4/5s',u'iphones',
u'iphone 5/5s',u'iphones',
u'iphone 6/6s',u'iphones',
u'iphone 6/6s plus',u'iphones',
u'iphone 6/6splus',u'iphones',
u'iphone 6/7s',u'iphones',
u'iphone 66s',u'iphones',
u'iphone 66s plus',u'iphones',
u'iphone 66splus',u'iphones',
u'iphone 7/6s',u'iphones',
u'iphone 7/7s',u'iphones',
u'iphone 7/7s plus',u'iphones',
u'iphone 75s',u'iphones',
u'iphone 76s',u'iphones',
u'iphone4/4s',u'iphones',
u'iphone5/5s',u'iphones',
u'iphone6/6s',u'iphones',
u'iphone6/6s plus',u'iphones',
u'iphone6/6splus',u'iphones',
u'iphone7/7s',u'iphones',
u'iphone7/7s plus',u'iphones',

u'samsung galaxy s6 edge',u'samsunggalaxys6edge',
u'samsung galaxy s6edge', u'samsunggalaxys6edge',
u'samsung galaxy s7 edge',u'samsunggalaxys7edge',
u'samsung galaxy s7edge',u'samsunggalaxys7edge',
u'samsung galaxy s8 plus',u'samsunggalaxys8plus',
u'samsung galaxy s8plus',u'samsunggalaxys8plus',
u'samsung s8 edge',u'samsunggalaxys8edge',
u'samsung s8 plus',u'samsunggalaxys8plus',
u'samsung s6 edge',u'samsunggalaxys6edge',
u'samsung s7 edge',u'samsunggalaxys7edge',
u'samsung galaxy s2',u'samsunggalaxys2',
u'samsung galaxy s3',u'samsunggalaxys3',
u'samsung galaxy s4',u'samsunggalaxys4',
u'samsung galaxy s5',u'samsunggalaxys5',
u'samsung galaxy s6',u'samsunggalaxys6',
u'samsung galaxy s7',u'samsunggalaxys7',
u'samsung galaxy s8',u'samsunggalaxys8',
u'samsung galaxys4',u'samsunggalaxys4',
u'samsung galaxys5',u'samsunggalaxys5',
u'samsung s2',u'samsunggalaxys2',
u'samsung s3',u'samsunggalaxys3',
u'samsung s4',u'samsunggalaxys4',
u'samsung s5',u'samsunggalaxys5',
u'samsung s6',u'samsunggalaxys6',
u'samsung s7',u'samsunggalaxys7',
u'samsung s8',u'samsunggalaxys8edge',


u'iphone 6 plus',u'iphone6plus',
u'iphone 6plus',u'iphone6plus',
u'iphone 6s plus',u'iphone6splus',
u'iphone 6splus',u'iphone6splus',
u'iphone 7 plus',u'iphone7plus',
u'iphone 7plus',u'iphone7plus',
u'iphone 7s plus',u'iphone7splus',
u'iphone6 plus',u'iphone6plus',
u'iphone6plus',u'iphone6plus',
u'iphone6s plus',u'iphone6splus',
u'iphone6splus',u'iphone6splus',
u'iphone7 plus',u'iphone7plus',
u'iphone7plus',u'iphone7plus',


u'iphone 3s',u'iphone3s',
u'iphone 4s',u'iphone4s',
u'iphone 5s',u'iphone5s',
u'iphone 6s',u'iphone6s',
u'iphone 7s',u'iphone7s',


u'iphone 1',u'iphone1',
u'iphone 2',u'iphone2',
u'iphone 3',u'iphone3',
u'iphone 4',u'iphone4',
u'iphone 5',u'iphone5',
u'iphone 6',u'iphone6',
u'iphone 7',u'iphone7',
u'iphone 8',u'iphone8',
u'iphone x',u'iphonex',



u'ipad generation',u'ipad',
u'ipad 1st generation',u'ipad1',
u'ipad 2nd generation',u'ipad2',
u'ipad 3rd generation',u'ipad3',
u'ipad 4th generation',u'ipad4',
u'ipad air 1st generation',u'ipadair',

u'ipad 1st gen',u'ipad1',
u'ipad 2nd gen',u'ipad2',
u'ipad 3rd gen',u'ipad3',
u'ipad 4th gen',u'ipad4',
u'ipad 5th gen',u'ipad5',
u'ipad air 1st gen',u'ipadair',
u'ipad air 5th gen',u'ipadair5',
u'ipad air 1st gen',u'ipadair',
u'ipad air 5th gen',u'ipadair5',

u'ipad 1st',u'ipad1',
u'ipad air',u'ipadair',
u'ipad mini','ipadmini',
u'ipad gen',u'ipad',
u'ipad touch',u'ipadtouch',
u'ipad pro',u'ipadpro',


u'ipod 1st generation',u'ipod1',
u'ipod 2nd generation',u'ipod2',
u'ipod 3rd generation',u'ipod3',
u'ipod 4th generation',u'ipod4',
u'ipod 5th generation',u'ipod5',
u'ipod 6th generation',u'ipod6',

u'ipod nano 1st generation',u'ipodnano1',
u'ipod nano 2nd generation',u'ipodnano2',
u'ipod nano 3rd generation',u'ipodnano3',
u'ipod nano 4th generation',u'ipodnano4',
u'ipod nano 5th generation',u'ipodnano5',
u'ipod nano 6th generation',u'ipodnano6',
u'ipod nano 7th generation',u'ipodnano7',
u'ipod shuffle 2nd generation',u'ipodshuffle2',
u'ipod shuffle 3rd generation',u'ipodshuffle3',
u'ipod shuffle 4th generation',u'ipodshuffle4',
u'ipod touch 2nd generation',u'ipodtouch2',
u'ipod touch 3rd generation',u'ipodtouch3',
u'ipod touch 4th generation',u'ipodtouch3',
u'ipod touch 5th generation',u'ipodtouch6',
u'ipod touch 6th generation',u'ipodtouch6',


u'ipod nano 2nd gen',u'ipodnano2',
u'ipod nano 3rd gen',u'ipodnano3',
u'ipod nano 4th gen',u'ipodnano4',
u'ipod nano 5th gen',u'ipodnano5',
u'ipod nano 5thgen',u'ipodnano5',
u'ipod nano 6th gen',u'ipodnano6',
u'ipod nano 7th gen',u'ipodnano7',
u'ipod nano gen',u'ipodnano',
u'ipod shuffle 3rdgen',u'ipodshuffle3',
u'ipod shuffle 4th gen',u'ipodshuffle4',
u'ipod touch 1st gen',u'ipodtouch1',
u'ipod touch 2nd gen',u'ipodtouch2',
u'ipod touch 3rd gen',u'ipodtouch3',
u'ipod touch 4th gen',u'ipodtouch4',
u'ipod touch 5th gen',u'ipodtouch5',
u'ipod touch 6th gen',u'ipodtouch6',
u'ipod touch 7th gen',u'ipodtouch7',
u'ipod touch 2nd',u'ipodtouch2',
u'ipod touch gen',u'ipodtouch',
u'ipod shuffle',u'ipodshuffle',
u'ipod touch',u'ipodtouch',
u'ipod 2nd gen',u'ipod2',
u'ipod 3rd gen',u'ipod3',
u'ipod 4th gen',u'ipod4',
u'ipod 5th gen',u'ipod5',
u'ipod 6th gen',u'ipod6',
u'ipod 7th gen',u'ipod7',
u'ipod gen',u'ipod',
u'ipod nano',u'ipodnano',
u'ipod 1st',u'ipod1',
u'ipod 5th',u'ipod5',


u'new macbook 13.3 inch',u'newapplemacbook13',
u'new macbook air',u'newapplemacbookair',
u'new macbook pro',u'newapplemacbookpro',

u'apple macbook 13inch',u'applemacbook13',
u'apple macbook 13.3"',u'applemacbook13',
u'apple macbook 13.3 in',u'applemacbook13',
u'apple macbook 13.3',u'applemacbook13',
u'apple macbook 12"',u'applemacbook12',
u'apple macbook 13"',u'applemacbook13',
u'apple macbook 13',u'applemacbook13',


u'apple macbook air',u'applemacbookair',
u'apple macbook pro',u'applemacbookpro',
u'macbook 12 inch',u'applemacbook12',
u'macbook 12"',u'applemacbook12',
u'macbook 13',u'applemacbook13',
u'macbook 13 inch',u'applemacbook13',
u'macbook 13.3"',u'applemacbook13',
u'macbook 13.3',u'applemacbook13',
u'macbook 13"',u'applemacbook13',
u'macbook 13in',u'applemacbook13',
u'macbook air',u'applemacbookair',
u'macbook pro',u'applemacbookpro',
u'macbookair',u'applemacbookair',
u'macbookpro',u'applemacbookpro',


u'32 gb',u'32gb',
u'8 gb',u'8gb',
u'64 gb',u'64gb',
u'80 gb',u'80gb',
u'128 gb',u'128gb',
u'250 gb',u'250gb',
u'256 gb',u'256gb',
u'512 gb',u'512gb',
u'10 ft',u'10ft',
u'fl oz',u'oz',
u'1 t',u'1t',

u'xbox one s',u'xboxones',
u'xbox one',u'xboxone',
u'xbox 360',u'xbox360',
u'gta 5',u'gta5',

u'1.7 ghz', u'1.7ghz',
u'2.00ghz', u'2.0ghz',
u'2.400 ghz', u'2.4ghz',
u'2.40 ghz', u'2.4ghz',
u'2.40ghz', u'2.4ghz',
u'3.20 ghz', u'3.2ghz', 
u'3.20ghz', u'3.2ghz',
u'3.30ghz', u'3.3ghz',
u'3.60 ghz', u'3.6ghz',
u'3.8 ghz', u'3.8ghz',

u'cases',u'case',
u'cell phone case',u'case',
u'phone case',u'case',
]


def replace_cat(x):
    if type(x)==np.float:
        return 'NULL'
    else:
        for k,v in replace_categories.items():
            if k in x:
                x=x.replace(k,v)
        return x

        
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def normalized_category(text):
    if type(text)==np.float:
        return ("No Label", "No Label", "No Label")
    for k,v in replace_categories.items():
        text=text.replace(k,v)
    try:
        s=text.split("/")
        if len(s)==3:
            return s
        else:
            return ("No Label", "No Label", "No Label")
    except:
        return ("No Label", "No Label", "No Label")


def normalized_brand(line):
    brand = line[0]
    name = line[1]
    namesplit = name.split(' ')
    if brand == 'missing':
        for x in namesplit:
            if x in all_brands:
                return x
    if name in all_brands:
        return name
    return brand


def normalized_name(text,use_replaced=True,remove_stopwords=False,stem_words=False): 
    
    text=str(text).lower()
    if use_replaced:
        for i in range(len(replace_list)//2):
            text=text.replace(replace_list[2*i],replace_list[2*i+1])
        
    text = non_alphanums.sub(' ',text).strip().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    if len(text)==0:
        return 'NULL'
    text = u" ".join(text)
    if stem_words:
        text = text.split()
        stemmed_words = [stem(word) for word in text]
        text = u" ".join(stemmed_words)
    
    return text


def normalized_desc(text,use_replaced=True,remove_stopwords=False, stem_words=False): 
    
    text=str(text).lower()
    if use_replaced:
        for i in range(len(replace_list)//2):
            text=text.replace(replace_list[2*i],replace_list[2*i+1])
        
    text = non_alphanums.sub(' ',text).strip().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        
    if len(text)==0:
        return 'NULL'
    
    text = u" ".join(text)
    if stem_words:
        text = text.split()
        stemmed_words = [stem(word) for word in text]
        text = u" ".join(stemmed_words)  

    return text
    
def object_size_in_mb(obj):
    size_in_bytes = sys.getsizeof(obj)
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    return size_in_mb


def introduce_new_unseen_words(desc):
    desc = desc.split(' ')
    if random.randrange(0, 10) == 0: # 10% chance of adding an unseen word
        new_word = ''.join(random.sample(string.ascii_letters, random.randrange(3, 15)))
        desc.insert(0, new_word)
    return ' '.join(desc)
    
def preprocessing(dataset):
    
    dataset['brand_name'].fillna(value='missing', inplace=True)
    
    for v in ['brand_name','category_name','name','item_description']:
        dataset[v] = dataset[v].map(lambda x:str(x).lower())
            
    var_ = ['category_level1','category_level2','category_level3']

    def get_version(name):

        name = ''.join(name.split())
        name = name.replace('+',"plus")
        res = ''
        for product_name in ['4','5','6','7','8']:
            for s in ['s','c','']:
                for plus in ['plus','','edge']:
                    if 'iphone'+product_name+s+plus in name or 'galaxy'+product_name+s+plus in name:
                        if len(res) < len(product_name+s+plus):
                            res = product_name+s+plus
        if res == '':
            return 'missing'
        return res

    # def get_dataset_productt(dataset,cate):
        # if cate == 'electronics/video games & consoles/games':
            # return
        # idx = dataset.index[dataset.category_name==cate]
        # dataset.loc[idx, 'version'] = dataset.loc[idx, 'name'].apply(lambda x: get_version(x))

    # cates = list(set(dataset['category_name'].values))
    # cates = [x for x in cates if 'electronics' in x]
    
    # for cate in cates:
        # get_dataset_productt(dataset,cate)
        
    dataset['version'] = 'missing'
    idx = np.logical_and(dataset['category_name'].str.contains('electronics'),
                         dataset['category_name'] != 'electronics/video games & consoles/games')
    idx = dataset.index[idx]
    dataset.loc[idx, 'version'] = dataset.loc[idx, 'name'].apply(lambda x: get_version(x))

    def get_necklace_version(name):

        name = ''.join(name.split())
        for product_name in ['10k','14k','18k','22k','24k','999','925']:
            if product_name in name:
                return product_name

        return 'missing'

    # def get_neck_dataset_productt(dataset,cate):
        # idx = dataset.index[dataset['category_name']==cate]
        # dataset.loc[idx, 'version'] = dataset.loc[idx,'name'].apply(lambda x: get_necklace_version(x))
        # idx = dataset.index[np.logical_and(dataset['category_name']==cate, dataset['version']=='missing')]
        # dataset.loc[idx, 'version'] = dataset.loc[idx, 'item_description'].apply(lambda x: get_necklace_version(x))

    # cates = list(set(dataset['category_name'].values))
    # cates = [x for x in cates if 'jewelry' in x]

    # for cate in cates:
        # get_neck_dataset_productt(dataset,cate)
    print('size dataset 2: {}'.format(object_size(dataset)))
    
    idx = dataset['category_name'].str.contains('jewelry')
    idx = dataset.index[idx]
    dataset.loc[idx, 'version'] = dataset.loc[idx, 'name'].apply(lambda x: get_necklace_version(x))
    
    idx = np.logical_and(dataset['category_name'].str.contains('jewelry'),
                         dataset['version'] == 'missing')
    idx = dataset.index[idx]
    dataset.loc[idx, 'version'] = dataset.loc[idx, 'item_description'].apply(lambda x: get_necklace_version(x))
    
    def get_leggings_version(name):
        name = ' '.join(name.split())
        res = ''
        if 'os' in name or 'one size' in name:
            res += 'os'
        if 'tc' in name:
            res += 'tc'
        return res
    
    idx = dataset.index[dataset['category_name']=='women/athletic apparel/pants, tights, leggings']
    dataset.loc[idx, 'version'] = dataset.loc[idx, 'name'].apply(lambda x: get_leggings_version(x))
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].replace(to_replace='No description yet', value='missing', inplace=True)
    print('size dataset 3: {}'.format(object_size(dataset)))
    dataset['name'] = dataset['name'].map(normalized_name)

    dataset['category_level1'],dataset['category_level2'],dataset['category_level3'] = \
    zip(*dataset['category_name'].map(normalized_category))

    dataset.drop('category_name',axis=1,inplace=True)

    dataset['item_description'] = dataset['item_description'].map(normalized_desc)
    print('size dataset 4: {}'.format(object_size(dataset)))
    return dataset
    
    
def prepare_data():
    
    start_time=time.time()
    train=pd.read_table('../input/train.tsv')
    test=pd.read_table('../input/test.tsv')
    
    if USE_TESTX6:
        print('test size before: {}'.format(test.shape))
        test = pd.concat([test.copy() for _ in range(6)])
        test = test.reset_index(drop=True)
        test['item_description'] = test['item_description'].apply(introduce_new_unseen_words)
        print('test size after: {}'.format(test.shape))
    
    print('[{}] data load'.format(time.time() - start_time))
    #global all_brands
    #all_brands=pd.concat([train.brand_name,test.brand_name]).value_counts().index.values[:4000]
    #all_brands = AC_machine(list(all_brands))
    #all_brands.make_automaton()
    global category_select
    category_select = pd.concat([train.category_name,test.category_name]).value_counts().index.values[:400]
    
    test.index = test.index.values + len(train)
    df = pd.concat([train,test])
    
    print('size df: {}'.format(object_size(df)))
    train_len = train.shape[0]
    del train, test
    p = Pool(processes=4)
    df = p.map(preprocessing,np.array_split(df,4))
    p.close(); p.join();
    df = pd.concat(df,axis=0)
    print('size df: {}'.format(object_size(df)))
    
    # df = preprocessing(df)

    df['len_desc'] = np.log(df['item_description'].apply(lambda x: len(str(x).split())))
    buckets = np.linspace(np.min(df['len_desc']), np.max(df['len_desc']), 10)
    df['len_desc'] = [np.sum(buckets <= length) for length in df['len_desc'].values]
    
    print('[{}] data preprocessing'.format(time.time() - start_time)) 
    le = LabelEncoder()
    le.fit(df.len_desc)
    df['len_desc'] = le.transform(df.len_desc)
    le.fit(df.category_level1)
    df['category_level1'] = le.transform(df.category_level1)
    le.fit(df.category_level2)
    df['category_level2'] = le.transform(df.category_level2)
    le.fit(df.category_level3)
    df['category_level3'] = le.transform(df.category_level3)
    le.fit(df.brand_name)
    df['brand_name'] = le.transform(df.brand_name)
    le.fit(df.item_condition_id)
    df['item_condition_id'] = le.transform(df.item_condition_id)
    # le.fit(df.productt)
    # df['productt'] = le.transform(df.productt)
    # le.fit(df.rom)
    # df['rom'] = le.transform(df.rom)
    le.fit(df.version)
    df['version'] = le.transform(df.version)
    # le.fit(df.ppairs)
    # df['ppairs'] = le.transform(df.ppairs)
    del le
    print('[{}] labelencode preprocessing'.format(time.time() - start_time)) 
    return df,train_len

USE_VAL=True
USE_TESTX6=False
df,train_len = prepare_data()    



if USE_VAL:     
    skf=KFold(train_len,n_folds=5,shuffle=True,random_state=1123)
    for _tr_ind,_te_ind  in skf:
        tr_ind=_tr_ind
        te_ind=_te_ind
        break
else:
    tr_ind=range(train_len)
    te_ind=range(train_len,df.shape[0])



#############FM_FTRL################   
import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,LabelBinarizer
def get_wb(df,idxs):
    df = df.loc[idxs]
    par = {"hash_ngrams": 2,
           "hash_ngrams_weights": [1.5, 1.0],
           "hash_size": 2 ** 22,
           "norm": 'l2',
           "tf": 'binary',
           "idf": None}
           
    wb = wordbatch.WordBatch(normalize_text=None, extractor=(WordBag, par), procs=8)
    wb.dictionary_freeze= True
    wb.fit(df['name'])

    wb_c1 = LabelBinarizer(sparse_output=True)
    wb_c2 = LabelBinarizer(sparse_output=True)
    wb_c3 = LabelBinarizer(sparse_output=True)
    wb_c4 = LabelBinarizer(sparse_output=True)
    wb_c5 = LabelBinarizer(sparse_output=True)
    wb_c6 = LabelBinarizer(sparse_output=True)
    wb_c7 = LabelBinarizer(sparse_output=True)
    wb_c8 = LabelBinarizer(sparse_output=True)
    wb_c1.fit(df.category_level1.values)
    wb_c2.fit(df.category_level2.values)
    wb_c3.fit(df.category_level3.values)
    wb_c4.fit(df['brand_name'])
    wb_c5.fit(df['item_condition_id'])
    wb_c6.fit(df['shipping'])
    wb_c7.fit(df['version'])
    wb_c8.fit(df['len_desc'])
    
    ct = CountVectorizer(max_features=50000,ngram_range=(1,1),min_df=5,binary=True)
    ct.fit(df['name'])   
    
    par = {"hash_ngrams": 2,
           "hash_ngrams_weights": [1.0, 1.0],
           "hash_size": 2 ** 28,
           "norm": "l2",
           "tf": 1.0,
           "idf": None}
    
    wb_1 = wordbatch.WordBatch(normalize_text=None, extractor=(WordBag, par), procs=8)
    wb_1.dictionary_freeze= True
    wb_1.fit(df['item_description'])
    return wb,wb_c1,wb_c2,wb_c3,wb_c4,wb_c5,wb_c6,wb_c7,wb_c8,ct,wb_1

def get_feature(df,idxs,wblist,masklist=None):
    wb,wb_c1,wb_c2,wb_c3,wb_c4,wb_c5,wb_c6,wb_c7,wb_c8,ct,wb_1 = wblist
    df = df.loc[idxs]
    
    X_name = wb.transform(df['name'])
    if masklist is None:
        mask_1 = np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    else:
        mask_1 = masklist[0]
    X_name = X_name[:,mask_1 ]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))
    
    X_category1 = wb_c1.transform(df.category_level1.values)
    X_category2 = wb_c2.transform(df.category_level2.values)
    X_category3 = wb_c3.transform(df.category_level3.values)
    X_brand = wb_c4.transform(df['brand_name'])
    X_item = wb_c5.transform(df['item_condition_id'])
    X_ship = wb_c6.transform(df['shipping'])
    X_version = wb_c7.transform(df['version'])
    X_len_desc = wb_c8.transform(df['len_desc'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))
    
    X_name_2 = ct.transform(df['name'])
    
    X_description = wb_1.transform(df['item_description'])
    if masklist is None:
        mask_2 = np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    else:
        mask_2 = masklist[1]
    X_description = X_description[:,mask_2 ]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
    
    print(X_name_2.shape, X_item.shape, X_ship.shape, X_description.shape,
          X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_name.shape, X_version.shape, X_len_desc.shape)
    
    sparse_merge = hstack((X_name,
                           X_name_2,
                           X_len_desc,
                           X_item,
                           X_ship,
                           # X_productt,
                           # X_ppairs,
                           # X_rom,
                           X_version,
                           X_brand,
                           X_category1,
                           X_category2,
                           X_category3,
                           X_description)).tocsr()


    print(sparse_merge.shape)
    if masklist is None:
        mask_3 = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    else:
        mask_3 = masklist[2]
    sparse_merge = sparse_merge[:, mask_3]
    print(sparse_merge.shape)
    
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))
    del X_name_2, X_item,X_len_desc, X_ship, 
    del X_version, X_description, X_brand, X_category1
    del X_category2, X_category3, X_name
    gc.collect()   
    y=np.log1p(df.price.values)
    if masklist is None: 
        masklist = [mask_1,mask_2,mask_3]
        return sparse_merge,y,masklist
    return sparse_merge,y
import psutil
import os
def get_memory_use():
    info = psutil.virtual_memory()
    print (u'RAM USED：',psutil.Process(os.getpid()).memory_info().rss*1./1024/1024/1024,'GB')
    print (u'RAM RATIO：',info.percent    )
def get_FMpreds(df):
    get_memory_use()
    print('begin TRAIN')
    wb,wb_c1,wb_c2,wb_c3,wb_c4,wb_c5,wb_c6,wb_c7,wb_c8,ct,wb_1 = get_wb(df,tr_ind)
    train_X,train_y,masklist = get_feature(df,tr_ind,[wb,wb_c1,wb_c2,wb_c3,wb_c4,wb_c5,wb_c6,wb_c7,wb_c8,ct,wb_1])
    if USE_VAL:
        valid_X,valid_y = get_feature(df,te_ind,[wb,wb_c1,wb_c2,wb_c3,wb_c4,wb_c5,wb_c6,wb_c7,wb_c8,ct,wb_1],masklist)
    print(train_X.shape,train_y.shape)
    
    gc.collect()

    model = FM_FTRL(alpha=0.03,
                    beta=0.01,
                    L1=0.00001,
                    L2=0.1,
                    D=train_X.shape[1],
                    alpha_fm=0.1,
                    L2_fm=0.01,
                    init_fm=0.01,
                    D_fm=300,
                    e_noise=0.0001,
                    iters=6,
                    inv_link="identity",
                    threads=4,
                    weight_fm=1.0)


    
    model.fit(train_X,train_y)
    get_memory_use()
    del train_X, train_y; gc.collect()                  
    print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))
    if USE_VAL:
        predFM = model.predict(X=valid_X)
        print("FM_FTRL dev RMSLE :", rmsle(np.expm1(valid_y), np.expm1(predFM)))
        del valid_y, valid_X;gc.collect()
    else:
        l = len(te_ind)
        dl = int(l/10)
        now = 0 
        predFM = []
        for i in range(10):
            print('###sub test:%d'%(i))
            if i == 9: dl = l - now
            test_X,test_y = get_feature(df,te_ind[now:now+dl],[wb,wb_c1,wb_c2,wb_c3,wb_c4,wb_c5,wb_c6,wb_c7,wb_c8,ct,wb_1],masklist)
            predFM.append(model.predict(test_X))
            now += dl
            del test_X,test_y
            gc.collect()
        predFM = np.concatenate(predFM)
    print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))  
    #model.fit(train_X, train_y)
    #predFM=model.predict(X=X_test)
    #print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
    
    return predFM
 
 
 
# predFM = get_FMpreds(df)

################################ NN model ######################
import pyximport
pyximport.install()
import os
import random
import numpy as np
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '10000'
np.random.seed(520)
random.seed(520)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=1)
from keras import backend
tf.set_random_seed(520)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU
from keras.layers import Embedding, Flatten, Activation,Convolution1D,GlobalMaxPooling1D,GlobalAveragePooling1D
# from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras.layers import BatchNormalization


token=Tokenizer(num_words=50000)
raw_name=df['name'].astype(str)
token.fit_on_texts(raw_name[:train_len])
raw_name=token.texts_to_sequences(raw_name)
name_word=50000+10
token=Tokenizer(num_words=60000)
raw_desc=df['item_description'].astype(str)
token.fit_on_texts(raw_desc[:train_len])
raw_desc=token.texts_to_sequences(raw_desc)
desc_word=60000+10
del token
gc.collect()

brand_word=df.brand_name.max()+1
condition_word=df.item_condition_id.max()+1

#productt_word=df.productt.max()+1
#ppairs_word=df.ppairs.max()+1
#rom_word=df['rom'].max()+1
#version_word=df['version'].max()+1
len_word=df['len_desc'].max()+1

cate_level1_word=df.category_level1.max()+1
cate_level2_word=df.category_level2.max()+1
cate_level3_word=df.category_level3.max()+1

name_max_len=18
desc_max_len=60
raw_name=pad_sequences(raw_name,maxlen=name_max_len)
raw_desc=pad_sequences(raw_desc,maxlen=desc_max_len)

print('[{}] token preprocessing'.format(time.time() - start_time))




from keras.layers import Layer,Lambda
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class GlobalAveragePooling1DMasked(Layer):

    def __init__(self,**kwargs):
        super(GlobalAveragePooling1DMasked, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GlobalAveragePooling1DMasked, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        if mask != None:
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], )
        

def fastext():
    # Inputs
    name = Input(shape=[name_max_len], name="name")
    item_desc = Input(shape=[desc_max_len], name="item_desc")
    
    shipping = Input(shape=[1], name='shipping')
    
    brand_name = Input(shape=[1], name="brand_name")   # category
    item_condition = Input(shape=[1], name="item_condition")  #category
    cat_level1 = Input(shape=[1], name="cat_level1")        #category
    cat_level2 = Input(shape=[1], name="cat_level2")        #category
    cat_level3 = Input(shape=[1], name="cat_level3")        #category
    len_desc = Input(shape=[1], name='len_desc')         #category
    
#     emb_layer = Embedding(name_word+desc_word,16,embeddings_initializer='uniform')
#     # Embeddings layers (adjust outputs to help model)
#     emb_name = emb_layer(name)
#     emb_item_desc = emb_layer(item_desc)
    
        
    # Embeddings layers (adjust outputs to help model)
    emb_name = Embedding(name_word,64,embeddings_initializer='uniform',mask_zero=False)(name)
    emb_item_desc = Embedding(desc_word,64,embeddings_initializer='uniform',mask_zero=False)(item_desc)
    
#     # Embeddings layers (adjust outputs to help model)
#     emb_name = Embedding(name_word,16,embeddings_initializer='uniform')(name)
#     emb_item_desc = Embedding(desc_word, 16,embeddings_initializer='uniform')(item_desc)
    
    emb_brand_name = Embedding(brand_word, 32,embeddings_initializer='uniform')(brand_name)
    emb_item_condition = Embedding(condition_word, 6,embeddings_initializer='uniform')(item_condition)

    emb_cat_level1 = Embedding(cate_level1_word, 6, embeddings_initializer='uniform')(cat_level1)
    emb_cat_level2 = Embedding(cate_level2_word, 16, embeddings_initializer='uniform')(cat_level2)
    emb_cat_level3 = Embedding(cate_level3_word, 32, embeddings_initializer='uniform')(cat_level3)
    emb_len_desc = Embedding(desc_word, 8, embeddings_initializer='uniform')(len_desc)
    
    
    warppers = []

    for emb in [emb_name, emb_item_desc]:
        avg = GlobalAveragePooling1D()(emb)
        #m = GlobalMaxPooling1D()(emb)
        #avg = GlobalAveragePooling1DMasked()(emb)
        #avg = AvgMasked(emb)
        warppers.append(avg)
        #warppers.append(m)
    textlayer = concatenate(warppers)    
        
    

    rnn_layer1 = GRU(8,name='item_desc_gru') (emb_item_desc)
    rnn_layer2 = GRU(16,name='name_gru') (emb_name)
    #cnn_layer=concatenate(cnn_warppers,axis=1)
    #cnn_layer=Attention(68)(cnn_layer)
    main = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_item_condition)
        , Flatten() (emb_cat_level1)
        , Flatten() (emb_cat_level2)
        , Flatten() (emb_cat_level3)
        , Flatten() (emb_len_desc)
        , shipping
        #, cnn_layer
        , textlayer
        , rnn_layer1
        , rnn_layer2
        # , att_emb_name
        # , att_emb_desc
    ])
    # main = Dropout(0.2)(main)
    # (incressing the nodes or adding layers does not effect the time quite as much as the rnn layers)
    main = BatchNormalization()(main)
    fc = Dropout(0.05)(Dense(256,activation='relu') (main))
    fc = Dropout(0.05)(Dense(128,activation='relu') (fc))
    # the output layer.
    output = Dense(1, activation="linear") (fc)
    
    model = Model([name, 
                   item_desc,
                   shipping, 
                   len_desc,
                   brand_name , item_condition, 
                   cat_level1,cat_level2,cat_level3], output)
 
    #opt=Adam(lr=0.002)
    model.compile(loss = 'mse', optimizer = 'adam')

    return model 
    
def fastext_v2():
    # Inputs
    name = Input(shape=[name_max_len], name="name")
    item_desc = Input(shape=[desc_max_len], name="item_desc")
    
    shipping = Input(shape=[1], name='shipping')
    
    brand_name = Input(shape=[1], name="brand_name")   # category
    item_condition = Input(shape=[1], name="item_condition")  #category
    cat_level1 = Input(shape=[1], name="cat_level1")        #category
    cat_level2 = Input(shape=[1], name="cat_level2")        #category
    cat_level3 = Input(shape=[1], name="cat_level3")        #category
    len_desc = Input(shape=[1],name='len_desc')    

    
        
    # Embeddings layers (adjust outputs to help model)
    emb_name = Embedding(name_word,158,embeddings_initializer='uniform')(name)
    emb_item_desc = Embedding(desc_word,158,embeddings_initializer='uniform')(item_desc)
    
#     # Embeddings layers (adjust outputs to help model)
#     emb_name = Embedding(name_word,16,embeddings_initializer='uniform')(name)
#     emb_item_desc = Embedding(desc_word, 16,embeddings_initializer='uniform')(item_desc)
    
    emb_brand_name = Embedding(brand_word, 16,embeddings_initializer='uniform')(brand_name)
    emb_item_condition = Embedding(condition_word, 3,embeddings_initializer='uniform')(item_condition)
    emb_len_desc = Embedding(len_word, 8, embeddings_initializer='uniform')(len_desc)

    emb_cat_level1 = Embedding(cate_level1_word, 3, embeddings_initializer='uniform')(cat_level1)
    emb_cat_level2 = Embedding(cate_level2_word, 8, embeddings_initializer='uniform')(cat_level2)
    emb_cat_level3 = Embedding(cate_level3_word, 16, embeddings_initializer='uniform')(cat_level3)
    
    
    warppers = []
    for emb in [emb_name, emb_item_desc]:
        avg = GlobalAveragePooling1D()(emb)
        warppers.append(avg)
    textlayer = concatenate(warppers)  
    
    flat_emb_name = Flatten()(emb_name)
    flat_emb_brand_name =   Flatten() (emb_brand_name)
    flat_emb_item_condition = Flatten() (emb_item_condition)
    flat_emb_len_desc = Flatten() (emb_len_desc)
    flat_emb_cat_level1 = Flatten() (emb_cat_level1)
    flat_emb_cat_level2 = Flatten() (emb_cat_level2)
    flat_emb_cat_level3 = Flatten() (emb_cat_level3)
    main = concatenate([
        flat_emb_brand_name
        , flat_emb_item_condition
        , flat_emb_len_desc
        , flat_emb_cat_level1
        , flat_emb_cat_level2
        , flat_emb_cat_level3
        , shipping
        #, cnn_layer
        , textlayer
        # , att_emb_name
        # , att_emb_desc
    ])
  
    main = BatchNormalization()(main)
    fc = Dropout(0.05)(Dense(256,activation='relu') (main))
    nfc = Dropout(0.05)(Dense(128,activation='relu') (fc))
    output = Dense(1, activation="linear") (nfc)
    
    model = Model([name, 
                   item_desc,   
                   shipping,len_desc,
                   brand_name , item_condition, 
                   cat_level1,cat_level2,cat_level3], output)
 
    #opt=Adam(lr=0.002)
    model.compile(loss = 'mse', optimizer = 'adam')

    return model

def fastext_v3():
    # Inputs
    name = Input(shape=[name_max_len], name="name")
    item_desc = Input(shape=[desc_max_len], name="item_desc")
    
    shipping = Input(shape=[1], name='shipping')
    
    brand_name = Input(shape=[1], name="brand_name")   # category
    item_condition = Input(shape=[1], name="item_condition")  #category
    cat_level1 = Input(shape=[1], name="cat_level1")        #category
    cat_level2 = Input(shape=[1], name="cat_level2")        #category
    cat_level3 = Input(shape=[1], name="cat_level3")        #category
    len_desc = Input(shape=[1], name='len_desc')         #category
    
#     emb_layer = Embedding(name_word+desc_word,16,embeddings_initializer='uniform')
#     # Embeddings layers (adjust outputs to help model)
#     emb_name = emb_layer(name)
#     emb_item_desc = emb_layer(item_desc)
    
        
    # Embeddings layers (adjust outputs to help model)
    emb_name = Embedding(name_word,64,embeddings_initializer='uniform',mask_zero=False)(name)
    emb_item_desc = Embedding(desc_word,64,embeddings_initializer='uniform',mask_zero=False)(item_desc)
    
#     # Embeddings layers (adjust outputs to help model)
#     emb_name = Embedding(name_word,16,embeddings_initializer='uniform')(name)
#     emb_item_desc = Embedding(desc_word, 16,embeddings_initializer='uniform')(item_desc)
    
    emb_brand_name = Embedding(brand_word, 32,embeddings_initializer='uniform')(brand_name)
    emb_item_condition = Embedding(condition_word, 6,embeddings_initializer='uniform')(item_condition)

    emb_cat_level1 = Embedding(cate_level1_word, 6, embeddings_initializer='uniform')(cat_level1)
    emb_cat_level2 = Embedding(cate_level2_word, 16, embeddings_initializer='uniform')(cat_level2)
    emb_cat_level3 = Embedding(cate_level3_word, 32, embeddings_initializer='uniform')(cat_level3)
    emb_len_desc = Embedding(desc_word, 8, embeddings_initializer='uniform')(len_desc)
    
    
    warppers = []

    for emb in [emb_name, emb_item_desc]:
        avg = GlobalAveragePooling1D()(emb)
        #m = GlobalMaxPooling1D()(emb)
        #avg = GlobalAveragePooling1DMasked()(emb)
        #avg = AvgMasked(emb)
        warppers.append(avg)
        #warppers.append(m)
    textlayer = concatenate(warppers)    
        
    
    # cnn1d layers (GRUs are faster than LSTMs and speed is important here)
    cnn_warppers=[]
    for n_gram in [2, 3]:
        cnn_layer1 = Convolution1D(16,kernel_size=n_gram,strides=1,activation='relu',padding='same')(emb_name)
        a = GlobalAveragePooling1D()(cnn_layer1)
        #m = GlobalMaxPooling1D()(cnn_layer1)
        # att =  Attention(name_max_len)(cnn_layer1)
        cnn_warppers.append(a)
        #cnn_warppers.append(m)
    #    cnn_warppers.append(cnn_layer1)
        cnn_layer1 = Convolution1D(16,kernel_size=n_gram,strides=1,activation='relu',padding='same')(emb_item_desc)
        a = GlobalAveragePooling1D()(cnn_layer1)
        #m = GlobalMaxPooling1D()(cnn_layer1)
        # at =  Attention(emb_item_desc)(cnn_layer1)
        cnn_warppers.append(a)
        #cnn_warppers.append(m)
    #    cnn_warppers.append(cnn_layer1)
        


    
    
    cnn_layer=concatenate(cnn_warppers,axis=1)
    #rnn_layer1 = GRU(32,name='item_desc_gru') (emb_item_desc)
    #rnn_layer2 = GRU(32,name='name_gru') (emb_name)
    #cnn_layer=concatenate(cnn_warppers,axis=1)
    #cnn_layer=Attention(68)(cnn_layer)
    main = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_item_condition)
        , Flatten() (emb_cat_level1)
        , Flatten() (emb_cat_level2)
        , Flatten() (emb_cat_level3)
        , Flatten() (emb_len_desc)
        , shipping
        , cnn_layer
        , textlayer
        # , att_emb_name
        # , att_emb_desc
    ])
    # main = Dropout(0.2)(main)
    # (incressing the nodes or adding layers does not effect the time quite as much as the rnn layers)
    main = BatchNormalization()(main)
    fc = Dropout(0.05)(Dense(256,activation='relu') (main))
    fc = Dropout(0.05)(Dense(128,activation='relu') (fc))
    # the output layer.
    output = Dense(1, activation="linear") (fc)
    
    model = Model([name, 
                   item_desc,
                   shipping, 
                   len_desc,
                   brand_name , item_condition, 
                   cat_level1,cat_level2,cat_level3], output)
 
    #opt=Adam(lr=0.002)
    model.compile(loss = 'mse', optimizer = 'adam')

    return model 

print(len(tr_ind),len(te_ind))

y=np.log1p(df.price.values[:train_len])
mean = y.mean()
std = y.std()
y = (y - mean) / std


train_name=raw_name[tr_ind]
train_desc=raw_desc[tr_ind]
train_shipping=df.shipping.values[tr_ind]

#train_productt=df.productt.values[tr_ind]
#train_ppairs=df.ppairs.values[tr_ind]
#train_rom=df['rom'].values[tr_ind]
#train_version=df['version'].values[tr_ind]
train_len_desc=df['len_desc'].values[tr_ind]

train_brand=df.brand_name.values[tr_ind]
train_condition=df.item_condition_id.values[tr_ind]
train_cat_level1=df.category_level1.values[tr_ind]
train_cat_level2=df.category_level2.values[tr_ind]
train_cat_level3=df.category_level3.values[tr_ind]
train_y=y[tr_ind]

X_train=[train_name,train_desc,train_shipping,
         #train_productt,train_ppairs,train_rom,train_version,
         train_len_desc,
         train_brand,train_condition,train_cat_level1,train_cat_level2,train_cat_level3]


val_name=raw_name[te_ind]
val_desc=raw_desc[te_ind]
val_shipping=df.shipping.values[te_ind]

#val_productt=df.productt.values[te_ind]
#val_ppairs=df.ppairs.values[te_ind]
#val_rom=df['rom'].values[te_ind]
#val_version=df['version'].values[te_ind]
val_len_desc=df['len_desc'].values[te_ind]

val_brand=df.brand_name.values[te_ind]
val_condition=df.item_condition_id.values[te_ind]
val_cat_level1=df.category_level1.values[te_ind]
val_cat_level2=df.category_level2.values[te_ind]
val_cat_level3=df.category_level3.values[te_ind]

if USE_VAL:
    val_y=y[te_ind]

X_val=[val_name,val_desc,val_shipping,
       #val_productt,val_ppairs,val_rom,val_version,
       val_len_desc,
       val_brand,val_condition,val_cat_level1,val_cat_level2,val_cat_level3]
       

############# NN model 1 ###########
def scheduler(epoch):
    if epoch<=1:
        return 0.0055
    if epoch==2:
        return 0.0008
    else:
        return 0.0003 
        

lrs = LearningRateScheduler(scheduler)
# model = fastext()

if USE_VAL:
    model.fit(
        X_train, train_y, epochs=3, batch_size=2048,
        callbacks=[lrs],
        shuffle=True,
        validation_data=(X_val,val_y), verbose=2,
    )
else:
    model.fit(
        X_train, train_y, epochs=3, batch_size=2048,
        callbacks=[lrs],
        shuffle=True,
        verbose=2,
    )
    print('[{}] fasttext model train'.format(time.time() - start_time))

predNN = model.predict(X_val,batch_size=4096)
predNN =  predNN * std + mean

print(predNN.shape)
print('[{}] fasttext model predict'.format(time.time() - start_time))
if USE_VAL:
    print('[{}] fasttext val rmsle'.format(rmsle(np.expm1(val_y*std+mean), np.expm1(predNN).flatten())))
    
    
#############NN model 2 ##############

def scheduler(epoch):
    if epoch<=1:
        return 0.0055
    if epoch==2:
        return 0.0008
    else:
        return 0.0003 
        

lrs=LearningRateScheduler(scheduler)
# model=fastext_v3()

if USE_VAL:
    model.fit(
        X_train, train_y, epochs=3, batch_size=2048,
        callbacks=[lrs],
        shuffle=True,
        validation_data=(X_val,val_y), verbose=2,
    )
else:
    model.fit(
        X_train, train_y, epochs=3, batch_size=2048,
        callbacks=[lrs],
        shuffle=True,
        verbose=2,
    )
    print('[{}] fasttext2 model train'.format(time.time() - start_time))

predNN_v2 =  model.predict(X_val,batch_size=4096)
predNN_v2 =  predNN_v2 * std + mean

print(predNN.shape)
print('[{}] fasttext2 model predict'.format(time.time() - start_time))
if USE_VAL:
    print('[{}] fasttext2 val rmsle'.format(rmsle(np.expm1(val_y*std+mean), np.expm1(predNN_v2).flatten())))


if USE_VAL:    
    SUB=pd.DataFrame()
    SUB['test_id']=df['train_id'].values[te_ind].astype(int)
    SUB['price']=np.expm1(predNN.flatten()*0.275+predNN_v2.flatten()*0.225+predFM.flatten()*0.5)
    print('[{}] ensemble val rmsle'.format(rmsle(np.expm1(val_y*std+mean), SUB['price'].values)))
  
    #SUB.to_csv('val_ensembleCNN_RNN_FM_2_1.csv',header=True,index=False)


if not USE_VAL:    
    SUB=pd.DataFrame()
    SUB['test_id']=df['test_id'].values[train_len:].astype(int)
    SUB['price']=np.expm1(predNN.flatten()*0.275+predNN_v2.flatten()*0.225+predFM.flatten()*0.5)
    SUB.to_csv('ensemble_NN_NN_FM_FINAL_V0_3.csv',header=True,index=False)
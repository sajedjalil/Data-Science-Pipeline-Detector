import gc
import json
import numpy as np 
import pandas as pd
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import Levenshtein 

from multiprocessing import Pool
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from scipy import spatial


n_samples = 30000 # Number of samples to read from the train.json
n_workers = 4
batch_size = 32

html_tags = ['<P>', '</P>', '<Table>', '</Table>', '<Tr>', '</Tr>', '<Ul>', '<Ol>', '<Dl>', '</Ul>', '</Ol>', \
             '</Dl>', '<Li>', '<Dd>', '<Dt>', '</Li>', '</Dd>', '</Dt>']
r_buf = ['is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can', 'the', 'a', 'an', 'of', 'in', 'and', 'on', \
         'what', 'where', 'when', 'which']


def clean(x, stop_words=[]):
    for r in html_tags:
        x = x.replace(r, '')
    for r in stop_words:
        x = x.replace(r, '')
    x = x.lower()
    x = re.sub(' +', ' ', x)
    return x


feature_names = [
    'qa_cos_d', 'qd_cos_d', 'ad_cos_d', 
    'qa_euc_d', 'qd_euc_d', 'ad_euc_d',
    'qa_lev_d', 'qa_lev_r', 'qa_jar_s', 'qa_jaw_s',
    'qa_tfidf_score', 'qd_tfidf_score', 'ad_tfidf_score', 
    'document_tfidf_sum', 'question_tfidf_sum', 'answer_tfidf_sum'
]

def extract_features(document_tfidf, question_tfidf, answer_tfidf, document, question, answer):
    qa_cos_d = spatial.distance.cosine(question_tfidf, answer_tfidf)
    qd_cos_d = spatial.distance.cosine(question_tfidf, document_tfidf)
    ad_cos_d = spatial.distance.cosine(answer_tfidf, document_tfidf)

    qa_euc_d = np.linalg.norm(question_tfidf - answer_tfidf)
    qd_euc_d = np.linalg.norm(question_tfidf - document_tfidf)
    ad_euc_d = np.linalg.norm(answer_tfidf - document_tfidf)
    
    qa_lev_d = Levenshtein.distance(question, answer)
    qa_lev_r = Levenshtein.ratio(question, answer)
    qa_jar_s = Levenshtein.jaro(question, answer) 
    qa_jaw_s = Levenshtein.jaro_winkler(question, answer)
    
    qa_tfidf_score = np.sum(question_tfidf*answer_tfidf.T)
    qd_tfidf_score = np.sum(question_tfidf*document_tfidf.T)
    ad_tfidf_score = np.sum(answer_tfidf*document_tfidf.T)
    
    document_tfidf_sum = np.sum(document_tfidf)
    question_tfidf_sum = np.sum(question_tfidf)
    answer_tfidf_sum = np.sum(answer_tfidf)
    
    f = [
        qa_cos_d, qd_cos_d, ad_cos_d, 
        qa_euc_d, qd_euc_d, ad_euc_d,
        qa_lev_d, qa_lev_r, qa_jar_s, qa_jaw_s,
        qa_tfidf_score, qd_tfidf_score, ad_tfidf_score, 
        document_tfidf_sum, question_tfidf_sum, answer_tfidf_sum
    ]       
    return f


def process_sample(args):
    json_data, annotated = args
    
    ids = []
    candidates_str = []
    targets = []
    targets_str = []
    targets_str_short = []
    features = []
    rank_features = []

    document = json_data['document_text']
        
    # TFIDF for document
    stop_words = text.ENGLISH_STOP_WORDS.union(["book"])
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
    tfidf.fit([document])

    document_tfidf = tfidf.transform([document]).todense()

    # TFIDF for question
    question = json_data['question_text']
    question_tfidf = tfidf.transform([question]).todense()

    if annotated:
        # Collect annotations
        start_token_true = json_data['annotations'][0]['long_answer']['start_token']
        end_token_true = json_data['annotations'][0]['long_answer']['end_token']

        # Collect short annotations
        if json_data['annotations'][0]['yes_no_answer'] == 'NONE':
            if len(json_data['annotations'][0]['short_answers']) > 0:
                s_ans = str(json_data['annotations'][0]['short_answers'][0]['start_token']) + ':' + \
                    str(json_data['annotations'][0]['short_answers'][0]['end_token'])
            else:
                s_ans = ''
        else:
            s_ans = json_data['annotations'][0]['yes_no_answer']

    cos_d_buf = []
    euc_d_buf = []
    lev_d_buf = []

    doc_tokenized = json_data['document_text'].split(' ')
    candidates = json_data['long_answer_candidates']
    candidates = [c for c in candidates if c['top_level'] == True]

    if not annotated or start_token_true != -1:
        for c in candidates:
            ids.append(str(json_data['example_id']))

            # TFIDF for candidate answer
            start_token = c['start_token']
            end_token = c['end_token']
            answer = ' '.join(doc_tokenized[start_token:end_token])
            answer_tfidf = tfidf.transform([answer]).todense()

            # Extract some features
            f = extract_features(document_tfidf, question_tfidf, answer_tfidf, 
                                 clean(document), clean(question, stop_words=r_buf), clean(answer))

            cos_d_buf.append(f[0])
            euc_d_buf.append(f[3])
            lev_d_buf.append(f[6])

            features.append(f)

            if annotated:
                targets_str.append(str(start_token_true) + ':' + str(end_token_true))
                targets_str_short.append(s_ans)
                # Get target
                if start_token == start_token_true and end_token == end_token_true:
                    target = 1
                else:
                    target = 0
                targets.append(target)
                
            candidates_str.append(str(start_token) + ':' + str(end_token))
            
        features = np.array(features)
        
        rank_cos_d = np.argsort(np.argsort(cos_d_buf))
        rank_euc_d = np.argsort(np.argsort(euc_d_buf))
        rank_lev_d = np.argsort(np.argsort(lev_d_buf))
        rank_cos_d_ismin = (cos_d_buf == np.nanmin(cos_d_buf)).astype(int)
        rank_euc_d_ismin = (euc_d_buf == np.nanmin(euc_d_buf)).astype(int)
        rank_lev_d_ismin = (lev_d_buf == np.nanmin(lev_d_buf)).astype(int)
        rank_features = np.array([rank_cos_d, rank_euc_d, rank_lev_d, \
                                       rank_cos_d_ismin, rank_euc_d_ismin, rank_lev_d_ismin]).T

    return {
        'ids': ids,
        'candidates_str': candidates_str,
        'targets': targets,
        'targets_str': targets_str,
        'targets_str_short': targets_str_short,
        'features': features,
        'rank_features': rank_features
    }


def get_train():
    ids = []
    candidates_str = []
    targets = []
    targets_str = []
    targets_str_short = []
    features = []
    rank_features = []

    # Read data from train.json and prepare features
    with open('/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl', 'r') as json_file:
        cnt = 0

        batch = []
        batch_cnt = 0
        for line in tqdm(json_file):
            json_data = json.loads(line) 
            batch.append(json_data)

            batch_cnt += 1
            if batch_cnt == batch_size:
                with Pool(processes=n_workers) as pool:  
                    results = pool.map_async(process_sample, zip(batch, [True]*len(batch)))
                    results = results.get()

                for r in results:
                    if len(r['ids']) > 0:
                        ids += r['ids']
                        candidates_str += r['candidates_str']
                        targets += r['targets']
                        targets_str += r['targets_str']
                        targets_str_short += r['targets_str_short']
                        features.append(r['features'])
                        rank_features.append(r['rank_features'])

                batch = []
                batch_cnt = 0

            cnt += 1
            if cnt >= n_samples:
                break
                
        if len(batch) > 0:
            with Pool(processes=n_workers) as pool:  
                results = pool.map_async(process_sample, zip(batch, [True]*len(batch)))
                results = results.get()

            for r in results:
                if len(r['ids']) > 0:
                    ids += r['ids']
                    candidates_str += r['candidates_str']
                    targets += r['targets']
                    targets_str += r['targets_str']
                    targets_str_short += r['targets_str_short']
                    features.append(r['features'])
                    rank_features.append(r['rank_features'])

            batch = []
            batch_cnt = 0

    train = pd.DataFrame()
    train['example_id'] = ids
    train['target'] = targets
    train['CorrectString'] = targets_str
    train['CorrectString_short'] = targets_str_short
    train['CandidateString'] = candidates_str

    features = np.concatenate(features, axis=0)
    features_df = pd.DataFrame(features)
    features_df.columns = feature_names
    train = pd.concat([train, features_df], axis=1)

    rank_features = np.concatenate(rank_features, axis=0)
    rank_features_df = pd.DataFrame(rank_features)
    rank_features_df.columns = [f'rank_feature_{i}' for i in range(rank_features.shape[1])]
    train = pd.concat([train, rank_features_df], axis=1)

    del features, features_df, \
        rank_features, rank_features_df
    gc.collect()

    return train
    

def get_test():
    ids = []
    question_tfidfs = []
    answer_tfidfs = []
    candidates_str = []
    features = []
    rank_features = []

    with open('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl', 'r') as json_file:   
        batch = []
        batch_cnt = 0
        for line in tqdm(json_file):
            json_data = json.loads(line) 
            batch.append(json_data)

            batch_cnt += 1
            if batch_cnt == batch_size:
                with Pool(processes=n_workers) as pool:  
                    results = pool.map_async(process_sample,  zip(batch, [False]*len(batch)))
                    results = results.get()
                    
                for r in results:
                    if len(r['ids']) > 0:
                        ids += r['ids']
                        candidates_str += r['candidates_str']
                        features.append(r['features'])
                        rank_features.append(r['rank_features'])

                batch = []
                batch_cnt = 0

        if len(batch) > 0:
            with Pool(processes=n_workers) as pool:  
                results = pool.map_async(process_sample, zip(batch, [False]*len(batch)))
                results = results.get()
                
            for r in results:
                if len(r['ids']) > 0:
                    ids += r['ids']
                    candidates_str += r['candidates_str']
                    features.append(r['features'])
                    rank_features.append(r['rank_features'])


    test = pd.DataFrame()
    test['example_id'] = ids
    test['CandidateString'] = candidates_str

    features = np.concatenate(features, axis=0)
    features_df = pd.DataFrame(features)
    features_df.columns = feature_names
    test = pd.concat([test, features_df], axis=1)

    rank_features = np.concatenate(rank_features, axis=0)
    rank_features_df = pd.DataFrame(rank_features)
    rank_features_df.columns = [f'rank_feature_{i}' for i in range(rank_features.shape[1])]
    test = pd.concat([test, rank_features_df], axis=1)

    del features, features_df, rank_features, rank_features_df
    gc.collect()
    
    return test


if __name__ == '__main__':
#     train = get_train()
#     train.to_csv('train_data.csv', index=False)
#     print(f'train.shape: {train.shape}')
#     print(f'Mean target: {train.target.mean()}')
    
#     test = get_test()
#     test.to_csv('test_data.csv', index=False)
#     print(f'test.shape: {test.shape}')
    pass
    
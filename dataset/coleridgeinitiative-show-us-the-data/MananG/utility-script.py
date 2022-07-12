# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-25T21:45:39.306096Z","iopub.execute_input":"2021-05-25T21:45:39.306501Z","iopub.status.idle":"2021-05-25T21:45:39.311366Z","shell.execute_reply.started":"2021-05-25T21:45:39.306415Z","shell.execute_reply":"2021-05-25T21:45:39.310324Z"}}
import os
import re
import json
import pandas as pd
import numpy as np

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-25T22:50:06.384284Z","iopub.execute_input":"2021-05-25T22:50:06.384768Z","iopub.status.idle":"2021-05-25T22:50:06.467363Z","shell.execute_reply.started":"2021-05-25T22:50:06.384727Z","shell.execute_reply":"2021-05-25T22:50:06.466185Z"}}
import random
from ast import literal_eval
class Data():
    def __init__(self):
        self.train_path='/kaggle/input/coleridgeinitiative-show-us-the-data/train/'
        self.test_path='/kaggle/input/coleridgeinitiative-show-us-the-data/test/'
        self.label_col = 'cleaned_label' #dataset_label cleaned_label
        self.label_df = self.get_label_df()
        self.grouped_label_df = self.get_grouped_label_df()
        self.MAX_SENTENCES_PER_ARTICLE = 2000 #check if there are articles more than this
        self.data_file = "train.csv"
        self.lines_col = "lines"
        self.labels_col = "labels"
        self.dataset_found_col = "dataset_found"
        self.article_id = "article_id"
        self.section_title = "section_title"
        self.dataset_name = "dataset_name"
        self.max_line_length = 64
        
        self.unique_tags = set(tag for tag in [0.0, 1.0])
        self.tag2id = {tag: id for id, tag in enumerate(self.unique_tags)}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}
        
    def get_label_df(self):
        df = pd.read_csv('../input/coleridgeinitiative-show-us-the-data/train.csv', index_col='Id')
        df[self.label_col] = df[self.label_col].apply(self.clean_text)
        return df
    
    def get_grouped_label_df(self):
        tmp = self.label_df[[self.label_col]].groupby('Id')[self.label_col].apply(list).reset_index()
        tmp = tmp.set_index('Id')
        return tmp
        
    def clean_text(self, txt):
        return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()

    def read_file(self, file_path):
        with open(file_path) as f:
            file_data= json.load(f)
        return file_data
    
    def merge_sections(self, article_json):
        article_string = ""
        for section in article_json:
            article_string += section['section_title'] + " " + section['text'] + " "
        return article_string
    
    def preprocess_to_avoid_split_mistakes(self, text):
        if type(text) != str:
            return ''
        text = text.replace('U.S.A.', 'USA')
        text = text.replace('U.S.', 'US')
        text = re.sub(' +', ' ', text)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        return text
    
    def clean_sections(self, article_json):
        try:
            for section in article_json:
                section['text'] = self.preprocess_to_avoid_split_mistakes(section['text'])
        except:
            print('exception n in clean_sections()')
            return {}
        return article_json
    
#     def get_clean_article_from_filepath(self, file_path):
#         article_json = self.read_file(file_path)
#         article_string = self.merge_sections(article_json)
#         article_string = self.preprocess_to_avoid_split_mistakes(article_string)
#         return article_string

    def get_article_json_from_filepath(self, file_path):
        article_json = self.read_file(file_path)
        article_json = self.clean_sections(article_json)
#         article_string = self.merge_sections(article_json)
#         article_string = self.clean_text(article_string)
        return article_json

#     def read_data(self, read_train, max_files_to_read= None, yield_data=False):
#         if read_train:
#             data_path = self.train_path
#         else:
#             data_path = self.test_path
#         data = {}
#         read_count = 0
#         for file_name in os.listdir(data_path):
#             file_path = os.path.join(data_path, file_name)
#             article_string = self.get_clean_article_from_filepath(file_path)
            
#             read_count += 1
#             if yield_data:
#                 yield file_name[:-5], article_string
#             else:
#                 data[file_name[:-5]] = article_string #[:10000]
#             if max_files_to_read is not None:
#                 if read_count == max_files_to_read:
#                     break
#         return data
    
    def yield_data(self, read_train, max_files_to_read= None):
        if read_train:
            data_path = self.train_path
        else:
            data_path = self.test_path
        read_count = 0
        for file_name in os.listdir(data_path):
            file_path = os.path.join(data_path, file_name)
            article_json = self.get_article_json_from_filepath(file_path)
#             print(article_json)
            read_count += 1
            yield file_name[:-5], article_json
            
            if max_files_to_read is not None and read_count == max_files_to_read:
                break

    def read_all_files(self, read_train, max_files_to_read= None):
        data = {}
        for file_name, article_json  in self.yield_data(read_train, max_files_to_read=max_files_to_read):
            data[file_name] = article_json
        return data    
        
    #Taken from SO: https://stackoverflow.com/a/2251638/2350203
    def find_sublist(self, sub, bigger):  
        if not bigger:
            return -1
        if not sub:
            return 0
        first, rest = sub[0], sub[1:]
        pos = 0
        try:
            while True:
                pos = bigger.index(first, pos) + 1
                if not rest or bigger[pos:pos+len(rest)] == rest:
                    return pos - 1
        except ValueError:
            return -1

    def is_sublist(self, sub, bigger): 
        return find_sublist(sub, bigger) >= 0

    # data = list('abcdfghdesdkflksdkeeddefaksda')
    # print(find_sublist(list('def'), data))
    
    
# return_offsets_mapping=True
    def encode_tags(self, tags, encodings):
        labels = [[self.tag2id[tag] for tag in doc] for doc in tags]
#         print(labels)
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * 1
            arr_offset = np.array(doc_offset)
#             print(len(arr_offset))
#             print(len(doc_labels))
#             print(len(doc_enc_labels))
#             print((arr_offset[:,0] == 0) & (arr_offset[:,1] != 0))
            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels)

        return encoded_labels
    
    def get_tokenized(self, article_dict): 
        orig_texts = []
        orig_keys = []
        for i in article_dict.items():
            orig_texts.append(i[1])
            orig_keys.append(i[0])

        tokenized = tokenizer(orig_texts, max_length=512, padding=True, return_tensors="tf")
        return tokenized, orig_texts, orig_keys

    def assign_label_to_string(self, dataset_list, article_str, section_title):
        article_str = self.clean_text(section_title + ' ' + article_str)
        article_splits = article_str.split()
        labels = np.zeros(len(article_splits))
        dataset_found = False
        dataset_name = ''
        for dataset in dataset_list: 
            dataset_splits = dataset.split()#TODO do this at higher level to speed it up!
#             print(dataset_splits)
            mention_index = self.find_sublist(dataset_splits, article_splits)
            if mention_index > -1:
                labels[mention_index: mention_index+len(dataset_splits)] = 1 # when you find index, set it to 1
                dataset_found = True
                dataset_name = dataset
                
        return article_splits, labels, dataset_found, dataset_name
    
    def assign_label_to_article_string(self, dataset_list, article_str, section_title, keep_all = True):
        article_splits = article_str.split('.')
        total_dataset_found = 0
        dataset_found_lst = []
        dataset_lst = []
        section_title_lst = []
        line_splits_lst = []
        labels_lst = []
        for line in article_splits: 
            random.shuffle(dataset_list)
            line_splits, line_labels, dataset_found, dataset_name = self.assign_label_to_string(dataset_list, line, section_title)
#             print(line_splits)
            append = True
            if dataset_found:
                total_dataset_found+= 1
#                 print("dataset_found", dataset_name, "\narticle_splits", line)
            elif not keep_all:
                #drop 95% of th values here 
                rand = random.randint(1,101)
                if rand < 50:
                    append = False
            if append:
                line_splits_lst.append(line_splits)
                labels_lst.append(list(line_labels))
                dataset_found_lst.append(dataset_found)
                dataset_lst.append(dataset_name)
                section_title_lst.append(section_title)
        
#         if total_dataset_found > 0:
#             print("total_dataset_found", total_dataset_found, "total lines", len(article_splits))
        
        return line_splits_lst, labels_lst, dataset_found_lst, dataset_lst, section_title_lst
    
#     def assign_label_to_data(self, data, keep_all=True):
#         label_col = 'cleaned_label' #'dataset_label'
#         article_labels={}
#         df = pd.DataFrame(data={self.lines_col:[], self.labels_col:[], self.dataset_found_col:[]})
#         df.to_csv(self.data_file, header=True, index= False)
#         for article_id, article_str in data.items():
#             dataset_mentions = self.grouped_label_df[self.grouped_label_df.index == article_id][self.label_col][0]
#             line_splits_lst, labels_lst, dataset_found_lst = self.assign_label_to_article_string(dataset_mentions, article_str, keep_all=keep_all)
# #             article_labels[article_id] = (line_splits_lst, labels_lst, dataset_found_lst)

#             df_train_data = pd.DataFrame(data= {self.lines_col:line_splits_lst, self.labels_col:labels_lst, self.dataset_found_col:dataset_found_lst}) #TODO may be just directly write to csv instead of storing everything
#             df_train_data.to_csv(self.data_file, mode='a', header=False, index= False)
#         return article_labels

    def assign_label_to_article_json(self, article_id, article_json, keep_all=True):
        if article_json is None: 
            return None
        
        article_labels={}
        if not os.path.isfile(self.data_file):
            df = pd.DataFrame(data={self.lines_col:[], self.labels_col:[], self.dataset_found_col:[],  self.dataset_name:[], self.section_title:[]})
            df.to_csv(self.data_file, header=True, index= False)
        
        dataset_mentions = self.grouped_label_df[self.grouped_label_df.index == article_id][self.label_col][0]
#         print('expected', dataset_mentions)
        for section_dict in article_json:
            section_title, article_str = section_dict['section_title'], section_dict['text']
            line_splits_lst, labels_lst, dataset_found_lst, dataset_name_lst, section_title_lst = self.assign_label_to_article_string(dataset_mentions, article_str, section_title, keep_all=keep_all)
#             article_labels[article_id] = (line_splits_lst, labels_lst, dataset_found_lst)

            df_train_data = pd.DataFrame(data= {self.lines_col:line_splits_lst, self.labels_col:labels_lst, self.dataset_found_col:dataset_found_lst, self.dataset_name:dataset_name_lst, self.section_title:section_title_lst}) #TODO may be just directly write to csv instead of storing everything
            df_train_data.to_csv(self.data_file, mode='a', header=False, index= False)
        return article_labels

    def get_data(self, data, labelled_df=None):
        tokenized_data, orig_texts, orig_keys = get_tokenized(data)
        labels = None
        if labelled_df is not None:
            article_labels = self.assign_label_to_data(tokenized_data, orig_keys, labelled_df)
            labels = [list(i) for i in article_labels.values()] # TODO improve this after training works
        return tokenized_data, orig_keys, labels
    
    def clean_df(self, df):
        df['lines'] = df['lines'].apply(lambda x: x[:self.max_line_length])
        if 'labels' in df.columns:
            df['labels'] = df['labels'].apply(lambda x: x[:self.max_line_length])
        
        df = df.loc[df['lines'].apply(lambda x: len(x) > 0), :]
        df.reset_index(inplace=True, drop=True)
        
        return df
    
    def read_train_from_file(self, nrows=None):
        if nrows is not None:
            df = pd.read_csv("/kaggle/input/articles-to-labelled-csv-for-easy-handling/train.csv", nrows=nrows)
        else:
            df = pd.read_csv("/kaggle/input/articles-to-labelled-csv-for-easy-handling/train.csv")
        df[self.lines_col]= df[self.lines_col].apply(literal_eval)
        df[self.labels_col]= df[self.labels_col].apply(literal_eval)
        df[self.lines_col] = df[self.lines_col].apply(lambda x: x[:self.max_line_length])
        df[self.labels_col] = df[self.labels_col].apply(lambda x: x[:self.max_line_length])
        df = self.clean_df(df)
        
        return df
    
    def get_dataset_names_from_outputs(self, tokens, output, tokenizer, filenames=None):
        ans = np.argmax(output['logits'], axis = 2)
        datasets = []
    #     print(output)
        for i in range(len(ans)):
            first_index = -1
            for j in range(1, len(ans[i])):
                if tokens[i][j] == 102: #[SEP] token
                    if first_index != -1:
                        dataset_name = tokenizer.decode(tokens[i][first_index: j])
                        if not dataset_name.startswith("##"):
                            #print(dataset_name)
                            datasets.append(dataset_name)
                    break
                if ans[i][j] == 1:
                    if first_index == -1:
                        first_index = j
                elif first_index != -1:
                    dataset_name = tokenizer.decode(tokens[i][first_index: j])
                    if not dataset_name.startswith("##"):
                        #print(dataset_name)
                        datasets.append(dataset_name)
    #               submission[filenames[i]].add(dataset_name)
                    #if with hash, either add current word or remove it completetly -> do some experiments on that
                    first_index = -1    
        return datasets
 
    def read_one_file(self, file_path, train=True):
        if train:
            data_path= "/kaggle/input/coleridgeinitiative-show-us-the-data/train"
        else:
            data_path="/kaggle/input/coleridgeinitiative-show-us-the-data/test"
        file_path = os.path.join(data_path, file_path + ".json")
        article_id, article_json = self.get_article_json_from_filepath(file_path)
        return article_json

    def write_submission_file(self, ids, predictions):
        labels = []
        for preds in predictions:
            label = []
            for dataset_title in preds:
                label.append(self.clean_text(dataset_title))
            labels.append('|'.join(label))

        submission = pd.DataFrame()
        submission['Id'] = ids
        submission['PredictionString'] = labels
        print('Writing submission file')
        submission.to_csv('submission.csv', index=False)
        return submission
        
    
 # %% [code]
from transformers import TFBertForTokenClassification
from transformers import BertTokenizer
from transformers import PreTrainedTokenizerFast
import tensorflow as tf

class MyBERT():
    def __init__(self, train = True):
        self.train = train
        self.MAX_LENGTH_PER_SENTENCE=200
        self.BATCH_SIZE=128
        if self.train:
            self.model = TFBertForTokenClassification.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.fast_tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-uncased', model_max_length=512)
            self.fast_tokenizer.add_special_tokens({'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})
        else:
            self.model = TFBertForTokenClassification.from_pretrained('/kaggle/input/notebookb555ddcd5f/model_file')
            self.tokenizer = BertTokenizer.from_pretrained('/kaggle/input/notebookb555ddcd5f/tokenizer_file')
            #TODO fix fast tokenizer as it does not work in test setting
#             self.fast_tokenizer = PreTrainedTokenizerFast.from_pretrained('/kaggle/input/notebookb555ddcd5f/tokenizer_file/', model_max_length=512)
#             self.fast_tokenizer.add_special_tokens({'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})
        
    def compile(self):
        if self.train == False:
            print('model not set to train!')
            return 
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        # model.layers[-1].activation = tf.keras.activations.softmax
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric]) #, 
        print(self.model.summary())
    
    def train_model(self, train_data, labels, epochs=1):
        history = self.model.fit(x=(train_data['input_ids'],train_data['token_type_ids'],train_data['attention_mask']), y=labels, batch_size=10, epochs = epochs)
        return history
    
    def get_dataset_names_from_outputs(self, tokens, output, filenames=None):
        ans = np.argmax(output['logits'], axis = 2)
        datasets = []
        for i in range(len(ans)):
            first_index = -1
            for j in range(1, len(ans[i])):
                if tokens[i][j] == 102: #[SEP] token
                    if first_index != -1:
                        dataset_name = self.decode(tokens[i][first_index: j])
                        if not dataset_name.startswith("##"):
                            print(dataset_name)
                            datasets.append(dataset_name)
                    break
                if ans[i][j] == 1:
                    if first_index == -1:
                        first_index = j
                elif first_index != -1:
                    dataset_name = self.decode(tokens[i][first_index: j])
                    if not dataset_name.startswith("##"):
                        print(dataset_name)
                        datasets.append(dataset_name)
                    #if with hash, either add current word or remove it completetly -> do some experiments on that
                    first_index = -1    
        return datasets
    
    def predict_on_df(self, train_df):
        train_df.dropna(inplace=True)
        i = 0
        all_datasets = []
        if train_df.shape[0] == 0:
            return all_datasets
            
        for chunk in np.array_split(train_df, int(train_df.shape[0]/self.BATCH_SIZE)):
            encodings = self.tokenize(chunk['lines'].tolist())
            op = self.model(encodings)
            datasets = self.get_dataset_names_from_outputs(encodings['input_ids'], op)
            all_datasets.extend(datasets)
        return all_datasets
    
    def tokenize_and_train_on_dict(self, data, labelled_df):
        tokenized_data, filename_data, labels = get_tokenized_labelled_data(data, self.tokenizer, labelled_df)
        return self.train(tokenized_data, labels)
    
    def tokenize(self, data):
        if self.train:
            encodings = self.fast_tokenizer(data, max_length=self.MAX_LENGTH_PER_SENTENCE, padding=True, truncation=True, return_offsets_mapping=True,is_split_into_words=True, return_tensors='np') #return_offsets_mapping=True # ,
        else:
            encodings = self.tokenizer(data, max_length=self.MAX_LENGTH_PER_SENTENCE, padding=True, truncation=True, is_split_into_words=True, return_tensors='np') #return_offsets_mapping=True # ,
        return encodings
    
    def decode(self, data):
        if self.train:
            encodings = self.fast_tokenizer.decode(data)
        else:
            encodings = self.tokenizer.decode(data)
        return encodings

    def predict(self, data):
        pass
    
    def tokenize_and_predict_on_dict(self, data, batch_size=32):
        submission_dict = {key: '' for key in data.keys()}

        for item in chunks(test_data, batch_size):
            try:
                tokenized_data, filename_data, _ = get_tokenized_data(item, self.tokenizer)
                output = self.model(tokenized_test)
                pred_dict = get_dataset_names_from_outputs(tokenized_data['input_ids'], output, mymodel.tokenizer, filename_data)
                for i in pred_dict.keys():
                    submission_dict[i] = pred_dict[i]
            except:
                pass
        return submission_dict
    
    def save(self, data):
        pass

# %% [code] {"execution":{"iopub.status.busy":"2021-05-25T22:51:49.123204Z","iopub.execute_input":"2021-05-25T22:51:49.123554Z","iopub.status.idle":"2021-05-25T22:51:51.624415Z","shell.execute_reply.started":"2021-05-25T22:51:49.123524Z","shell.execute_reply":"2021-05-25T22:51:51.623497Z"}}
# from tqdm import tqdm
# data = Data()
# # train_data = data.read_all_files(read_train=True, max_files_to_read= 3) #, max_files_to_read= 10 # TODO: more trial and error : check if merging articles have any effect
# for article_id, article_json  in tqdm(data.yield_data(read_train=True), total=15000):
#     article_labels = data.assign_label_to_article_json(article_id, article_json, keep_all=False)

# %% [code] {"execution":{"iopub.status.busy":"2021-05-25T22:54:07.102874Z","iopub.execute_input":"2021-05-25T22:54:07.10324Z","iopub.status.idle":"2021-05-25T22:54:07.853151Z","shell.execute_reply.started":"2021-05-25T22:54:07.103208Z","shell.execute_reply":"2021-05-25T22:54:07.852138Z"}}
# !ls

# %% [code]
# TODO
#article_splits = article_str.split('.') #TODO: check if the line is too small, it could be merged with the previous line or the next one (e.g. U.S.A)
# what if we do not remove any special chars from the data and train it like that? and clean_text after the predictions? it must improve like 10%
# 
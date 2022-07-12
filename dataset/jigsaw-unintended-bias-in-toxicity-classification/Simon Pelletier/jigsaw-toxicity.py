import os
import sys
import csv
csv.field_size_limit(sys.maxsize)
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import torch.nn as nn
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

bs = 128



def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true.round(), y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    flen = file_len(data_path)
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        with tqdm(total=flen) as pbar:
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx.lower()
                    text += " "
                sent_list = sent_tokenize(text)
                sent_length_list.append(len(sent_list))

                for sent in sent_list:
                    word_list = word_tokenize(sent)
                    word_length_list.append(len(word_list))
                pbar.update(1)
        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]

class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50):
        super(WordAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        output = self.lookup(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, 0)
        output = element_wise_mul(f_output, output.permute(1, 0))

        return output, h_output


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        f_output, h_output = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, 0)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output, h_output


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input):

        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)

        return output


class MyTestset(Dataset):

    def __init__(self, df, dict_path, max_length_sentences=30, max_length_word=35, test=False):
        super(MyTestset, self).__init__()

        print("Loading dataset...")
        texts = np.array(df.iloc[:, 1])
        self.texts = texts
        self.num_classes = 2
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE, usecols=[0]).values
        self.dict = [word[0] for word in self.dict]

        print("Loading dictionary")
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE, usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = int(max_length_sentences)
        self.max_length_word = int(max_length_word)
        self.num_classes = 2


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        document_encode = [
        [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
        in
            sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64)



def make_csv():
    saved_path = "../input/jigsawtoxicity/"
    batch_size = 128
    maxiter = 100
    word_hidden_size = 200
    sent_hidden_size = 200
    max_word_length, max_sent_length = 27, 27

    word2vec_path = "../input/glovedata/glove.6B.200d.txt"

    model = HierAttNet(word_hidden_size, sent_hidden_size, batch_size, 2,
                       word2vec_path, max_sent_length, max_word_length)
    model.cuda()
    model.load_state_dict(torch.load(saved_path + "toxicity_model.state_dict"))
    test_set = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

    ids = torch.Tensor(test_set.iloc[:, 0])

    test_params = {"batch_size": bs,
                   "shuffle": False,
                   "drop_last": False}


    test_set = MyTestset(test_set, word2vec_path, max_sent_length, max_word_length, test=True)
    test_generator = DataLoader(test_set, **test_params)

    te_pred_ls = torch.Tensor([]).cuda()
    with tqdm(total=len(test_generator), unit_scale=True, unit_divisor=bs) as pbar:
        for iter, te_feature in enumerate(test_generator):
            if torch.cuda.is_available():
                te_feature = te_feature.cuda()
            with torch.no_grad():
                model._init_hidden_state(len(te_feature))
                te_predictions = model(te_feature)
                prediction = F.softmax(te_predictions, 0)
                max_prob, max_prob_index = torch.max(prediction, dim=-1)
                te_pred_ls = torch.cat((te_pred_ls, max_prob_index.float().clone()))

            pbar.update(1)

    matrix = torch.cat((ids[:len(te_pred_ls)].reshape(-1, 1).cuda(), te_pred_ls.reshape(-1, 1)), 1)
    dframe = pd.DataFrame(matrix.cpu().numpy(), columns=["id", "prediction"])
    dframe.id = dframe.id.astype(int)
    dframe.to_csv("submission.csv", header=True, index=False)
    print("DONE.")
if __name__ == "__main__":
    make_csv()

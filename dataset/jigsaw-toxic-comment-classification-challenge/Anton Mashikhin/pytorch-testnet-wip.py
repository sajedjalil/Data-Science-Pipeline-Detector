# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import sys, os

sys.path.insert(0,'..')
from config import USE_CUDA, SSD_FOLDER

TEST_PRINT = False

class TestNet(nn.Module):

    def __init__(self, hidden_dim = 50, use_gpu = USE_CUDA):
        super().__init__()
        label_size = 6
        self.num_layers = 1
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu

        # embeds
        self.read_embed()
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.load_embed()

        self.lstm = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers = self.num_layers, bidirectional=True)
        self.linear = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.linear2 = nn.Linear(2 * self.hidden_dim, label_size)
        self.drop = nn.Dropout(p = 0.3)
        self.activation = nn.ReLU()

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.transpose(1,0).contiguous()
        if TEST_PRINT : print('embeds', embeds.shape) #embeds torch.Size([600, 4, 300])
        lstm_out, self.hidden = self.lstm(embeds)
        if TEST_PRINT: print('lstm_out', lstm_out.shape, "hidden", self.hidden.shape) #lstm_out torch.Size([600, 4, 100]) hidden torch.Size([2, 4, 50])
        x = lstm_out
        if TEST_PRINT: print("before max", x.shape) # before max torch.Size([600, 4, 100])
        x, _ = torch.max(x, dim=0)
        if TEST_PRINT: print("after max", x.shape) #after max torch.Size([4, 100])
        x  = self.linear(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        return y

    def read_embed(self):
        # self.np_embed = np.random.randint(962233, size=(962233,300))
        self.np_embed = np.load(SSD_FOLDER + "pretrained_embeds.npy") #size=(962233,300)
        self.vocab_size , self.embedding_dim = self.np_embed.shape
        print(self.vocab_size, self.embedding_dim)

    def load_embed(self):
        self.word_embeddings.weight.data.copy_(torch.from_numpy(self.np_embed))
        # self.word_embeddings.weight.requires_grad=False

if __name__ == "__main__":
    TEST_PRINT = True
    USE_CUDA = True
    batch = 4
    net = TestNet(use_gpu = True )
    net.cuda()

    x = Variable(torch.LongTensor(torch.from_numpy(np.random.randint(600, size=(batch,600))) ))
    if USE_CUDA:
        x = x.cuda()
    y = net(x)
    print('output', y.size())

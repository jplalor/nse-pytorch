# NSE Implementation in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 


class NSE(nn.Module):
    """docstring"""

    def __init__(self, config): #n_outputs, dim_size, gpu, fix_embeds, p):
        super(NSE, self).__init__()
        self.embed = nn.Embedding(config.num_embed, config.dim_size)
       #self.dropout = nn.Dropout(config.p)
        self.h_x = nn.Linear(2 * config.dim_size, 2 * config.dim_size)
        self.h_lstm = nn.LSTM(config.dim_size, config.dim_size, dropout=config.p)
        self.c_lstm = nn.LSTM(2 * config.dim_size, config.dim_size)
        self.h_x1 = nn.Linear(3 * config.dim_size, 3 * config.dim_size)
        self.h_lstm1 = nn.LSTM(config.dim_size, config.dim_size)
        self.c_lstm1 = nn.LSTM(3 * config.dim_size, 3 * config.dim_size)
        self.h_l1 = nn.Linear(4 * config.dim_size, 1024)
        self.l_y = nn.Linear(1024, 3)

        self.__dim_size = config.dim_size
        self.__gpu = config.gpu
        self.__fix_embeds = config.fix_embeds
        if self.__gpu > 0:
            self.cuda()

        self.init_weights()

    def init_weights(self):
        params = list(self.parameters())
        for param in params:
            param.weight.data.uniform(-0.1, 0.1)

    def forward(self, batch, train):
        n_units = self.n_units
        premise = self.embed(batch.premise)
        hypothesis = self.embed(batch.hypothesis)
        if self.__fix_embeds:
            premise = Variable(premise.data)
            hypothesis = Variable(hypothesis.data)

        list_h1 = torch.transpose(premise, 1, 2)
        # push through premise
        h2 = self.h_lstm(premise)
        h1_w = torch.bmm(list_h1, h2)
        h1_a = nn.Softmax(h1_w)
        h2_r = torch.bmm(h1_a, list_h1)
        lr = self.h_x(torch.cat((h2, h2_r), 1))
        h3 = self.c_lstm(lr)
        list_h1 = (1 - h1_a)




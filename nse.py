# NSE Implementation in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from torch.autograd import Variable


class NSE(nn.Module):
    """docstring"""

    def __init__(self, config): #n_outputs, dim_size, gpu, fix_embeds, p):
        super(NSE, self).__init__()
        self.embed = nn.Embedding(config.n_embed, config.dim_size)
       #self.dropout = nn.Dropout(config.p)
        self.h_x = nn.Linear(2 * config.dim_size, 2 * config.dim_size)
        self.h_lstm = nn.LSTM(config.dim_size, config.dim_size, dropout=config.p)
        self.softmax = nn.Softmax()
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

        self.config = config

        #self.init_weights()

    def init_weights(self):
        params = list(self.parameters())
        for param in params:
            nn.init.uniform(param, -0.1, 0.1)

    def forward(self, batch):
        print('forward pass')
        print('premise size: {}'.format(batch.premise.size()))
        premise = self.embed(batch.premise)
        hypothesis = self.embed(batch.hypothesis)
        if self.__fix_embeds:
            premise = Variable(premise.data)
            hypothesis = Variable(hypothesis.data)

        x_len, batch_size, dim_size = premise.size()
        x_in = premise
        list_h1 = torch.transpose(x_in, 0, 2)
        print('x_in size:{}\nlist_h1 size: {}'.format(x_in.size(),
                                                       list_h1.size()))

        # first, premise
        h2, states_h2 = self.h_lstm(x_in)
        print('h2 size: {}'.format(h2.size()))

        # reshape for bmm
        h1_w = torch.bmm(h2.view(batch_size, x_len, dim_size),
                         list_h1.view(batch_size, dim_size, x_len))
        print('h1_w size: {}'.format(h1_w.size()))
        h1_a = self.softmax(h1_w) #.view(batch_size, -1))
        h1_a_m = h1_a.view(batch_size, dim_size)
        print('h1_a len: {}, {}'.format(h1_a.size(), h1_a_m.size()))
        h2_r = torch.bmm(h1_a_m, list_h1).view(batch_size, dim_size)
        print('h2_r size: {}'.format(h2_r.size()))
        concat = torch.cat((h2, h2_r), 2)
        print('concat size: {}'.format(concat.size()))
        lr = self.h_x(concat.view(-1, 2 * dim_size))
        print('lr size: {}'.format(lr.size()))
        h3, states_h3 = self.c_lstm(lr.view(batch_size, 2 * dim_size))
        print('h3 size: {}'.format(h3.size()))
        list_h1 = (1 - h1_a_m) * list_h1
        list_h1 += torch.bmm(h3, h1_a)
        print('list_h1: {}'.format(list_h1.size()))




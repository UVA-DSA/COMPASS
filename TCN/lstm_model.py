from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb

# This module should be tested carefully 

class LSTM_Layer(nn.Module):
    def __init__(self, input_size, num_class,hidden_size, num_layers,device,
                 bi_dir=False, use_gru=False):
        super(LSTM_Layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_class = num_class
        self.bi_dir = bi_dir
        self.use_gru = use_gru
        self.device = device

        if self.use_gru:
            self.lstm = nn.GRU(input_size, hidden_size, num_layers, 
                                batch_first=True, bidirectional=bi_dir)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                batch_first=True, bidirectional=bi_dir)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size
        self.fc = nn.Linear(self.hidden_size,self.num_class)

    def forward(self, x): # x: (batch,feature,seq)
        
        #x = x.permute(0, 2, 1) 

        batch_size = x.size(0)
        
        x, _ = self.lstm(x, self.__get_init_state(batch_size)) # x: (batch,seq,hidden)
        
        #x = x.permute(0, 2, 1) 
            # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        #x = x.contiguous().view(batch_size,-1)
        x = self.fc(x)
        
        return x

    def __get_init_state(self, batch_size):

        if self.bi_dir:
            nl_x_nd = 2 * self.num_layers
        else:
            nl_x_nd = 1 * self.num_layers

        h0 = torch.zeros(nl_x_nd, batch_size, self.hidden_size)
        h0 = h0.to(self.device).cuda()

        if self.use_gru:
            return h0
        else:
            c0 = torch.zeros(nl_x_nd, batch_size, self.hidden_size)
            c0 = c0.to(self.device).cuda()
            return (h0, c0)


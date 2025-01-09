import torch
import torch.nn.functional as F
from config import *
import math
import torch.nn as nn   

class MultichannelLinear(nn.Module): #maybe this is missing the bias term
    def __init__(self, channels, in_features, out_features,  project = 1, up = False):
        super(MultichannelLinear, self).__init__()
        self.up = up
        self.project = project
        if not up:
            self.weight_pw = nn.Parameter(torch.empty(int(math.ceil(channels/project)), out_features, in_features*project))
            self.weight_bias = nn.Parameter(torch.empty(int(math.ceil(channels/project)), out_features))
        else:
            self.weight_pw = nn.Parameter(torch.empty(channels, out_features*project, in_features))
            self.weight_bias = nn.Parameter(torch.empty(channels, out_features*project))
        nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features*project), b=1/math.sqrt(in_features*project))
        nn.init.uniform_(self.weight_bias, a=-1/math.sqrt(in_features*project), b=1/math.sqrt(in_features*project))

    def __call__(self, x):
        if not self.up:
            if not self.project ==1:   
                #reshape x to (batchsize, num_pcas/down_project, dim_pcas*down_project)
                if x.shape[1] % self.project != 0:
                    x = F.pad(x, (0,0,0,self.project - x.shape[1] % self.project))
                x = x.reshape(x.shape[0], int(x.shape[1]/self.project), x.shape[2]*self.project)
                
            x = torch.matmul(self.weight_pw.unsqueeze(0),x.unsqueeze(-1)).squeeze(-1) + self.weight_bias.unsqueeze(0)
        else:
            x = torch.matmul(self.weight_pw.unsqueeze(0),x.unsqueeze(-1)).squeeze(-1) + self.weight_bias.unsqueeze(0)
            #reshape x from (batchisze, channels,in_features)to (batchsize, channels * project, out_features)
            x = x.reshape(x.shape[0], int(x.shape[1]*self.project), int(x.shape[2]/self.project))
     #   print(x.shape)
        return x

class MethylMLP(nn.Module): 
    def __init__(self, num_classes, num_inputs, num_lin_blocks, hidden_dim ):
        super().__init__()
        self.input = nn.Linear(num_inputs, hidden_dim)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_lin_blocks*2)])
       # self.out = nn.ModuleList([nn.Linear(hidden_dim, c) for c in num_classes])
        self.out_emb = nn.Linear(hidden_dim, num_inputs)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, cls = False):
        x = F.gelu(self.input(x))
        for i in range(len(self.linears)//2):
            x2 = x
            x = F.gelu(self.linears[i*2](x))
            x = F.gelu(self.linears[i*2+1](x)) +x2

        
        if cls:
            return self.out(x)
        else:
            return torch.nn.functional.sigmoid(self.out_emb(x))

class EncoderModelPreTrain(nn.Module):
    #input should be (batchsize, num_pcas, dim_pcas)
    def __init__(self, num_classes, num_tokens, hidden_dim ,n_layers , compression):
        super().__init__()
        
        self.EmbeddingLayer = MultichannelLinear(num_tokens, 1, hidden_dim,compression)
        self.module_list = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=hidden_dim//64,dim_feedforward=4*hidden_dim, batch_first=True, activation='gelu') for i in range(n_layers)])
        self.classification_token = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        nn.init.uniform_(self.classification_token, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))
        #self.outs = nn.ModuleList([nn.Linear(hidden_dim, c) for c in num_classes])
        self.out = nn.Linear(hidden_dim, num_classes)
        self.DeEmbeddingLayer = MultichannelLinear(num_tokens//compression, hidden_dim, 1,compression, up = True)


    def forward(self, x, cls = False):
        x = self.EmbeddingLayer(x.unsqueeze(-1))

        if cls:
            classification_token = torch.stack([self.classification_token.unsqueeze(0) for _ in range(x.shape[0])])
            x = torch.cat((classification_token,x),dim = 1)

        for layer in self.module_list:
            x = layer(x)
    
        if cls:
            classification_token = x[:,0,:]
            return self.out(classification_token)
        else:
            return torch.nn.functional.sigmoid(self.DeEmbeddingLayer(x)).squeeze(-1)
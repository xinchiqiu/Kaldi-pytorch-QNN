import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from distutils.util import strtobool
import sys
import math
from quaternion_layers import *
from quaternion_ops    import q_normalize

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def act_fun(act_type):

 if act_type=="relu":
    return nn.ReLU()

 if act_type=="prelu":
    return nn.PReLU()

 if act_type=="tanh":
    return nn.Tanh()

 if act_type=="sigmoid":
    return nn.Sigmoid()

 if act_type=="hardtanh":
    return nn.Hardtanh()

 if act_type=="leaky_relu":
    return nn.LeakyReLU(0.2)

 if act_type=="elu":
    return nn.ELU()

 if act_type=="softmax":
    return nn.LogSoftmax(dim=1)

 if act_type=="linear":
     return nn.LeakyReLU(1) # initializzed like this, but not used in forward!


class QMLP(nn.Module):
    def __init__(self, options,inp_dim):
        super(QMLP, self).__init__()

        self.input_dim=inp_dim
        self.dnn_lay=list(map(int, options['dnn_lay'].split(',')))
        self.dnn_drop=list(map(float, options['dnn_drop'].split(',')))
        self.dnn_act=options['dnn_act'].split(',')
        self.quaternion_init=str(options['quaternion_init'])
        self.autograd=strtobool(options['autograd'])
        self.proj =strtobool(options['proj_layer'])
        self.proj_dim = int(options['proj_dim'])
        self.proj_norm = strtobool(options['proj_norm'])
        self.quaternion_norm = strtobool(options['quaternion_norm'])

        self.wx  = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        self.N_dnn_lay=len(self.dnn_lay)

        if self.proj:
            current_input=self.proj_dim
        else:
            current_input=self.input_dim

        # Project layer
        if self.proj:
            self.proj_layer = nn.Linear(self.input_dim, self.proj_dim, bias=True)
            nn.init.zeros_(self.proj_layer.bias)
            torch.nn.init.xavier_normal_(self.proj_layer.weight)

        # Initialization of hidden layers

        for i in range(self.N_dnn_lay):

             # dropout
             self.drop.append(nn.Dropout(p=self.dnn_drop[i]))

             # activation
             self.act.append(act_fun(self.dnn_act[i]))

             add_bias=True

             # Linear operations
             if self.autograd:
                self.wx.append(QuaternionLinearAutograd(current_input, self.dnn_lay[i], bias=add_bias, weight_init=self.quaternion_init))
             else:
                self.wx.append(QuaternionLinear(current_input, self.dnn_lay[i], bias=add_bias, weight_init=self.quaternion_init))

             current_input=self.dnn_lay[i]

        self.out_dim=current_input

    def forward(self, x):

      if self.proj:
            x = nn.tanh(self.proj_layer(x))
            if self.proj_norm:
                x = nn.Dropout(p=0.2)(x)

      for i in range(self.N_dnn_lay):
        if self.quaternion_norm:
            x = self.drop[i](self.act[i](q_normalize(self.wx[i](x))))
        else:
            x = self.drop[i](self.act[i](self.wx[i](x)))
      return x

class QLSTM(nn.Module):

    def __init__(self, options,inp_dim):
        super(QLSTM, self).__init__()

        # Reading parameters
        self.input_dim=inp_dim
        self.lstm_lay=list(map(int, options['lstm_lay'].split(',')))
        self.lstm_drop=list(map(float, options['lstm_drop'].split(',')))
        self.lstm_act=options['lstm_act'].split(',')
        self.quaternion_init=str(options['quaternion_init'])
        self.bidir=strtobool(options['lstm_bidir'])
        self.use_cuda=strtobool(options['use_cuda'])
        self.autograd=strtobool(options['autograd'])
        self.to_do=options['to_do']
        self.proj =strtobool(options['proj_layer'])
        self.proj_dim = int(options['proj_dim'])
        self.proj_norm = strtobool(options['proj_norm'])
        self.proj_act  = str(options['proj_act'])
        self.quaternion_norm = strtobool(options['quaternion_norm'])


        if self.to_do=='train':
            self.test_flag=False
        else:
            self.test_flag=True
        # Project layer
        if self.proj:
            self.proj_layer = nn.Linear(self.input_dim, self.proj_dim, bias=True)
            nn.init.zeros_(self.proj_layer.bias)
            torch.nn.init.xavier_normal_(self.proj_layer.weight)
        # List initialization
        self.wfx  = nn.ModuleList([]) # Forget
        self.ufh  = nn.ModuleList([]) # Forget

        self.wix  = nn.ModuleList([]) # Input
        self.uih  = nn.ModuleList([]) # Input

        self.wox  = nn.ModuleList([]) # Output
        self.uoh  = nn.ModuleList([]) # Output

        self.wcx  = nn.ModuleList([]) # Cell state
        self.uch  = nn.ModuleList([])  # Cell state

        self.act  = nn.ModuleList([]) # Activations

        self.N_lstm_lay=len(self.lstm_lay)

        if self.proj:
            current_input=self.proj_dim
        else:
            current_input=self.input_dim
        # Initialization of hidden layers

        for i in range(self.N_lstm_lay):

             # Activations
             self.act.append(act_fun(self.lstm_act[i]))

             add_bias=True

             if(self.autograd):

                 # Feed-forward connections
                 self.wfx.append(QuaternionLinearAutograd(current_input, self.lstm_lay[i],bias=add_bias, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.wix.append(QuaternionLinearAutograd(current_input, self.lstm_lay[i],bias=add_bias, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.wox.append(QuaternionLinearAutograd(current_input, self.lstm_lay[i],bias=add_bias, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.wcx.append(QuaternionLinearAutograd(current_input, self.lstm_lay[i],bias=add_bias, weight_init=self.quaternion_init, init_criterion='glorot'))

                 # Recurrent connections
                 self.ufh.append(QuaternionLinearAutograd(self.lstm_lay[i], self.lstm_lay[i],bias=False, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.uih.append(QuaternionLinearAutograd(self.lstm_lay[i], self.lstm_lay[i],bias=False, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.uoh.append(QuaternionLinearAutograd(self.lstm_lay[i], self.lstm_lay[i],bias=False, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.uch.append(QuaternionLinearAutograd(self.lstm_lay[i], self.lstm_lay[i],bias=False, weight_init=self.quaternion_init, init_criterion='glorot'))
             else:

                # Feed-forward connections
                 self.wfx.append(QuaternionLinear(current_input, self.lstm_lay[i],bias=add_bias, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.wix.append(QuaternionLinear(current_input, self.lstm_lay[i],bias=add_bias, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.wox.append(QuaternionLinear(current_input, self.lstm_lay[i],bias=add_bias, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.wcx.append(QuaternionLinear(current_input, self.lstm_lay[i],bias=add_bias, weight_init=self.quaternion_init, init_criterion='glorot'))

                 # Recurrent connections
                 self.ufh.append(QuaternionLinear(self.lstm_lay[i], self.lstm_lay[i],bias=False, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.uih.append(QuaternionLinear(self.lstm_lay[i], self.lstm_lay[i],bias=False, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.uoh.append(QuaternionLinear(self.lstm_lay[i], self.lstm_lay[i],bias=False, weight_init=self.quaternion_init, init_criterion='glorot'))
                 self.uch.append(QuaternionLinear(self.lstm_lay[i], self.lstm_lay[i],bias=False, weight_init=self.quaternion_init, init_criterion='glorot'))
             if self.bidir:
                 current_input=2*self.lstm_lay[i]
             else:
                 current_input=self.lstm_lay[i]

        self.out_dim=self.lstm_lay[i]+self.bidir*self.lstm_lay[i]

    def forward(self, x):

        if self.proj:

            x = act_fun(self.proj_act)((self.proj_layer(x)))
            if self.proj_norm:
                x = q_normalize(x)

            if self.test_flag==False:
                drop_mask=torch.bernoulli(torch.Tensor(x.shape[1],self.proj_dim).fill_(1-0.2))
                drop_mask=drop_mask.cuda()
                x = x*drop_mask

        #print(x.shape)
        for i in range(self.N_lstm_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2*x.shape[1], self.lstm_lay[i])
                x=torch.cat([x,flip(x,0)],1)
            else:
                h_init = torch.zeros(x.shape[1],self.lstm_lay[i])


            # Drop mask initilization (same mask for all time steps)
            if self.test_flag==False:
                drop_mask=torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.lstm_drop[i]))
            else:
                drop_mask=torch.FloatTensor([1-self.lstm_drop[i]])

            if self.use_cuda:
               h_init=h_init.cuda()
               drop_mask=drop_mask.cuda()


            # Feed-forward affine transformations (all steps in parallel)
            if self.quaternion_norm:
                wfx_out=q_normalize(self.wfx[i](x))
                wix_out=q_normalize(self.wix[i](x))
                wox_out=q_normalize(self.wox[i](x))
                wcx_out=q_normalize(self.wcx[i](x))
            else:
                wfx_out=self.wfx[i](x)
                wix_out=self.wix[i](x)
                wox_out=self.wox[i](x)
                wcx_out=self.wcx[i](x)

            # Processing time steps
            hiddens = []
            ct=h_init
            ht=h_init

            for k in range(x.shape[0]):

                # LSTM equations
                ft=torch.sigmoid(wfx_out[k]+self.ufh[i](ht))
                it=torch.sigmoid(wix_out[k]+self.uih[i](ht))
                ot=torch.sigmoid(wox_out[k]+self.uoh[i](ht))
                ct=it*self.act[i](wcx_out[k]+self.uch[i](ht))*drop_mask+ft*ct
                ht=ot*self.act[i](ct)

                hiddens.append(ht)

            # Stacking hidden states
            h=torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f=h[:,0:int(x.shape[1]/2)]
                h_b=flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
                h=torch.cat([h_f,h_b],2)

            # Setup x for the next hidden layer
            x=h


        return x



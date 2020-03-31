##########################################################
# pytorch-kaldi v.0.1
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from distutils.util import strtobool
import sys
import math
from quaternion_layers import *
from quaternion_ops    import q_normalize
from neural_networks   import SincConv

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

class QliGRU(nn.Module):

    def __init__(self, options,inp_dim):
        super(QliGRU, self).__init__()

        # Reading parameters
        self.input_dim=inp_dim
        self.ligru_lay=list(map(int, options['ligru_lay'].split(',')))
        self.ligru_drop=list(map(float, options['ligru_drop'].split(',')))
        self.ligru_use_batchnorm=list(map(strtobool, options['ligru_use_batchnorm'].split(',')))
        self.ligru_use_laynorm=list(map(strtobool, options['ligru_use_laynorm'].split(',')))
        self.ligru_use_laynorm_inp=strtobool(options['ligru_use_laynorm_inp'])
        self.ligru_use_batchnorm_inp=strtobool(options['ligru_use_batchnorm_inp'])
        self.ligru_orthinit=strtobool(options['ligru_orthinit'])
        self.ligru_act=options['ligru_act'].split(',')
        self.bidir=strtobool(options['ligru_bidir'])
        self.use_cuda=strtobool(options['use_cuda'])
        self.to_do=options['to_do']

        if self.to_do=='train':
            self.test_flag=False
        else:
            self.test_flag=True


        # List initialization
        self.wh  = nn.ModuleList([])
        self.uh  = nn.ModuleList([])

        self.wz  = nn.ModuleList([]) # Update Gate
        self.uz  = nn.ModuleList([]) # Update Gate


        self.ln  = nn.ModuleList([]) # Layer Norm
        self.bn_wh  = nn.ModuleList([]) # Batch Norm
        self.bn_wz  = nn.ModuleList([]) # Batch Norm



        self.act  = nn.ModuleList([]) # Activations

        self.N_ligru_lay=len(self.ligru_lay)

        current_input=self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_ligru_lay):

             # Activations
             self.act.append(act_fun(self.ligru_act[i]))

             add_bias=True


             #if self.ligru_use_laynorm[i] or self.ligru_use_batchnorm[i]:
            #     add_bias=False


             # Feed-forward connections
             self.wh.append(QuaternionLinearAutograd(current_input, self.ligru_lay[i],bias=add_bias))
             self.wz.append(QuaternionLinearAutograd(current_input, self.ligru_lay[i],bias=add_bias))

             # Recurrent connections
             self.uh.append(QuaternionLinearAutograd(self.ligru_lay[i], self.ligru_lay[i],bias=False))
             self.uz.append(QuaternionLinearAutograd(self.ligru_lay[i], self.ligru_lay[i],bias=False))


             # batch norm initialization
             self.bn_wh.append(QuaternionBatchNorm1d(self.ligru_lay[i]))
             self.bn_wz.append(QuaternionBatchNorm1d(self.ligru_lay[i]))


             #self.ln.append(LayerNorm(self.ligru_lay[i]))

             if self.bidir:
                 current_input=2*self.ligru_lay[i]
             else:
                 current_input=self.ligru_lay[i]

        self.out_dim=self.ligru_lay[i]+self.bidir*self.ligru_lay[i]



    def forward(self, x):

        for i in range(self.N_ligru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2*x.shape[1], self.ligru_lay[i])
                x=torch.cat([x,flip(x,0)],1)
            else:
                h_init = torch.zeros(x.shape[1],self.ligru_lay[i])


            # Drop mask initilization (same mask for all time steps)
            if self.test_flag==False:
                drop_mask=torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.ligru_drop[i]))
            else:
                drop_mask=torch.FloatTensor([1-self.ligru_drop[i]])

            if self.use_cuda:
               h_init=h_init.cuda()
               drop_mask=drop_mask.cuda()


            # Feed-forward affine transformations (all steps in parallel)
            wh_out=self.wh[i](x)
            wz_out=self.wz[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.ligru_use_batchnorm[i]:

                wh_out_bn=self.bn_wh[i](wh_out.view(wh_out.shape[0]*wh_out.shape[1],wh_out.shape[2]))
                wh_out=wh_out_bn.view(wh_out.shape[0],wh_out.shape[1],wh_out.shape[2])

                wz_out_bn=self.bn_wz[i](wz_out.view(wz_out.shape[0]*wz_out.shape[1],wz_out.shape[2]))
                wz_out=wz_out_bn.view(wz_out.shape[0],wz_out.shape[1],wz_out.shape[2])



            # Processing time steps
            hiddens = []
            ht=h_init

            for k in range(x.shape[0]):

                # ligru equation
                zt=torch.sigmoid(wz_out[k]+self.uz[i](ht))
                at=wh_out[k]+self.uh[i](ht)
                hcand=self.act[i](at)*drop_mask
                ht=(zt*ht+(1-zt)*hcand)


                if self.ligru_use_laynorm[i]:
                    ht=self.ln[i](ht)

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

class liQGRU(nn.Module):

    def __init__(self, options,inp_dim):
        super(liQGRU, self).__init__()

        # Reading parameters
        self.input_dim=inp_dim
        self.ligru_lay=list(map(int, options['ligru_lay'].split(',')))
        self.ligru_drop=list(map(float, options['ligru_drop'].split(',')))
        self.ligru_act=options['ligru_act'].split(',')
        self.bidir=strtobool(options['ligru_bidir'])
        self.use_cuda=strtobool(options['use_cuda'])
        self.to_do=options['to_do']
        self.quaternion_init=str(options['quaternion_init'])
        self.autograd=strtobool(options['autograd'])
        self.proj =strtobool(options['proj_layer'])
        self.proj_dim = int(options['proj_dim'])
        self.proj_norm = strtobool(options['proj_norm'])
        self.quaternion_norm = strtobool(options['quaternion_norm'])

        if self.to_do=='train':
            self.test_flag=False
        else:
            self.test_flag=True


        # List initialization
        self.wh  = nn.ModuleList([])
        self.uh  = nn.ModuleList([])

        self.wz  = nn.ModuleList([]) # Update Gate
        self.uz  = nn.ModuleList([]) # Update Gate


        self.act  = nn.ModuleList([]) # Activations

        self.N_ligru_lay=len(self.ligru_lay)

        if self.proj:
            current_input=self.proj_dim
        else:
            current_input=self.input_dim

        # Project layer
        if self.proj:
            self.proj_layer = nn.Linear(self.input_dim, self.proj_dim, bias=True)
            torch.nn.init.xavier_normal(self.proj_layer.weight)

        # Initialization of hidden layers

        for i in range(self.N_ligru_lay):

             # Activations
             self.act.append(act_fun(self.ligru_act[i]))

             add_bias=True

             if self.autograd:
                 # Feed-forward connections
                 self.wh.append(QuaternionLinearAutograd(current_input, self.ligru_lay[i],bias=add_bias,weight_init=self.quaternion_init))
                 self.wz.append(QuaternionLinearAutograd(current_input, self.ligru_lay[i],bias=add_bias,weight_init=self.quaternion_init))


                 # Recurrent connections
                 self.uh.append(QuaternionLinearAutograd(self.ligru_lay[i], self.ligru_lay[i],bias=False,weight_init=self.quaternion_init))
                 self.uz.append(QuaternionLinearAutograd(self.ligru_lay[i], self.ligru_lay[i],bias=False,weight_init=self.quaternion_init))
             else:
                 # Feed-forward connections
                 self.wh.append(QuaternionLinear(current_input, self.ligru_lay[i],bias=add_bias,weight_init=self.quaternion_init))
                 self.wz.append(QuaternionLinear(current_input, self.ligru_lay[i],bias=add_bias,weight_init=self.quaternion_init))


                 # Recurrent connections
                 self.uh.append(QuaternionLinear(self.ligru_lay[i], self.ligru_lay[i],bias=False,weight_init=self.quaternion_init))
                 self.uz.append(QuaternionLinear(self.ligru_lay[i], self.ligru_lay[i],bias=False,weight_init=self.quaternion_init))


             if self.bidir:
                 current_input=2*self.ligru_lay[i]
             else:
                 current_input=self.ligru_lay[i]

        self.out_dim=self.ligru_lay[i]+self.bidir*self.ligru_lay[i]



    def forward(self, x):

        if self.proj:
            x = nn.Hardtanh()(self.proj_layer(x))
            if self.proj_norm:
                x = q_normalize(x)

        for i in range(self.N_ligru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2*x.shape[1], self.ligru_lay[i])
                x=torch.cat([x,flip(x,0)],1)
            else:
                h_init = torch.zeros(x.shape[1],self.ligru_lay[i])


            # Drop mask initilization (same mask for all time steps)
            if self.test_flag==False:
                drop_mask=torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.ligru_drop[i]))
            else:
                drop_mask=torch.FloatTensor([1-self.ligru_drop[i]])

            if self.use_cuda:
               h_init=h_init.cuda()
               drop_mask=drop_mask.cuda()


            # Feed-forward affine transformations (all steps in parallel)
            wh_out=self.wh[i](x)
            wz_out=self.wz[i](x)

            if(self.quaternion_norm):
                wh_out = q_normalize(wh_out)
                wz_out = q_normalize(wz_out)


            # Processing time steps
            hiddens = []
            ht=h_init

            for k in range(x.shape[0]):

                # ligru equation
                zt=torch.sigmoid(wz_out[k]+self.uz[i](ht))
                at=wh_out[k]+self.uh[i](ht)
                hcand=self.act[i](at)*drop_mask
                ht=(zt*ht+(1-zt)*hcand)

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

class QSincNet(nn.Module):

    def __init__(self,options,inp_dim):
       super(QSincNet,self).__init__()

       # Reading parameters
       self.input_dim=inp_dim
       self.sinc_N_filt=list(map(int, options['sinc_N_filt'].split(',')))
       self.quaternion_norm = strtobool(options['quaternion_norm'])

       self.sinc_len_filt=list(map(int, options['sinc_len_filt'].split(',')))
       self.sinc_max_pool_len=list(map(int, options['sinc_max_pool_len'].split(',')))

       self.sinc_act=options['sinc_act'].split(',')
       self.sinc_drop=list(map(float, options['sinc_drop'].split(',')))

       self.sinc_use_laynorm=list(map(strtobool, options['sinc_use_laynorm'].split(',')))
       self.sinc_use_batchnorm=list(map(strtobool, options['sinc_use_batchnorm'].split(',')))
       self.sinc_use_laynorm_inp=strtobool(options['sinc_use_laynorm_inp'])
       self.sinc_use_batchnorm_inp=strtobool(options['sinc_use_batchnorm_inp'])

       self.N_sinc_lay=len(self.sinc_N_filt)

       self.sinc_sample_rate=int(options['sinc_sample_rate'])
       self.sinc_min_low_hz=int(options['sinc_min_low_hz'])
       self.sinc_min_band_hz=int(options['sinc_min_band_hz'])


       self.conv  = nn.ModuleList([])
       self.bn  = nn.ModuleList([])
       self.ln  = nn.ModuleList([])
       self.act = nn.ModuleList([])
       self.drop = nn.ModuleList([])


       if self.sinc_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)

       if self.sinc_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)

       current_input=self.input_dim

       for i in range(self.N_sinc_lay):

         N_filt=int(self.sinc_N_filt[i])
         len_filt=int(self.sinc_len_filt[i])

         # dropout
         self.drop.append(nn.Dropout(p=self.sinc_drop[i]))

         # activation
         self.act.append(act_fun(self.sinc_act[i]))

         # layer norm initialization
         self.ln.append(LayerNorm([N_filt,int((current_input-self.sinc_len_filt[i]+1)/self.sinc_max_pool_len[i])]))

         self.bn.append(nn.BatchNorm1d(N_filt,int((current_input-self.sinc_len_filt[i]+1)/self.sinc_max_pool_len[i]),momentum=0.05))

         if i==0:
          self.conv.append(SincConv(1, N_filt, len_filt,sample_rate=self.sinc_sample_rate, min_low_hz=self.sinc_min_low_hz, min_band_hz=self.sinc_min_band_hz))
         else:
          self.conv.append(QuaternionConv(self.sinc_N_filt[i-1], self.sinc_N_filt[i], self.sinc_len_filt[i], operation='convolution1d'))

         current_input=int((current_input-self.sinc_len_filt[i]+1)/self.sinc_max_pool_len[i])


       self.out_dim=current_input*N_filt



    def forward(self, x):

       batch=x.shape[0]
       seq_len=x.shape[1]

       if bool(self.sinc_use_laynorm_inp):
        x=self.ln0(x)

       if bool(self.sinc_use_batchnorm_inp):
        x=self.bn0(x)

       x=x.view(batch,1,seq_len)

       for i in range(self.N_sinc_lay):

         if i==0 and self.sinc_use_laynorm[i]:
          x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i]))))

         if i==0 and self.sinc_use_batchnorm[i]:
          x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i]))))

         if i!=0:
            if self.quaternion_norm:
                x = self.drop[i](self.act[i](F.max_pool1d(q_normalize(self.conv[i](x),channel=1), self.sinc_max_pool_len[i])))
            else:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i])))


       x = x.view(batch,-1)

       return x

class QCNN(nn.Module):

    def __init__(self,options,inp_dim):
       super(QCNN,self).__init__()

       # Reading parameters
       self.input_dim=inp_dim
       self.cnn_N_filt=list(map(int, options['cnn_N_filt'].split(',')))

       self.cnn_len_filt=list(map(int, options['cnn_len_filt'].split(',')))
       self.cnn_max_pool_len=list(map(int, options['cnn_max_pool_len'].split(',')))

       self.cnn_act=options['cnn_act'].split(',')
       self.cnn_drop=list(map(float, options['cnn_drop'].split(',')))

       self.N_cnn_lay=len(self.cnn_N_filt)
       self.conv  = nn.ModuleList([])
       self.act = nn.ModuleList([])
       self.drop = nn.ModuleList([])

       current_input=int(self.input_dim /4)

       for i in range(self.N_cnn_lay):

         N_filt=int(self.cnn_N_filt[i])
         len_filt=int(self.cnn_len_filt[i])

         # dropout
         self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

         # activation
         self.act.append(act_fun(self.cnn_act[i]))


         if i==0:
          self.conv.append(nn.Conv1d(4, N_filt, len_filt))

         else:
          self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))

         current_input=int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])

       self.out_dim=current_input*N_filt



    def forward(self, x):

       batch=x.shape[0]
       seq_len=x.shape[1]

       x=x.view(batch,4,int(seq_len/4))

       for i in range(self.N_cnn_lay):
            x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

       x = x.view(batch,-1)

       return x

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

class QRNN(nn.Module):

    def __init__(self, options,inp_dim):
        super(QRNN, self).__init__()

        # Reading parameters
        self.input_dim=inp_dim
        self.rnn_lay=list(map(int, options['rnn_lay'].split(',')))
        self.rnn_drop=list(map(float, options['rnn_drop'].split(',')))
        self.rnn_act=options['rnn_act'].split(',')
        self.bidir=strtobool(options['rnn_bidir'])
        self.use_cuda=strtobool(options['use_cuda'])
        self.to_do=options['to_do']
        self.quaternion_init=str(options['quaternion_init'])
        self.autograd=strtobool(options['autograd'])
        self.proj =strtobool(options['proj_layer'])
        self.proj_dim = int(options['proj_dim'])
        self.proj_norm = strtobool(options['proj_norm'])
        self.proj_act  = str(options['proj_act'])
        self.quaternion_norm = strtobool(options['quaternion_norm'])

        if self.proj:
            self.proj_layer = nn.Linear(self.input_dim, self.proj_dim, bias=True)
            nn.init.zeros_(self.proj_layer.bias)
            torch.nn.init.xavier_normal_(self.proj_layer.weight)

        if self.to_do=='train':
            self.test_flag=False
        else:
            self.test_flag=True


        # List initialization
        self.wh  = nn.ModuleList([])
        self.uh  = nn.ModuleList([])

        self.act  = nn.ModuleList([]) # Activations

        self.N_rnn_lay=len(self.rnn_lay)

        if self.proj:
            current_input=self.proj_dim
        else:
            current_input=self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_rnn_lay):

             # Activations
             self.act.append(act_fun(self.rnn_act[i]))

             add_bias=True

             if self.autograd:

                 # Feed-forward connections
                 self.wh.append(QuaternionLinearAutograd(current_input, self.rnn_lay[i],bias=add_bias, weight_init=self.quaternion_init))

                 # Recurrent connections
                 self.uh.append(QuaternionLinearAutograd(self.rnn_lay[i], self.rnn_lay[i],bias=False, weight_init=self.quaternion_init))
             else:
                 # Feed-forward connections
                 self.wh.append(QuaternionLinear(current_input, self.rnn_lay[i],bias=add_bias, weight_init=self.quaternion_init))
                 self.uh.append(QuaternionLinear(self.rnn_lay[i], self.rnn_lay[i],bias=False, weight_init=self.quaternion_init))

                 # Recurrent connections

             if self.bidir:
                 current_input=2*self.rnn_lay[i]
             else:
                 current_input=self.rnn_lay[i]

        self.out_dim=self.rnn_lay[i]+self.bidir*self.rnn_lay[i]



    def forward(self, x):


        for i in range(self.N_rnn_lay):

            #if self.proj:

            #    x = act_fun(self.proj_act)((self.proj_layer(x)))
            #    if self.proj_norm:
            #        x = q_normalize(x)

            #    if self.test_flag==False:
            #        drop_mask=torch.bernoulli(torch.Tensor(x.shape[1],self.proj_dim).fill_(1-0.2))
            #        drop_mask=drop_mask.cuda()
            #        x = x*drop_mask

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2*x.shape[1], self.rnn_lay[i])
                x=torch.cat([x,flip(x,0)],1)
            else:
                h_init = torch.zeros(x.shape[1],self.rnn_lay[i])


            # Drop mask initilization (same mask for all time steps)
            if self.test_flag==False:
                drop_mask=torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.rnn_drop[i]))
            else:
                drop_mask=torch.FloatTensor([1-self.rnn_drop[i]])

            if self.use_cuda:
               h_init=h_init.cuda()
               drop_mask=drop_mask.cuda()


            # Feed-forward affine transformations (all steps in parallel)
            wh_out=self.wh[i](x)

            # Processing time steps
            hiddens = []
            ht=h_init

            for k in range(x.shape[0]):

                # rnn equation
                at=wh_out[k]+self.uh[i](ht)
                ht=self.act[i](at)*drop_mask

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

class QGRU(nn.Module):

    def __init__(self, options,inp_dim):
        super(QGRU, self).__init__()

        # Reading parameters
        self.input_dim=inp_dim
        self.gru_lay=list(map(int, options['gru_lay'].split(',')))
        self.gru_drop=list(map(float, options['gru_drop'].split(',')))
        self.gru_act=options['gru_act'].split(',')
        self.quaternion_init=str(options['quaternion_init'])
        self.bidir=strtobool(options['gru_bidir'])
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
            current_input=self.proj_dim
        else:
            current_input=self.input_dim

        # List initialization
        self.wh  = nn.ModuleList([])
        self.uh  = nn.ModuleList([])

        self.wz  = nn.ModuleList([]) # Update Gate
        self.uz  = nn.ModuleList([]) # Update Gate

        self.wr  = nn.ModuleList([]) # Reset Gate
        self.ur  = nn.ModuleList([]) # Reset Gate


        self.act  = nn.ModuleList([]) # Activations

        self.N_gru_lay=len(self.gru_lay)

        # Initialization of hidden layers
        for i in range(self.N_gru_lay):

             # Activations
             self.act.append(act_fun(self.gru_act[i]))

             add_bias=True

             if(self.autograd):
                 # Feed-forward connections
                 self.wh.append(QuaternionLinearAutograd(current_input, self.gru_lay[i],bias=add_bias))
                 self.wz.append(QuaternionLinearAutograd(current_input, self.gru_lay[i],bias=add_bias))
                 self.wr.append(QuaternionLinearAutograd(current_input, self.gru_lay[i],bias=add_bias))

                 # Recurrent connections
                 self.uh.append(QuaternionLinearAutograd(self.gru_lay[i], self.gru_lay[i],bias=False))
                 self.uz.append(QuaternionLinearAutograd(self.gru_lay[i], self.gru_lay[i],bias=False))
                 self.ur.append(QuaternionLinearAutograd(self.gru_lay[i], self.gru_lay[i],bias=False))
             else:
                  # Feed-forward connections
                  self.wh.append(QuaternionLinear(current_input, self.gru_lay[i],bias=add_bias))
                  self.wz.append(QuaternionLinear(current_input, self.gru_lay[i],bias=add_bias))
                  self.wr.append(QuaternionLinear(current_input, self.gru_lay[i],bias=add_bias))

                  # Recurrent connections
                  self.uh.append(QuaternionLinear(self.gru_lay[i], self.gru_lay[i],bias=False))
                  self.uz.append(QuaternionLinear(self.gru_lay[i], self.gru_lay[i],bias=False))
                  self.ur.append(QuaternionLinear(self.gru_lay[i], self.gru_lay[i],bias=False))


             #self.ln.append(LayerNorm(self.gru_lay[i]))

             if self.bidir:
                 current_input=2*self.gru_lay[i]
             else:
                 current_input=self.gru_lay[i]

        self.out_dim=self.gru_lay[i]+self.bidir*self.gru_lay[i]



    def forward(self, x):

        if self.proj:
            x = act_fun(self.proj_act)((self.proj_layer(x)))

            if self.proj_norm:
                x = q_normalize(x)
            #if self.test_flag==False:
            #    drop_mask=torch.bernoulli(torch.Tensor(x.shape[1],self.proj_dim).fill_(1-0.2))
            #    drop_mask=drop_mask.cuda()
            #        x = x*drop_mask

        for i in range(self.N_gru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2*x.shape[1], self.gru_lay[i])
                x=torch.cat([x,flip(x,0)],1)
            else:
                h_init = torch.zeros(x.shape[1],self.gru_lay[i])


            # Drop mask initilization (same mask for all time steps)
            if self.test_flag==False:
                drop_mask=torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.gru_drop[i]))
            else:
                drop_mask=torch.FloatTensor([1-self.gru_drop[i]])

            if self.use_cuda:
               h_init=h_init.cuda()
               drop_mask=drop_mask.cuda()


            # Feed-forward affine transformations (all steps in parallel)
            wh_out=self.wh[i](x)
            wz_out=self.wz[i](x)
            wr_out=self.wr[i](x)



            # Processing time steps
            hiddens = []
            ht=h_init

            for k in range(x.shape[0]):

                # gru equation
                zt=torch.sigmoid(wz_out[k]+self.uz[i](ht))
                rt=torch.sigmoid(wr_out[k]+self.ur[i](ht))
                at=wh_out[k]+self.uh[i](rt*ht)
                hcand=self.act[i](at)*drop_mask
                ht=(zt*ht+(1-zt)*hcand)

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

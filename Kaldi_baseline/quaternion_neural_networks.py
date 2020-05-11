import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.nn import Module
import numpy as np
from scipy.stats import chi
from numpy.random import RandomState
from distutils.util import strtobool
import math

#################### quaternion_ops ################################################
def check_input(input):

    if input.dim() not in {2, 3}:
        raise RuntimeError(
            "quaternion linear accepts only input of dimension 2 or 3."
            " input.dim = " + str(input.dim())
        )

    nb_hidden = input.size()[-1]

    if nb_hidden % 4 != 0:
        raise RuntimeError(
            "Quaternion Tensors must be divisible by 4."
            " input.size()[1] = " + str(nb_hidden)
        )
#
# Getters
#
def get_r(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, 0, nb_hidden // 4)
    elif input.dim() == 3:
        return input.narrow(2, 0, nb_hidden // 4)


def get_i(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 4, nb_hidden // 4)

def get_j(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 2, nb_hidden // 4)

def get_k(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden - nb_hidden // 4, nb_hidden // 4)


def get_modulus(input, vector_form=False):
    check_input(input)
    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)
    if vector_form:
        return torch.sqrt(r * r + i * i + j * j + k * k)
    else:
        return torch.sqrt((r * r + i * i + j * j + k * k).sum(dim=0))

def q_normalize(input, channel=-1):

    if channel != -1:
        dim   = int(input.dim()) - 1
        input = input.transpose(dim,channel)

    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)

    norm = torch.sqrt(r*r + i*i + j*j + k*k + 0.00001)
    r = r / norm
    i = i / norm
    j = j / norm
    k = k / norm

    input = torch.cat([r,i,j,k], dim=-1)

    if channel != -1:
        dim = int(input.dim()) - 1
        return input.transpose(dim, channel)
    else:
        return input

def get_normalized(input, eps=0.0001):
    check_input(input)
    data_modulus = get_modulus(input)
    if input.dim() == 2:
        data_modulus_repeated = data_modulus.repeat(1, 4)
    elif input.dim() == 3:
        data_modulus_repeated = data_modulus.repeat(1, 1, 4)
    return input / (data_modulus_repeated.expand_as(input) + eps)




def quaternion_linear(input, r_weight, i_weight, j_weight, k_weight, bias):

    """
    Applies a quaternion linear transformation to the incoming data:
    It is important to notice that the forward phase of a QNN is defined
    as W * Inputs (with * equal to the Hamilton product). The constructed
    cat_kernels_4_quaternion is a modified version of the quaternion representation
    so when we do torch.mm(Input,W) it's equivalent to W * Inputs.
    """

    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
    cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
    cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
    cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
    cat_kernels_4_quaternion   = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)

    if input.dim() == 2 :

        if bias is not None:
            return torch.addmm(bias, input, cat_kernels_4_quaternion)
        else:
            return torch.mm(input, cat_kernels_4_quaternion)
    else:
        output = torch.matmul(input, cat_kernels_4_quaternion)
        if bias is not None:
            return output+bias
        else:
            return output

# Custom AUTOGRAD for lower VRAM consumption
class QuaternionLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, r_weight, i_weight, j_weight, k_weight, bias=None):
        ctx.save_for_backward(input, r_weight, i_weight, j_weight, k_weight, bias)
        check_input(input)
        cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
        cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
        cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
        cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
        cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)
        if input.dim() == 2 :
            if bias is not None:
                return torch.addmm(bias, input, cat_kernels_4_quaternion)
            else:
                return torch.mm(input, cat_kernels_4_quaternion)
        else:
            output = torch.matmul(input, cat_kernels_4_quaternion)
            if bias is not None:
                return output+bias
            else:
                return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, r_weight, i_weight, j_weight, k_weight, bias = ctx.saved_tensors
        grad_input = grad_weight_r = grad_weight_i = grad_weight_j = grad_weight_k = grad_bias = None

        input_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
        input_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
        input_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
        input_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
        cat_kernels_4_quaternion_T = Variable(torch.cat([input_r, input_i, input_j, input_k], dim=1).permute(1,0), requires_grad=False)

        r = get_r(input)
        i = get_i(input)
        j = get_j(input)
        k = get_k(input)
        input_r = torch.cat([r, -i, -j, -k], dim=0)
        input_i = torch.cat([i,  r, -k, j], dim=0)
        input_j = torch.cat([j,  k, r, -i], dim=0)
        input_k = torch.cat([k,  -j, i, r], dim=0)
        input_mat = Variable(torch.cat([input_r, input_i, input_j, input_k], dim=1), requires_grad=False)

        r = get_r(grad_output)
        i = get_i(grad_output)
        j = get_j(grad_output)
        k = get_k(grad_output)
        input_r = torch.cat([r, i, j, k], dim=1)
        input_i = torch.cat([-i,  r, k, -j], dim=1)
        input_j = torch.cat([-j,  -k, r, i], dim=1)
        input_k = torch.cat([-k,  j, -i, r], dim=1)
        grad_mat = torch.cat([input_r, input_i, input_j, input_k], dim=0)

        if ctx.needs_input_grad[0]:
            grad_input  = grad_output.mm(cat_kernels_4_quaternion_T)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_mat.permute(1,0).mm(input_mat).permute(1,0)
            unit_size_x = r_weight.size(0)
            unit_size_y = r_weight.size(1)
            grad_weight_r = grad_weight.narrow(0,0,unit_size_x).narrow(1,0,unit_size_y)
            grad_weight_i = grad_weight.narrow(0,0,unit_size_x).narrow(1,unit_size_y,unit_size_y)
            grad_weight_j = grad_weight.narrow(0,0,unit_size_x).narrow(1,unit_size_y*2,unit_size_y)
            grad_weight_k = grad_weight.narrow(0,0,unit_size_x).narrow(1,unit_size_y*3,unit_size_y)
        if ctx.needs_input_grad[5]:
            grad_bias   = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight_r, grad_weight_i, grad_weight_j, grad_weight_k, grad_bias
'''
def hamilton_product(q0, q1):
    """
    Applies a Hamilton product q0 * q1:
    Shape:
        - q0, q1 should be (batch_size, quaternion_number)
        (rr' - xx' - yy' - zz')  +
        (rx' + xr' + yz' - zy')i +
        (ry' - xz' + yr' + zx')j +
        (rz' + xy' - yx' + zr')k +
    """

    q1_r = get_r(q1)
    q1_i = get_i(q1)
    q1_j = get_j(q1)
    q1_k = get_k(q1)

    # rr', xx', yy', and zz'
    r_base = torch.mul(q0, q1)
    # (rr' - xx' - yy' - zz')
    r   = get_r(r_base) - get_i(r_base) - get_j(r_base) - get_k(r_base)

    # rx', xr', yz', and zy'
    i_base = torch.mul(q0, torch.cat([q1_i, q1_r, q1_k, q1_j], dim=1))
    # (rx' + xr' + yz' - zy')
    i   = get_r(i_base) + get_i(i_base) + get_j(i_base) - get_k(i_base)

    # ry', xz', yr', and zx'
    j_base = torch.mul(q0, torch.cat([q1_j, q1_k, q1_r, q1_i], dim=1))
    # (rx' + xr' + yz' - zy')
    j   = get_r(j_base) - get_i(j_base) + get_j(j_base) + get_k(j_base)

    # rz', xy', yx', and zr'
    k_base = torch.mul(q0, torch.cat([q1_k, q1_j, q1_i, q1_r], dim=1))
    # (rx' + xr' + yz' - zy')
    k   = get_r(k_base) + get_i(k_base) - get_j(k_base) + get_k(k_base)

    return torch.cat([r, i, j, k], dim=1)
'''
#
# PARAMETERS INITIALIZATION
#

def unitary_init(in_features, out_features, rng, kernel_size=None, criterion='he'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    s = np.sqrt(3.0) * s

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-s,s,number_of_weights)
    v_i = np.random.uniform(-s,s,number_of_weights)
    v_j = np.random.uniform(-s,s,number_of_weights)
    v_k = np.random.uniform(-s,s,number_of_weights)

    # Unitary quaternion
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i]**2 + v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
        v_r[i]/= norm
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    return (v_r, v_i, v_j, v_k)

def random_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(0.0,1.0,number_of_weights)
    v_i = np.random.uniform(0.0,1.0,number_of_weights)
    v_j = np.random.uniform(0.0,1.0,number_of_weights)
    v_k = np.random.uniform(0.0,1.0,number_of_weights)

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    weight_r = v_r * s
    weight_i = v_i * s
    weight_j = v_j * s
    weight_k = v_k * s
    return (weight_r, weight_i, weight_j, weight_k)


def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState(np.random.randint(1,1234))


    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4,loc=0,scale=s,size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.normal(0,1.0,number_of_weights)
    v_j = np.random.normal(0,1.0,number_of_weights)
    v_k = np.random.normal(0,1.0,number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
    	norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2 +0.0001)
    	v_i[i]/= norm
    	v_j[i]/= norm
    	v_k[i]/= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i*np.sin(phase)
    weight_j = modulus * v_j*np.sin(phase)
    weight_k = modulus * v_k*np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)

def create_dropout_mask(dropout_p, size, rng, as_type, operation='linear'):
    if operation == 'linear':
        mask = rng.binomial(n=1, p=1-dropout_p, size=size)
        return Variable(torch.from_numpy(mask).type(as_type))
    else:
         raise Exception("create_dropout_mask accepts only 'linear'. Found operation = "
                        + str(operation))

def affect_init(r_weight, i_weight, j_weight, k_weight, init_func, rng, init_criterion):
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
    r_weight.size() != k_weight.size() :
         raise ValueError('The real and imaginary weights '
                 'should have the same size . Found: r:'
                 + str(r_weight.size()) +' i:'
                 + str(i_weight.size()) +' j:'
                 + str(j_weight.size()) +' k:'
                 + str(k_weight.size()))

    elif r_weight.dim() != 2:
        raise Exception('affect_init accepts only matrices. Found dimension = '
                        + str(r_weight.dim()))
    kernel_size = None
    r, i, j, k  = init_func(r_weight.size(0), r_weight.size(1), rng, kernel_size, init_criterion)
    r, i, j, k  = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def affect_init_conv(r_weight, i_weight, j_weight, k_weight, kernel_size, init_func, rng,
                     init_criterion):
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
    r_weight.size() != k_weight.size() :
         raise ValueError('The real and imaginary weights '
                 'should have the same size . Found: r:'
                 + str(r_weight.size()) +' i:'
                 + str(i_weight.size()) +' j:'
                 + str(j_weight.size()) +' k:'
                 + str(k_weight.size()))

    elif 2 >= r_weight.dim():
        raise Exception('affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = '
                        + str(real_weight.dim()))

    r, i, j, k = init_func(
        r_weight.size(1),
        r_weight.size(0),
        rng=rng,
        kernel_size=kernel_size,
        criterion=init_criterion
    )
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)

def get_kernel_and_weight_shape(operation, in_channels, out_channels, kernel_size):
    if operation == 'convolution1d':
        if type(kernel_size) is not int:
            raise ValueError(
                """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                must be integer in the case. Found kernel_size = """ + str(kernel_size)
            )
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels) + tuple((ks,))
    else:# in case it is 2d or 3d.
        if   operation == 'convolution2d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size)
        elif operation == 'convolution3d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size, kernel_size)
        elif type(kernel_size) is not int:
            if   operation == 'convolution2d' and len(kernel_size) != 2:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                    must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                )
            elif operation == 'convolution3d' and len(kernel_size) != 3:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                    must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                )
            else:
                ks = kernel_size
        w_shape = (out_channels, in_channels) + (*ks,)
    return ks, w_shape
####################################################################################
###################### quaternion_layers ###########################################
class QuaternionLinearAutograd(Module):
    r"""Applies a quaternion linear transformation to the incoming data.
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=False, quaternion_format=False):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features       = in_features//4
        self.out_features      = out_features//4
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.bias = torch.zeros(self.out_features*4)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0,1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init, 'unitary': unitary_init, 'random': random_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return quaternion_linear(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'

class QuaternionLinear(Module):
    r"""A custom Autograd function is call to drastically reduce the VRAM consumption. Nonetheless, computing
    time is also slower compared to QuaternionLinearAutograd().
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinear, self).__init__()
        self.in_features  = in_features//4
        self.out_features = out_features//4
        self.r_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias     = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init    = weight_init
        self.seed           = seed if seed is not None else np.random.randint(0,1234)
        self.rng            = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.view(T * N, C)
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'
####################################################################################
####################### QLSTM network ##############################################
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

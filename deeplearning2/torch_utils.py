import torch
from torch import optim, nn, FloatTensor as FT
import torch.nn.parallel
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms, utils as vutils
from torch.autograd import Variable

import operator

def unit_prefix(x, n=1):
    for i in range(n): x = x.unsqueeze(0)
    return x

def align(x, y, start_dim=2):
    xd, yd = x.dim(), y.dim()
    if xd > yd: y = unit_prefix(y, xd - yd)
    elif yd > xd: x = unit_prefix(x, yd - xd)

    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd-i-1
        if   ys[td]==1: ys[td] = xs[td]
        elif xs[td]==1: xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)

def dot(x, y):
    assert(1<y.dim()<5)
    x, y = align(x, y)
    
    if y.dim() == 2: return x.mm(y)
    elif y.dim() == 3: return x.bmm(y)
    else:
        xs,ys = x.size(), y.size()
        res = torch.zeros(*(xs[:-1] + (ys[-1],)))
        for i in range(xs[0]): res[i].baddbmm_(x[i], (y[i]))
        return res


def aligned_op(x,y,f): return f(*align(x,y,0))

def add(x, y): return aligned_op(x, y, operator.add)
def sub(x, y): return aligned_op(x, y, operator.sub)
def mul(x, y): return aligned_op(x, y, operator.mul)
def div(x, y): return aligned_op(x, y, operator.truediv)
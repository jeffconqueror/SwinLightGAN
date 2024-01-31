import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary

def prune(args, model, test_loader, device):
    model = model.to(device)
    summary(model.cuda(), (3,224,224))

    # print(model)
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
    # print('factor is : ', bn.numpy())
    # print('factor is : ', bn_avg)
    y, i = torch.sort(bn)
    thre_index = int(total * args['percent'])
    thre = y[thre_index]
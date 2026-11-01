import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolution_att(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(55, 55))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        return output


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, is_resi=True):
        super().__init__()
        self.is_resi = is_resi
        self.gc1 = GraphConvolution_att(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(55 * in_features)
        self.gc2 = GraphConvolution_att(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(55 * in_features)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        return y + x if self.is_resi else y


class GCN_muti_att(nn.Module):
    def __init__(self, input_feature, hidden_feature, num_class, p_dropout, num_stage=1):
        super().__init__()
        self.num_stage = num_stage
        self.gc1 = GraphConvolution_att(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(55 * hidden_feature)
        self.gcbs = nn.ModuleList(
            [GC_Block(hidden_feature, p_dropout=p_dropout) for _ in range(num_stage)]
        )
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        for gcb in self.gcbs:
            y = gcb(y)
        out = torch.mean(y, dim=1)
        return self.fc_out(out)

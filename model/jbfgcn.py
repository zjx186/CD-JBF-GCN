import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=0, padding=True):
        super(unit_tcn, self).__init__()
        if padding:
            if dilation==0:
                pad = int((kernel_size - 1) / 2)
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                                      stride=(stride, 1))
            else:
                pad = int((kernel_size - 1) / 2) * dilation
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                                      stride=(stride, 1), dilation=dilation)
        else:
            if dilation==0:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(0, 0),
                                      stride=(stride, 1))
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(0, 0),
                                      stride=(stride, 1), dilation=dilation)

        self.bn = nn.BatchNorm2d(out_channels)

        # if act:
        #     self.act = nn.ReLU()
        # else:
        #     self.act = lambda x:x
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, act=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        if act:
            self.act = nn.ReLU()
        else:
            self.act = lambda x:x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        # batchsize, channel, t, node_num
        N, C, T, V = x.size()
        #print(N, C, T, V)
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.act(y)


class unit_jb_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, B, coff_embedding=4, num_subset=3, act=True):
        super(unit_jb_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_f = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_f.append(nn.Conv2d(2 * in_channels, out_channels, 1))

        self.PB = nn.Parameter(torch.from_numpy(B.astype(np.float32)))
        nn.init.constant_(self.PB, 1e-6)
        self.B = Variable(torch.from_numpy(B.astype(np.float32)), requires_grad=False)

        self.b_conv_a = nn.ModuleList()
        self.b_conv_b = nn.ModuleList()
        for i in range(self.num_subset):
            self.b_conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.b_conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        if act:
            self.act = nn.ReLU()
        else:
            self.act = lambda x:x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_f[i], self.num_subset)

    def forward(self, x, b):
        # batchsize, channel, t, node_num
        N, C, T, V = x.size()
        N_b, C_b, T_b, V_b = b.size()
        #print(N, C, T, V)
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        B = self.B.cuda(b.get_device())
        B = B + self.PB                      # V_b*V

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            zx = torch.matmul(A2, A1).view(N, C, T, V)

            B1 = self.b_conv_a[i](b).permute(0, 3, 1, 2).contiguous().view(N_b, V_b, self.inter_c * T_b)    # V_b
            B2 = self.b_conv_b[i](x).view(N, self.inter_c * T, V)                               # V
            B1 = self.soft(torch.matmul(B1, B2) / B1.size(-1))  # N V_b V
            B1 = B1 + B[i]
            B2 = b.view(N_b, C_b * T_b, V_b)
            zb = torch.matmul(B2, B1).view(N, C, T, V)

            # ---------------------------------------------------------------------
            zcat =  torch.cat([zx, zb],dim=1)
            z =  self.conv_f[i](zcat)
            # ---------------------------------------------------------------------
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.act(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, dilation=0, t_kernel=9, padding=True, act=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, act=act)
        self.tcn1 = unit_tcn(out_channels, out_channels, kernel_size=t_kernel, stride=stride, dilation=dilation, padding=padding)
        if act:
            self.act = nn.ReLU()
        else:
            self.act = lambda x:x

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride, dilation=dilation)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.act(x)

class jb_TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, C, stride=1, residual=True, dilation=0, t_kernel=9, padding=True, act=True):
        super(jb_TCN_GCN_unit, self).__init__()
        self.gcn_a = unit_gcn(in_channels, out_channels, A, act=act)
        self.gcn_b = unit_gcn(in_channels, out_channels, C, act=act)
        self.tcn_a = unit_tcn(out_channels, out_channels, kernel_size=t_kernel, stride=stride, dilation=dilation, padding=padding)
        self.tcn_b = unit_tcn(out_channels, out_channels, kernel_size=t_kernel, stride=stride, dilation=dilation, padding=padding)
        if act:
            self.act = nn.ReLU()
        else:
            self.act = lambda x:x

        if not residual:
            self.residual_a = lambda x: 0
            self.residual_b = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual_a = lambda x: x
            self.residual_b = lambda x: 0

        else:
            self.residual_a = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride, dilation=dilation)
            self.residual_b = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride, dilation=dilation)

    def forward(self, x, b):
        x = self.tcn_a(self.gcn_a(x)) + self.residual_a(x)
        b = self.tcn_b(self.gcn_b(b)) + self.residual_b(b)
        return self.act(x), self.act(b)


class jbf_TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, B, C, D, stride=1, residual=True, dilation=0, t_kernel=9, padding=True, act=True):
        super(jbf_TCN_GCN_unit, self).__init__()
        self.gcn_a = unit_jb_gcn(in_channels, out_channels, A, B, act=act)
        self.gcn_b = unit_jb_gcn(in_channels, out_channels, C, D, act=act)
        self.tcn_a = unit_tcn(out_channels, out_channels, kernel_size=t_kernel, stride=stride, dilation=dilation, padding=padding)
        self.tcn_b = unit_tcn(out_channels, out_channels, kernel_size=t_kernel, stride=stride, dilation=dilation, padding=padding)
        if act:
            self.act = nn.ReLU()
        else:
            self.act = lambda x:x

        if not residual:
            self.residual_a = lambda x: 0
            self.residual_b = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual_a = lambda x: x
            self.residual_b = lambda x: 0

        else:
            self.residual_a = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride, dilation=dilation)
            self.residual_b = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride, dilation=dilation)

    def forward(self, x, b):
        x_ori = x
        x = self.tcn_a(self.gcn_a(x, b)) + self.residual_a(x)
        b = self.tcn_b(self.gcn_b(b, x_ori)) + self.residual_b(b)
        return self.act(x), self.act(b)


class GGRU(nn.Module):
    def __init__(self, in_channels, hid_channels, A, dropout=0.3):
        super(GGRU, self).__init__()
        self.input_r = nn.Linear(in_channels, hid_channels, bias=True)
        self.input_i = nn.Linear(in_channels, hid_channels, bias=True)
        self.input_n = nn.Linear(in_channels, hid_channels, bias=True)

        self.hidden_r = nn.Linear(hid_channels, hid_channels, bias=False)
        self.hidden_i = nn.Linear(hid_channels, hid_channels, bias=False)
        self.hidden_h = nn.Linear(hid_channels, hid_channels, bias=False)

        self.out_fc1 = nn.Linear(hid_channels, hid_channels)
        self.out_fc2 = nn.Linear(hid_channels, hid_channels)
        self.out_fc3 = nn.Linear(hid_channels, 3)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        self.hid_gcn = TCN_GCN_unit(hid_channels, hid_channels, A)

    def step_forward(self, x_last, hidden):
        # x(N*M,V,C) hidden(N*M,C,V)
        N, V, C_x = x_last.size()
        msg = self.hid_gcn(hidden.unsqueeze(2)).squeeze(2)
        msg, hidden = msg.permute(0, 2, 1), hidden.permute(0, 2, 1)     # N*M, V, C

        r = torch.sigmoid(self.input_r(x_last) + self.hidden_r(msg))  # r:N, V, hid
        z = torch.sigmoid(self.input_i(x_last) + self.hidden_i(msg))  # z:N, V, hid
        n = torch.tanh(self.input_n(x_last) + r * self.hidden_h(msg))  # n:N, V, hid
        hidden = (1 - z) * n + z * hidden  # hidden: [N, V, hid]

        hidd = hidden
        hidd = self.dropout1(self.leaky_relu(self.out_fc1(hidd)))
        hidd = self.dropout2(self.leaky_relu(self.out_fc2(hidd)))
        pred = self.out_fc3(hidd)
        pred_ = x_last[:,:,:3] + pred  # pred_: [64, 21, 3]
        hidden = hidden.permute(0, 2, 1)  # hidden: [64, 256, 21] for next convolution
        return pred_, hidden  # pred: [64, 21, 3], hidden: [64, 256, 21]

    def forward(self, inputs, inputs_previous, inputs_previous2, hidden, t):  # inputs:[64, 1, 63];  hidden:[64, 256, 21]
        pred_all = []

        N, C, T, V = inputs.size()
        inputs = inputs.permute(0,2,3,1).contiguous().view(N, T, V, -1)  # [N, 1, V, 3]
        inputs_previous = inputs_previous.permute(0,2,3,1).contiguous().view(N, T, V, -1)
        inputs_previous2 = inputs_previous2.permute(0,2,3,1).contiguous().view(N, T, V, -1)

        for step in range(0, t):
            if step < 1:
                ins_p = inputs[:, 0, :, :]  # ins_p: [N, V, 3]
                ins_v = (inputs_previous - inputs_previous2)[:, 0, :, :]  # ins_v: [N, V, 3]
                ins_a = ins_p - inputs_previous[:, 0, :, :] - ins_v
            elif step == 1:
                ins_p = pred_all[step - 1]
                ins_v = (inputs - inputs_previous)[:, 0, :, :]
                ins_a = ins_p - inputs[:, 0, :, :] - ins_v
            elif step == 2:
                ins_p = pred_all[step - 1]
                ins_v = pred_all[step - 2] - inputs[:, 0, :, :]
                ins_a = ins_p - pred_all[step - 2] - ins_v
            else:
                ins_p = pred_all[step - 1]
                ins_v = pred_all[step - 2] - pred_all[step - 3]
                ins_a = ins_p - pred_all[step - 2] - ins_v

            n = torch.randn(ins_p.size()).cuda(inputs.get_device()) * 0.0005
            ins = torch.cat((ins_p + n, ins_v, ins_a), dim=-1)
            pred_, hidden = self.step_forward(ins, hidden)
            pred_all.append(pred_)  # [t, N, V, 3]

        preds = torch.stack(pred_all, dim=1)  # [N, T, V, C]

        return preds.permute(0, 3, 1, 2).contiguous()  # [N, C, T, V]


    def forward(self, feature, hidden):
        # N, C, T, V
        N, C, T, V = feature.size()
        feature = feature.permute(0, 2, 3, 1).view(N, T*V, C)
        msg = self.hid_gcn(hidden)
        msg = msg.permute(0, 2, 3, 1).view(N, T*V, C)
        hidden = hidden.permute(0, 2, 3, 1).view(N, T * V, C)

        r = torch.sigmoid(self.input_r(feature) + self.hidden_r(msg))  # r:N, V, hid
        z = torch.sigmoid(self.input_i(feature) + self.hidden_i(msg))  # z:N, V, hid
        n = torch.tanh(self.input_n(feature) + r * self.hidden_h(msg))  # n:N, V, hid
        hidden = (1 - z) * n + z * hidden  # hidden: [N, V, hid]
        hidden = hidden.view(N, T, V, C).permute(0,3,1,2)

        return hidden  # pred: [64, 21, 3], hidden: [64, 256, 21]


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        self.num_point = num_point
        self.num_bone = num_point-1
        A = self.graph.A
        B = self.graph.B
        C = self.graph.C
        D = self.graph.D
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn_b = nn.BatchNorm1d(num_person * in_channels * self.num_bone)

        # encoder
        self.l1 = jb_TCN_GCN_unit(3, 64, A, C, residual=False)
        self.l2 = jbf_TCN_GCN_unit(64, 64, A, B, C, D,)
        self.l3 = jbf_TCN_GCN_unit(64, 64, A, B, C, D,)
        self.l4 = jbf_TCN_GCN_unit(64, 64, A, B, C, D,)
        self.l5 = jb_TCN_GCN_unit(64, 128, A, C, stride=2)
        self.l6 = jbf_TCN_GCN_unit(128, 128, A, B, C, D)
        self.l7 = jbf_TCN_GCN_unit(128, 128, A, B, C, D)
        self.l8 = jb_TCN_GCN_unit(128, 256, A, C, stride=2)   # 75
        self.l9 = jb_TCN_GCN_unit(256, 256, A, C)
        self.l10 = jb_TCN_GCN_unit(256, 256, A, C)


        # classifier
        self.fc = nn.Linear(256, num_class)
        self.fc_b = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc_b.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        #self.drop_out = nn.Dropout(0.5)

        # predictor
        self.p1 = jb_TCN_GCN_unit(256, 256, A, C, stride=2)     # 39
        self.p2 = jb_TCN_GCN_unit(256, 256, A, C, stride=2)     # 19
        self.ggru = GGRU(9, 256, A)
        self.ggru_b = GGRU(9, 256, C)

    def forward(self, x, b, x_last, x_last_p1, x_last_p2, x_target, b_last, b_last_p1, b_last_p2, b_target):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        N_b, C_b, T_b, V_b, M_b = b.size()
        b = b.permute(0, 4, 3, 1, 2).contiguous().view(N_b, M_b * V_b * C_b, T_b)
        b = self.data_bn_b(b)
        b = b.view(N_b, M_b, V_b, C_b, T_b).permute(0, 1, 3, 4, 2).contiguous().view(N_b * M_b, C_b, T_b, V_b)

        x_last = x_last.permute(0, 4, 1, 2, 3).contiguous().view(-1, 3, 1, self.num_point)
        x_last_p1 = x_last_p1.permute(0, 4, 1, 2, 3).contiguous().view(-1, 3, 1, self.num_point)
        x_last_p2 = x_last_p2.permute(0, 4, 1, 2, 3).contiguous().view(-1, 3, 1, self.num_point)

        b_last = b_last.permute(0, 4, 1, 2, 3).contiguous().view(-1, 3, 1, self.num_bone)
        b_last_p1 = b_last_p1.permute(0, 4, 1, 2, 3).contiguous().view(-1, 3, 1, self.num_bone)
        b_last_p2 = b_last_p2.permute(0, 4, 1, 2, 3).contiguous().view(-1, 3, 1, self.num_bone)

        x, b = self.l1(x, b)
        x, b = self.l2(x, b)
        x, b = self.l3(x, b)
        x, b = self.l4(x, b)
        x, b = self.l5(x, b)
        x, b = self.l6(x, b)
        x, b = self.l7(x, b)
        x, b = self.l8(x, b)
        x, b = self.l9(x, b)
        x, b = self.l10(x, b)

        # N*M,C,T,V
        c_new = x.size(1)
        x_class = x.view(N, M, c_new, -1)
        x_class = x_class.mean(3).mean(1)
        b_class = b.view(N, M, c_new, -1)
        b_class = b_class.mean(3).mean(1)
        x_class = self.fc(x_class)
        b_class = self.fc_b(b_class)

        p, p_b = self.p1(x, b)
        p, p_b = self.p2(p, p_b)          # T=19
        x_pred = p.mean(2)      # N*M, C, V
        pred = self.ggru(x_last, x_last_p1, x_last_p2, x_pred, 10)
        x_target = x_target.permute(0, 4, 1, 2, 3).contiguous().view(-1, 3, 10, V)   # bs, 3, t, node_num

        b_pred = p_b.mean(2)  # N*M, C, V
        b_pred = self.ggru_b(b_last, b_last_p1, b_last_p2, b_pred, 10)
        b_target = b_target.permute(0, 4, 1, 2, 3).contiguous().view(-1, 3, 10, V_b)  # bs, 3, t, node_num

        return x_class, b_class, pred[::2], x_target[::2], b_pred[::2], b_target[::2]
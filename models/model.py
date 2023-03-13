from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

import math


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, is_residual=True, attention_activaion=None, return_atten=False):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.attention_activaion = attention_activaion
        self.is_residual = is_residual
        self.return_atten = return_atten
        self.linear_q = nn.Linear(self.dim_q, self.dim_k, bias=False)
        self.linear_k = nn.Linear(self.dim_q, self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_q, self.dim_v, bias=False)

        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        batch, seq_len, dim_q = x.shape
        assert dim_q == self.dim_q
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        if self.attention_activaion is not None:
            act = nn.ReLU()
            q = act(q)
            k = act(k)
            v = act(v)

        score = torch.matmul(q, k.transpose(-2, -1)) * self._norm_fact

        score = torch.softmax(score, dim=-1)
        att = torch.matmul(score, v)
        if self.is_residual:
            att += v

        if self.return_atten:
            return att, score
        return att


class Translater(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Translater, self).__init__()
        self.attention = SelfAttention(hidden_size, hidden_size, hidden_size, is_residual=True,
                                       attention_activaion='relu')
        self.linear1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.linear2 = nn.Linear(int(hidden_size / 2), 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        value = self.attention(input)
        value = self.linear1(value)

        value = self.linear2(value)
        return value


class BGL(nn.Module):

    def __init__(self, in1_features, in2_features, out_features, bias=(True, True, True)):
        super(BGL, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)
        self.linear_1 = nn.Linear(in_features=2 * self.in1_features + 1,
                                  out_features=self.out_features,
                                  bias=False)
        self.linear_2 = nn.Linear(in_features=2 * self.in1_features + 1,
                                  out_features=self.out_features,
                                  bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        U = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        W1 = np.zeros((self.out_features, 1 + 2 * self.in1_features), dtype=np.float32)
        W2 = np.zeros((self.out_features, 1 + 2 * self.in1_features), dtype=np.float32)

        self.linear.weight.data.copy_(torch.from_numpy(U))
        self.linear_1.weight.data.copy_(torch.from_numpy(W1))
        self.linear_2.weight.data.copy_(torch.from_numpy(W2))

    def forward(self, input1, input2):
        input1 = input1.unsqueeze(dim=1)
        input2 = input2.unsqueeze(dim=1)
        batch_size, _, dim1 = input1.size()
        batch_size, _, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, 1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
        if self.bias[1]:
            ones = input2.data.new(batch_size, 1, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)

        affine = self.linear(input1)

        affine = affine.view(batch_size, self.out_features, dim2)

        bgl = torch.transpose(torch.bmm(affine, input2), 1, 2)

        bgl = bgl.contiguous().view(batch_size, 1, 1, self.out_features)

        return bgl.squeeze(dim=1).squeeze(dim=1)


class BTDM(BertPreTrainedModel):
    def __init__(self, config):
        super(BTDM, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.w1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w3 = nn.Linear(self.hidden_size, self.hidden_size)

        self.s_classier = nn.Linear(self.hidden_size, 2)
        self.s_classier_from_o = Translater(self.hidden_size, config.hidden_dropout_prob)

        self.o_classier = nn.Linear(config.hidden_size, 2)
        self.o_classier_from_s = Translater(self.hidden_size, config.hidden_dropout_prob)
        self.bgl = BGL(config.hidden_size, config.hidden_size, config.num_p)

        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, token_ids, mask_token_ids, s2_mask, o2_mask, s3_mask, o3_mask):
        head, tail, rel, cls = self.get_embed(token_ids, mask_token_ids)
        s1_pred = self.s_pred(head, cls=cls)
        o1_pred = self.o_pred(tail, cls=cls)
        o2_pred = self.o_pred_from_s(head, tail, s2_mask, rel, cls)
        s2_pred = self.s_pred_from_o(head, tail, o2_mask, rel, cls)

        p_pred = self.p_pred(rel, s3_mask, o3_mask)

        return s1_pred, o1_pred, s2_pred, o2_pred, p_pred

    def get_embed(self, token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed = bert_out[0]
        head = self.w1(embed)
        tail = self.w2(embed)
        rel = self.w3(embed)
        cls = bert_out[1]
        head = head + tail[:, 0, :].unsqueeze(dim=1)
        tail = tail + head[:, 0, :].unsqueeze(dim=1)

        head, tail, rel, cls = self.dropout(head), self.dropout(tail), self.dropout(rel), self.dropout(cls)
        return head, tail, rel, cls

    def extract_entity(self, input, mask):
        '''
        取首尾平均
        :param input:BLH
        :param mask:BL
        :return: BH
        '''
        _, _, dim = input.shape
        entity = input * mask.unsqueeze(dim=-1)
        entity = entity.sum(dim=1) / mask.sum(dim=-1, keepdim=True)
        return entity

    def s_pred(self, head, cls):
        s_logist = self.s_classier(head + cls.unsqueeze(dim=1))
        s_pred = self.sigmoid(s_logist)
        return s_pred

    def o_pred(self, tail, cls):
        o_logist = self.o_classier(tail + cls.unsqueeze(dim=1))
        o_pred = self.sigmoid(o_logist)
        return o_pred

    def o_pred_from_s(self, head, tail, s_mask, rel, cls):
        s_entity = self.extract_entity(head, s_mask)
        s2o_embed = tail * s_entity.unsqueeze(dim=1) + rel
        o_logist = self.o_classier_from_s(s2o_embed + cls.unsqueeze(dim=1))
        o_pred = self.sigmoid(o_logist)
        return o_pred

    def s_pred_from_o(self, head, tail, o_mask, rel, cls):
        o_entity = self.extract_entity(tail, o_mask)
        o2s_embed = head * o_entity.unsqueeze(dim=1) + rel
        s_logist = self.s_classier_from_o(o2s_embed + cls.unsqueeze(dim=1))
        s_pred = self.sigmoid(s_logist)
        return s_pred

    def p_pred(self, rel, s_mask, o_mask):
        s_entity = self.extract_entity(rel, s_mask)
        o_entity = self.extract_entity(rel, o_mask)
        logist = self.bgl(s_entity, o_entity)
        r_pred = self.sigmoid(logist)
        return r_pred

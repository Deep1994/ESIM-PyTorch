# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:27:24 2020

@author: del
"""

import torch
import torch.nn as nn
from utils import get_mask, replace_masked, init_esim_weights
from layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention

"""
定义网络时，需要继承nn.Module，并实现它的forward方法。
把网络中具有可学习参数的层放在构造函数__init__中。
如果某一层(如ReLU)不具有可学习的参数，则既可以放在构造函数中，也可以不放，
但建议不放在其中，而在forward中使用nn.functional代替。
"""
class ESIM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embeding_dim,
                 hihdden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=2,
                 device="gpu"):
        
        super(ESIM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embeding_dim
        self.hidden_size = hihdden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        
        self.word_embedding = nn.Embedding(self.vocab_size,
                                           self.embedding_dim,
                                           padding_idx=padding_idx,
                                           _weight=embeddings)
        
        self.word_embedding.weight.requires_grad = False
        
        # 做batchnormalization之前需要先转换维度
        self.bn_embedding = nn.BatchNorm1d(self.embedding_dim)
        # self.bn_1 = nn.BatchNorm1d(2*4*self.hidden_size)
        # self.bn_2 = nn.BatchNorm1d(self.hidden_size)
        # self.bn_3 = nn.BatchNorm1d(self.hidden_size // 2)
        
        if self.dropout:
            self.rnn_dropout = RNNDropout(p=self.dropout)
        
        self.first_rnn = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim, 
                                        self.hidden_size, 
                                        bidirectional=True)
        
        self.projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                  self.hidden_size),
                                        nn.ReLU())
        
        self.attention = SoftmaxAttention()
        
        self.second_rnn = Seq2SeqEncoder(nn.LSTM,
                                        self.hidden_size, 
                                        self.hidden_size, 
                                        bidirectional=True)

        # self.linear_1 = nn.Sequential(nn.Linear(2*4*self.hidden_size,
        #                               self.hidden_size),
        #                               nn.ReLU())
        # self.linear_2 = nn.Sequential(nn.Linear(self.hidden_size,
        #                               self.hidden_size // 2),
        #                               nn.ReLU())

        # self.linear_3 = nn.Linear(self.hidden_size // 2,
        #                           self.num_classes)
        
        self.classification = nn.Sequential(nn.Linear(2*4*self.hidden_size,
                                                      self.hidden_size),
                                            nn.ReLU(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(self.hidden_size,
                                                      self.hidden_size // 2),
                                            nn.ReLU(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(self.hidden_size // 2,
                                                      self.num_classes))
        # Initialize all weights and biases in the model.
        self.apply(init_esim_weights)           
        
        
    def forward(self, q1, q1_lengths, q2, q2_lengths):
        
        q1_mask = get_mask(q1, q1_lengths).to(self.device)
        q2_mask = get_mask(q2, q2_lengths).to(self.device)
        
        # embed + BatchNormalization
        # q1_embed = self.bn_embedding(self.word_embedding(q1).transpose(1, 2).contiguous()).transpose(1, 2)
        # q2_embed = self.bn_embedding(self.word_embedding(q2).transpose(1, 2).contiguous()).transpose(1, 2)
        
        q1_embed = self.word_embedding(q1)
        q2_embed = self.word_embedding(q2)
        
        if self.dropout:
            q1_embed = self.rnn_dropout(q1_embed)
            q2_embed = self.rnn_dropout(q2_embed)
        
        # 双向lstm编码
        q1_encoded = self.first_rnn(q1_embed, q1_lengths)
        q2_encoded = self.first_rnn(q2_embed, q2_lengths)
        
        # atention
        q1_aligned, q2_aligned = self.attention(q1_encoded, q1_mask,
                                                q2_encoded, q2_mask)
        # concat
        q1_combined = torch.cat([q1_encoded,
                                 q1_aligned,
                                 q1_encoded - q1_aligned,
                                 q1_encoded * q1_aligned],
                                 dim=-1)
        q2_combined = torch.cat([q2_encoded,
                                 q2_aligned,
                                 q2_encoded - q2_aligned,
                                 q2_encoded * q2_aligned],
                                 dim=-1)
        
        # 映射一下
        projected_q1 = self.projection(q1_combined)
        projected_q2 = self.projection(q2_combined)
        
        if self.dropout:
            projected_q1 = self.rnn_dropout(projected_q1)
            projected_q2 = self.rnn_dropout(projected_q2)
        
        # 再次经过双向RNN
        q1_compare = self.second_rnn(projected_q1, q1_lengths)
        q2_compare = self.second_rnn(projected_q2, q2_lengths)
        
        # 平均池化 + 最大池化
        q1_avg_pool = torch.sum(q1_compare * q1_mask.unsqueeze(1).transpose(2, 1), dim=1)\
                                / torch.sum(q1_mask, dim=1, keepdim=True)
        q2_avg_pool = torch.sum(q2_compare * q2_mask.unsqueeze(1).transpose(2, 1), dim=1)\
                                / torch.sum(q2_mask, dim=1, keepdim=True)

        q1_max_pool, _ = replace_masked(q1_compare, q1_mask, -1e7).max(dim=1)
        q2_max_pool, _ = replace_masked(q2_compare, q2_mask, -1e7).max(dim=1)
        
        # 拼接成最后的特征向量
        merged = torch.cat([q1_avg_pool, q1_max_pool, q2_avg_pool, q2_max_pool], dim=1)
        
        # 分类
        # dense = self.bn_1(merged)
        # dense = self.linear_1(dense)
        # dense = self.bn_2(dense)
        # dense = nn.Dropout(p=self.dropout)(dense)
        # dense = self.linear_2(dense)
        # dense = self.bn_3(dense)
        # dense = nn.Dropout(p=self.dropout)(dense)
        # logits = self.linear_3(dense)

        # probabilities = nn.functional.softmax(dense, dim=-1)

        logits = self.classification(merged)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


        
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
import tensorflow.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable

from iep.models.layers import ResidualBlock
from iep.embedding import expand_embedding_vocab


class StackedAttention(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim):
        super(StackedAttention, self).__init__()
        self.Wv = tf.keras.layers.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1), padding=0)
        self.Wu = nn.Linear(input_dim, hidden_dim)
        self.Wp = nn.Conv2d(hidden_dim, 1, kernel_size=(1, 1), padding=0)
        self.hidden_dim = hidden_dim
        self.attention_maps = None

    def call(self, v, u):
        """
        Input:
        - v: N x D x H x W
        - u: N x D

        Returns:
        - next_u: N x D
        """
        N, K = v.size(0), self.hidden_dim
        D, H, W = v.size(1), v.size(2), v.size(3)
        v_proj = self.Wv(v)  # N x K x H x W
        u_proj = self.Wu(u)  # N x K
        u_proj_expand = u_proj.reshape(N, K, 1, 1).expand(N, K, H, W)
        h = nn.tanh(v_proj + u_proj_expand)
        p = nn.softmax(self.Wp(h).reshape(N, H * W)).reshape(N, 1, H, W)
        self.attention_maps = tf.identity(p.data)

        v_tilde = (p.expand_as(v) * v).sum(2).sum(3).reshape(N, D)
        next_u = u + v_tilde
        return next_u


class LstmEncoder(tf.keras.Model):
    def __init__(self, token_to_idx, wordvec_dim=300,
                 rnn_dim=256, rnn_num_layers=2, rnn_dropout=0):
        super(LstmEncoder, self).__init__()
        self.token_to_idx = token_to_idx
        self.NULL = token_to_idx['<NULL>']
        self.START = token_to_idx['<START>']
        self.END = token_to_idx['<END>']

        self.embed = tf.get_variable("embed", [len(token_to_idx), wordvec_dim])
        self.embed = nn.Embedding(len(token_to_idx), wordvec_dim)
        self.rnn = nn.LSTM(wordvec_dim, rnn_dim, rnn_num_layers,
                           dropout=rnn_dropout, batch_first=True)  # CHANGE

    def expand_vocab(self, token_to_idx, word2vec=None, std=0.01):
        expand_embedding_vocab(self.embed, token_to_idx,
                               word2vec=word2vec, std=std)

    def call(self, x):
        N, T = x.size()
        idx = tf.Tensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence
        x_cpu = x.data.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data).long()
        idx = tf.Variable(idx)

        hs, _ = self.rnn(self.embed(x))
        idx = idx.reshape(N, 1, 1).expand(N, 1, hs.size(2))
        H = hs.size(2)
        return hs.gather(1, idx).reshape(N, H)

def build_cnn(feat_dim=(1024, 14, 14),
              res_block_dim=128,
              num_res_blocks=0,
              proj_dim=512,
              pooling='maxpool2'):
    C, H, W = feat_dim
    layers = []
    if num_res_blocks > 0:
        layers.append(tf.keras.layers.Conv2d(C, res_block_dim, kernel_size=(3, 3), padding=1))
        layers.append(nn.ReLU())
        C = res_block_dim
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(C))
    if proj_dim > 0:
        layers.append(tf.keras.layers.Conv2d(C, proj_dim, kernel_size=(1, 1), padding=0))
        layers.append(nn.ReLU())
        C = proj_dim
    if pooling == 'maxpool2':
        layers.append(tf.keras.layers.MaxPool2d(kernel_size=(2, 2), stride=2))
        H, W = H // 2, W // 2
    return nn.Sequential(*layers), (C, H, W)

def build_mlp(input_dim, hidden_dims, output_dim,
              use_batchnorm=False, dropout=0):
    layers = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(tf.keras.layers.batch_normalization(input_dim))
    for dim in hidden_dims:
        layers.append(nn.Linear(D, dim))  # CHANGE
        if use_batchnorm:
            layers.append(tf.keras.layers.batch_normalization(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(tf.keras.layers.batch_normalization.ReLU())
        D = dim
    layers.append(nn.Linear(D, output_dim))  # CHANGE
    return tf.keras.layers.Sequential(layers)


class LstmModel(tf.keras.Model):
    def __init__(self, vocab,
                 rnn_wordvec_dim=300, rnn_dim=256, rnn_num_layers=2, rnn_dropout=0,
                 fc_use_batchnorm=False, fc_dropout=0, fc_dims=(1024,)):
        rnn_kwargs = {
            'token_to_idx': vocab['question_token_to_idx'],
            'wordvec_dim': rnn_wordvec_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }
        self.rnn = LstmEncoder(**rnn_kwargs)

        classifier_kwargs = {
            'input_dim': rnn_dim,
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answer_token_to_idx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }
        self.classifier = build_mlp(**classifier_kwargs)

    def forward(self, questions, feats):
        q_feats = self.rnn(questions)
        scores = self.classifier(q_feats)
        return scores


class CnnLstmModel(tf.keras.Model):
    def __init__(self, vocab,
                 rnn_wordvec_dim=300, rnn_dim=256, rnn_num_layers=2, rnn_dropout=0,
                 cnn_feat_dim=(1024, 14, 14),
                 cnn_res_block_dim=128, cnn_num_res_blocks=0,
                 cnn_proj_dim=512, cnn_pooling='maxpool2',
                 fc_dims=(1024,), fc_use_batchnorm=False, fc_dropout=0):
        rnn_kwargs = {
            'token_to_idx': vocab['question_token_to_idx'],
            'wordvec_dim': rnn_wordvec_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }
        self.rnn = LstmEncoder(**rnn_kwargs)

        cnn_kwargs = {
            'feat_dim': cnn_feat_dim,
            'res_block_dim': cnn_res_block_dim,
            'num_res_blocks': cnn_num_res_blocks,
            'proj_dim': cnn_proj_dim,
            'pooling': cnn_pooling,
        }
        self.cnn, (C, H, W) = build_cnn(**cnn_kwargs)

        classifier_kwargs = {
            'input_dim': C * H * W + rnn_dim,
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answer_token_to_idx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }
        self.classifier = build_mlp(**classifier_kwargs)

    def forward(self, questions, feats):
        N = questions.size(0)
        assert N == feats.size(0)
        q_feats = self.rnn(questions)
        img_feats = self.cnn(feats)
        cat_feats = tf.concat([q_feats, img_feats.reshape(N, -1)], 1)
        scores = self.classifier(cat_feats)
        return scores


class CnnLstmSaModel(tf.keras.Model):
    def __init__(self, vocab,
                 rnn_wordvec_dim=300, rnn_dim=256, rnn_num_layers=2, rnn_dropout=0,
                 cnn_feat_dim=(1024, 14, 14),
                 stacked_attn_dim=512, num_stacked_attn=2,
                 fc_use_batchnorm=False, fc_dropout=0, fc_dims=(1024,)):

        rnn_kwargs = {
            'token_to_idx': vocab['question_token_to_idx'],
            'wordvec_dim': rnn_wordvec_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }
        self.rnn = LstmEncoder(**rnn_kwargs)

        C, H, W = cnn_feat_dim
        self.image_proj = nn.Conv2d(C, rnn_dim, kernel_size=(1, 1), padding=0)
        self.stacked_attns = []
        for i in range(num_stacked_attn):
            sa = StackedAttention(rnn_dim, stacked_attn_dim)
            self.stacked_attns.append(sa)
            self.add_module('stacked-attn-%d' % i, sa)

        classifier_args = {
            'input_dim': rnn_dim,
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answer_token_to_idx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }
        self.classifier = build_mlp(**classifier_args)

    def forward(self, questions, feats):
        u = self.rnn(questions)
        v = self.image_proj(feats)

        for sa in self.stacked_attns:
            u = sa(v, u)

        scores = self.classifier(u)
        return scores


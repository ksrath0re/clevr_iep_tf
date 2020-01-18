import json
import tensorflow as tf


# !/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import tensorflow as tf
from iep.models import LstmModel, CnnLstmModel, CnnLstmSaModel


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>']  == 2
    return vocab

#TODO find replacement of loac_cpu one
def load_cpu(path):
    """
    Loads a checkpoint, remapping all Tensors to CPU
    """
    saver = tf.train.Saver()

    with tf.Session() as sess:
        return saver.restore(sess, path)


def load_baseline(path):
    model_cls_dict = {
        'LSTM': LstmModel,
        'CNN+LSTM': CnnLstmModel,
        'CNN+LSTM+SA': CnnLstmSaModel,
    }
    checkpoint = load_cpu(path)
    baseline_type = checkpoint['baseline_type']
    kwargs = checkpoint['baseline_kwargs']
    state = checkpoint['baseline_state']

    model = model_cls_dict[baseline_type](**kwargs)
    model.load_state_dict(state)
    return model, kwargs

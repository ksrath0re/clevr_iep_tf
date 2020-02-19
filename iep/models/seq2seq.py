#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf
from iep.embedding import expand_embedding_vocab
import numpy as np


class Encoder(tf.keras.Model):
    def __init__(
            self,
            encoder_vocab_size,
            wordvec_dim,
            hidden_dim,
            rnn_dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Embedding(
            encoder_vocab_size, wordvec_dim)
        self.encoder_rnn1 = tf.keras.layers.LSTM(
            hidden_dim, dropout=rnn_dropout, return_sequences=True, return_state=True)
        self.encoder_rnn2 = tf.keras.layers.LSTM(
            hidden_dim, dropout=rnn_dropout, return_sequences=True, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        x, state_h, state_c = self.encoder_rnn1(x)
        #x = self.encoder_rnn1(x)
        #print("size of output : ", len(x))
        #for i, item in enumerate(x):
        #    print("Shape of ", i, " : ", item.shape)
        state = [state_h, state_c]
        output, state_h, state_c = self.encoder_rnn2(x, initial_state=state)
        return output, state_h, state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.hidden_dim, self.hidden_dim))
        # return tf.eye(self.hidden_dim, batch_shape=[64])


class Decoder(tf.keras.Model):
    def __init__(
            self,
            decoder_vocab_size,
            wordvec_dim,
            hidden_dim,
            rnn_dropout):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.decoder_embedding = tf.keras.layers.Embedding(decoder_vocab_size, wordvec_dim)
        self.decoder_rnn1 = tf.keras.layers.LSTM(hidden_dim, dropout=rnn_dropout, return_sequences=True, return_state=True)
        self.decoder_rnn2 = tf.keras.layers.LSTM(hidden_dim, dropout=rnn_dropout, return_sequences=True, return_state=True)
        self.decoder_linear = tf.keras.layers.Dense(decoder_vocab_size, input_shape=(hidden_dim,))

    def call(self, hidden, y, encoded, N, H, T_out, V_out):
        print("T_out : ", T_out)
        y_embed = self.decoder_embedding(y)
        print("y_embed ka shape after embedding : ", y_embed.shape)
        y_embed = tf.reshape(y_embed, [y_embed.shape[0], -1, y_embed.shape[-1]])
        # encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
        print("Encoded ka shape : ", encoded.shape)
        encoded_repeat = tf.broadcast_to(tf.reshape(encoded, [N, 1, H]), [N, T_out, H])
        print("Encoded Repeat ka shape : ", encoded_repeat.shape, " y_embed ka shape : ", y_embed.shape)
        rnn_input = tf.concat([encoded_repeat, y_embed], 2)

        x, state_h, state_c = self.decoder_rnn1(rnn_input)
        state = [state_h, state_c]
        output, state_h, state_c = self.decoder_rnn2(x, initial_state=state)
        output_2d = tf.reshape(output, [N*T_out, H])
        output_logprobs = tf.reshape(self.decoder_linear(output_2d), [N, T_out, V_out])
        return output_logprobs, state_h, state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.hidden_dim, self.hidden_dim))
        # return tf.eye(self.hidden_dim, batch_shape=[64])



class Seq2Seq(tf.keras.Model):
    def __init__(self,
                 encoder_vocab_size=100,
                 decoder_vocab_size=100,
                 wordvec_dim=300,
                 hidden_dim=256,
                 rnn_num_layers=2,
                 rnn_dropout=0,
                 null_token=0,
                 start_token=1,
                 end_token=2,
                 encoder_embed=None
                 ):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = rnn_num_layers
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.wordvec_dim = wordvec_dim
        self.rnn_dropout = rnn_dropout
        self.encoder_embed2 = tf.keras.layers.Embedding(
            encoder_vocab_size, wordvec_dim)
        input_tensor = tf.keras.Input(shape=(46, 300))
        input_shape = [64, 46, 300]

        self.decoder_embed2 = tf.keras.layers.Embedding(
             decoder_vocab_size, wordvec_dim)
        # self.decoder_rnn = tf.keras.layers.LSTM(hidden_dim, dropout=rnn_dropout) for _ in range(rnn_num_layers)
        # self.decoder_rnn = tf.keras.layers.StackedRNNCells(
        #   decoder_cells, input_shape=(wordvec_dim + hidden_dim,))
        #    LSTM(wordvec_dim + hidden_dim, hidden_dim, rnn_num_layers,
        # dropout=rnn_dropout, batch_first=True)
        # self.decoder_linear = tf.keras.layers.Dense(
        #     decoder_vocab_size, input_shape=(hidden_dim,))
        self.NULL = null_token
        self.START = start_token
        self.END = end_token
        self.multinomial_outputs = None

    def expand_encoder_vocab(self, token_to_idx, word2vec=None, std=0.01):
        expand_embedding_vocab(self.encoder_embed, token_to_idx,
                               word2vec=word2vec, std=std)

    def get_dims(self, x=None, y=None):
        V_in = self.encoder_embed2.input_dim
        V_out = self.decoder_embed2.input_dim
        D = self.encoder_embed2.output_dim
        H = self.hidden_size
        L = self.num_layers

        N = tf.shape(x)[0] if x is not None else None
        N = tf.shape(y)[0] if N is None and y is not None else N
        T_in = tf.shape(x)[1] if x is not None else None
        #T_out = tf.shape(tf.reshape(y, (1, -1)))[1] if y is not None else None
        T_out = tf.shape(y)[1] if y is not None else None
        return V_in, V_out, D, H, L, N, T_in, T_out

    def before_rnn(self, x, replace=0):
        # TODO: Use PackedSequence instead of manually plucking out the last
        # non-NULL entry of each sequence; it is cleaner and more efficient.
        N, T = tf.shape(x)
        idx = tf.dtypes.cast(tf.fill([N], (T - 1)), dtype=tf.int64)
        idx_np = idx.numpy()
        # print("Idx np size :", len(idx_np))

        # Find the last non-null element in each sequence. Is there a clean
        # way to do this?
        for i in range(N):
            for t in range(T - 1):
                if x[i, t] != self.NULL and x[i, t + 1] == self.NULL:
                    idx_np[i] = t
                    break
        idx = tf.convert_to_tensor(idx_np, dtype=tf.int64)
        idx = tf.dtypes.cast(idx, dtype=tf.int64)
        # print("Idx np tensor size :", idx.shape)
        x_np = x.read_value().numpy()
        for i, v in enumerate(x_np):
            if (v.any() == 0):
                x_np[i] = 0
        x = tf.Variable(tf.convert_to_tensor(x_np, dtype=tf.float32))
        return x, tf.Variable(idx)

    def encoder(self, x):
        print("Shape of X :  ", x.shape)
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)
        print("Shape of X after before_rnn:", x.shape)
        encoder_ob = Encoder(
            self.encoder_vocab_size,
            self.wordvec_dim,
            self.hidden_size,
            self.rnn_dropout)
        hidden = encoder_ob.initialize_hidden_state()
        out, _, _ = encoder_ob(x, hidden)

        # Pull out the hidden state for the last non-null value in each input
        print("idx Shape :", idx.shape, " out shape : ", out.shape)
        idx = tf.broadcast_to(tf.reshape(idx, [N, 1, 1]), [N, 1, H])
        print("idx Shape after broadcast ", idx.shape)
        t = gather_numpy(out, 1, idx)
        print(" t ka shape : ", t.shape)
        
        return tf.reshape(t, [N, H])

    def decoder(self, encoded, y, h0=None, c0=None):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
        print("T_out in decoder starting : ", T_out)
        print("Shape of Y after before_rnn:", y.shape)
        if T_out > 1:
            y, _ = self.before_rnn(y)
        print("Shape of Y after before_rnn:", y.shape)

        # if h0 is None:
        #     h0 = tf.Variable(tf.zeros([L, N, H], encoded.dtype))
        # if c0 is None:
        #     c0 = tf.Variable(tf.zeros([L, N, H], encoded.dtype))
        decoder_ob = Decoder(
            self.decoder_vocab_size,
            self.wordvec_dim,
            self.hidden_size,
            self.rnn_dropout)
        hidden = decoder_ob.initialize_hidden_state()
        output_logprobs, ht, ct = decoder_ob(hidden, y, encoded, N, H, T_out, V_out)

        return output_logprobs, ht, ct

    def compute_loss(self, output_logprobs, y):
        """
    Compute loss. We assume that the first element of the output sequence y is
    a start token, and that each element of y is left-aligned and right-padded
    with self.NULL out to T_out. We want the output_logprobs to predict the
    sequence y, shifted by one timestep so that y[0] is fed to the network and
    then y[1] is predicted. We also don't want to compute loss for padded
    timesteps.

    Inputs:
    - output_logprobs: Variable of shape (N, T_out, V_out)
    - y: LongTensor Variable of shape (N, T_out)
    """
        self.multinomial_outputs = None
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
        mask = y.data != self.NULL
        y_mask_tf = tf.dtypes.cast(tf.fill([N, T_out], 0), dtype=mask.dtype)
        y_mask = tf.Variable(y_mask_tf)
        y_mask[:, 1:] = mask[:, 1:]
        y_masked = y[y_mask]
        out_mask_tf = tf.dtypes.cast(tf.fill([N, T_out], 0), dtype=mask.dtype)
        out_mask = tf.Variable(out_mask_tf)
        out_mask[:, :-1] = mask[:, 1:]
        out_mask = out_mask.view(N, T_out, 1).expand(N, T_out, V_out)
        out_masked = output_logprobs[out_mask].reshape(-1, V_out)
        loss = tf.nn.softmax_cross_entropy_with_logits(out_masked, y_masked)
        return loss

    def __call__(self, x, y):
        encoded = self.encoder(x)
        print("Encoded ka shape : ", encoded.shape)
        output_logprobs, _, _ = self.decoder(encoded, y)
        print("output logprobs ka shape :", output_logprobs.shape)
        loss = self.compute_loss(output_logprobs, y)
        return loss

    def sample(self, x, max_length=50):
        # TODO: Handle sampling for minibatch inputs
        # TODO: Beam search?
        self.multinomial_outputs = None
        assert x.shape(0) == 1, "Sampling minibatches not implemented"
        encoded = self.encoder(x)
        y = [self.START]
        h0, c0 = None, None
        while True:
            cur_y_tf = tf.dtypes.cast(tf.convert_to_tensor(
                [y[-1]]).reshape(1, 1), dtype=x.data.dtype)
            cur_y = tf.Variable(cur_y_tf)
            logprobs, h0, c0 = self.decoder(encoded, cur_y, h0=h0, c0=c0)
            _, next_y = logprobs.data.max(2)
            y.append(next_y[0, 0, 0])
            if len(y) >= max_length or y[-1] == self.END:
                break
        return y

    def reinforce_sample(
            self,
            x,
            max_length=30,
            temperature=1.0,
            argmax=False):
        N, T = x.shape(0), max_length
        encoded = self.encoder(x)
        y = tf.dtypes.cast(tf.fill([N, T], self.NULL), dtype=tf.int64)
        done = tf.dtypes.cast(tf.fill([N], 0), dtype=tf.int8)
        cur_input = tf.Variable(tf.dtypes.cast(
            tf.fill([N, 1], self.START), dtype=x.data.dtype))
        h, c = None, None
        self.multinomial_outputs = []
        self.multinomial_probs = []
        for t in range(T):
            # logprobs is N x 1 x V
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            logprobs = logprobs / temperature
            probs = tf.nn.softmax(logprobs.view(N, -1))  # Now N x V
            if argmax:
                _, cur_output = probs.max(1)
            else:
                cur_output = probs.multinomial()  # Now N x 1
            self.multinomial_outputs.append(cur_output)
            self.multinomial_probs.append(probs)
            cur_output_data = cur_output.data
            not_done = logical_not(done)
            y[:, t][not_done] = cur_output_data[not_done]
            done = logical_or(done, cur_output_data.cpu() == self.END)
            cur_input = cur_output
            if done.sum() == N:
                break
        return tf.Variable(tf.dtypes.cast(y, dtype=x.data.dtype))

    def reinforce_backward(self, reward, output_mask=None):  # TODO make necessary changes
        """
    If output_mask is not None, then it should be a FloatTensor of shape (N, T)
    giving a multiplier to the output.
    """
        assert self.multinomial_outputs is not None, 'Must call reinforce_sample first'
        grad_output = []

        def gen_hook(mask):
            def hook(grad):
                return grad * mask.contiguous().view(-1, 1).expand_as(grad)

            return hook

        if output_mask is not None:
            for t, probs in enumerate(self.multinomial_probs):
                mask = tf.Variable(output_mask[:, t])
                probs.register_hook(gen_hook(mask))

        for sampled_output in self.multinomial_outputs:
            sampled_output.reinforce(reward)
            grad_output.append(None)
        # torch.autograd.backward(self.multinomial_outputs, grad_output,
        # retain_variables=True) #CHANGE

def gather_numpy(t, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = t.shape[:dim] + t.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    data_swaped = np.swapaxes(t, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    #gathered = np.choose(index_swaped, data_swaped)
    gathered = np.take_along_axis(data_swaped, index_swaped, axis=0)
    return np.swapaxes(gathered, 0, dim)


def logical_and(x, y):
    return x * y


def logical_or(x, y):
    return (x + y).clamp_(0, 1)


def logical_not(x):
    return x == 0

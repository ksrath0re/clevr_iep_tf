#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf
from iep.embedding import expand_embedding_vocab
import numpy as np
import tensorflow_probability as tfp


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
            hidden_dim,
            dropout=rnn_dropout,
            recurrent_dropout=rnn_dropout,
            return_sequences=True,
            return_state=True)
        self.encoder_rnn2 = tf.keras.layers.LSTM(
            hidden_dim,
            dropout=rnn_dropout,
            recurrent_dropout=rnn_dropout,
            return_sequences=True,
            return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        #print("embed : ", x)
        x, state_h, state_c = self.encoder_rnn1(x)
        state = [state_h, state_c]
        output, state_h, state_c = self.encoder_rnn2(x, initial_state=state)
        #print("out : ", output)
        return output, state_h, state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.hidden_dim, self.hidden_dim))


class Decoder(tf.keras.Model):
    def __init__(
            self,
            decoder_vocab_size,
            wordvec_dim,
            hidden_dim,
            rnn_dropout):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.decoder_embedding = tf.keras.layers.Embedding(
            decoder_vocab_size, wordvec_dim)
        self.decoder_rnn1 = tf.keras.layers.LSTM(
            hidden_dim,
            dropout=rnn_dropout,
            return_sequences=True,
            return_state=True)
        self.decoder_rnn2 = tf.keras.layers.LSTM(
            hidden_dim,
            dropout=rnn_dropout,
            return_sequences=True,
            return_state=True)
        self.decoder_linear = tf.keras.layers.Dense(
            decoder_vocab_size, input_shape=(hidden_dim,))

    def call(self, hidden, y, encoded, N, H, T_out, V_out):
        y_embed = self.decoder_embedding(y)
        y_embed = tf.reshape(
            y_embed, [y_embed.shape[0], -1, y_embed.shape[-1]])
        encoded_repeat = tf.broadcast_to(
            tf.reshape(
                encoded, [
                    N, 1, H]), [
                N, T_out, H])
        rnn_input = tf.concat([encoded_repeat, y_embed], 2)

        x, state_h, state_c = self.decoder_rnn1(rnn_input)
        state = [state_h, state_c]
        output, state_h, state_c = self.decoder_rnn2(x, initial_state=state)
        output_2d = tf.reshape(output, [N * T_out, H])
        output_logprobs = tf.reshape(
            self.decoder_linear(output_2d), [
                N, T_out, V_out])
        return output_logprobs, state_h, state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.hidden_dim, self.hidden_dim))


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

        # Find the last non-null element in each sequence. Is there a clean
        # way to do this?
        for i in range(N):
            for t in range(T - 1):
                if x[i, t] != self.NULL and x[i, t + 1] == self.NULL:
                    idx_np[i] = t
                    break
        idx = tf.convert_to_tensor(idx_np, dtype=tf.int64)
        #idx = tf.dtypes.cast(idx, dtype=x.dtype)
        #x_np = x.read_value().numpy()
        #print("x before condition : ", x)
        #for i, v in enumerate(x_np):
        #    if (v.any() == 0):
        #        x_np[i] = 0
        #print("x after condition : ", x_np)
        #x = tf.Variable(tf.convert_to_tensor(x_np, dtype=tf.int64))
        #print("x after condition : ", x)
        #print("idx : ", idx)
        return x, tf.Variable(idx)

    def encoder(self, x):
        #print("Shape of X :  ", x.shape)
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)
        #print("Shape of X after before_rnn:", x.shape)
        encoder_ob = Encoder(
            self.encoder_vocab_size,
            self.wordvec_dim,
            self.hidden_size,
            self.rnn_dropout)
        hidden = encoder_ob.initialize_hidden_state()
        out, _, _ = encoder_ob(x, hidden)
        #print("out : ", out)
        # Pull out the hidden state for the last non-null value in each input
        idx = tf.broadcast_to(tf.reshape(idx, [N, 1, 1]), [N, 1, H])
        #print("idx value ", idx)
        t = tf.convert_to_tensor(np.take_along_axis(out.numpy(), idx.numpy(), axis=1))
        #t = gather_numpy(out, 1, idx)
        #print("t : ", t)
        return tf.reshape(t, [N, H])

    def decoder(self, encoded, y, h0=None, c0=None):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
        if T_out > 1:
            y, _ = self.before_rnn(y)
        decoder_ob = Decoder(
            self.decoder_vocab_size,
            self.wordvec_dim,
            self.hidden_size,
            self.rnn_dropout)
        hidden = decoder_ob.initialize_hidden_state()
        output_logprobs, ht, ct = decoder_ob(
            hidden, y, encoded, N, H, T_out, V_out)

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
        n = y.read_value().numpy()
        mask = tf.convert_to_tensor(np.where(n != 0, 1, 0), dtype=tf.int32)
        y_mask_tf = tf.dtypes.cast(tf.fill([N, T_out], 0), dtype=mask.dtype)
        y_mask = tf.Variable(y_mask_tf)
        y_mask = y_mask[:, 1:].assign(mask[:, 1:])
        y_masked = tf.boolean_mask(y, y_mask)
        out_mask_tf = tf.dtypes.cast(tf.fill([N, T_out], 0), dtype=mask.dtype)
        out_mask = tf.Variable(out_mask_tf)
        out_mask = out_mask[:, :-1].assign(mask[:, 1:])
        out_mask = tf.broadcast_to(
            tf.reshape(
                out_mask, [
                    N, T_out, 1]), [
                N, T_out, V_out])
        output_logprobs = tf.boolean_mask(output_logprobs, out_mask)
        out_masked = tf.reshape(output_logprobs, [-1, V_out])
        y_masked = tf.dtypes.cast(y_masked, dtype=tf.int32)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_masked, labels=y_masked))

        return loss

    def __call__(self, x, y):
        encoded = self.encoder(x)
        #print("Encoded ka type : ", type(encoded))
        output_logprobs, _, _ = self.decoder(encoded, y)

        loss = self.compute_loss(output_logprobs, y)
        return loss

    def sample(self, x, max_length=50):
        # TODO: Handle sampling for minibatch inputs
        # TODO: Beam search?
        self.multinomial_outputs = None
        print("x : ", x)
        x = tf.dtypes.cast(x, dtype=tf.int32)
        #assert x.shape(0) == 1, "Sampling minibatches not implemented"
        encoded = self.encoder(x)
        print("encoded :", encoded)
        y = [self.START]
        h0, c0, i  = None, None, 0
        while True:
            cur_y_tf = tf.reshape(tf.dtypes.cast(tf.convert_to_tensor([y[-1]]), dtype=x.dtype), [1, 1])
            cur_y = tf.Variable(cur_y_tf)
            if i == 0 : print("cur_y ", cur_y)
            logprobs, h0, c0 = self.decoder(encoded, cur_y, h0=h0, c0=c0)
            #_, next_y = logprobs.max(2)
            if i ==0 : print("logprobs : ", logprobs)
            next_y = tf.math.argmax(logprobs, axis=2, output_type=tf.dtypes.int32)
            next_y = tf.reshape(next_y, [1, 1, -1])
            if i == 0 : print("next_y : ", next_y.shape, next_y)
            y.append(next_y.numpy()[0, 0, 0])
            i = 1
            if len(y) >= max_length or y[-1] == self.END:
                break
        return y

    def reinforce_sample(self, x, max_length=30, temperature=1.0, argmax=False):
        N, T = tf.shape(x)[0], max_length
        #print("N, T  : ", N, T)
        encoded = self.encoder(x)
        #print("encoded shape : ", encoded.shape)
        y = tf.dtypes.cast(tf.fill([N, T], self.NULL), dtype=tf.int64)
        done = tf.dtypes.cast(tf.fill([N], 0), dtype=tf.int8)
        #print("y shape : ", y.shape," done shape : ", done.shape)
        cur_input = tf.Variable(tf.dtypes.cast(tf.fill([N, 1], self.START), dtype=x.dtype))
        #print("cur_iput shape : ", cur_input.shape)
        h, c = None, None
        self.multinomial_outputs = []
        self.multinomial_probs = []
        for t in range(T):
            # logprobs is N x 1 x V
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            #print("logprobs shape and type: ", logprobs.shape, logprobs.dtype)
            logprobs = logprobs / temperature
            probs = tf.nn.softmax(tf.reshape(logprobs, [N, -1]))  # Now N x V
            #print("probs shape and type: ", probs.shape, probs.dtype)
            if argmax:
                _, cur_output = probs.max(1)
            else:
                probs_np = probs.numpy().astype(float)
                probs_np /= probs_np.sum()
                cur_output = tf.convert_to_tensor(np.random.multinomial(1, probs_np))  # Now N x 1
            #print("cur_output shape and type: ", cur_output.shape, cur_output.dtype)
            self.multinomial_outputs.append(cur_output)
            self.multinomial_probs.append(probs)
            cur_output_data = cur_output
            not_done = logical_not(done)
            if isinstance(y,  np.ndarray):
                continue
            else:
                y = y.numpy()
            not_done = not_done.numpy()
            cur_output_data = cur_output_data.numpy()
            y[:, t][not_done] = cur_output_data[not_done]
            #y = y[:, t][not_done].assign(cur_output_data[not_done])
            done = logical_or(done, tf.convert_to_tensor(np.where(cur_output_data == self.END, 1, 0), dtype=tf.int8))
            cur_input = tf.reshape(cur_output, [-1, 1])
            if done.numpy().sum() == N:
                break
        return tf.Variable(tf.convert_to_tensor(y, dtype=x.dtype))

    def reinforce_backward(self, reward, output_mask=None):  # TODO make necessary changes
        """
    If output_mask is not None, then it should be a FloatTensor of shape (N, T)
    giving a multiplier to the output.
    """
        assert self.multinomial_outputs is not None, 'Must call reinforce_sample first'
        grad_output = []
        print("in reinforce backward --- ")
        # def gen_hook(mask):
        #     def hook(grad):
        #         return grad * mask.contiguous().view(-1, 1).expand_as(grad)
        #
        #     return hook

        # if output_mask is not None:
        #     for t, probs in enumerate(self.multinomial_probs):
        #         mask = tf.Variable(output_mask[:, t])
        #         probs.register_hook(gen_hook(mask))

        N = len(self.multinomial_outputs)
        for sampled_output in self.multinomial_probs:
            print("sampled output : ", sampled_output)
            sample_output = tf.dtypes.cast(sampled_output, dtype=tf.float32)
            m = tfp.distributions.Multinomial(N, probs=sampled_output)
            loss = m.log_prob(sampled_output) * reward

            #grad_output.append(None)
        return loss, self.multinomial_outputs
        #torch.autograd.backward(self.multinomial_outputs, grad_output, retain_variables=True) #CHANGE


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
        raise ValueError(
            "Except for dimension " +
            str(dim) +
            ", all dimensions of index and self should be the same size")
    data_swaped = np.swapaxes(t, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    #gathered = np.choose(index_swaped, data_swaped)
    gathered = np.take_along_axis(data_swaped, index_swaped, axis=0)
    return np.swapaxes(gathered, 0, dim)


def logical_and(x, y):
    return x * y


def logical_or(x, y):
    return tf.clip_by_value(tf.dtypes.cast((x + y), dtype=tf.int32), clip_value_min=0, clip_value_max=1)


def logical_not(x):
    return x == 0

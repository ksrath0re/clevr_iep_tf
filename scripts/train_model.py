#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

import argparse
import json
import random
import shutil

import numpy as np
import h5py
import tensorflow as tf
import iep.utils as utils
import iep.preprocess
from iep.data import ClevrDataLoader
from iep.models.seq2seq import Seq2Seq
from iep.models.module_net import ModuleNet
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint

parser = argparse.ArgumentParser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Input data
parser.add_argument('--train_question_h5', default='data/train_questions.h5')
parser.add_argument('--train_features_h5', default='data/train_features.h5')
parser.add_argument('--val_question_h5', default='data/val_questions.h5')
parser.add_argument('--val_features_h5', default='data/val_features.h5')
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--vocab_json', default='data/vocab.json')

parser.add_argument('--loader_num_workers', type=int, default=0)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=10000, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)

# What type of model to use and which parts to train
parser.add_argument(
    '--model_type',
    default='PG+EE',
    choices=[
        'PG',
        'EE',
        'PG+EE',
        'LSTM',
        'CNN+LSTM',
        'CNN+LSTM+SA'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# RNN options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument('--cnn_pooling', default='maxpool2',
                    choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
                    choices=['maxpool2', 'maxpool4', 'none'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0, type=float)

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=100, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='data/checkpoint.pt')
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--record_loss_every', type=int, default=1)
parser.add_argument('--checkpoint_every', default=10000, type=int)


class LossAndErrorPrintingCallback(Callback):
    """A custom callback class to print accuracy at batch end"""

    def on_train_batch_end(self, batch, logs=None):
        print('Accuracy: {} Loss: {}'.format(logs['acc'], logs['loss']))


def main(args):
    if args.randomize_checkpoint_path == 1:
        name, ext = os.path.splitext(args.checkpoint_path)
        num = random.randint(1, 1000000)
        args.checkpoint_path = '%s_%06d%s' % (name, num, ext)

    vocab = utils.load_vocab(args.vocab_json)

    if args.use_local_copies == 1:
        shutil.copy(args.train_question_h5, '/tmp/train_questions.h5')
        shutil.copy(args.train_features_h5, '/tmp/train_features.h5')
        shutil.copy(args.val_question_h5, '/tmp/val_questions.h5')
        shutil.copy(args.val_features_h5, '/tmp/val_features.h5')
        args.train_question_h5 = '/tmp/train_questions.h5'
        args.train_features_h5 = '/tmp/train_features.h5'
        args.val_question_h5 = '/tmp/val_questions.h5'
        args.val_features_h5 = '/tmp/val_features.h5'

    question_families = None
    if args.family_split_file is not None:
        with open(args.family_split_file, 'r') as f:
            question_families = json.load(f)

    train_loader_kwargs = {
        'question_h5': args.train_question_h5,
        'feature_h5': args.train_features_h5,
        'vocab': vocab,
        'batch_size': args.batch_size,
        'shuffle': args.shuffle_train_data == 1,
        'question_families': question_families,
        'max_samples': args.num_train_samples,
        'num_workers': args.loader_num_workers,
    }
    val_loader_kwargs = {
        'question_h5': args.val_question_h5,
        'feature_h5': args.val_features_h5,
        'vocab': vocab,
        'batch_size': args.batch_size,
        'question_families': question_families,
        'max_samples': args.num_val_samples,
        'num_workers': args.loader_num_workers,
    }

    train_loader = ClevrDataLoader(**train_loader_kwargs)
    val_loader = ClevrDataLoader(**val_loader_kwargs)
    train_loop(args, train_loader, val_loader)

    if args.use_local_copies == 1 and args.cleanup_local_copies == 1:
        os.remove('/tmp/train_questions.h5')
        os.remove('/tmp/train_features.h5')
        os.remove('/tmp/val_questions.h5')
        os.remove('/tmp/val_features.h5')


def batch_creater(data, batch_size, drop_last):
    q, a, f, ans, pro, l = ([] for i in range(6))
    batch = (q, a, f, ans, pro, l)
    j = 0
    big_batch = []
    #print("Length of data set : ", len(data))
    trimmed_data = len(data)/60
    #print("trimmed length : ", int(trimmed_data))
    for i in range(int(trimmed_data)):
        for k in range(6):
            batch[k].append(data[i][k])
        if len(batch[0]) == batch_size:
            big_batch.append(batch)
            q, a, f, ans, pro, l = ([] for i in range(6))
            batch = (q, a, f, ans, pro, l)
            j += 1
    if len(batch) > 0 and not drop_last:
        big_batch.append(batch)
    print("Batching Completed.")
    return big_batch


def to_tensor(nd_array):
    return tf.convert_to_tensor(nd_array, dtype=tf.float32)


def loss_function(pred, real):
    # mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=real, logits=pred)
    return tf.reduce_mean(loss_)


def train_loop(args, train_loader, val_loader):
    vocab = utils.load_vocab(args.vocab_json)
    program_generator, pg_kwargs, pg_optimizer = None, None, None
    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    baseline_model, baseline_kwargs, baseline_optimizer = None, None, None
    baseline_type = None

    pg_best_state, ee_best_state, baseline_best_state = None, None, None

    # pg_checkpoint = ModelCheckpoint(args.checkpoint_path,
    #                              monitor='val_accuracy',
    #                              verbose=1,
    #                              save_best_only=True,
    #                              mode='min',
    #                              load_weights_on_restart=True)
    # ee_checkpoint = ModelCheckpoint(args.checkpoint_path,
    #                                 monitor='val_accuracy',
    #                                 verbose=1,
    #                                 save_best_only=True,
    #                                 mode='min',
    #                                 load_weights_on_restart=True)
    pg_checkpoint_dir = './pg_training_checkpoints'
    pg_checkpoint_prefix = os.path.join(pg_checkpoint_dir, "ckpt")
    ee_checkpoint_dir = './ee_training_checkpoints'
    ee_checkpoint_prefix = os.path.join(ee_checkpoint_dir, "ckpt")
    # Set up model
    if args.model_type == 'PG' or args.model_type == 'PG+EE':
        program_generator, pg_kwargs = get_program_generator(args)
        pg_optimizer = optimizers.Adam(args.learning_rate)
        print('Here is the program generator:')
        checkpoint = tf.train.Checkpoint(optimizer=pg_optimizer,
                                         program_generator=program_generator)
        # program_generator.build(input_shape=[46,])
        # program_generator.compile(optimizer='adam', loss='mse')
        print(program_generator)
    if args.model_type == 'EE' or args.model_type == 'PG+EE':
        execution_engine, ee_kwargs = get_execution_engine(args)
        ee_optimizer = optimizers.Adam(args.learning_rate)
        print('Here is the execution engine:')
        print(execution_engine)
        checkpoint = tf.train.Checkpoint(optimizer=ee_optimizer,
                                         execution_engine=execution_engine)

    stats = {
        'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
        'train_accs': [], 'val_accs': [], 'val_accs_ts': [],
        'best_val_acc': -1, 'model_t': 0,
    }
    t, epoch, reward_moving_average = 0, 0, 0
    batch_size = 64

    # set_mode('train', [program_generator, execution_engine, baseline_model])

    print('train_loader has %d samples' % len(train_loader))
    print('val_loader has %d samples' % len(val_loader))
    train_data_load = batch_creater(train_loader, batch_size, False)
    val_data_load = batch_creater(val_loader, batch_size, False)
    print("train data load length :", len(train_data_load))

    while t < args.num_iterations:
        total_loss = 0
        epoch += 1
        print('Starting epoch %d' % epoch)
        #print("value of t :", t)
        for run_num, batch in enumerate(train_data_load):
            batch_loss = 0
            t += 1
            questions, _, feats, answers, programs, _ = to_tensor(
                batch[0]), batch[1], to_tensor(
                batch[2]), to_tensor(
                batch[3]), to_tensor(
                batch[4]), batch[5]

            questions_var = tf.Variable(questions)
            feats_var = tf.Variable(feats, trainable=True)
            answers_var = tf.Variable(answers, trainable=True)
            if programs[0] is not None:
                programs_var = tf.Variable(programs, trainable=True)

            reward = None
            if args.model_type == 'PG':
                # Train program generator with ground-truth programs+++
                with tf.GradientTape() as tape:
                    batch_loss = program_generator(questions_var, programs_var)
                total_loss += batch_loss
                variables = program_generator.variables
                gradients = tape.gradient(batch_loss, variables)
                pg_optimizer.apply_gradients(zip(gradients), variables)

            if args.model_type == 'EE':
                # Train program generator with ground-truth programs+++
                feats = tf.transpose(feats_var, perm=[0, 2, 3, 1])
                feats_var = tf.Variable(feats)
                with tf.GradientTape() as tape:
                    scores = execution_engine(feats_var, programs_var)
                    answers_var = tf.dtypes.cast(answers_var, dtype=tf.int32)
                    batch_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=answers_var))
                total_loss += batch_loss
                grads = tape.gradient(batch_loss, execution_engine.trainable_variables)
                gradients = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(execution_engine.trainable_variables, grads)]
                #TODO Might need some changes for gradients

                ee_optimizer.apply_gradients(zip(gradients, execution_engine.trainable_variables))

            elif args.model_type == 'PG+EE':
                print("in PG EE -----------------")
                feats = tf.transpose(feats_var, perm=[0, 2, 3, 1])
                feats_var = tf.Variable(feats)
                with tf.GradientTape() as pg_tape, tf.GradientTape() as ee_tape:
                    programs_pred = program_generator.reinforce_sample(questions_var)
                    print("shape of programs_pred : ", programs_pred.shape)
                    scores = execution_engine(feats_var, programs_pred)
                    answers_var = tf.dtypes.cast(answers_var, dtype=tf.int32)
                    batch_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=answers_var))
                    #_, preds = scores.data.max(1)
                    print("dim of score :", scores.shape)
                    preds = tf.math.reduce_max(scores, axis=1, keepdims=True)
                    print("dim of pred :", preds.shape)
                    # raw_reward = (preds == answers).float()
                    raw_reward = tf.cast((preds == answers), dtype=tf.float32)
                    reward_moving_average *= args.reward_decay
                    reward_moving_average += (1.0 - args.reward_decay) * raw_reward.numpy().mean()
                    centered_reward = raw_reward - reward_moving_average

                if args.train_execution_engine == 1:
                    grads = ee_tape.gradient(batch_loss, execution_engine.trainable_variables)
                    gradients = [grad if grad is not None else tf.zeros_like(var) for var, grad in
                                 zip(execution_engine.trainable_variables, grads)]
                    # TODO Might need some changes for gradients

                    ee_optimizer.apply_gradients(zip(gradients, execution_engine.trainable_variables))

                if args.train_program_generator == 1:
                    loss, multinomial_outputs = program_generator.reinforce_backward(centered_reward)
                    multinomial_outputs = tf.concat(multinomial_outputs, 0)
                    multinomial_outputs = tf.Variable(multinomial_outputs)
                    print("multi op shape new : ", multinomial_outputs.shape)
                    grads = pg_tape.gradient(loss, multinomial_outputs)
                    pg_optimizer.apply_gradients(grads, multinomial_outputs)
            print('Epoch {} Batch No. {} Loss {:.4f}'.format(epoch, run_num, batch_loss.numpy()))

            # if t == args.num_iterations:
            #     break

            if t % (args.record_loss_every * 2) == 0:
                #print(t, batch_loss)
                stats['train_losses'].append(batch_loss)
                stats['train_losses_ts'].append(t)
                if reward is not None:
                    stats['train_rewards'].append(reward)

            if t % args.checkpoint_every == 0:
                print('Checking training accuracy ... ')
                train_acc = check_accuracy(args, program_generator, execution_engine, train_data_load)
                print('train accuracy is', train_acc)
                print('Checking validation accuracy ...')
                val_acc = check_accuracy(args, program_generator, execution_engine, val_data_load)
                print('val accuracy is ', val_acc)
                stats['train_accs'].append(train_acc)
                stats['val_accs'].append(val_acc)
                stats['val_accs_ts'].append(t)

                if val_acc > stats['best_val_acc']:
                    stats['best_val_acc'] = val_acc
                    stats['model_t'] = t
                    if args.model_type == 'PG':
                        checkpoint = tf.train.Checkpoint(optimizer=pg_optimizer, model=program_generator)
                        checkpoint.save(file_prefix=pg_checkpoint_prefix)
                    if args.model_type == 'EE':
                        checkpoint = tf.train.Checkpoint(optimizer=ee_optimizer, model=execution_engine)
                        checkpoint.save(file_prefix=ee_checkpoint_prefix)

        if t == args.num_iterations:
            break


def parse_int_list(s):
    return tuple(int(n) for n in s.split(','))


def get_state(m):
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state


def get_program_generator(args):
    vocab = utils.load_vocab(args.vocab_json)
    kwargs = {
        'encoder_vocab_size': len(vocab['question_token_to_idx']),
        'decoder_vocab_size': len(vocab['program_token_to_idx']),
        'wordvec_dim': args.rnn_wordvec_dim,
        'hidden_dim': args.rnn_hidden_dim,
        'rnn_num_layers': args.rnn_num_layers,
        'rnn_dropout': args.rnn_dropout,
    }
    pg = Seq2Seq(**kwargs)
    return pg, kwargs


def get_execution_engine(args):
    print("Initiating Execution Engine process")
    vocab = utils.load_vocab(args.vocab_json)
    kwargs = {
        'vocab': vocab,
        'feature_dim': parse_int_list(args.feature_dim),
        'stem_batchnorm': args.module_stem_batchnorm == 1,
        'stem_num_layers': args.module_stem_num_layers,
        'module_dim': args.module_dim,
        'module_residual': args.module_residual == 1,
        'module_batchnorm': args.module_batchnorm == 1,
        'classifier_proj_dim': args.classifier_proj_dim,
        'classifier_downsample': args.classifier_downsample,
        'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
        'classifier_batchnorm': args.classifier_batchnorm == 1,
        'classifier_dropout': args.classifier_dropout,
    }
    ee = ModuleNet(**kwargs)
    return ee, kwargs


def set_mode(mode, models):
    assert mode in ['train', 'eval']
    for m in models:
        if m is None:
            continue
        if mode == 'train':
            m.train()
        if mode == 'eval':
            m.eval()


def check_accuracy(args, program_generator, execution_engine, loader):
    #set_mode('eval', [program_generator, execution_engine])
    num_correct, num_samples = 0, 0
    for run_num, batch in enumerate(loader):

        questions, _, feats, answers, programs, _ = to_tensor(
            batch[0]), batch[1], to_tensor(
            batch[2]), to_tensor(
            batch[3]), to_tensor(
            batch[4]), batch[5]

        questions_var = tf.Variable(questions)
        feats_var = tf.Variable(feats, trainable=True)
        answers_var = tf.Variable(answers, trainable=True)
        if programs[0] is not None:
            programs_var = tf.Variable(programs, trainable=True)

        scores = None  # Use this for everything but PG
        if args.model_type == 'PG':
            vocab = utils.load_vocab(args.vocab_json)
            for i in range(tf.shape(questions)[0]):
                q = tf.Variable(questions[i:i + 1])
                q = tf.dtypes.cast(q.read_value(), dtype=tf.int32)
                program_pred = program_generator.sample(tf.Variable(q))
                #print("program_pred : ", program_pred)
                program_pred_str = iep.preprocess.decode(program_pred, vocab['program_idx_to_token'])
                #print("program_pred_str : ", program_pred_str)
                #print("program__i : ", programs[i])
                program_str = iep.preprocess.decode(programs[i].numpy(), vocab['program_idx_to_token'])
                #print("program__str : ", program_str)
                if program_pred_str == program_str:
                    num_correct += 1
                num_samples += 1
        elif args.model_type == 'EE':
            feats = tf.transpose(feats_var, perm=[0, 2, 3, 1])
            feats_var = tf.Variable(feats)
            scores = execution_engine(feats_var, programs_var)
        elif args.model_type == 'PG+EE':
            programs_pred = program_generator.reinforce_sample(questions_var, argmax=True)
            scores = execution_engine(feats_var, programs_pred)

        if scores is not None:
            #_, preds = scores.max(1)
            preds = tf.math.argmax(scores, axis=1, output_type=tf.dtypes.int32)
            #num_correct += (preds == answers).sum()
            num_correct += (np.where(np.equal(preds.numpy(), answers.numpy()), 1, 0)).sum()
            num_samples += tf.shape(preds)[0].numpy()

        if num_samples >= args.num_val_samples:
            break

    acc = float(num_correct) / num_samples
    return acc


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

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
    print("Length of data set : ", len(data))
    trimmed_data = len(data)/60
    print("trimmed length : ", int(trimmed_data))
    # print("Batch Size : ", batch_size)
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

    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 load_weights_on_restart=True)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
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
    # train_loader = train_loader[:256]
    print('train_loader has %d samples' % len(train_loader))
    print('val_loader has %d samples' % len(val_loader))
    # data_sampler = iter(range(len(train_loader)))
    data_load = batch_creater(train_loader, batch_size, False)
    print("Data load length :", len(data_load))
    # print(data_load[0][0])

    while t < args.num_iterations:
        total_loss = 0
        epoch += 1
        print('Starting epoch %d' % epoch)
        print("value of t :", t)
        # train_loader_data = get_data(train_loader)
        # print("train data loader length :", len(train_loader_data))
        # print(train_loader[0].shape)
        # print(train_loader[0])
        for run_num, batch in enumerate(data_load):
            batch_loss = 0
            with tf.GradientTape() as tape:
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
                    batch_loss = program_generator(questions_var, programs_var)
                    total_loss += batch_loss
                    variables = program_generator.variables
                    gradients = tape.gradient(batch_loss, variables)
                    pg_optimizer.apply_gradients(zip(gradients), variables)

                if args.model_type == 'EE':
                    # Train program generator with ground-truth programs+++
                    print("Training program generator with ground-truth programs ... ")
                    print("shape of features before train : ", feats_var.shape)
                    feats = tf.transpose(feats_var, perm=[0, 2, 3, 1])
                    #feats_var.assign(feats)
                    feats_var = tf.Variable(feats)
                    print("shape of reshaped features before train : ", feats_var.shape)
                    print("type of feats_var: ", type(feats_var,)," and of program_var :", type(programs_var))
                    scores = execution_engine(feats_var, programs_var)
                    scores = tf.dtypes.cast(scores, dtype=tf.float32)
                    answers_var = tf.dtypes.cast(answers_var, dtype=tf.int32)
                    #answers_var = answers_var.read_value()
                    print("Shape of score var and ans_var : ", scores.shape, answers_var.shape)
                    print("type of score and ans_var : ", type(scores), type(answers_var))
                    scores = tf.Variable(scores)
                    answers_var = tf.Variable(answers_var)
                    #with tf.GradientTape() as tape2:
                    #batch_loss = loss_function(scores, answers_var)
                    batch_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=answers_var))
                    total_loss += batch_loss
                    variables = execution_engine.variables
                    #variables = execution_engine.trainable_variables
                    gradients = tape.gradient(batch_loss, variables)
                    print("loss : ", batch_loss)
                    #print("variables : ", variables)
                    #print("gradient :", gradients)
                    ee_optimizer.apply_gradients(zip(gradients, variables))
            print(
                'Epoch {} Batch No. {} Loss {:.4f}'.format(
                    epoch, run_num, batch_loss.numpy()))
        if epoch % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        if t == args.num_iterations:
            break
            # program_generator.compile(optimizer=pg_optimizer, loss=loss)
            # ques = np.asarray(questions_var.read_value())
            # prog = np.asarray(programs_var.read_value())
            # history = program_generator.fit(
            #     x=ques,
            #     y=prog,
            #     batch_size=args.batch_size,
            #     epochs=10,
            #     verbose=0,
            #     callbacks=[LossAndErrorPrintingCallback(), checkpoint])

            # elif args.model_type == 'EE':
            #     # Train execution engine with ground-truth programs
            #     scores = execution_engine(feats_var, programs_var)
            #     loss = tf.nn.softmax_cross_entropy_with_logits(
            #         scores, answers_var)
            #     execution_engine.compile(optimizer=ee_optimizer, loss=loss)
            #     history = execution_engine.fit(
            #         questions_var,
            #         to_categorical(answers_var),
            #         batch_size=args.batch_size,
            #         epochs=10,
            #         verbose=0,
            #         callbacks=[LossAndErrorPrintingCallback(), checkpoint])

            # elif args.model_type == 'PG+EE':
            #     programs_pred = program_generator.reinforce_sample(questions_var)
            #     scores = execution_engine(feats_var, programs_pred)
            #
            #     loss = tf.nn.softmax_cross_entropy_with_logits(scores, answers_var)
            #     _, preds = scores.data.max(1)
            #     # raw_reward = (preds == answers).float()
            #     raw_reward = tf.cast((preds == answers), dtype=tf.float32)
            #     reward_moving_average *= args.reward_decay
            #     reward_moving_average += (1.0 - args.reward_decay) * raw_reward.mean()
            #     centered_reward = raw_reward - reward_moving_average
            #
            #     if args.train_execution_engine == 1:
            #         ee_optimizer.zero_grad()
            #         loss.backward()
            #         ee_optimizer.step()
            #
            #     if args.train_program_generator == 1:
            #         pg_optimizer.zero_grad()
            #         program_generator.reinforce_backward(centered_reward.cuda())
            #         pg_optimizer.step()

            # if t % args.record_loss_every == 0:
            #     print(t, loss.data[0])
            #     stats['train_losses'].append(loss.data[0])
            #     stats['train_losses_ts'].append(t)
            #     if reward is not None:
            #         stats['train_rewards'].append(reward)
            #
            # if t % args.checkpoint_every == 0:
            #     print('Checking training accuracy ... ')
            #     train_acc = check_accuracy(args, program_generator, execution_engine,
            #                                baseline_model, train_loader)
            #     print('train accuracy is', train_acc)
            #     print('Checking validation accuracy ...')
            #     val_acc = check_accuracy(args, program_generator, execution_engine,
            #                              baseline_model, val_loader)
            #     print('val accuracy is ', val_acc)
            #     stats['train_accs'].append(train_acc)
            #     stats['val_accs'].append(val_acc)
            #     stats['val_accs_ts'].append(t)
            #
            #     if val_acc > stats['best_val_acc']:
            #         stats['best_val_acc'] = val_acc
            #         stats['model_t'] = t
            #         best_pg_state = get_state(program_generator)
            #         best_ee_state = get_state(execution_engine)
            #         best_baseline_state = get_state(baseline_model)
            #
            #     checkpoint = {
            #         'args': args.__dict__,
            #         'program_generator_kwargs': pg_kwargs,
            #         'program_generator_state': best_pg_state,
            #         'execution_engine_kwargs': ee_kwargs,
            #         'execution_engine_state': best_ee_state,
            #         'baseline_kwargs': baseline_kwargs,
            #         'baseline_state': best_baseline_state,
            #         'baseline_type': baseline_type,
            #         'vocab': vocab
            #     }
            #     for k, v in stats.items():
            #         checkpoint[k] = v
            #     print('Saving checkpoint to %s' % args.checkpoint_path)
            #     torch.save(checkpoint, args.checkpoint_path)
            #     del checkpoint['program_generator_state']
            #     del checkpoint['execution_engine_state']
            #     del checkpoint['baseline_state']
            #     with open(args.checkpoint_path + '.json', 'w') as f:
            #         json.dump(checkpoint, f)

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


def check_accuracy(
        args,
        program_generator,
        execution_engine,
        baseline_model,
        loader):
    set_mode('eval', [program_generator, execution_engine, baseline_model])
    num_correct, num_samples = 0, 0
    for batch in loader:
        questions, _, feats, answers, programs, _ = batch

        questions_var = tf.Variable(questions.cuda(), volatile=True)
        feats_var = tf.Variable(feats.cuda(), volatile=True)
        answers_var = tf.Variable(feats.cuda(), volatile=True)
        if programs[0] is not None:
            programs_var = tf.Variable(programs.cuda(), volatile=True)

        scores = None  # Use this for everything but PG
        if args.model_type == 'PG':
            vocab = utils.load_vocab(args.vocab_json)
            for i in range(questions.size(0)):
                program_pred = program_generator.sample(
                    tf.Variable(questions[i:i + 1].cuda(), volatile=True))
                program_pred_str = iep.preprocess.decode(
                    program_pred, vocab['program_idx_to_token'])
                program_str = iep.preprocess.decode(
                    programs[i], vocab['program_idx_to_token'])
                if program_pred_str == program_str:
                    num_correct += 1
                num_samples += 1
        elif args.model_type == 'EE':
            scores = execution_engine(feats_var, programs_var)
        elif args.model_type == 'PG+EE':
            programs_pred = program_generator.reinforce_sample(
                questions_var, argmax=True)
            scores = execution_engine(feats_var, programs_pred)
        elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
            scores = baseline_model(questions_var, feats_var)

        if scores is not None:
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == answers).sum()
            num_samples += preds.size(0)

        if num_samples >= args.num_val_samples:
            break

    set_mode('train', [program_generator, execution_engine, baseline_model])
    acc = float(num_correct) / num_samples
    return acc


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

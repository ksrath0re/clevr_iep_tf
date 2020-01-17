#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import json
import random
import shutil
import utils
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from iep.data import ClevrDataLoader, get_data
from models.baselines import LstmModel, CnnLstmModel, CnnLstmSaModel
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_question_h5', default='data/train_questions.h5')
parser.add_argument('--train_features_h5', default='data/train_features.h5')
parser.add_argument('--val_question_h5', default='data/val_questions.h5')
parser.add_argument('--val_features_h5', default='data/val_features.h5')
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--vocab_json', default='data/vocab.json')
parser.add_argument('--train_images', default='../input/train/images', type=str)
parser.add_argument('--val_images', default='../input/val/images', type=str)

parser.add_argument('--loader_num_workers', type=int, default=1)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=10000, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG',
                    choices=['PG', 'EE', 'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA'])
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
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='data/checkpoint.pt')
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--record_loss_every', type=int, default=1)
parser.add_argument('--checkpoint_every', default=10000, type=int)


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
        # 'batch_size': args.batch_size,
        'shuffle': args.shuffle_train_data == 1,
        'question_families': question_families,
        'max_samples': args.num_train_samples,
        'num_workers': args.loader_num_workers,
        'images_path': args.train_images
    }
    val_loader_kwargs = {
        'question_h5': args.val_question_h5,
        'feature_h5': args.val_features_h5,
        'vocab': vocab,
        # 'batch_size': args.batch_size,
        'question_families': question_families,
        'max_samples': args.num_val_samples,
        'num_workers': args.loader_num_workers,
        'images_path': args.val_images
    }

    with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
            ClevrDataLoader(**val_loader_kwargs) as val_loader:
        train_loop(args, train_loader, val_loader)

    if args.use_local_copies == 1 and args.cleanup_local_copies == 1:
        os.remove('/tmp/train_questions.h5')
        os.remove('/tmp/train_features.h5')
        os.remove('/tmp/val_questions.h5')
        os.remove('/tmp/val_features.h5')


def train_loop(args, train_loader, val_loader):
    vocab = utils.load_vocab(args.vocab_json)

    program_generator, pg_kwargs, pg_optimizer = None, None, None
    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    baseline_model, baseline_kwargs, baseline_optimizer = None, None, None
    baseline_type = None

    pg_best_state, ee_best_state, baseline_best_state = None, None, None

    # Set up model

    if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
        baseline_model, baseline_kwargs = get_baseline_model(args)
        print('Here is the baseline model')
        print(baseline_model.summary())

        baseline_optimizer = torch.optim.Adam(args.learning_rate)

        baseline_type = args.model_type

    stats = {
        'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
        'train_accs': [], 'val_accs': [], 'val_accs_ts': [],
        'best_val_acc': -1, 'model_t': 0,
    }
    t, epoch, reward_moving_average = 0, 0, 0

    baseline_model.compile(optimizer=baseline_optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    '''Fetch data for the validation data loader'''
    train_questions, _, train_feats, train_answers, train_programs, _ = \
        get_data(train_loader)

    '''Fetch data for the validation data loader'''
    val_questions, _, val_feats, val_answers, val_programs, _ = \
        get_data(val_loader)

    '''
    questions for train dataset have length of 33, while validation dataset has
    a length of 29. Pad at the end of the validation questions to make it of the
    same length as of training dataset.
    '''

    val_questions = pad_sequences(
        val_questions,
        value=train_loader.vocab['question_token_to_idx']['<PAD>'],
        padding='post',
        maxlen=train_loader.all_questions.shape[1])

    '''Checkpoint callback'''
    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 load_weights_on_restart=True)

    '''
    Initial image size is (224, 224, 3). Resize them to (28,28,3)
    '''


    width = 28
    height = 28
    dim = (width, height)

    '''list to hold the corresponding dataset images'''
    train_images = []
    val_images = []

    for train_image in sorted(os.listdir(args.train_images)):
        train_image = cv2.imread(os.path.join(args.train_images, train_image),
                                 cv2.IMREAD_COLOR)
        train_image = cv2.resize(train_image, dim)
        train_image = cv2.normalize(train_image, None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        '''
        every image has 10 questions. So the question tensor is of the shape
        (#images * 10, 33).
        To make image tensor in sync with the question tensor shape, add the
        same image 10 times to the image dataset.
        '''
        for i in range(10):
            train_images.append(train_image)

    for val_image in sorted(os.listdir(args.val_images)):
        val_image = cv2.imread(os.path.join(args.val_images, val_image),
                               cv2.IMREAD_COLOR)
        val_image = cv2.resize(val_image, dim)
        val_image = cv2.normalize(val_image, None, alpha=0, beta=1,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        for i in range(10):
            val_images.append(val_image)

    train_images = np.asarray(train_images, dtype=np.float32)
    val_images = np.asarray(val_images, dtype=np.float32)

    for i in range(args.num_iterations):
        epoch += 1
        print('Starting epoch %d' % epoch)

        if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
            if args.model_type == 'LSTM':
                inputs = [train_questions]
                validation_data = [val_questions]
            elif args.model_type == 'CNN+LSTM':
                inputs = [train_questions, train_images]
                validation_data = [val_questions, val_images]

            history = baseline_model.fit(
                inputs,
                to_categorical(train_answers),
                batch_size=args.batch_size,
                epochs=10,
                verbose=0,
                validation_data=(validation_data, to_categorical(val_answers)),
                callbacks=[LossAndErrorPrintingCallback(), checkpoint])

    baseline_model.save_weights(args.model_save_path)

    '''list all data in history'''
    print(history.history.keys())

    '''summarize history for accuracy'''
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../data/accuracy.png', dpi=100)
    plt.show()

    '''summarize history for loss'''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../data/val_loss.png', dpi=100)
    plt.show()


def check_accuracy(args, program_generator, execution_engine,
                   baseline_model, loader):
    num_correct, num_samples = 0, 0
    for batch in loader:
        questions, _, feats, answers, programs, _ = batch

        # Convert the above values to tensors

        scores = None
        if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
            scores = baseline_model(questions, feats)

        if scores is not None:
            _, preds = scores.data.max(1)
            num_correct += (preds == answers).sum()
            num_samples += preds.size(0)

        if num_samples >= args.num_val_samples:
            break

    acc = float(num_correct) / num_samples
    return acc


def parse_int_list(s):
    return tuple(int(n) for n in s.split(','))


def get_baseline_model(args):
    vocab = utils.load_vocab(args.vocab_json)


    if args.baseline_start_from is not None:
        model, kwargs = utils.load_baseline(args.baseline_start_from)
    elif args.model_type == 'LSTM':
        kwargs = {
            'vocab': vocab,
            'rnn_wordvec_dim': args.rnn_wordvec_dim,
            'rnn_dim': args.rnn_hidden_dim,
            'rnn_num_layers': args.rnn_num_layers,
            'rnn_dropout': args.rnn_dropout,
            'fc_dims': parse_int_list(args.classifier_fc_dims),
            'fc_use_batchnorm': args.classifier_batchnorm == 1,
            'fc_dropout': args.classifier_dropout,
        }
        model = LstmModel(**kwargs)
    elif args.model_type == 'CNN+LSTM':
        kwargs = {
            'vocab': vocab,
            'rnn_wordvec_dim': args.rnn_wordvec_dim,
            'rnn_dim': args.rnn_hidden_dim,
            'rnn_num_layers': args.rnn_num_layers,
            'rnn_dropout': args.rnn_dropout,
            'cnn_feat_dim': parse_int_list(args.feature_dim),
            'cnn_num_res_blocks': args.cnn_num_res_blocks,
            'cnn_res_block_dim': args.cnn_res_block_dim,
            'cnn_proj_dim': args.cnn_proj_dim,
            'cnn_pooling': args.cnn_pooling,
            'fc_dims': parse_int_list(args.classifier_fc_dims),
            'fc_use_batchnorm': args.classifier_batchnorm == 1,
            'fc_dropout': args.classifier_dropout,
        }
        model = CnnLstmModel(**kwargs)
    elif args.model_type == 'CNN+LSTM+SA':
        kwargs = {
            'vocab': vocab,
            'rnn_wordvec_dim': args.rnn_wordvec_dim,
            'rnn_dim': args.rnn_hidden_dim,
            'rnn_num_layers': args.rnn_num_layers,
            'rnn_dropout': args.rnn_dropout,
            'cnn_feat_dim': parse_int_list(args.feature_dim),
            'stacked_attn_dim': args.stacked_attn_dim,
            'num_stacked_attn': args.num_stacked_attn,
            'fc_dims': parse_int_list(args.classifier_fc_dims),
            'fc_use_batchnorm': args.classifier_batchnorm == 1,
            'fc_dropout': args.classifier_dropout,
        }
        model = CnnLstmSaModel(**kwargs)
    if model.rnn.token_to_idx != vocab['question_token_to_idx']:
        # Make sure new vocab is superset of old
        for k, v in model.rnn.token_to_idx.items():
            assert k in vocab['question_token_to_idx']
            assert vocab['question_token_to_idx'][k] == v
        for token, idx in vocab['question_token_to_idx'].items():
            model.rnn.token_to_idx[token] = idx
        kwargs['vocab'] = vocab
        model.rnn.expand_vocab(vocab['question_token_to_idx'])

    return model.model, kwargs


if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    main(args)

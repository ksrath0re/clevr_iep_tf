#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import h5py
import tensorflow as tf
import re
import iep.programs
import collections
import collections.abc

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class Dataset(object):
    """An abstract class representing a Dataset.

  All other datasets should subclass it. All subclasses should override
  ``__len__``, that provides the size of the dataset, and ``__getitem__``,
  supporting integer indexing in range from 0 to len(self) exclusive.
  """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DataLoader(object):
    r"""
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraies
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use :func:`torch.initial_seed()` to access the PyTorch seed for
              each worker in :attr:`worker_init_fn`, and use it to set other
              seeds before data loading.

    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """

    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=None, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if self.num_workers < 0:
            raise ValueError('num_workers option cannot be negative; '
                             'use num_workers=0 to disable multiprocessing.')

        # if batch_sampler is None:
        #     if sampler is None:
        #         if shuffle:
        #             sampler = RandomSampler(dataset)
        #         else:
        #             sampler = SequentialSampler(dataset)
        #     batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(DataLoader, self).__setattr__(attr, val)

    # def __iter__(self):
    #     return _DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])

    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return tf.stack([tf.convert_to_tensor(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return tf.convert_to_tensor(batch, dtype=tf.int64)
    elif isinstance(batch[0], float):
        return tf.convert_to_tensor(batch, dtype=tf.float64)
    elif isinstance(batch[0], (str, bytes)):
        return batch
    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def _dataset_to_tensor(dset, mask=None):
    arr = np.asarray(dset, dtype=np.int64)
    if mask is not None:
        arr = arr[mask]
    tensor = tf.convert_to_tensor(arr, dtype=tf.int64)
    return tensor


class ClevrDataset(Dataset):
    def __init__(self, question_h5, feature_h5, vocab, mode='prefix',
                 image_h5=None, max_samples=None, question_families=None,
                 image_idx_start_from=None):
        mode_choices = ['prefix', 'postfix']
        if mode not in mode_choices:
            raise ValueError('Invalid mode "%s"' % mode)
        self.image_h5 = image_h5
        self.vocab = vocab
        self.feature_h5 = feature_h5
        self.mode = mode
        self.max_samples = max_samples

        mask = None
        if question_families is not None:
            # Use only the specified families
            all_families = np.asarray(question_h5['question_families'])
            N = all_families.shape[0]
            print(question_families)
            target_families = np.asarray(question_families)[:, None]
            mask = (all_families == target_families).any(axis=0)
        if image_idx_start_from is not None:
            all_image_idxs = np.asarray(question_h5['image_idxs'])
            mask = all_image_idxs >= image_idx_start_from

        # Data from the question file is small, so read it all into memory
        print('Reading question data into memory')
        self.all_questions = _dataset_to_tensor(question_h5['questions'], mask)
        self.all_image_idxs = _dataset_to_tensor(question_h5['image_idxs'], mask)
        self.all_programs = None
        if 'programs' in question_h5:
            self.all_programs = _dataset_to_tensor(question_h5['programs'], mask)
        self.all_answers = _dataset_to_tensor(question_h5['answers'], mask)

    def __getitem__(self, index):
        question = self.all_questions[index]
        image_idx = self.all_image_idxs[index]
        answer = self.all_answers[index]
        program_seq = None
        if self.all_programs is not None:
            program_seq = self.all_programs[index]

        image = None
        if self.image_h5 is not None:
            image = self.image_h5['images'][image_idx]
            image = tf.convert_to_tensor(np.asarray(image, dtype=np.float32))

        feats = self.feature_h5['features'][image_idx]
        feats = tf.convert_to_tensor(np.asarray(feats, dtype=np.float32))

        program_json = None
        if program_seq is not None:
            program_json_seq = []
            for fn_idx in program_seq:
                fn_str = self.vocab['program_idx_to_token'][fn_idx]
                if fn_str == '<START>' or fn_str == '<END>': continue
                fn = iep.programs.str_to_function(fn_str)
                program_json_seq.append(fn)
            if self.mode == 'prefix':
                program_json = iep.programs.prefix_to_list(program_json_seq)
            elif self.mode == 'postfix':
                program_json = iep.programs.postfix_to_list(program_json_seq)

        return (question, image, feats, answer, program_seq, program_json)

    def __len__(self):
        if self.max_samples is None:
            return self.all_questions.size(0)
        else:
            return min(self.max_samples, self.all_questions.size(0))


class ClevrDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if 'question_h5' not in kwargs:
            raise ValueError('Must give question_h5')
        if 'feature_h5' not in kwargs:
            raise ValueError('Must give feature_h5')
        if 'vocab' not in kwargs:
            raise ValueError('Must give vocab')

        feature_h5_path = kwargs.pop('feature_h5')
        print('Reading features from ', feature_h5_path)
        self.feature_h5 = h5py.File(feature_h5_path, 'r')

        self.image_h5 = None
        if 'image_h5' in kwargs:
            image_h5_path = kwargs.pop('image_h5')
            print('Reading images from ', image_h5_path)
            self.image_h5 = h5py.File(image_h5_path, 'r')

        vocab = kwargs.pop('vocab')
        mode = kwargs.pop('mode', 'prefix')

        question_families = kwargs.pop('question_families', None)
        max_samples = kwargs.pop('max_samples', None)
        question_h5_path = kwargs.pop('question_h5')
        image_idx_start_from = kwargs.pop('image_idx_start_from', None)
        print('Reading questions from ', question_h5_path)
        with h5py.File(question_h5_path, 'r') as question_h5:
            self.dataset = ClevrDataset(question_h5, self.feature_h5, vocab, mode,
                                        image_h5=self.image_h5,
                                        max_samples=max_samples,
                                        question_families=question_families,
                                        image_idx_start_from=image_idx_start_from)
        kwargs['collate_fn'] = clevr_collate
        super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)

    def close(self):
        if self.image_h5 is not None:
            self.image_h5.close()
        if self.feature_h5 is not None:
            self.feature_h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def clevr_collate(batch):
    transposed = list(zip(*batch))
    question_batch = default_collate(transposed[0])
    image_batch = transposed[1]
    if any(img is not None for img in image_batch):
        image_batch = default_collate(image_batch)
    feat_batch = transposed[2]
    if any(f is not None for f in feat_batch):
        feat_batch = default_collate(feat_batch)
    answer_batch = default_collate(transposed[3])
    program_seq_batch = transposed[4]
    if transposed[4][0] is not None:
        program_seq_batch = default_collate(transposed[4])
    program_struct_batch = transposed[5]
    return [question_batch, image_batch, feat_batch, answer_batch, program_seq_batch, program_struct_batch]

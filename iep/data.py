import h5py
import numpy as np
import sys
import iep.programs as programs
import tensorflow as tf


def _dataset_to_tensor(dset, mask=None):
    arr = np.asarray(dset, dtype=np.int64)
    if mask is not None:
        arr = arr[mask]
    tensor = tf.convert_to_tensor(arr, dtype=tf.int64)
    return tensor


class ClevrDataLoader:
    def __init__(self, **kwargs):
        if 'question_h5' not in kwargs:
            raise ValueError('question_h5 was not provided.')
        if 'feature_h5' not in kwargs:
            raise ValueError('feature_h5 was not provided.')
        if 'vocab' not in kwargs:
            raise ValueError('vocab was not provided.')

        self.mode = kwargs.pop('mode', 'prefix')
        mode_choices = ['prefix', 'postfix']
        if self.mode not in mode_choices:
            raise ValueError('Invalid mode "%s"' % self.mode)

        feature_h5_path = kwargs.pop('feature_h5')
        print('Reading features from ', feature_h5_path)
        self.feature_h5 = h5py.File(feature_h5_path, 'r')
        print(self.feature_h5)

        self.image_h5 = None
        if 'image_h5' in kwargs:
            image_h5_path = kwargs.pop('image_h5')
            print('Reading images from ', image_h5_path)
            self.image_h5 = h5py.File(image_h5_path, 'r')

        self.vocab = kwargs.pop('vocab')
        #print("---------------", self.vocab)

        question_h5_path = kwargs.pop('question_h5')
        print('Reading questions from ', question_h5_path)
        self.question_h5 = h5py.File(question_h5_path, 'r')
        mask = None

        question_families = kwargs.pop('question_families', None)
        if question_families is not None:
            '''Use only the specified families'''
            all_families = np.asarray(self.question_h5['question_families'])
            print(question_families)
            target_families = np.asarray(question_families)[:, None]
            mask = (all_families == target_families).any(axis=0)

        image_idx_start_from = kwargs.pop('image_idx_start_from', None)
        if image_idx_start_from is not None:
            all_image_idxs = np.asarray(self.question_h5['image_idxs'])
            mask = all_image_idxs >= image_idx_start_from

        self.max_samples = kwargs.pop('max_samples', None)

        '''Data from the question file is small, so read it all into memory'''
        print('Reading question data into memory')
        # self.size = self.question_h5['questions'].shape[0]
        self.all_questions = _dataset_to_tensor(
            self.question_h5['questions'], mask)
        self.all_image_idxs = _dataset_to_tensor(
            self.question_h5['image_idxs'], mask)
        self.all_programs = None

        if 'programs' in self.question_h5:
            self.all_programs = _dataset_to_tensor(
                self.question_h5['programs'], mask)

        self.all_answers = _dataset_to_tensor(
            self.question_h5['answers'], mask)

    def __len__(self):
        if self.max_samples is None:
            return self.all_questions.shape[0]
        else:
            return min(self.max_samples, self.all_questions.shape[0])

    def __enter__(self):
        pass

    def __exit__(self, ext, exv, trb):
        if ext is not None:
            print('An error has been caught...\n')
            print(exv)
        sys.exit(1)

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

        features = self.feature_h5['features'][image_idx]
        features = tf.convert_to_tensor(np.asarray(features, dtype=np.float32))

        program_json = None
        #print("Program Sequence : ")
        #print(program_seq)
        pr = np.asarray(program_seq)
        if pr is not None:
            program_json_seq = []
            for fn_idx in pr:
                fn_str = self.vocab['program_idx_to_token'][fn_idx]
                if fn_str == '<START>' or fn_str == '<END>':
                    continue
                fn = programs.str_to_function(fn_str)
                program_json_seq.append(fn)

            if self.mode == 'prefix':
                program_json = programs.prefix_to_list(program_json_seq)
            elif self.mode == 'postfix':
                program_json = programs.postfix_to_list(program_json_seq)

        return (question, image, features, answer, program_seq, program_json)

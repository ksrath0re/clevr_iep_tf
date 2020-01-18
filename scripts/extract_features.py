# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, os, json
import h5py
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import cv2
import tensorflow.keras.applications.resnet50 as resnet50
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import eval

parser = argparse.ArgumentParser()
parser.add_argument('--extract_features_for', default='train', type=str,
                    help='Please provide in the type of dataset for which you'
                         'want to extract features. The allowed dataset are'
                         'train, val and test.')
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_file', required=True)

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)


def build_model(args):
    if not hasattr(resnet50, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if 'resnet' not in args.model.lower():
        raise ValueError('Feature extraction only supports ResNets')

    '''
    Transfer learning: it helps to utilise previously trained network, rather
    than training new complex models by using their learned weights and then use
    standard training methods to learn the remaining, non-reused parameters.
    '''
    original_model = getattr(tf.keras.applications.resnet50, 'ResNet50')(
        weights='imagenet')

    bottleneck_input = original_model.get_layer(index=0).input
    bottleneck_output = original_model.get_layer(index=-2).output
    bottleneck_model = Model(inputs=bottleneck_input, outputs=bottleneck_output)

    '''Set every layer in the model to be non-trainable'''
    for layer in bottleneck_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(bottleneck_model)
    '''This output of the model is flattened  into (2048, ).'''

    '''Check the link 
    https://towardsdatascience.com/how-to-train-your-model-dramatically-faster-9ad063f0f718
    for fitting
    '''
    return model


def run_batch(cur_batch, model):
    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = tf.keras.utils.normalize(image_batch)
    image_batch = tf.convert_to_tensor(image_batch)

    features = model(image_batch)
    return eval(features)


def main(args):
    input_paths = []
    idx_set = set()
    for fn in os.listdir(args.input_image_dir):
        if not fn.endswith('.png'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        input_paths.append((os.path.join(args.input_image_dir, fn), idx))
        idx_set.add(idx)
    input_paths.sort(key=lambda x: x[1])
    assert len(idx_set) == len(input_paths)
    assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
    if args.max_images is not None:
        input_paths = input_paths[:args.max_images]
    print(input_paths[0])
    print(input_paths[-1])

    model = build_model(args)

    with h5py.File(args.output_h5_file, 'w') as f:
        features_dataset = None
        i0 = 0
        current_batch = []
        for i, (path, idx) in enumerate(input_paths):
            img = cv2.imread(path)
            img = img[None]
            current_batch.append(img)
            if len(current_batch) == args.batch_size:
                features = run_batch(current_batch, model)
                if features_dataset is None:
                    N = len(input_paths)
                    _, C = features.shape
                    features_dataset = f.create_dataset('features', (N, C),
                                                        dtype=np.float32)
                i1 = i0 + len(current_batch)
                features_dataset[i0:i1] = features
                i0 = i1
                print('Processed %d / %d images' % (i1, len(input_paths)))
                current_batch.clear()

        if len(current_batch) > 0:
            features = run_batch(current_batch, model)
            i1 = i0 + len(current_batch)
            features_dataset[i0:i1] = features
            print('Processed %d / %d images' % (i1, len(input_paths)))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.extract_features_for == 'train' and args.input_image_dir is None:
        args.input_image_dir = '../../clevr-dataset-gen/output/train/images/'
        args.output_h5_file = '../output/features/train_features.h5'
    elif args.extract_features_for == 'val' and args.input_image_dir is None:
        args.input_image_dir = '../../clevr-dataset-gen/output/val/images/'
        args.output_h5_file = '../output/features/val_features.h5'
    elif args.extract_features_for == 'test' and args.input_image_dir is None:
        args.input_image_dir = '../../clevr-dataset-gen/output/test/images/'
        args.output_h5_file = '../output/features/test_features.h5'
    main(args)

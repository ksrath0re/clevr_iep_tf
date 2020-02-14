# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json
import h5py
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import tensorflow_hub as hub
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_file', required=True)

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)


def build_model(args, img_size):
    if 'resnet' not in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    print("Fetching resnet model .....")
    full_model = tf.keras.applications.ResNet101(
        input_shape=img_size, include_top=False, weights='imagenet')
    #print(full_model.summary())
    layer_output=full_model.get_layer('conv4_block23_out').output
    intermediate_model=tf.keras.models.Model(inputs=full_model.input,outputs=layer_output)
    #return full_model
    return intermediate_model


def run_batch(cur_batch, model):
    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)

    '''Normalise the input RGB image batch'''
    image_batch = tf.keras.utils.normalize(image_batch)
    image_batch = tf.convert_to_tensor(image_batch)

    '''Pass the input batch to the model to obtain the features'''
    if image_batch.shape[0] == 128:
        image_batch = tf.reshape(image_batch, [128, -1, 224, 3])
    else:
        image_batch = tf.reshape(
            image_batch, [
                image_batch.shape[0], -1, 224, 3])
    print("Image batch Shape Before Feature Extraction : ", image_batch.shape)
    features = model(image_batch)
    features = tf.reshape(features, [-1, 1024, 14, 14])
    #layer_output=model.get_layer('conv4_block23_out').output
    #intermediate_model=tf.keras.models.Model(inputs=image_batch,outputs=layer_output)
    #features = model.get_layer('conv4_block23_out').output
    print("Feature batch Shape after Feature Extraction : ", features.shape)
    return features


def main(args):
    input_paths = []
    idx_set = set()
    for fn in os.listdir(args.input_image_dir):
        if not fn.endswith('.png'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        input_paths.append((os.path.join(args.input_image_dir, fn), idx))
        idx_set.add(idx)
    input_paths.sort(key=lambda x: x[1])
    assert len(idx_set) == len(input_paths)
    #assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
    if args.max_images is not None:
        input_paths = input_paths[:args.max_images]
    print(input_paths[0])
    print(input_paths[-1])
    img_size = (args.image_height, args.image_width, 3)
    model = build_model(args, img_size)

    with h5py.File(args.output_h5_file, 'w') as f:
        feat_dset = None
        i0 = 0
        cur_batch = []
        for i, (path, idx) in enumerate(input_paths):
            img = imread(path, mode='RGB')
            img = imresize(img, img_size, interp='bicubic')
            img = img.transpose(2, 0, 1)[None]
            #print("image shapoe : ", img.shape)
            cur_batch.append(img)
            if len(cur_batch) == args.batch_size:
                feats = run_batch(cur_batch, model)
                if feat_dset is None:
                    N = len(input_paths)
                    _, C, H, W = feats.shape
                    print("N :", N)
                    print("Feats shape :", feats.shape)
                    feat_dset = f.create_dataset('features', (N, C, H, W),
                                                 dtype=np.float32)
                i1 = i0 + len(cur_batch)
                feat_dset[i0:i1] = feats
                i0 = i1
                print('Processed %d / %d images' % (i1, len(input_paths)))
                cur_batch = []
        if len(cur_batch) > 0:
            feats = run_batch(cur_batch, model)
            i1 = i0 + len(cur_batch)
            feat_dset[i0:i1] = feats
            print('Processed %d / %d images' % (i1, len(input_paths)))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

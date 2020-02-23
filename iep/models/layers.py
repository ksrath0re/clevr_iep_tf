import tensorflow as tf
import tensorflow.nn as nn


class ResidualBlock(tf.keras.Model):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True):
        if out_dim is None:
            out_dim = in_dim
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3), padding=1)
        self.conv2 = tf.keras.layers.Conv2D(in_channels=out_dim, out_channels=out_dim, kernel_size=(3, 3), padding=1)
        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn1 = tf.nn.batch_normalization(out_dim)
            self.bn2 = tf.nn.batch_normalization(out_dim)
        self.with_residual = with_residual
        if in_dim == out_dim or not with_residual:
            self.proj = None
        else:
            self.proj = tf.keras.layers.Conv2D(in_channels=in_dim, out_channels=out_dim, kernel_size=(1, 1))

    def __call__(self, x):
        if self.with_batchnorm:
            out = tf.nn.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.conv2(tf.nn.relu(self.conv1(x)))
        res = x if self.proj is None else self.proj(x)
        if self.with_residual:
            out = tf.nn.relu(res + out)
        else:
            out = tf.nn.relu(out)
        return out


class GlobalAveragePool(tf.keras.Model):
    def __call__(self, x):
        N, C = tf.shape(x)
        return tf.squeeze(tf.reshape(x, [N, C, -1]).mean(2), [2])


class Flatten(tf.keras.Model):
    def __call__(self, x):
        return tf.reshape(x, [x.size(0), -1])

import tensorflow as tf


class ResidualBlock(tf.keras.Model):
    def __init__(self, out_dim=None, with_residual=True, with_batchnorm=True):
        #if out_dim is None:
        #    out_dim = in_dim
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_dim, kernel_size=(3, 3), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(out_dim, kernel_size=(3, 3), padding='same')
        self.relu_layer = tf.keras.layers.ReLU()
        self.with_batchnorm = with_batchnorm
        if with_batchnorm:
            self.bn1 = tf.nn.batch_normalization(out_dim)
            self.bn2 = tf.nn.batch_normalization(out_dim)
        self.with_residual = with_residual
        if not with_residual:
            self.proj = None
        else:
            self.proj = tf.keras.layers.Conv2D(out_dim, kernel_size=(1, 1))

    def __call__(self, x):
        #print("x ka shape : ", x.shape)
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        #print("x ka shape  after transpose: ", x.shape)
        if self.with_batchnorm:
            out = tf.nn.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            #print("in if with batchnorm")
        else:
            out = self.conv2(tf.nn.relu(self.conv1(x)))
            #print("in else with batchnorm")
        res = x if self.proj is None else self.proj(x)
        if self.with_residual:
            #print(res.shape, out.shape)
            out = tf.nn.relu(tf.math.add(res, out))
            #print(res.shape, out.shape)
            #print("in if with residual")
            #out = self.relu_layer(res + out)
        else:
            out = tf.nn.relu(out)
            #print("in else with reidual")
            #out = self.relu_layer(out)
        #print("out ka shape : ", out.shape)
        out = tf.transpose(out, perm=[0, 3, 1, 2])
        #print("out ka shape  after transpose: ", out.shape)
        return out


class GlobalAveragePool(tf.keras.Model):
    def __call__(self, x):
        N, C = tf.shape(x)
        return tf.squeeze(tf.reshape(x, [N, C, -1]).mean(2), [2])


class Flatten(tf.keras.Model):
    def __call__(self, x):
        return tf.reshape(x, [tf.shape(x)[0], -1])

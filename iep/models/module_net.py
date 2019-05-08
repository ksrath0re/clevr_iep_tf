from iep.models.layers import ResidualBlock, GlobalAveragePool, Flatten
import iep.programs
import tensorflow as tf

class ConcatBlock(nn.Module):
  def __init__(self, dim, with_residual=True, with_batchnorm=True):
    #super(ConcatBlock, self).__init__()
    self.proj = tf.keras.layers.Conv2d(2 * dim, dim, kernel_size=(1,1), padding=0)
    self.res_block = ResidualBlock(dim, with_residual=with_residual,
                        with_batchnorm=with_batchnorm)

  def forward(self, x, y):
    out = tf.concat([x, y], 1) # Concatentate along depth
    out = tf.nn.relu(self.proj(out))
    out = self.res_block(out)
    return out


def build_stem(feature_dim, module_dim, num_layers=2, with_batchnorm=True):
  layers = []
  prev_dim = feature_dim
  for i in range(num_layers):
    layers.append(tf.keras.layers.Conv2d(prev_dim, module_dim, kernel_size=(3,3), padding=1))
    if with_batchnorm:
      layers.append(tf.keras.layers.BatchNorm2d(module_dim))
    layers.append(tf.keras.layers.ReLU())
    prev_dim = module_dim
  return tf.keras.layers.Sequential(layers)



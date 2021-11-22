import tensorflow as tf
from tensorflow.python.keras.engine import training


@tf.keras.utils.register_keras_serializable(package='CG_Layers', name=None)
class DownsampleBlock(tf.keras.layers.Layer):
#downsample block layer, implemented from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    def __init__(self, filters, kernel_size, norm_type='batchnorm', apply_norm=True, **kwargs):
        super(DownsampleBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.apply_norm = apply_norm

        self.conv_initializer = tf.random_normal_initializer(0., 0.02)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same',kernel_initializer=self.conv_initializer, use_bias=False)
        if apply_norm:
            if self.norm_type.lower() == 'batchnorm':
                self.norm = tf.keras.layers.BatchNormalization()
            elif self.norm_type.lower() == 'instancenorm':
                self.norm = InstanceNormalization()
            else:
                self.norm = lambda x: x
        else:
            self.norm = lambda x: x
        self.act = tf.keras.layers.LeakyReLU()
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.norm(x)
        return self.act(x)
    def get_config(self):
        config = super(ResnetBlock, self).get_config()
        config.update({"filters": self.filters, "kernel_size":self.kernel_size, "norm_type":self.norm_type, "apply_norm":self.apply_norm})
        return config

def DownsampleStack(inp, filters, size, norm_type='batchnorm', apply_norm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        conv = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(inp)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                norm = tf.keras.layers.BatchNormalization()(conv)
            elif norm_type.lower() == 'instancenorm':
                norm = InstanceNormalization()(conv)
            else:
                norm = conv
        else:
            norm = conv
        leaky_relu = tf.keras.layers.LeakyReLU()(norm)
        return leaky_relu

@tf.keras.utils.register_keras_serializable(package='CG_Layers', name=None)
#upsample block layer, implemented from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, norm_type = 'batchnorm', apply_dropout=False, **kwargs):
        super(UpsampleBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.apply_dropout = apply_dropout

        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv = tf.keras.layers.Conv2DTranspose(self.filters, self.kernel_size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
        if self.norm_type.lower() =='batchnorm':
            self.norm = tf.keras.layers.BatchNormalization()
        elif self.norm_type.lower() == 'instancenorm':
            self.norm = InstanceNormalization()
        else:
            self.norm = lambda x: x

        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)
        else:
            self.dropout = lambda x: x
        self.act = tf.keras.layers.ReLU()

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.norm(x)
        if training:
            x = self.dropout(x)
        return self.act(x)
    def get_config(self):
        config = super(ResnetBlock, self).get_config()
        config.update({"filters": self.filters, "kernel_size":self.kernel_size, "norm_type":self.norm_type, "apply_dropout":self.apply_dropout})
        return config

def UpsampleStack(inp, filters, size, norm_type = 'batchnorm', apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        deconv =  tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(inp)
        if norm_type.lower() =='batchnorm':
            norm = tf.keras.layers.BatchNormalization()(deconv)
        elif norm_type.lower() == 'instancenorm':
            norm = InstanceNormalization()(deconv)
        else:
            norm = deconv

        if apply_dropout:
            dropout = tf.keras.layers.Dropout(0.5)(norm)
        else:
            dropout = norm
        act = tf.keras.layers.ReLU()(dropout)
        return act

@tf.keras.utils.register_keras_serializable(package='CG_Layers', name=None)
class ResnetBlock(tf.keras.layers.Layer):
# resnetBlock, inspired from https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
    def __init__(self, filters, norm_type='instancenorm', **kwargs):
        super(ResnetBlock, self).__init__( **kwargs)

        self.filters = filters
        self.norm_type = norm_type


        self.conv2d_1 = tf.keras.layers.Conv2D(self.filters, (3,3), padding='same')
        if self.norm_type.lower() == 'instancenorm':
            self.norm_1 = InstanceNormalization()
        elif self.norm_type.lower() == 'batchnorm':
            self.norm_1 = tf.keras.layers.BatchNormalization()
        else:
            self.norm_1 = lambda x: x
        self.act1 = tf.keras.layers.ReLU()

        self.conv2d_2 = tf.keras.layers.Conv2D(self.filters, (3,3), padding='same')

        if self.norm_type.lower() == 'instancenorm':
            self.norm_2 = InstanceNormalization()
        elif self.norm_type.lower() == 'batchnorm':
            self.norm_2 = tf.keras.layers.BatchNormalization()
        else:
            self.norm_2 = lambda x: x
    def call(self, inputs, training=None):
        x = self.conv2d_1(inputs)
        x = self.norm_1(x)
        x = self.act1(x)
        x = self.conv2d_2(x)
        x = self.norm_2(x)
        return x+inputs
    def get_config(self):
        config = super(ResnetBlock, self).get_config()
        config.update({"filters": self.filters, "norm_type":self.norm_type})
        return config

        


def ResnetStack(inp, filters, norm_type='instancenorm'):
# resnetBlock, inspired from https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
    initializer = tf.random_normal_initializer(0., 0.02)
    conv2d_1 = tf.keras.layers.Conv2D(filters, (3,3), padding='same', kernel_initializer=initializer)(inp)
    if norm_type.lower() == 'instancenorm':
        norm_1 = InstanceNormalization()(conv2d_1)
    elif norm_type.lower() == 'batchnorm':
        norm_1 = tf.keras.layers.BatchNormalization()(conv2d_1)
    else:
        norm_1 = conv2d_1
    relu = tf.keras.layers.ReLU()(norm_1)
    conv2d_2 = tf.keras.layers.Conv2D(filters, (3,3), padding='same', kernel_initializer=initializer)(relu)
    if norm_type.lower() == 'instancenorm':
        norm_2 = InstanceNormalization()(conv2d_2)
    elif norm_type.lower() == 'batchnorm':
        norm_2 = tf.keras.layers.BatchNormalization()(conv2d_2)
    else:
        norm_2 = conv2d_2
    concat = tf.keras.layers.Concatenate()([norm_2, inp])

    return concat

@tf.keras.utils.register_keras_serializable(package='CG_Layers', name=None)
#instance norm layer, implemented from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

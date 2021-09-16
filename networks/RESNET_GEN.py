import tensorflow as tf
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.util.tf_export import InvalidSymbolNameError
from .MODEL_CLASS import MODEL
from .CG_Layers import InstanceNormalization, ResnetBlock


class RESNET_GENERATOR(MODEL):
# resnet generator, based off of https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
    def __init__(self, output_dir, name='resnet_gen'):
        super(RESNET_GENERATOR, self).__init__(
            output_dir,
            name=name,
            model_args={'n_resnet':9, 'norm_type':'instancenorm'},
            optimizer_args={'lr':2e-4, 'beta_1':0.5}
        )

    def _build_model(self, n_resnet=9, norm_type='instancenorm'):
        init = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.Input(shape=[None, None, 3])

        #conv block 1
        conv_1 = tf.keras.layers.Conv2D(64, (7,7), padding='same', kernel_initializer=init)(inp)
        if norm_type.lower() == 'batchnorm':
            norm_1 = tf.keras.layers.BatchNormalization()(conv_1)
        elif norm_type.lower() == 'instancenorm':
            norm_1 = InstanceNormalization()(conv_1)
        act_1 = tf.keras.layers.ReLU()(norm_1)

        #conv block 2
        conv_2 = tf.keras.layers.Conv2D(128, (3,3), strides=2, padding='same', kernel_initializer=init)(act_1)
        if norm_type.lower() == 'batchnorm':
            norm_2 = tf.keras.layers.BatchNormalization()(conv_2)
        elif norm_type.lower() == 'instancenorm':
            norm_2 = InstanceNormalization()(conv_2)
        act_2 = tf.keras.layers.ReLU()(norm_2)

        #conv block 3
        conv_3 = tf.keras.layers.Conv2D(256, (3,3), strides=2, padding='same', kernel_initializer=init)(act_2)
        if norm_type.lower() == 'batchnorm':
            norm_3 = tf.keras.layers.BatchNormalization()(conv_3)
        elif norm_type.lower() == 'instancenorm':
            norm_3 = InstanceNormalization()(conv_3)
        act_3 = tf.keras.layers.ReLU()(norm_3)

        #resnet blocks
        output_filters = 256
        resnet_blk = ResnetBlock(256, input_shape=[None, None, output_filters], norm_type=norm_type)(act_3)
        for _ in range(1, n_resnet):
            output_filters += 256
            resnet_blk = ResnetBlock(256, input_shape=[None, None, output_filters], norm_type=norm_type)(resnet_blk)

        #deconv block 1
        deconv_1 = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', kernel_initializer=init)(resnet_blk)
        if norm_type.lower() == 'batchnorm':
            norm_4 = tf.keras.layers.BatchNormalization()(deconv_1)
        elif norm_type.lower() == 'instancenorm':
            norm_4 = InstanceNormalization()(deconv_1)
        act_4 = tf.keras.layers.ReLU()(norm_4)

        deconv_2 = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', kernel_initializer=init)(act_4)
        if norm_type.lower() == 'batchnorm':
            norm_5 = tf.keras.layers.BatchNormalization()(deconv_2)
        elif norm_type.lower() == 'instancenorm':
            norm_5 = InstanceNormalization()(deconv_2)
        act_5 = tf.keras.layers.ReLU()(norm_5)

        conv_4 = tf.keras.layers.Conv2D(3, (7,7), padding='same', kernel_initializer=init)(act_5)
        if norm_type.lower() == 'batchnorm':
            norm_6 = tf.keras.layers.BatchNormalization()(conv_4)
        elif norm_type.lower() == 'instancenorm':
            norm_6 = InstanceNormalization()(conv_4)
        outp = tf.keras.layers.Activation('tanh')(norm_6)

        return tf.keras.Model(inputs=inp, outputs=outp)

    def _build_optimizer(self, lr=2e-4, beta_1=0.5):
        return tf.keras.optimizers.Adam(lr, beta_1=beta_1)


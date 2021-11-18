import tensorflow as tf
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.util.tf_export import InvalidSymbolNameError
from .MODEL_CLASS import MODEL
from .CG_Layers import InstanceNormalization, ResnetBlock
from .SG_LR_SCHEDULER import StarGANSchedule


class RESNET_GENERATOR(MODEL):
# resnet generator, based off of https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
    def __init__(self, input_shape, n_labels, output_dir, lr=StarGANSchedule(0.0001, 10, 10), name='resnet_gen'):
        super(RESNET_GENERATOR, self).__init__(
            output_dir,
            name=name,
            model_args={'input_shape':input_shape, 'n_labels':n_labels, 'n_resnet':6},
            optimizer_args={'lr':lr, 'beta_1':0.5, 'beta_2':0.999}
        )

    def _build_model(self, input_shape=[None, None, 3], n_labels=None, n_resnet=6):
        init = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.Input(shape=input_shape)
        inp_labels = tf.keras.Input(shape=[n_labels])
        labels = tf.keras.layers.RepeatVector(input_shape[-3]*input_shape[-2])(inp_labels)
        labels = tf.keras.layers.Reshape((input_shape[-3], input_shape[-2], n_labels)) (labels)
        concat = tf.keras.layers.Concatenate()([inp, labels])

        #conv block 1
        conv_1 = tf.keras.layers.Conv2D(64, (7,7), padding='same', kernel_initializer=init)(concat)
        norm_1 = InstanceNormalization()(conv_1)
        act_1 = tf.keras.layers.ReLU()(norm_1)

        #conv block 2
        conv_2 = tf.keras.layers.Conv2D(128, (3,3), strides=2, padding='same', kernel_initializer=init)(act_1)
        norm_2 = InstanceNormalization()(conv_2)
        act_2 = tf.keras.layers.ReLU()(norm_2)

        #conv block 3
        conv_3 = tf.keras.layers.Conv2D(256, (3,3), strides=2, padding='same', kernel_initializer=init)(act_2)
        norm_3 = InstanceNormalization()(conv_3)
        act_3 = tf.keras.layers.ReLU()(norm_3)

        #resnet blocks
        resnet_blk = ResnetBlock(256, norm_type='instancenorm')(act_3)
        for _ in range(1, n_resnet):
            resnet_blk = ResnetBlock(256, norm_type='instancenorm')(resnet_blk)

        #deconv block 1
        deconv_1 = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', kernel_initializer=init)(resnet_blk)
        norm_4 = InstanceNormalization()(deconv_1)
        act_4 = tf.keras.layers.ReLU()(norm_4)

        deconv_2 = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', kernel_initializer=init)(act_4)
        norm_5 = InstanceNormalization()(deconv_2)
        act_5 = tf.keras.layers.ReLU()(norm_5)

        conv_4 = tf.keras.layers.Conv2D(3, (7,7), padding='same', kernel_initializer=init)(act_5)
        norm_6 = InstanceNormalization()(conv_4)
        outp = tf.keras.layers.Activation('tanh')(norm_6)

        return tf.keras.Model(inputs=[inp, inp_labels], outputs=outp, name=self.name)

    def _build_optimizer(self, lr=2e-4, beta_1=0.5, beta_2=0.999, **kwargs):
        return tf.keras.optimizers.Adam(lr, beta_1=beta_1, beta_2=beta_2, **kwargs)


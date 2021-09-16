import tensorflow as tf
from .MODEL_CLASS import MODEL
from .CG_Layers import InstanceNormalization, DownsampleBlock


class PIX2PIX_DISC(MODEL):
# pix2pix discriminator, implemented from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    def __init__(self, input_shape, output_dir, name='pix2pix_disc'):
        super(PIX2PIX_DISC, self).__init__(
            output_dir,
            name=name,
            model_args={'input_shape':input_shape, 'norm_type':'instancenorm', 'target':False},
            optimizer_args={'lr':2e-4, 'beta_1':0.5}
        )

    def _build_model(self, input_shape=[None, None, 3], norm_type='instancenorm', target=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
        x = inp
        model_input = inp
        if target:
            tar = tf.keras.layers.Input(shape=input_shape, name='target_image')
            x = tf.keras.layers.Concatenate([inp, tar])
            model_input = [inp, tar]
        down1 = DownsampleBlock(64, 4, norm_type, False) (x)
        down2 = DownsampleBlock(128, 4, norm_type, False) (down1)
        down3 = DownsampleBlock(256, 4, norm_type, False) (down2)

        zero_pad1=tf.keras.layers.ZeroPadding2D()(down3)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False) (zero_pad1)
        if norm_type.lower() == 'batchnorm':
            norm1 = tf.keras.layers.BatchNormalization()(conv)
        elif norm_type.lower() == 'instancenorm':
            norm1 = InstanceNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

        zero_pad2=tf.keras.layers.ZeroPadding2D()(leaky_relu)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

        return tf.keras.Model(inputs=model_input, outputs=last, name=self.name)

    def _build_optimizer(self, lr=2e-4, beta_1=0.5):
        return tf.keras.optimizers.Adam(lr, beta_1=beta_1)

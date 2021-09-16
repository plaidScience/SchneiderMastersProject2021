import tensorflow as tf
from .MODEL_CLASS import MODEL
from .CG_Layers import InstanceNormalization, DownsampleBlock, UpsampleBlock


class UNET_GENERATOR(MODEL):
#UNET Generator, implemented from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    def __init__(self, output_dir, input_shape, name='unet_gen'):
        super(UNET_GENERATOR, self).__init__(
            output_dir,
            name=name,
            model_args={'input_shape':input_shape, 'output_channels':3, 'norm_type':'instancenorm'},
            optimizer_args={'lr':2e-4, 'beta_1':0.5}
        )

    def _build_model(self, input_shape=[None, None, 3], output_channels=3, norm_type='instancenorm',):
        down_stack = [
            DownsampleBlock(64, 4, norm_type, apply_norm=False),
            DownsampleBlock(128, 4, norm_type),
            DownsampleBlock(256, 4, norm_type),
            DownsampleBlock(512, 4, norm_type),
            DownsampleBlock(512, 4, norm_type),
            DownsampleBlock(512, 4, norm_type),
            DownsampleBlock(512, 4, norm_type),
            DownsampleBlock(512, 4, norm_type),
        ]

        up_stack = [
            UpsampleBlock(512, 4, norm_type, apply_dropout=True),
            UpsampleBlock(512, 4, norm_type, apply_dropout=True),
            UpsampleBlock(512, 4, norm_type, apply_dropout=True),
            UpsampleBlock(512, 4, norm_type),
            UpsampleBlock(256, 4, norm_type),
            UpsampleBlock(128, 4, norm_type),
            UpsampleBlock(64, 4, norm_type),

        ]
        initializer = tf.random_normal_initializer(0., 0.02)

        last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same',
            kernel_initializer=initializer, activation='tanh')

        inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
        x = inp

        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inp, outputs=x)

    def _build_optimizer(self, lr=2e-4, beta_1=0.5):
        return tf.keras.optimizers.Adam(lr, beta_1=beta_1)

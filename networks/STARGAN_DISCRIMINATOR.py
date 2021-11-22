import tensorflow as tf
from .MODEL_CLASS import MODEL
from .CG_Layers import InstanceNormalization, DownsampleBlock
from .SG_LR_SCHEDULER import StarGANSchedule


class PIX2PIX_DISC(MODEL):
# pix2pix discriminator, implemented from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    def __init__(self, input_shape, n_classes, output_dir, lr=StarGANSchedule(0.0001, 10, 10), name='pix2pix_disc'):
        super(PIX2PIX_DISC, self).__init__(
            output_dir,
            name=name,
            model_args={'input_shape':input_shape, 'n_classes':n_classes},
            optimizer_args={'lr':lr, 'beta_1':0.5, 'beta_2':0.999}
        )

    def _build_model(self, input_shape=[None, None, 3], n_classes=3):

        norm_type='none'

        inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
        x = inp

        x = tf.keras.layers.ZeroPadding2D(1)(x)
        x = tf.keras.layers.Conv2D(64, (4,4), strides=2, padding='none')(x)
        x= tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.ZeroPadding2D(1)(x)
        x = tf.keras.layers.Conv2D(128, (4,4), strides=2, padding='none')(x)
        x= tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.ZeroPadding2D(1)(x)
        x = tf.keras.layers.Conv2D(256, (4,4), strides=2, padding='none')(x)
        x= tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.ZeroPadding2D(1)(x)
        x = tf.keras.layers.Conv2D(512, (4,4), strides=2, padding='none')(x)
        x= tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.ZeroPadding2D(1)(x)
        x = tf.keras.layers.Conv2D(1024, (4,4), strides=2, padding='none')(x)
        x= tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.ZeroPadding2D(1)(x)
        x = tf.keras.layers.Conv2D(2048, (4,4), strides=2, padding='none')(x)
        x= tf.keras.layers.LeakyReLU()(x)

        
        x1=tf.keras.layers.ZeroPadding2D()(x)
        src = tf.keras.layers.Conv2D(1, 3, strides=1)(x1)

        cls = tf.keras.layers.Conv2D(n_classes, (input_shape[-3]//64, input_shape[-2]//64), strides=1)(x)
        cls_flattened = tf.keras.layers.Flatten()(cls)

        return tf.keras.Model(inputs=inp, outputs=[src, cls_flattened], name=self.name)

    def _build_optimizer(self, lr=2e-4, beta_1=0.5, beta_2=0.999, **kwargs):
        return tf.keras.optimizers.Adam(lr, beta_1=beta_1, beta_2=beta_2, **kwargs)

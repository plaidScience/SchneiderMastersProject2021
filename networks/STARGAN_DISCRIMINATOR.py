import tensorflow as tf
from .MODEL_CLASS import MODEL
from .CG_Layers import InstanceNormalization, DownsampleBlock
from .SG_LR_SCHEDULER import StarGANSchedule


class PIX2PIX_DISC(MODEL):
# pix2pix discriminator, implemented from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    def __init__(self, input_shape, n_classes, output_dir, name='pix2pix_disc'):
        sg_schedule = StarGANSchedule(0.0001, 10, 10)
        super(PIX2PIX_DISC, self).__init__(
            output_dir,
            name=name,
            model_args={'input_shape':input_shape, 'n_classes':n_classes},
            optimizer_args={'lr':sg_schedule, 'beta_1':0.5, 'beta_2':0.999}
        )

    def _build_model(self, input_shape=[None, None, 3], n_classes=3):
        initializer = tf.random_normal_initializer(0., 0.02)

        norm_type='none'

        inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
        x = inp

        down1 = DownsampleBlock(64, 4, norm_type, False) (x)
        down2 = DownsampleBlock(128, 4, norm_type, False) (down1)
        down3 = DownsampleBlock(256, 4, norm_type, False) (down2)
        down4 = DownsampleBlock(512, 4, norm_type, False) (down3)
        down5 = DownsampleBlock(1024, 4, norm_type, False) (down4)
        down6 = DownsampleBlock(2048, 4, norm_type, False) (down5)

        
        zero_pad2=tf.keras.layers.ZeroPadding2D()(down6)
        src = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
        cls = tf.keras.layers.Conv2D(n_classes, (input_shape[-3]//64, input_shape[-2]//64), strides=1, kernel_initializer=initializer)(down6)

        return tf.keras.Model(inputs=inp, outputs=[src, cls], name=self.name)

    def _build_optimizer(self, lr=2e-4, beta_1=0.5):
        return tf.keras.optimizers.Adam(lr, beta_1=beta_1)

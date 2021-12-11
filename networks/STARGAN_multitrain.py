from re import T
from numpy.core.numeric import False_
from six import b
import tensorflow as tf

import datetime
import time

import os

import matplotlib.pyplot as plt
import io

from tensorflow.python.ops.gen_array_ops import one_hot

from . import STARGAN_GENERATOR as generator
from . import STARGAN_DISCRIMINATOR as discriminator
from .STARGAN import StarGAN


class StarGAN_MultiCycle(StarGAN):
    def __init__(self, input_shape, n_classes, output_dir, time_created=None, preprocess_model=None, strategy=None, restore_model_from_checkpoint = True):
        super(StarGAN_MultiCycle, self).__init__(
            input_shape,
            n_classes,
            output_dir,
            time_created=time_created,
            preprocess_model=preprocess_model,
            strategy=strategy,
            restore_model_from_checkpoint=restore_model_from_checkpoint
        )
        self.LAMBDA_multi = [1.0, 0.25, 0.125, 0.0625]

        self.gen_loss_keys = ['loss_adv', 'loss_cls', 'loss_cyc', 'loss_multicycle']
        self.disc_loss_keys = ['loss_adv', 'loss_cls', 'loss_gp']

    @tf.function
    def _train_preprocess(self, inp):
        real, cls = inp['image'], inp['label']
        targets = self.gen_targets(cls, len(self.LAMBDA_multi))
        if self.preprocess_model is not None:
            real = self.preprocess_model(real)
        return real, cls, targets

    @tf.function
    def _train_step_disc(self, real, cls, target):

        with tf.GradientTape() as d_tape:
            d_real, d_real_pred = self.disc(real)
            fake = self.gen([real, target])
            d_fake, d_fake_pred = self.disc(fake)

            d_loss_real = - self.adverserial_loss(d_real)
            d_loss_fake = self.adverserial_loss(d_fake)

            d_loss_cls = self.classifier_loss(cls, d_real_pred)*self.LAMBDA_class

            d_loss_gp = self.gradient_penalty(real, fake)*self.LAMBDA_gp


            d_loss = d_loss_real+d_loss_fake + d_loss_cls + d_loss_gp
            
        disc_gradients = d_tape.gradient(d_loss, self.disc.model.trainable_variables)
        self.disc.optimizer.apply_gradients(zip(disc_gradients, self.disc.model.trainable_variables))

        return (d_loss_real +d_loss_fake), d_loss_cls, d_loss_gp

    @tf.function
    def _train_step_gen(self, real, cls, targets):
        with tf.GradientTape() as g_tape:
            fake = self.gen([real, targets[0]])
            g_fake, g_fake_pred = self.disc(fake)

            g_loss_fake = - self.adverserial_loss(g_fake)*self.LAMBDA_multi[0]
            g_loss_cls = self.classifier_loss(targets[0], g_fake_pred)*self.LAMBDA_class*self.LAMBDA_multi[0]

            cycled = self.gen([fake, cls])
            g_loss_cycled = self.cycle_loss(real, cycled)*self.LAMBDA_cycle*self.LAMBDA_multi[0]

            g_loss_multi = 0
            for i in range(1, len(self.LAMBDA_multi)):
                fake = self.gen([fake, targets[i]])
                g_fake, g_fake_pred = self.disc(fake)

                g_loss_multi -= self.adverserial_loss(g_fake)*self.LAMBDA_multi[i]
                g_loss_multi += self.classifier_loss(targets[i], g_fake_pred)*self.LAMBDA_class*self.LAMBDA_multi[i]

                cycled = self.gen([fake, cls])
                g_loss_multi += self.cycle_loss(real, cycled)*self.LAMBDA_cycle*self.LAMBDA_multi[i]

            g_loss = g_loss_fake + g_loss_cls + g_loss_cycled + g_loss_multi

        gen_gradients = g_tape.gradient(g_loss, self.gen.model.trainable_variables)
        self.gen.optimizer.apply_gradients(zip(gen_gradients, self.gen.model.trainable_variables))

        return g_loss_fake, g_loss_cls, g_loss_cycled, g_loss_multi

    @tf.function
    def _train_step(self, data, step):

        imgs, cls, targets = self._train_preprocess(data)
        total_disc_loss = self._train_step_disc(imgs, cls, targets[0])
        total_gen_loss = self._train_step_gen(imgs, cls, targets) if (step+1)%self.gen_rate == 0 else (-1.0, -1.0, -1.0, -1.0)
        

        return total_gen_loss, total_disc_loss

    @tf.function
    def _test_preprocess(self, inp, is_training=False):
        real, cls = inp['image'], inp['label']
        targets = self.gen_targets(cls, len(self.LAMBDA_multi))
        if self.preprocess_model is not None:
            real = self.preprocess_model(real, training=is_training)
        return real, cls, targets

    @tf.function
    def _test_step_disc(self, real, cls, target, is_training=False):

        d_real, d_real_pred = self.disc(real, training=is_training)
        fake = self.gen([real, target], training=is_training)
        d_fake, d_fake_pred = self.disc(fake, training=is_training)

        d_loss_real = - self.adverserial_loss(d_real)
        d_loss_fake = self.adverserial_loss(d_fake)

        d_loss_cls = self.classifier_loss(cls, d_real_pred)*self.LAMBDA_class

        d_loss_gp = self.gradient_penalty(real, fake)*self.LAMBDA_gp

        return (d_loss_real+d_loss_fake), d_loss_cls, d_loss_gp

    @tf.function
    def _test_step_gen(self, real, cls, targets, is_training=False):
        fake = self.gen([real, targets[0]])
        g_fake, g_fake_pred = self.disc(fake)

        g_loss_fake = - self.adverserial_loss(g_fake)*self.LAMBDA_multi[0]
        g_loss_cls = self.classifier_loss(targets[0], g_fake_pred)*self.LAMBDA_class*self.LAMBDA_multi[0]

        cycled = self.gen([fake, cls])
        g_loss_cycled = self.cycle_loss(real, cycled)*self.LAMBDA_cycle*self.LAMBDA_multi[0]

        g_loss_multi=0

        for i in range(1, len(self.LAMBDA_multi)):
            fake = self.gen([fake, targets[i]])
            g_fake, g_fake_pred = self.disc(fake)

            g_loss_multi -= self.adverserial_loss(g_fake)*self.LAMBDA_multi[i]
            g_loss_multi += self.classifier_loss(targets[0], g_fake_pred)*self.LAMBDA_class*self.LAMBDA_multi[i]

            cycled = self.gen([fake, cls])
            g_loss_multi += self.cycle_loss(real, cycled)*self.LAMBDA_cycle*self.LAMBDA_multi[i]

        return g_loss_fake, g_loss_cls, g_loss_cycled, g_loss_multi

    @tf.function
    def _test_step(self, data, step, is_training=False):

        imgs, cls, targets = self._test_preprocess(data, is_training)
        total_disc_loss = self._test_step_disc(imgs, cls, targets[0], is_training)
        total_gen_loss = self._test_step_gen(imgs, cls, targets, is_training)
        
        return total_gen_loss, total_disc_loss
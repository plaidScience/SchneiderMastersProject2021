from re import T
from numpy.core.numeric import False_
import tensorflow as tf

import datetime
import time

import os

import matplotlib.pyplot as plt
import io

from . import STARGAN_GENERATOR as generator
from . import STARGAN_DISCRIMINATOR as discriminator

class StarganGAN():
    def __init__(self, input_shape, n_classes, ouput_dir, modeltype='resnet'):
        self.time_created = datetime.datetime.now().strftime("%m_%d/%H/")
        self.output_dir = os.path.join(ouput_dir, self.time_created)
        self.input_shape = input_shape
        self.n_classes = n_classes
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.gen, self.disc = self._create_models('generator', 'discriminator', modeltype)
        print("Generator:")
        self.gen.summary()
        print("Discriminator:")
        self.disc.summary()

        self.checkpoint_folder = os.path.join(self.output_dir, 'checkpoints')

        self.checkpoint = tf.train.Checkpoint(
            gen = self.gen.model,
            gen_optimizer = self.gen.optimizer,
            disc = self.disc.model,
            disc_optimizer = self.disc.optimizer,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, os.path.join(self.checkpoint_folder, 'checkpoint'), max_to_keep=5)

        self.xentropy_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.mae_loss_object = tf.keras.losses.MeanAbsoluteError()
        self.LAMBDA_class = 1
        self.LAMBDA_cycle = 1
        self.LAMBDA_gp = 1


    def _create_models(self, gen_name, disc_name, modeltype):
        gen = generator.RESNET_GENERATOR(self.input_shape, self.n_classes, self.output_dir, gen_name)

        disc = discriminator.PIX2PIX_DISC(self.input_shape, self.n_classes, self.output_dir, disc_name)

        return gen, disc

    def adverserial_loss(self, output):
        return tf.reduce_mean(output)
    
    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform((real.shape[0], 1, 1, 1), minval=0.0, maxval=1.0)
        x_hat = (alpha*real + (1-alpha)*fake)
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.disc(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx-1.0), ** 2)
        return d_regularizer

    def classifier_loss(self, images, classes):
        loss = self.xentropy_loss_object(images, classes)
        return loss

    def cycle_loss(self, real, cycled):
        loss = self.mae_loss_object(real, cycled)
        return loss

    def _train_preprocess(self, inp):
        real, cls = inp
        target = self.classes[tf.random.uniform(shape=1, minval=0, maxval=self.n_classes, dtype=tf.dtypes.int64)]
        return real, cls, target

    def _train_step_disc(self, real, cls, target):

        with tf.GradientTape() as d_tape:
            d_real, d_real_pred = self.disc(real)
            fake = self.gen(real, target)
            d_fake, d_fake_pred = self.disc(fake)

            d_loss_real = - self.adverserial_loss(d_real)
            d_loss_fake = self.adverserial_loss(d_fake)

            d_loss_cls = self.classifier_loss(cls, d_real_pred)

            d_loss_gp = self.gradient_penalty(real, fake)


            d_loss = d_loss_real+d_loss_fake+d_loss_cls*self.LAMBDA_class+d_loss_gp*self.LAMBDA_gp
            
        disc_gradients = d_tape.gradient(d_loss, self.disc.model.trainable_variables)
        self.disc.optimizer.apply_gradients(zip(disc_gradients, self.disc.model.trainable_variables))

        return d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp

    def _train_step_gen(self, real, cls, target):
        with tf.GradientTape() as g_tape:
            fake = self.gen(real, target)
            g_fake, g_fake_pred = self.disc(fake)
            
            cycled = self.gen(fake, cls)

            g_loss_fake = self.adverserial_loss(g_fake)
            g_loss_cls = self.classifier_loss(target, g_fake_pred)

            g_loss_cycled = self.cycle_loss(real, cycled)

            g_loss = g_loss_fake + g_loss_cls*self.LAMBDA_class + g_loss_cycled*self.LAMBDA_cycle

        gen_gradients = g_tape.gradient(g_loss, self.gen.model.trainable_variables)
        self.gen.optimizer.apply_gradients(zip(gen_gradients, self.gen.model.trainable_variables))

        return g_loss_fake, g_loss_cls, g_loss_cycled


    def _train_step(self, data):

        imgs, cls, target = self._train_preprocess(data)
        total_disc_loss = self._train_step_disc(imgs, cls, target)
        total_gen_loss = self._train_step_gen(imgs, cls, target) if self.epoch%5 == 0 else None
        

        return total_gen_loss, total_disc_loss

    def train(self, data, epochs, start_epoch=0, log_freq=1, gen_freq=5, checkpoint_freq=5):
        
        #preprocess data


        gen_loss = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
        disc_loss = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)


        for epoch in range(start_epoch, epochs):
            n = 0
            print(f'Epoch: {epoch+1}: Starting!', end='')
            start = time.time()
            for batch in data:
                batch_g, batch_d = self._train_step(batch)
                gen_loss(batch_g)
                disc_loss(batch_d)
                n+=1
                if n%5==0:
                    print(f'\rEpoch: {epoch+1}: Batch {n} completed!', end='')

            print(f'\rEpoch {epoch+1} completed in {time.time()-start:.0f}, Total {n} Batches completed!')
            print(f'\t[GENERATOR Loss]: {gen_loss.result():.04f}')
            print(f'\t[DISCRIMINATOR Loss]: {disc_loss.result():.04f}')

            if (epoch+1)%log_freq==0:
                with self.gen.logger.as_default():
                    tf.summary.scalar('loss', gen_loss.result(), step=epoch)
                with self.disc.logger.as_default():
                    tf.summary.scalar('loss', disc_loss.result(), step=epoch)

            if (epoch+1)%gen_freq==0:
                #log translated images in some form (maybe a NxN image tiling of translations?)
                self.log_images()


            if (epoch + 1) % checkpoint_freq == 0:
                ckpt_save_path = self.checkpoint_manager.save()
                print (f'Epoch {epoch+1}: Saving checkpoint at {ckpt_save_path}')



            gen_loss.reset_states()
            disc_loss.reset_states()

    def log_images(self):
        images=None

    def save_models(self, path=None):
        self.gen.save_model(path)
        self.disc.save_model(path)
    def load_models(self, path=None):
        self.gen.load_model(path)
        self.disc.load_model(path)


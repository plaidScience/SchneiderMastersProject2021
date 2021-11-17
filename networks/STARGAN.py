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


class StarGAN():
    def __init__(self, input_shape, n_classes, ouput_dir, preprocess_model=None, strategy=None):
        self.time_created = datetime.datetime.now().strftime("%m_%d/%H/")
        self.output_dir = os.path.join(ouput_dir, self.time_created)
        self.input_shape = input_shape
        self.n_classes = n_classes
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.gen, self.disc = self._create_models('generator', 'discriminator')
        self.train_step = self._train_step

        print("Generator:")
        self.gen.summary()
        print("Discriminator:")
        self.disc.summary()

        self.preprocess_model = preprocess_model

        self.checkpoint_folder = os.path.join(self.output_dir, 'checkpoints')

        self.checkpoint = tf.train.Checkpoint(
            gen = self.gen.model,
            gen_optimizer = self.gen.optimizer,
            disc = self.disc.model,
            disc_optimizer = self.disc.optimizer,
            preprocess_model = self.preprocess_model

        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, os.path.join(self.checkpoint_folder, 'checkpoint'), max_to_keep=5)

        self.xentropy_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.mae_loss_object = tf.keras.losses.MeanAbsoluteError()
        self.mse_loss_object = tf.keras.losses.MeanSquaredError()
        self.LAMBDA_class = 1
        self.LAMBDA_cycle = 10
        self.LAMBDA_gp = 10

        self.gen_rate=5


    def _create_models(self, gen_name, disc_name):
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
        d_regularizer = self.mse_loss_object(ddx, 1.0)
        return d_regularizer

    def classifier_loss(self, images, classes):
        loss = self.xentropy_loss_object(images, classes)
        return loss

    def cycle_loss(self, real, cycled):
        loss = self.mae_loss_object(real, cycled)
        return loss

    def gen_targets(self, cls, num=1):
        list_targets = []
        for i in range(num):
            target = tf.random.shuffle(cls)[0]
            target = tf.repeat([target], [tf.shape(cls)[0]], axis=0)
            list_targets.append(target)
        targets = tf.stack(list_targets, 0)
        return targets

    def gen_target(self, cls):
        target = tf.random.shuffle(cls)[0]
        target = tf.repeat([target], [tf.shape(cls)[0]], axis=0)
        return target
    
    def gen_onehot_target(self, batch_size):
        idx = tf.random.uniform(shape=[], maxval=self.n_classes, dtype=tf.int32)
        target = tf.one_hot([idx], self.n_classes)
        target = tf.repeat(target, batch_size, axis=0)
        return target

    @tf.function
    def _train_preprocess(self, inp):
        real, cls = inp['image'], inp['label']
        target = self.gen_target(cls)
        if self.preprocess_model is not None:
            real = self.preprocess_model(real)
        return real, cls, target

    @tf.function
    def _train_step_disc(self, real, cls, target):

        with tf.GradientTape() as d_tape:
            d_real, d_real_pred = self.disc(real)
            fake = self.gen([real, target])
            d_fake, d_fake_pred = self.disc(fake)

            d_loss_real = - self.adverserial_loss(d_real)
            d_loss_fake = self.adverserial_loss(d_fake)

            d_loss_cls = self.classifier_loss(cls, d_real_pred)

            d_loss_gp = self.gradient_penalty(real, fake)


            d_loss = d_loss_real+d_loss_fake+d_loss_cls*self.LAMBDA_class+d_loss_gp*self.LAMBDA_gp
            
        disc_gradients = d_tape.gradient(d_loss, self.disc.model.trainable_variables)
        self.disc.optimizer.apply_gradients(zip(disc_gradients, self.disc.model.trainable_variables))

        return d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp

    @tf.function
    def _train_step_gen(self, real, cls, target):
        with tf.GradientTape() as g_tape:
            fake = self.gen([real, target])
            g_fake, g_fake_pred = self.disc(fake)
            
            cycled = self.gen([fake, cls])

            g_loss_fake = self.adverserial_loss(g_fake)
            g_loss_cls = self.classifier_loss(target, g_fake_pred)

            g_loss_cycled = self.cycle_loss(real, cycled)

            g_loss = g_loss_fake + g_loss_cls*self.LAMBDA_class + g_loss_cycled*self.LAMBDA_cycle

        gen_gradients = g_tape.gradient(g_loss, self.gen.model.trainable_variables)
        self.gen.optimizer.apply_gradients(zip(gen_gradients, self.gen.model.trainable_variables))

        return g_loss_fake, g_loss_cls, g_loss_cycled

    @tf.function
    def _train_step(self, data, epoch):

        imgs, cls, target = self._train_preprocess(data)
        total_disc_loss = self._train_step_disc(imgs, cls, target)
        total_gen_loss = self._train_step_gen(imgs, cls, target) if epoch%self.gen_rate == 0 else None
        

        return total_gen_loss, total_disc_loss

    def train(self, data, label_strs, epochs, start_epoch=0, batch_size=8, log_freq=1, gen_freq=5, checkpoint_freq=5):
        
        #preprocess data
        data, sample_batch=self._preprocess_data(data, batch_size, True, True)

        one_hot_labels = tf.one_hot(range(self.n_classes), self.n_classes)




        #g_loss_fake, g_loss_cls, g_loss_cycled
        total_gen_loss = tf.keras.metrics.Mean('total_gen_loss', dtype=tf.float32)
        gen_adv_loss = tf.keras.metrics.Mean('gen_adv_loss', dtype=tf.float32)
        gen_cls_loss = tf.keras.metrics.Mean('gen_cls_loss', dtype=tf.float32)
        gen_cyc_loss = tf.keras.metrics.Mean('gen_cyc_loss', dtype=tf.float32)

        #d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp
        total_disc_loss = tf.keras.metrics.Mean('total_disc_loss', dtype=tf.float32)
        disc_adv_loss = tf.keras.metrics.Mean('disc_adv_loss', dtype=tf.float32)
        disc_cls_loss = tf.keras.metrics.Mean('disc_cls_loss', dtype=tf.float32)
        disc_grad_loss = tf.keras.metrics.Mean('disc_grad_loss', dtype=tf.float32)


        for epoch in range(start_epoch, epochs):
            n = 0
            print(f'Epoch: {epoch+1}: Starting!', end='')
            start = time.time()
            for batch in data:
                batch_g, batch_d = self._train_step(batch, epoch)

                if batch_g is not None:
                    g_fake, g_cls, g_cyc = batch_g
                    total_gen_loss((g_fake+g_cls*self.LAMBDA_class+g_cyc*self.LAMBDA_cycle))
                    gen_adv_loss(g_fake)
                    gen_cls_loss(g_cls*self.LAMBDA_class)
                    gen_cyc_loss(g_cyc*self.LAMBDA_cycle)

                d_real, d_fake, d_cls, d_gp = batch_d
                total_disc_loss((d_real+d_fake+d_cls*self.LAMBDA_class+d_gp*self.LAMBDA_gp))
                disc_adv_loss(d_real+d_fake)
                disc_cls_loss(d_cls*self.LAMBDA_class)
                disc_grad_loss(d_gp*self.LAMBDA_gp)
                n+=1
                if n%5==0:
                    print(f'\rEpoch: {epoch+1}: Batch {n} completed!', end='')

            print(f'\rEpoch {epoch+1} completed in {time.time()-start:.0f}, Total {n} Batches completed!')
            if (epoch)%self.gen_rate==0:
                print(f'\t[GENERATOR Loss]: {total_gen_loss.result():.04f}')
                print(f'\t\t[GENERATOR adv]: {gen_adv_loss.result():.04f}')
                print(f'\t\t[GENERATOR cls]: {gen_cls_loss.result():.04f}')
                print(f'\t\t[GENERATOR cyc]: {gen_cyc_loss.result():.04f}')
            print(f'\t[DISCRIMINATOR Loss]: {total_disc_loss.result():.04f}')
            print(f'\t\t[DISCRIMINATOR adv]: {disc_adv_loss.result():.04f}')
            print(f'\t\t[DISCRIMINATOR cls]: {disc_cls_loss.result():.04f}')
            print(f'\t\t[DISCRIMINATOR gp ]: {disc_grad_loss.result():.04f}')


            if (epoch+1)%log_freq==0:
                if epoch%self.gen_rate == 0:
                    with self.gen.logger.as_default():
                        tf.summary.scalar('loss', total_gen_loss.result(), step=epoch)
                        tf.summary.scalar('loss_adv', gen_adv_loss.result(), step=epoch)
                        tf.summary.scalar('loss_cls', gen_cls_loss.result(), step=epoch)
                        tf.summary.scalar('loss_cyc', gen_cyc_loss.result(), step=epoch)

                with self.disc.logger.as_default():
                    tf.summary.scalar('loss', total_disc_loss.result(), step=epoch)
                    tf.summary.scalar('loss_adv', disc_adv_loss.result(), step=epoch)
                    tf.summary.scalar('loss_cls', disc_cls_loss.result(), step=epoch)
                    tf.summary.scalar('loss_gp', disc_grad_loss.result(), step=epoch)

            if (epoch+1)%gen_freq==0:
                #log translated images in some form (maybe a NxN image tiling of translations?)
                for i in range(self.n_classes):
                    self.log_images(epoch, sample_batch, one_hot_labels[i], label_strs[i], num_images=batch_size)


            if (epoch + 1) % checkpoint_freq == 0:
                ckpt_save_path = self.checkpoint_manager.save()
                print (f'Epoch {epoch+1}: Saving checkpoint at {ckpt_save_path}')



            total_gen_loss.reset_states()
            gen_adv_loss.reset_states()
            gen_cls_loss.reset_states()
            gen_cyc_loss.reset_states()
            total_disc_loss.reset_states()
            disc_adv_loss.reset_states()
            disc_cls_loss.reset_states()
            disc_grad_loss.reset_states()

    def _preprocess_data(self, data, batch_size, shuffle=True, cache=True):
        
        data_count = tf.data.experimental.cardinality(data).numpy()
        
        if cache:
            data = data.cache()
        if shuffle:
            data = data.shuffle(data_count)
        
        data = data.batch(batch_size)

        sample_batch = next(iter(data))

        return data, sample_batch


    def log_images(self, epoch, batch, target_label, label_str, num_images=5):
        batch = batch['image'][0:num_images]
        if self.preprocess_model is not None:
            batch = self.preprocess_model(batch)
        image_size = tf.shape(batch)[-3:].numpy()
        target = tf.repeat([target_label], num_images, axis=0)
        predictions = self.gen([batch, target])
        predictions = (predictions+1.0)/2.0
        batch = (batch+1.0)/2.0
        dpi = 100.
        w_pad = 2/72.
        h_pad = 2/72.
        plot_width = (image_size[0]+2*w_pad*image_size[0])*2./dpi
        plot_height = (image_size[1]+2*h_pad*image_size[1])/dpi
        fig, ax = plt.subplots(1, 2, figsize=(plot_width, plot_height), dpi=dpi)

        title = ['input', label_str]

        img = [None, None]

        for i in range(2):
            ax[i].set_title(title[i])
            ax[i].axis('off')
            img[i] = ax[i].imshow(tf.zeros(image_size))

        images_list = []
        for image, prediction in zip(batch, predictions):
            img[0].set_data(image*0.5 + 0.5)
            img[1].set_data(prediction*0.5 + 0.5)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')

            image = tf.image.decode_png(buf.getvalue(), channels=3)
            images_list.append(image)
        out_images = tf.stack(images_list, axis=0)
        with self.gen.logger.as_default():
            tf.summary.image(label_str, out_images, max_outputs=num_images, step=epoch)
        plt.close()

    def save_models(self, path=None):
        self.gen.save_model(path)
        self.disc.save_model(path)
    def load_models(self, path=None):
        self.gen.load_model(path)
        self.disc.load_model(path)


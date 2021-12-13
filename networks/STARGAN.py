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
    def __init__(self, input_shape, n_classes, output_dir, time_created=None, preprocess_model=None, strategy=None, restore_model_from_checkpoint = True):
        if time_created is None:
            time_created = datetime.datetime.now().strftime("%m_%d/%H/")
            self.output_dir = os.path.join(output_dir, time_created)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            self.restore_model = False
        else:
            self.output_dir = os.path.join(output_dir, time_created)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                self.restore_model = False
            else:
                self.restore_model = True
        self.input_shape = input_shape
        self.n_classes = n_classes
        
        
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

        if self.restore_model:
            if restore_model_from_checkpoint:
                if self.checkpoint_manager.latest_checkpoint:
                    self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                else:
                    raise FileNotFoundError(f"Checkpoint File Can't Be Found in {self.checkpoint_folder}!")
            else:
                self.load_models()

        

        self.xentropy_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mae_loss_object = tf.keras.losses.MeanAbsoluteError()
        self.LAMBDA_class = 1
        self.LAMBDA_cycle = 10
        self.LAMBDA_gp = 10

        self.gen_rate=5

        self.gen_loss_keys = ['loss_adv', 'loss_cls', 'loss_cyc']
        self.disc_loss_keys = ['loss_adv', 'loss_cls', 'loss_gp']


    def _create_models(self, gen_name, disc_name):
        gen = generator.RESNET_GENERATOR(self.input_shape, self.n_classes, self.output_dir, lr=self._get_lr, name=gen_name)

        disc = discriminator.PIX2PIX_DISC(self.input_shape, self.n_classes, self.output_dir, lr=self._get_lr, name=disc_name)

        return gen, disc
    
    def _get_lr(self):
        computed = 0.0001-((self.epoch+1)-(self.epochs/2))*(0.0001/(self.epochs/2))
        return tf.clip_by_value(computed, 0.0, 0.0001)

    def adverserial_loss(self, output):
        return tf.reduce_mean(output)
    
    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform((real.shape[0], 1, 1, 1), minval=0.0, maxval=1.0)
        x_hat = (alpha*real + (1-alpha)*fake)
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat, _ = self.disc(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        d_regularizer = tf.reduce_mean(tf.square(ddx-1.0))
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
        idx = tf.random.uniform(shape=[], minval=0, maxval=self.n_classes, dtype=tf.int32)
        target = tf.one_hot([idx], self.n_classes)
        target = tf.repeat(target, batch_size, axis=0)
        return target
    
    def gen_inv_target(self, cls, mask):

        idxs = []
        for i, mask_i in enumerate(mask):
            if mask_i == 1.0:
                idxs.append(i)
        random_mask_val = tf.one_hot(tf.random.shuffle(idxs)[0], self.n_classes, 1, 0, dtype=tf.float32)
        targets = (1-cls)*(1-mask) + random_mask_val

        return targets

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

            d_loss_cls = self.classifier_loss(cls, d_real_pred)*self.LAMBDA_class

            d_loss_gp = self.gradient_penalty(real, fake)*self.LAMBDA_gp


            d_loss = d_loss_real+d_loss_fake + d_loss_cls + d_loss_gp
            
        disc_gradients = d_tape.gradient(d_loss, self.disc.model.trainable_variables)
        self.disc.optimizer.apply_gradients(zip(disc_gradients, self.disc.model.trainable_variables))

        return (d_loss_real+d_loss_fake), d_loss_cls, d_loss_gp

    @tf.function
    def _train_step_gen(self, real, cls, target):
        with tf.GradientTape() as g_tape:
            fake = self.gen([real, target])
            g_fake, g_fake_pred = self.disc(fake)
            
            cycled = self.gen([fake, cls])

            g_loss_fake = - self.adverserial_loss(g_fake)
            g_loss_cls = self.classifier_loss(target, g_fake_pred)*self.LAMBDA_class

            g_loss_cycled = self.cycle_loss(real, cycled)*self.LAMBDA_cycle

            g_loss = g_loss_fake + g_loss_cls + g_loss_cycled

        gen_gradients = g_tape.gradient(g_loss, self.gen.model.trainable_variables)
        self.gen.optimizer.apply_gradients(zip(gen_gradients, self.gen.model.trainable_variables))

        return g_loss_fake, g_loss_cls, g_loss_cycled

    @tf.function
    def _train_step(self, data, step):

        imgs, cls, target = self._train_preprocess(data)
        total_disc_loss = self._train_step_disc(imgs, cls, target)
        total_gen_loss = self._train_step_gen(imgs, cls, target) if (step+1)%self.gen_rate == 0 else (-1.0, -1.0, -1.0)
        

        return total_gen_loss, total_disc_loss

    @tf.function
    def _test_preprocess(self, inp, is_training=False):
        real, cls = inp['image'], inp['label']
        target = self.gen_target(cls)
        if self.preprocess_model is not None:
            real = self.preprocess_model(real, training=is_training)
        return real, cls, target

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
    def _test_step_gen(self, real, cls, target, is_training=False):
        fake = self.gen([real, target], training=is_training)
        g_fake, g_fake_pred = self.disc(fake, training=is_training)
            
        cycled = self.gen([fake, cls], training=is_training)

        g_loss_fake = - self.adverserial_loss(g_fake)
        g_loss_cls = self.classifier_loss(target, g_fake_pred)*self.LAMBDA_class

        g_loss_cycled = self.cycle_loss(real, cycled)*self.LAMBDA_cycle

        return g_loss_fake, g_loss_cls, g_loss_cycled

    @tf.function
    def _test_step(self, data, step, is_training=False):

        imgs, cls, target = self._test_preprocess(data, is_training)
        total_disc_loss = self._test_step_disc(imgs, cls, target, is_training)
        total_gen_loss = self._test_step_gen(imgs, cls, target, is_training)
        
        return total_gen_loss, total_disc_loss

    def train(self, data, label_strs, epochs, data_val=None, start_epoch=1, batch_size=8, log_freq=1, gen_freq=5, checkpoint_freq=5, save_freq = 1, log_lr=False):
        
        #preprocess data
        data, sample_batch=self._preprocess_data(data, batch_size, True, True, True)

        hair_locs = ['hair' in lstr.lower() for lstr in label_strs]

        one_hot_labels = tf.one_hot(range(self.n_classes), self.n_classes, dtype=tf.float32)




        #g_loss_fake, g_loss_cls, g_loss_cycled
        total_gen_loss = tf.keras.metrics.Mean('total_gen_loss', dtype=tf.float32)
        gen_losses = []
        for i in self.gen_loss_keys:
            gen_losses.append(tf.keras.metrics.Mean('gen_'+i, dtype=tf.float32))

        #d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp
        total_disc_loss = tf.keras.metrics.Mean('total_disc_loss', dtype=tf.float32)
        disc_losses = []
        for i in self.disc_loss_keys:
            disc_losses.append(tf.keras.metrics.Mean('disc_'+i, dtype=tf.float32))

        if data_val is not None:
            validate=True
            data_val, sample_val = self._preprocess_data(data_val, batch_size, True, True, True)
            #g_loss_fake, g_loss_cls, g_loss_cycled
            total_gen_loss_val = tf.keras.metrics.Mean('total_gen_loss_val', dtype=tf.float32)
            gen_val_losses = []
            for i in self.gen_loss_keys:
                gen_val_losses.append(tf.keras.metrics.Mean('gen_val_'+i, dtype=tf.float32))

            #d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp
            total_disc_loss_val = tf.keras.metrics.Mean('total_disc_loss_val', dtype=tf.float32)
            disc_val_losses = []
            for i in self.disc_loss_keys:
                disc_val_losses.append(tf.keras.metrics.Mean('disc_val_'+i, dtype=tf.float32))
        else:
            validate = False


        self.epochs = epochs
        for epoch in range(start_epoch-1, epochs):
            n = 0
            self.epoch = epoch
            print(f'Epoch: {epoch+1}: Starting!', end='')
            start = time.time()
            for batch_id, batch in data.enumerate():
                batch_g, batch_d = self._train_step(batch, batch_id)

                if (batch_id+1)%self.gen_rate == 0:
                    total_gen_loss(tf.math.reduce_sum(batch_g))
                    for i in range(len(gen_losses)):
                        gen_losses[i](batch_g[i])

                total_disc_loss(tf.math.reduce_sum(batch_d))
                for i in range(len(disc_losses)):
                    disc_losses[i](batch_d[i])
                n+=1
                if n%5==0:
                    print(f'\rEpoch: {epoch+1}: Batch {n} completed!', end='')

            print(f'\rEpoch {epoch+1} completed in {time.time()-start:.0f}, Total {n} Batches completed!')
            print(f'\t[GENERATOR Loss]: {total_gen_loss.result():.04f}')
            for i in range(len(gen_losses)):
                print(f'\t\t[GENERATOR {self.gen_loss_keys[i]}]: {gen_losses[i].result():.04f}')
            print(f'\t[DISCRIMINATOR Loss]: {total_disc_loss.result():.04f}')
            for i in range(len(disc_losses)):
                print(f'\t\t[DISCRIMINATOR {self.disc_loss_keys[i]}]: {disc_losses[i].result():.04f}')
            
            if validate:
                print(f'Validation: {epoch+1}: Starting!', end='')
                n=0
                start = time.time()
                for batch_id, batch in data_val.enumerate():
                    batch_g, batch_d = self._test_step(batch, batch_id)

                    total_gen_loss_val(tf.math.reduce_sum(batch_g))
                    for i in range(len(gen_val_losses)):
                        gen_val_losses[i](batch_g[i])

                    total_disc_loss_val(tf.math.reduce_sum(batch_d))
                    for i in range(len(disc_val_losses)):
                        disc_val_losses[i](batch_d[i])
                    n+=1
                    if n%5==0:
                        print(f'\rValidation: {epoch+1}: Batch {n} completed!', end='')

                print(f'\rValidation {epoch+1} completed in {time.time()-start:.0f}, Total {n} Batches completed!')
                print(f'\t[GENERATOR Loss]: {total_gen_loss_val.result():.04f}')
                for i in range(len(gen_val_losses)):
                    print(f'\t\t[GENERATOR {self.gen_loss_keys[i]}]: {gen_val_losses[i].result():.04f}')
                print(f'\t[DISCRIMINATOR Loss]: {total_disc_loss_val.result():.04f}')
                for i in range(len(disc_val_losses)):
                    print(f'\t\t[DISCRIMINATOR {self.disc_loss_keys[i]}]: {disc_val_losses[i].result():.04f}')


            if (epoch+1)%log_freq==0:
                with self.gen.logger.as_default():
                    tf.summary.scalar('loss', total_gen_loss.result(), step=epoch)
                    for i in range(len(gen_losses)):
                        tf.summary.scalar(self.gen_loss_keys[i], gen_losses[i].result(), step=epoch)

                with self.disc.logger.as_default():
                    tf.summary.scalar('loss', total_disc_loss.result(), step=epoch)
                    for i in range(len(disc_losses)):
                        tf.summary.scalar(self.disc_loss_keys[i], disc_losses[i].result(), step=epoch)
                
                if validate:
                    with self.gen.val_logger.as_default():
                        tf.summary.scalar('loss', total_gen_loss_val.result(), step=epoch)
                        for i in range(len(gen_val_losses)):
                            tf.summary.scalar(self.gen_loss_keys[i], gen_val_losses[i].result(), step=epoch)

                    with self.disc.val_logger.as_default():
                        tf.summary.scalar('loss', total_disc_loss_val.result(), step=epoch)
                        for i in range(len(disc_val_losses)):
                            tf.summary.scalar(self.disc_loss_keys[i], disc_val_losses[i].result(), step=epoch)


            if (epoch+1)%gen_freq==0:
                #log translated images in some form (maybe a NxN image tiling of translations?)
                self.log_image(epoch, sample_batch, one_hot_labels, label_strs, num_images=batch_size, hair_locs=hair_locs)
                #for i in range(self.n_classes):
                #    self.log_images(epoch, sample_batch, one_hot_labels[i], label_strs[i], num_images=batch_size//2, hair_locs=hair_locs)
                if validate:
                    self.log_image(epoch, sample_val, one_hot_labels, label_strs, num_images=batch_size, hair_locs=hair_locs, validation=True)
                    #for i in range(self.n_classes):
                    #    self.log_images(epoch, sample_val, one_hot_labels[i], label_strs[i], num_images=batch_size//2, hair_locs=hair_locs, validation=True)



            if (epoch + 1) % checkpoint_freq == 0:
                ckpt_save_path = self.checkpoint_manager.save()
                print (f'Epoch {epoch+1}: Saving checkpoint at {ckpt_save_path}')
            
            if (epoch+1)%save_freq == 0 or epoch == (epochs-1):
                self.save_models()
            
            if log_lr:
                with self.gen.logger.as_default():
                    tf.summary.scalar('lr', self.gen.optimizer.lr, step=epoch)
                with self.disc.logger.as_default():
                    tf.summary.scalar('lr', self.disc.optimizer.lr, step=epoch)



            total_gen_loss.reset_states()
            for i in range(len(gen_losses)):
                gen_losses[i].reset_states()
            total_disc_loss.reset_states()
            for i in range(len(disc_losses)):
                disc_losses[i].reset_states()
            if validate:
                total_gen_loss_val.reset_states()
                for i in range(len(gen_val_losses)):
                    gen_val_losses[i].reset_states()
                total_disc_loss_val.reset_states()
                for i in range(len(disc_val_losses)):
                    disc_val_losses[i].reset_states()
    
    def test(self, data, label_strs, batch_size=8, log_at=-1):
        
        #preprocess data
        data, sample_batch=self._preprocess_data(data, batch_size, True, True, True)

        hair_locs = ['hair' in lstr.lower() for lstr in label_strs]

        one_hot_labels = tf.one_hot(range(self.n_classes), self.n_classes, dtype=tf.float32)




        total_gen_loss = tf.keras.metrics.Mean('total_gen_loss', dtype=tf.float32)
        gen_losses = []
        for i in self.gen_loss_keys:
            gen_losses.append(tf.keras.metrics.Mean('gen_'+i, dtype=tf.float32))

        #d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp
        total_disc_loss = tf.keras.metrics.Mean('total_disc_loss', dtype=tf.float32)
        disc_losses = []
        for i in self.disc_loss_keys:
            disc_losses.append(tf.keras.metrics.Mean('disc_'+i, dtype=tf.float32))
        print(f'Testing: Starting!', end='')
        start = time.time()
        n=0
        for batch_id, batch in data.enumerate():

            batch_g, batch_d = self._test_step(batch, batch_id)

            
            total_gen_loss(tf.math.reduce_sum(batch_g))
            for i in range(len(gen_losses)):
                gen_losses[i](batch_g[i])

            total_disc_loss(tf.math.reduce_sum(batch_d))
            for i in range(len(disc_losses)):
                disc_losses[i](batch_d[i])
            n+=1
            if n%5==0:
                print(f'\rTesting: Batch {n} completed!', end='')

        print(f'\rTesting completed in {time.time()-start:.0f}, Total {n} Batches completed!')
        print(f'\t[GENERATOR Loss]: {total_gen_loss.result():.04f}')
        for i in range(len(gen_losses)):
            print(f'\t\t[GENERATOR {self.gen_loss_keys[i]}]: {gen_losses[i].result():.04f}')
        print(f'\t[DISCRIMINATOR Loss]: {total_disc_loss.result():.04f}')
        for i in range(len(disc_losses)):
            print(f'\t\t[DISCRIMINATOR {self.disc_loss_keys[i]}]: {disc_losses[i].result():.04f}')

        with self.gen.test_logger.as_default():
            tf.summary.scalar('loss', total_gen_loss.result(), step=log_at)
            for i in range(len(gen_losses)):
               tf.summary.scalar(self.gen_loss_keys[i], gen_losses[i].result(), step=log_at)

        with self.disc.test_logger.as_default():
            tf.summary.scalar('loss', total_disc_loss.result(), step=log_at)
            for i in range(len(disc_losses)):
                tf.summary.scalar(self.disc_loss_keys[i], disc_losses[i].result(), step=log_at)
            
        #log translated images in some form (maybe a NxN image tiling of translations?)
        self.log_image(log_at, sample_batch, one_hot_labels, label_strs, num_images=batch_size, hair_locs=hair_locs, testing=True)
        #for i in range(self.n_classes):
        #    self.log_images(log_at, sample_batch, one_hot_labels[i], label_strs[i], num_images=batch_size//2, hair_locs=hair_locs, testing=True)

        total_gen_loss.reset_states()
        for i in range(len(gen_losses)):
            gen_losses[i].reset_states()
        total_disc_loss.reset_states()
        for i in range(len(disc_losses)):
            disc_losses[i].reset_states()

    def _preprocess_data(self, data, batch_size, shuffle=True, shuffle_all_every_iteration=True, cache=True):
        
        data_count = tf.data.experimental.cardinality(data).numpy()
        
        if cache:
            data = data.cache()
        if shuffle:
            data = data.shuffle(data_count, reshuffle_each_iteration=shuffle_all_every_iteration)
        
        data = data.batch(batch_size)

        sample_batch = next(iter(data))

        if shuffle and (not shuffle_all_every_iteration):
            data = data.shuffle((data_count//batch_size)+1, reshuffle_each_iteration=True)

        return data, sample_batch

    def merge_labels(self, label_1, label_2, do_mask=False, mask=[]):
        if do_mask:
            label_1 *= mask
        final_label = tf.clip_by_value(label_1 + label_2, 0.0, 1.0) 
        return final_label


    def log_images(self, epoch, batch, target_label, label_str, num_images=5, hair_locs=[], validation=False, testing=False):
        batch_image = batch['image'][0:num_images]
        label = batch['label'][0:num_images]
        if self.preprocess_model is not None:
            batch_image = self.preprocess_model(batch_image)
        image_size = tf.shape(batch_image)[-3:].numpy()
        target = tf.repeat([target_label], num_images, axis=0)
        hair_mask = tf.repeat([[0.0 if hair else 1.0 for hair in hair_locs]], num_images, axis=0)
        target = self.merge_labels(label, target, 'hair' in label_str.lower(), hair_mask)
        predictions = self.gen([batch_image, target])
        cycled_imgs = self.gen([predictions, label])
        dpi = 100.
        w_pad = 2/72.
        h_pad = 2/72.
        plot_width = (image_size[0]+2*w_pad*image_size[0])*2./dpi
        plot_height = (image_size[1]+2*h_pad*image_size[1])/dpi
        fig, ax = plt.subplots(1, 3, figsize=(plot_width, plot_height), dpi=dpi)

        title = ['input', label_str, 'cycled']
        suptitle = target[0].numpy()
        fig.suptitle(f'Target: {suptitle}',  y=0.02, fontsize=8, va='bottom')

        img = [None, None, None]

        for i in range(3):
            ax[i].set_title(title[i])
            ax[i].axis('off')
            img[i] = ax[i].imshow(tf.zeros(image_size))

        images_list = []
        for image, prediction, target_lbl, cycled in zip(batch_image,predictions, target, cycled_imgs):
            img[0].set_data(image*0.5 + 0.5)
            img[1].set_data(prediction*0.5 + 0.5)
            img[2].set_data(cycled*0.5 + 0.5)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            fig.suptitle(f'Target: {target_lbl.numpy()}',  y=0.02, fontsize=8, va='bottom')

            image = tf.image.decode_png(buf.getvalue(), channels=3)
            images_list.append(image)
        out_images = tf.stack(images_list, axis=0)
        if validation: logger = self.gen.val_logger
        elif testing: logger = self.gen.test_logger
        else: logger = self.gen.logger
        with logger.as_default():
            tf.summary.image(label_str, out_images, max_outputs=num_images, step=epoch)
        plt.close()

    def log_image(self, epoch, batch, target_labels, label_strs, num_images=5, hair_locs=[], validation=False, testing=False):

        batch_image = batch['image'][0:num_images]
        label = batch['label'][0:num_images]
        if self.preprocess_model is not None:
            batch_image = self.preprocess_model(batch_image)
        image_size = tf.shape(batch_image)[-3:].numpy()
        dpi = 100.
        w_pad = 2/72.
        h_pad = 2/72.
        plot_width = (image_size[0]+2*w_pad*image_size[0])/dpi
        plot_height = (image_size[1]+2*h_pad*image_size[1])/dpi
        fig, ax = plt.subplots(num_images, len(label_strs)+1, figsize=(plot_width*(len(label_strs)+1), plot_height*num_images), dpi=dpi)

        for i in range(-1, len(label_strs)):
            if i == -1:
                title_str = 'Input'
                images = batch_image
            else:
                title_str = label_strs[i]
                target = tf.repeat([target_labels[i]], num_images, axis=0)
                hair_mask = tf.repeat([[0.0 if hair else 1.0 for hair in hair_locs]], num_images, axis=0)
                target = self.merge_labels(label, target, 'hair' in label_strs[i].lower(), hair_mask)
                images = self.gen([batch_image, target])
            title = None
            for j, image in enumerate(images):
                if title is None:
                    title = title_str
                    ax[j, i+1].set_title(title)
                ax[j, i+1].axis('off')
                ax[j, i+1].imshow(image*0.5+0.5)

        buf = io.BytesIO()     
        plt.savefig(buf, format='png')
        image = tf.image.decode_png(buf.getvalue(), channels=3)
        if validation: logger = self.gen.val_logger
        elif testing: logger = self.gen.test_logger
        else: logger = self.gen.logger
        with logger.as_default():
            tf.summary.image('images_graphed', tf.expand_dims(image, 0), step=epoch)
        plt.close()

    def save_models(self, path=None):
        self.gen.save_model(path)
        self.disc.save_model(path)
    def load_models(self, path=None):
        self.gen.load_model(path)
        self.disc.load_model(path)


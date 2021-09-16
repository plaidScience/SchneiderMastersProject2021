import tensorflow as tf

import datetime
import time

import os

import matplotlib.pyplot as plt
import io

from . import PIX2PIX_DISC as pix2pix
from . import UNET_GEN as unet
from . import RESNET_GEN as resnet

class CycleGAN():
    def __init__(self, ouput_dir, modeltype='resnet'):
        self.time_created = datetime.datetime.now().strftime("%m_%d/%H/")
        self.output_dir = os.path.join(ouput_dir, self.time_created)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.gen_f, self.disc_x = self._create_model_pair('gen_f', 'disc_x', modeltype)
        self.gen_g, self.disc_y = self._create_model_pair('gen_g', 'disc_y', modeltype)
        print("Generator F:")
        self.gen_f.summary()
        print("Generator G:")
        self.gen_g.summary()
        print("Discriminator X:")
        self.disc_x.summary()
        print("Discriminator Y:")
        self.disc_y.summary()

        self.checkpoint_folder = os.path.join(self.output_dir, 'checkpoints')

        self.checkpoint = tf.train.Checkpoint(
            gen_f = self.gen_f.model,
            gen_f_optimizer = self.gen_f.optimizer,
            gen_g = self.gen_g.model,
            gen_g_optimizer = self.gen_g.optimizer,
            disc_x = self.disc_x.model,
            disc_x_optimizer = self.disc_x.optimizer,
            disc_y = self.disc_y.model,
            disc_y_optimizer = self.disc_y.optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, os.path.join(self.checkpoint_folder, 'checkpoint'), max_to_keep=5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mae_loss_object = tf.keras.losses.MeanAbsoluteError()
        self.LAMBDA = 10


    def _create_model_pair(self, gen_name, disc_name, modeltype):
        if modeltype.lower() == 'resnet':
            gen = resnet.RESNET_GENERATOR(self.output_dir, gen_name)
        elif modeltype.lower() == 'unet':
            gen = unet.UNET_GENERATOR(self.output_dir, gen_name)
        else:
            raise NotImplementedError(f"Model Type {modeltype} NYI")

        disc = pix2pix.PIX2PIX_DISC(self.output_dir, disc_name)

        return gen, disc

    def disc_loss(self, real, generated):
        real_loss = self.loss_object(tf.ones_like(real), real)
        generated_loss = self.loss_object(tf.zeros_like(generated), generated)
        total_loss = real_loss + generated_loss
        return total_loss * 0.5

    def gen_loss(self, generated):
        return self.loss_object(tf.ones_like(generated), generated)

    def identity_loss(self, real, generated):
        loss = self.mae_loss_object(real, generated)
        return loss*self.LAMBDA

    def cycle_loss(self, real, generated):
        loss = self.mae_loss_object(real, generated)
        return loss*self.LAMBDA*0.5

    def _train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            #generator_output
            fake_y = self.gen_g(real_x, training=True)
            cycled_x = self.gen_f(fake_y, training=True)

            fake_x = self.gen_f(real_y, training=True)
            cycled_y = self.gen_g(fake_x, training=True)

            same_x = self.gen_f(real_x, training=True)
            same_y = self.gen_g(real_y, training=True)

            #discriminator_output
            disc_real_x = self.disc_x(real_x, training=True)
            disc_real_y = self.disc_y(real_y, training=True)

            disc_fake_x = self.disc_x(fake_x, training=True)
            disc_fake_y = self.disc_y(fake_y, training=True)

            #losses
            gen_f_loss = self.gen_loss(disc_fake_x)
            gen_g_loss = self.gen_loss(disc_fake_y)

            total_cycle_loss = self.cycle_loss(real_x, cycled_x) + self.cycle_loss(real_y, cycled_y)

            identity_f_loss = self.identity_loss(real_x, same_x)
            identity_g_loss = self.identity_loss(real_y, same_y)

            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_f_loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_g_loss

            disc_x_loss = self.disc_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.disc_loss(disc_real_y, disc_fake_y)

        gen_f_gradients = tape.gradient(total_gen_f_loss, self.gen_f.model.trainable_variables)
        gen_g_gradients = tape.gradient(total_gen_g_loss, self.gen_g.model.trainable_variables)
        disc_x_gradients = tape.gradient(disc_x_loss, self.disc_x.model.trainable_variables)
        disc_y_gradients = tape.gradient(disc_y_loss, self.disc_y.model.trainable_variables)

        self.gen_f.optimizer.apply_gradients(zip(gen_f_gradients, self.gen_f.model.trainable_variables))
        self.gen_g.optimizer.apply_gradients(zip(gen_g_gradients, self.gen_g.model.trainable_variables))
        self.disc_x.optimizer.apply_gradients(zip(disc_x_gradients, self.disc_x.model.trainable_variables))
        self.disc_y.optimizer.apply_gradients(zip(disc_y_gradients, self.disc_y.model.trainable_variables))

        return total_gen_f_loss, total_gen_g_loss, disc_x_loss, disc_y_loss

    def train(self, ds_x, ds_y, epochs, start_epoch=0, log_freq=1, gen_freq=5, checkpoint_freq=5):
        zipped_dataset = tf.data.Dataset.zip((ds_x, ds_y))
        test_x = next(iter(ds_x))
        test_y = next(iter(ds_y))

        f_loss = tf.keras.metrics.Mean('gen_f_loss', dtype=tf.float32)
        g_loss = tf.keras.metrics.Mean('gen_g_loss', dtype=tf.float32)
        x_loss = tf.keras.metrics.Mean('disc_x_loss', dtype=tf.float32)
        y_loss = tf.keras.metrics.Mean('disc_y_loss', dtype=tf.float32)


        for epoch in range(start_epoch, epochs):
            n = 1
            print(f'Epoch: {epoch+1}: Starting!', end='')
            start = time.time()
            for img_x, img_y in zipped_dataset:
                batch_f, batch_g, batch_x, batch_y = self._train_step(img_x, img_y)
                f_loss(batch_f)
                g_loss(batch_g)
                x_loss(batch_x)
                y_loss(batch_y)
                if n%5==0:
                    print(f'\rEpoch: {epoch+1}: Batch {n} completed!', end='')
                n+=1

            print(f'\rEpoch {epoch+1} completed in {time.time()-start:.0f}, Total {n} Batches completed!')
            print(f'\t[GENERATOR]: Gen F: {f_loss.result():.04f} Gen G: {g_loss.result():.04f}')
            print(f'\t[DISCRIMINATOR]: Disc X: {x_loss.result():.04f} Disc Y: {y_loss.result():.04f}')

            if (epoch+1)%log_freq==0:
                with self.gen_f.logger.as_default():
                    tf.summary.scalar('loss', f_loss.result(), step=epoch)
                with self.gen_g.logger.as_default():
                    tf.summary.scalar('loss', g_loss.result(), step=epoch)
                with self.disc_x.logger.as_default():
                    tf.summary.scalar('loss', x_loss.result(), step=epoch)
                with self.disc_y.logger.as_default():
                    tf.summary.scalar('loss', y_loss.result(), step=epoch)

            if (epoch+1)%gen_freq==0:
                self.log_image(test_x, self.gen_g, epoch, "Image X to Y")
                self.log_image(test_x, self.gen_f, epoch, "Image X to X")
                self.log_image(test_y, self.gen_f, epoch, "Image Y to X")
                self.log_image(test_y, self.gen_g, epoch, "Image Y to Y")

            if (epoch + 1) % checkpoint_freq == 0:
                ckpt_save_path = self.checkpoint_manager.save()
                print (f'Epoch {epoch+1}: Saving checkpoint at {ckpt_save_path}')



            f_loss.reset_states()
            g_loss.reset_states()
            x_loss.reset_states()
            y_loss.reset_states()

    def log_image(self, images, generator, epoch, log_str='Epoch Generated'):
        image_size = tf.shape(images)[-3:].numpy()
        predictions = generator(images, training=False)
        dpi = 100.
        w_pad = 2/72.
        h_pad = 2/72.
        plot_width = (image_size[0]+2*w_pad*image_size[0])*2./dpi
        plot_height = (image_size[1]+2*h_pad*image_size[1])/dpi
        fig, ax = plt.subplots(1, 2, figsize=(plot_width, plot_height), dpi=dpi)

        title = ['Input Image', 'Predicted Image']

        img = [None, None]

        for i in range(2):
            ax[i].set_title(title[i])
            ax[i].axis('off')
            img[i] = ax[i].imshow(tf.zeros(image_size))

        images_list = []
        for image, prediction in zip(images, predictions):
            img[0].set_data(image*0.5 + 0.5)
            img[1].set_data(prediction*0.5 + 0.5)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')

            image = tf.image.decode_png(buf.getvalue(), channels=3)
            images_list.append(image)
        out_images = tf.stack(images_list, axis=0)
        with generator.logger.as_default():
            tf.summary.image(log_str, out_images, step=epoch)

    def save_models(self, path=None):
        self.gen_f.save_model(path)
        self.gen_g.save_model(path)
        self.disc_y.save_model(path)
        self.disc_x.save_model(path)
    def load_models(self, path=None):
        self.gen_f.load_model(path)
        self.gen_g.load_model(path)
        self.disc_y.load_model(path)
        self.disc_x.load_model(path)

def main():
    import tensorflow_datasets as tfds
    AUTOTUNE = tf.data.AUTOTUNE

    dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)

    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']

    BUFFER_SIZE = 1000
    BATCH_SIZE = 4
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    EPOCHS=40

    def random_crop(image):
        croppped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
        return croppped_image
    def normalize(image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image
    def random_jitter(image):
        # resizing to 286 x 286 x 3
        image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        image = random_crop(image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)
        return image

    def preprocess_image_train(image, label):
        image = random_jitter(image)
        image = normalize(image)
        return image
    def preprocess_image_test(image, label):
        image = normalize(image)
        return image

    train_horses = train_horses.take(BUFFER_SIZE).cache().map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    train_zebras = train_zebras.take(BUFFER_SIZE).cache().map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_horses = test_horses.take(BUFFER_SIZE).map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_zebras = test_zebras.take(BUFFER_SIZE).map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    cycleGAN = CycleGAN('./OUTPUT/cycleGAN/', modeltype=str(input("Input Desired Model Type: ")))
    cycleGAN.train(train_horses, train_zebras, EPOCHS, start_epoch=0, log_freq=1, gen_freq=1, checkpoint_freq=5)

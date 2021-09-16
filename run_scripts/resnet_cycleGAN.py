import tensorflow_datasets as tfds
import tensorflow as tf

from networks import CYCLEGAN as CG

def main():
    AUTOTUNE = tf.data.AUTOTUNE

    dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)

    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']

    BUFFER_SIZE = 1000
    BATCH_SIZE = 1
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

    cycleGAN = CG.CycleGAN('./OUTPUT/cycleGAN_resnet/', modeltype='resnet')
    cycleGAN.train(train_horses, train_zebras, EPOCHS, start_epoch=0, log_freq=1, gen_freq=1, checkpoint_freq=5)

if __name__ =='__main__':
    main()
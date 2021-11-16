import tensorflow_datasets as tfds
import tensorflow as tf

from networks import STARGAN as SG
from data import celeba_local as CELEBA

def main():
    AUTOTUNE = tf.data.AUTOTUNE

    load_dir = "D:/celeba/"
    labels = ["Blond_Hair", "Brown_Hair", "Black_Hair", "Male", "Young"]

    dataset = CELEBA.load_celeba(load_dir, labels)

    BATCH_SIZE = 16
    IMG_WIDTH = 218
    IMG_HEIGHT = 178
    EPOCHS=20

    cycleGAN = SG.StarGAN([IMG_HEIGHT, IMG_WIDTH, 3], len(labels), './OUTPUT/starGAN_celeba/')
    cycleGAN.train(dataset, labels, EPOCHS, start_epoch=0, batch_size=BATCH_SIZE, log_freq=1, gen_freq=1, checkpoint_freq=5)

if __name__ =='__main__':
    main()
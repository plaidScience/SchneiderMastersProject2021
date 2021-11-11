import tensorflow as tf
import tensorflow_datasets as tfds



def load_celeba(load_dir, features):
    ds = tfds.load("celeb_a", data_dir=load_dir)
    return ds
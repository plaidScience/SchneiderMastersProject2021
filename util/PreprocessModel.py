import tensorflow as tf


def get_preprocess_model(crop_size, resize_dim, random_flip=True):
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.CenterCrop(crop_size, crop_size),
        tf.keras.layers.experimental.preprocessing.Resizing(resize_dim, resize_dim)
    ])
    if random_flip:
        model.add(tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'))
    return model
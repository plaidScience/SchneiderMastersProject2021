import tensorflow as tf

def calc_fid_activations(act1, act2, calc_mu_sigma=True):
    if calc_mu_sigma:
        mu1, sigma1 = tf.nn.moments(act1, axes=0)
        mu2, sigma2 = tf.nn.moments(act2, axes=0)
    else:
        mu1, sigma1 = act1
        mu2, sigma2 = act2

    ssdif = tf.reduce_sum(tf.square(mu1 - mu2))

    trace = tf.reduce_sum((sigma1 + sigma2) - 2.0 * tf.sqrt(sigma1 * sigma2))

    fid = ssdif + trace

    return fid

def get_model(input_shape):

    if input_shape != (299, 299, 3):
        model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(299, 299),
            tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
        ])
    else:
        model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    return model

def preprocess_input(x, data_format=None):
    return tf.keras.applications.inception_v3.preprocess_input(x, data_format=data_format)

def calc_mu_sigma(act):
    return tf.nn.moments(act, axes=0)

def calc_fid_model(imgs1, imgs2, model):
    act1 = model.predict(imgs1)
    act2 = model.predict(imgs2)
    fid = calc_fid_activations(act1, act2)
    return fid

import tensorflow as tf

def chooseOneGPU(chosenGPU):
    '''
    Limits tensorflow to one GPU
    Inputs:
        chosenGPU: the ID of the chosen GPU
    Outputs:
        none
    '''
    # Limit to ONE GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use ONE GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[chosenGPU], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
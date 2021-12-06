import tensorflow_datasets as tfds
import tensorflow as tf

from networks import STARGAN_multicycle as SG_multi
from data import celeba_local as CELEBA

from util import restrict_gpu, PreprocessModel

def main():
    restrict_gpu.chooseOneGPU(int(input("Input the ID of the GPU you'd Like to choose: ")))
    AUTOTUNE = tf.data.AUTOTUNE

    #load_dir = "D:/celeba/"
    load_dir = '/media/Data1/CELEBA/'
    labels = ["Blond_Hair", "Brown_Hair", "Black_Hair", "Male", "Young"]

    dataset, test_dataset, val_dataset = CELEBA.load_celeba(load_dir, labels, tt_split=True)

    BATCH_SIZE = 16
    IMG_WIDTH = 178
    IMG_HEIGHT = 218
    
    RESCALE_DIM = 128
    EPOCHS=20

    preporcess_model = PreprocessModel.get_preprocess_model(IMG_WIDTH, RESCALE_DIM)

    starGAN = SG_multi.StarGAN([RESCALE_DIM, RESCALE_DIM, 3], len(labels), './OUTPUT/sg_multi_celeba/', preprocess_model=preporcess_model)
    starGAN.train(dataset, labels, EPOCHS, data_val=val_dataset, start_epoch=0, batch_size=BATCH_SIZE, log_freq=1, gen_freq=1, checkpoint_freq=5, log_lr=True)
    starGAN.test(test_dataset, labels, BATCH_SIZE, log_at=EPOCHS)

if __name__ =='__main__':
    main()
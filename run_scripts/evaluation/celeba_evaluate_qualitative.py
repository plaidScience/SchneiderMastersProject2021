
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_array_ops import concat
from util import fid as FID
from util import restrict_gpu, PreprocessModel
from data import celeba_local as CELEBA

from util import restrict_gpu, PreprocessModel
from networks.STARGAN import StarGAN as SG_BASE
from networks.STARGAN_multicycle import StarGAN_MultiCycle as SG_MC
from networks.STARGAN_multitrain import StarGAN_MultiCycle as SG_MT

import os


def main():

    #restrict_gpu.chooseOneGPU(int(input("Input the ID of the GPU you'd Like to choose: ")))
    AUTOTUNE = tf.data.AUTOTUNE

    load_dir = "D:/celeba/"
    #load_dir = '/media/Data1/CELEBA/'
    labels = ["Blond_Hair", "Brown_Hair", "Black_Hair", "Male", "Young"]
    label_mask = tf.constant([1.0 if 'hair' in label.lower() else 0.0 for label in labels])

    dataset, test_dataset, val_dataset = CELEBA.load_celeba(load_dir, labels, tt_split=True)

    BATCH_SIZE = 16
    IMG_WIDTH = 178
    IMG_HEIGHT = 218
    
    RESCALE_DIM = 128
    EPOCHS=20

    preprocess_model = PreprocessModel.get_preprocess_model(IMG_WIDTH, RESCALE_DIM, random_flip=False)
    fid_model = FID.get_model([RESCALE_DIM, RESCALE_DIM, 3])
    model_idx = int(input("Input the Num of the model you want to eval on (1 for Stargan, 2 for SG_MC, 3 for SG_MT): "))

    if model_idx < 0:
        take_first = int(input("Take First N Data For Testing: "))
        dataset = dataset.take(take_first)
        val_dataset = val_dataset.take(take_first)
        test_dataset= test_dataset.take(take_first)
        model_idx = -model_idx

    if model_idx == 1:
        model_date = "12_06/16/"
        model_folder = './FINAL_OUTPUT/starGAN_celeba/'
        model = SG_BASE(
            [RESCALE_DIM, RESCALE_DIM, 3], len(labels),
            model_folder,
            time_created=model_date,
            preprocess_model=preprocess_model,
            restore_model_from_checkpoint=False
            )
    elif model_idx == 2:
        model_date = "12_08/18/"
        model_folder = './FINAL_OUTPUT/sg_multi_celeba/'
        model = SG_MC(
            [RESCALE_DIM, RESCALE_DIM, 3], len(labels),
            model_folder,
            time_created=model_date,
            preprocess_model=preprocess_model,
            restore_model_from_checkpoint=False
            )
    elif model_idx == 3:
        model_date = "12_11/22/"
        model_folder = './FINALOUTPUT/sg_multitrain_celeba/'
        model = SG_MT(
            [RESCALE_DIM, RESCALE_DIM, 3], len(labels),
            model_folder,
            time_created=model_date,
            preprocess_model=preprocess_model,
            restore_model_from_checkpoint=False
            )
    else:
        raise(ValueError("Model ID out of Range!"))

    if input("Log Model Image? [y/N]: ").lower().startswith("y"):
        tf.keras.utils.plot_model(model.gen.model, to_file=os.path.join(model_folder, 'qualitative_results/', 'gen_model.png'), show_shapes=True, expand_nested=True)
        tf.keras.utils.plot_model(model.disc.model, to_file=os.path.join(model_folder, 'qualitative_results/', 'disc_model.png'), show_shapes=True, expand_nested=True)

    
    def get_generated_imgs(batch_image, batch_labels, first_out=None, hair_mask=tf.constant([1.0, 1.0, 1.0, 0.0, 0.0]), concat_along=2):

        for i, hair_id in enumerate(hair_mask):
            if hair_id == 1.0:
                first_masked = i
                break
        if first_out is None:
            output = [batch_image]
        else:
            output=[first_out]
        trg_labels = model.set_targets_mask(batch_labels, hair_mask, first_masked)
        gen_imgs = model.gen.model.predict([batch_image, trg_labels])
        output.append(gen_imgs)

        for i, hair_id in enumerate(hair_mask):
            if hair_id==0.0:
                #no hair here, continue to generate
                mask = tf.one_hot(i, len(labels))
                trg_labels = model.inv_selected_targets(batch_labels, mask)
                gen_imgs = model.gen.model.predict([batch_image, trg_labels])
                output.append(gen_imgs)
        hair_mask_2 = hair_mask
        move_id = 0
        for i, hair_id in enumerate(hair_mask):
            if hair_id==0.0:
                for j, hair_id_2 in enumerate(hair_mask_2):
                    j+=move_id
                    if hair_id_2==0.0 and not i==j:
                        mask = tf.one_hot(i, len(labels)) + tf.one_hot(j, len(labels))
                        trg_labels = model.inv_selected_targets(batch_labels, mask)
                        gen_imgs = model.gen.model.predict([batch_image, trg_labels])
                        output.append(gen_imgs)
                trg_labels = model.set_targets_mask(batch_labels, hair_mask, first_masked)
                mask = tf.one_hot(i, len(labels))
                trg_labels = model.inv_selected_targets(trg_labels, mask)
                gen_imgs = model.gen.model.predict([batch_image, trg_labels])
                output.append(gen_imgs)
            hair_mask_2 = hair_mask_2[1:]
            move_id +=1
        
        mask = 1.0-hair_mask
        trg_labels = model.set_targets_mask(batch_labels, hair_mask, first_masked)
        trg_labels = model.inv_selected_targets(trg_labels, mask)
        gen_imgs = model.gen.model.predict([batch_image, trg_labels])
        output.append(gen_imgs)
    
        return tf.concat(output, axis=concat_along)

    def save_generated(base_dataset, save_to=os.path.join(model_folder, 'qualitative_results/'), n_cycles=0, trailing_zeros=5, batch_img_out=False):
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        

        base_dataset = base_dataset.batch(BATCH_SIZE)
        for i, batch in base_dataset.enumerate():
            print(f"generating batch {i}", end="\r")
            images = batch['image']
            labels = batch['label']
            preprocessed = preprocess_model.predict(images)
            predict_on = preprocessed
            cycle_results = []
            cycle_results.append(get_generated_imgs(predict_on, labels, hair_mask=label_mask, concat_along=2))
            for n in range(n_cycles):
                cycle_labels = model.gen_target(labels)
                predict_on = model.gen.model.predict([predict_on, cycle_labels])
                cycle_results.append(get_generated_imgs(predict_on, labels, first_out=tf.ones_like(predict_on), hair_mask=label_mask,  concat_along=2))

            
            imgs = tf.concat(cycle_results, axis=1)
            for j, img in enumerate(imgs):
                img= tf.clip_by_value(img*0.5+0.5, 0.0, 1.0)
                img = tf.image.convert_image_dtype(img, tf.uint8)
                encoded = tf.io.encode_jpeg(img, format='rgb')
                tf.io.write_file(os.path.join(save_to, "images_2/", ("image{:0"+str(trailing_zeros)+"d}.jpg").format(i*BATCH_SIZE+j)), encoded)
            if batch_img_out:
                batch_img = tf.concat([imgs[i] for i in range(imgs.shape[0])], axis=0)
                batch_img= tf.clip_by_value(batch_img*0.5+0.5, 0.0, 1.0)
                batch_img = tf.image.convert_image_dtype(batch_img, tf.uint8)
                encoded = tf.io.encode_jpeg(batch_img, format='rgb')
                tf.io.write_file(os.path.join(save_to, "batches_2/", ("batch{:0"+str(trailing_zeros)+"d}.jpg").format(i)), encoded)

        return
    
    save_generated(test_dataset, n_cycles=2, trailing_zeros=4, batch_img_out=True)

        
if __name__ =='__main__':
    main()
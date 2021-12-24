
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.backend import dtype
from util import fid as FID
from util import restrict_gpu, PreprocessModel
from data import celeba_local as CELEBA

from util import restrict_gpu, PreprocessModel
from networks.STARGAN import StarGAN as SG_BASE
from networks.STARGAN_multicycle import StarGAN_MultiCycle as SG_MC
from networks.STARGAN_multitrain import StarGAN_MultiCycle as SG_MT

import os


def main(use_lpips=False):

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
        dataset = dataset.shard(80, 1)
        val_dataset = val_dataset.shard(80, 1)
        test_dataset= test_dataset.shard(80, 1)
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
        model_folder = './FINAL_OUTPUT/sg_multitrain_celeba/'
        model = SG_MT(
            [RESCALE_DIM, RESCALE_DIM, 3], len(labels),
            model_folder,
            time_created=model_date,
            preprocess_model=preprocess_model,
            restore_model_from_checkpoint=False
            )
    else:
        raise(ValueError("Model ID out of Range!"))

    if use_lpips:
        import util.lpips as lpips
        lpips_model = lpips.get_lpips_model(RESCALE_DIM)

    def get_fids(base_dataset, n_cycles=0, test_lpips=True, save_imgs = False, save_to=model_folder, trailing_zeros=5):

        if save_imgs and not os.path.exists(save_to):
            os.makedirs(save_to)

        base_dataset = base_dataset.batch(BATCH_SIZE)

        if test_lpips:
            lpips_avg = tf.keras.metrics.Mean("lpips average", dtype=tf.float32)
            lpips_recon = tf.keras.metrics.Mean("lpips reconstructive average", dtype=tf.float32)

        base_acts = []
        predicted_acts = []
        reconstructed_acts = []
        label_accuracy = tf.keras.metrics.BinaryAccuracy("label_acc", dtype=tf.float32)
        recon_accuracy = tf.keras.metrics.BinaryAccuracy('recon_acc', dtype=tf.float32)
        for i, batch in base_dataset.enumerate():
            print(f"\rCalculating Activations for images {i*BATCH_SIZE} through {((i+1)*BATCH_SIZE ) - 1}", end="")
            images = batch['image']
            labels = batch['label']
            preprocessed = preprocess_model.predict(images)
            base_acts.append(fid_model.predict(preprocessed))
            predict_on = preprocessed
            for n in range(n_cycles):
                #print(i)
                cycle_labels = model.gen_target(labels)
                predict_on = model.gen.model.predict([predict_on, cycle_labels])
            inv_labels = model.gen_inv_target(labels, label_mask)
            predicted = tf.clip_by_value(model.gen.model.predict([predict_on, inv_labels]), -1.0, 1.0)
            predicted_acts.append(fid_model.predict(predicted))
            reconstructed = tf.clip_by_value(model.gen.model.predict([predict_on, labels]), -1.0, 1.0)
            reconstructed_acts.append(fid_model.predict(reconstructed))
            inv_pred_labels = tf.math.sigmoid(model.disc.model.predict(predicted)[1])
            recon_pred_labels = tf.math.sigmoid(model.disc.model.predict(reconstructed)[1])
            label_accuracy(inv_labels, inv_pred_labels)
            recon_accuracy(labels, recon_pred_labels)
            if save_imgs or save_imgs is tuple:
                if save_imgs is not tuple:
                    save_predicted = save_imgs
                    save_base = save_imgs
                else:
                    save_base, save_predicted = save_imgs
                for j, (image, predicted_image, reconstructed_image) in enumerate(zip(preprocessed*0.5+0.5, predicted*0.5+0.5, reconstructed*0.5+0.5)):
                    if save_base:
                        image = tf.clip_by_value(image, 0.0, 1.0)
                        image = tf.image.convert_image_dtype(image, tf.uint8)
                        encoded = tf.io.encode_jpeg(image, format='rgb')
                        tf.io.write_file(os.path.join(save_to, 'images/', ("image{:0"+str(trailing_zeros)+"d}.jpg").format(i*BATCH_SIZE+j)), encoded)
                    if save_predicted:
                        predicted_image = tf.clip_by_value(predicted_image, 0.0, 1.0)
                        predicted_image = tf.image.convert_image_dtype(predicted_image, tf.uint8)
                        predicted_encoded = tf.io.encode_jpeg(predicted_image, format='rgb')
                        tf.io.write_file(os.path.join(save_to, f'images_cycled_{n_cycles}/', ("image{:0"+str(trailing_zeros)+"d}.jpg").format(i*BATCH_SIZE+j)), predicted_encoded)
                        reconstructed_image = tf.clip_by_value(reconstructed_image, 0.0, 1.0)
                        reconstructed_image = tf.image.convert_image_dtype(reconstructed_image, tf.uint8)
                        reconstructed_encoded = tf.io.encode_jpeg(reconstructed_image, format='rgb')
                        tf.io.write_file(os.path.join(save_to, f'images_cycled_{n_cycles}_reconstructed/', ("image{:0"+str(trailing_zeros)+"d}.jpg").format(i*BATCH_SIZE+j)), reconstructed_encoded)
            if test_lpips:
                lpips_img = tf.cast(tf.image.convert_image_dtype((preprocessed*0.5+0.5), tf.uint8), tf.float32)
                lpips_predicted = tf.cast(tf.image.convert_image_dtype((predicted*0.5+0.5), tf.uint8), tf.float32)
                lpips_reconstructed = tf.cast(tf.image.convert_image_dtype((reconstructed*0.5+0.5), tf.uint8), tf.float32)
                lpips_batch = lpips_model([lpips_img, lpips_predicted])
                lpips_rec_batch = lpips_model([lpips_img, lpips_reconstructed])

                lpips_batch = tf.reduce_mean(lpips_batch[~np.isnan(lpips_batch)])
                lpips_avg(lpips_batch)

                lpips_rec_batch = tf.reduce_mean(lpips_rec_batch[~np.isnan(lpips_rec_batch)])
                lpips_recon(lpips_rec_batch)
        base_acts = tf.concat(base_acts, axis=0)
        predicted_acts = tf.concat(predicted_acts, axis=0)
        reconstructed_acts = tf.concat(reconstructed_acts, axis=0)

        base_moments = FID.calc_mu_sigma(base_acts)
        predicted_moments = FID.calc_mu_sigma(predicted_acts)
        reconstructed_moments = FID.calc_mu_sigma(reconstructed_acts)

        fid = FID.calc_fid_activations(base_moments, predicted_moments, calc_mu_sigma=False)
        fid_recon = FID.calc_fid_activations(base_moments, reconstructed_moments, calc_mu_sigma=False)
        print(f"\n\nFID for inverse with {n_cycles} cycles is: {fid}")
        print(f"FID for reconstructed with {n_cycles} cycles is: {fid_recon}")
        if test_lpips:
            print(f"LPIPS Score for inverse with {n_cycles} cycles is: {lpips_avg.result()}")
            print(f"LPIPS Score for reconstructed with {n_cycles} cycles is: {lpips_recon.result()}")
            lpips_avg.reset_states()
            lpips_recon.reset_states()
        print(f"Label Prediction Accuracy for inverse with {n_cycles} cycles is: {label_accuracy.result()}")
        print(f"Labal Preciction Accuracy for reconstructed with {n_cycles} cycles is: {recon_accuracy.result()}")
        label_accuracy.reset_states()
        recon_accuracy.reset_states()

    
    get_fids(test_dataset, n_cycles=0, test_lpips=use_lpips, save_imgs=(True, True), save_to=os.path.join(model_folder, 'test_imgs/'), trailing_zeros=5)
    get_fids(test_dataset, n_cycles=1, test_lpips=use_lpips, save_imgs=(False, True), save_to=os.path.join(model_folder, 'test_imgs/'), trailing_zeros=5)
    get_fids(test_dataset, n_cycles=2, test_lpips=use_lpips, save_imgs=(False, True), save_to=os.path.join(model_folder, 'test_imgs/'), trailing_zeros=5)
    get_fids(test_dataset, n_cycles=3, test_lpips=use_lpips, save_imgs=(False, True), save_to=os.path.join(model_folder, 'test_imgs/'), trailing_zeros=5)

    model.save_models()

        
if __name__ =='__main__':
    main()
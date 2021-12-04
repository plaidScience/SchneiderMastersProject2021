import tensorflow as tf
import pandas as pd
from tensorflow.python.training.tracking.base import TrackableReference

def load_celeba(load_dir, features, image_shape=(218, 178, 3), tt_split = False):
    filename = load_dir+"list_attr_celeba.txt"

    with open(filename) as f:
        content = f.readlines()
        header = content[1].replace(' \n', '').split(' ')
        header.insert(0, 'image_id')
    data = pd.read_csv(filename, header=None, names=header, delimiter='\s+', skiprows=[0, 1])

    if tt_split:
        eval_fname = load_dir+"list_eval_partition.txt"
        split_data = pd.read_csv(eval_fname, header=None, names=['image_id', 'split'], delimiter='\s+')
        data = data.merge(split_data, on='image_id', how='left', validate='one_to_one')

        data_train = data[data['split'] == 0]
        labels_train = ((data_train[features]+1)/2).to_numpy()
        images_train = data_train["image_id"].to_numpy()

        data_test = data[data['split'] == 1]
        labels_test = ((data_test[features]+1)/2).to_numpy()
        images_test = data_test["image_id"].to_numpy()

        data_val = data[data['split'] == 2]
        
        labels_val = ((data_val[features]+1)/2).to_numpy()
        images_val = data_val["image_id"].to_numpy()

        ds_train = create_dataset_celeba(labels_train, images_train, load_dir, image_shape)
        ds_test = create_dataset_celeba(labels_test, images_test, load_dir, image_shape)
        ds_val = create_dataset_celeba(labels_val, images_val, load_dir, image_shape)


        return ds_train, ds_val, ds_test

    else:
        labels = ((data[features]+1)/2).to_numpy()
        images = data["image_id"].to_numpy()
        ds = create_dataset_celeba(labels, images, load_dir, image_shape)

    return ds



def create_dataset_celeba(labels, images, load_dir, image_shape):
    dataset = tf.data.Dataset.from_tensor_slices({"image": images, 'label': labels})

    def mapping_func (ds_dict_item):
        image_id = ds_dict_item['image']
        label = ds_dict_item['label']
        label = tf.cast(label, tf.float32)

        raw = tf.io.read_file(load_dir+'img_align_celeba/'+image_id)
        image = tf.io.decode_jpeg(raw, channels=image_shape[2])
        image = tf.image.convert_image_dtype(image, tf.float32)

        image = tf.image.resize(image, image_shape[0:2])
        image = (image*2.0)-1.0
        
        return {"image": image, "label":label}
    mapped = dataset.map(mapping_func, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    return mapped


def main():
    import os
    logger_train = tf.summary.create_file_writer(os.path.join("./OUTPUT", 'logs/', 'IMAGE_OUT', 'train'))
    logger_test = tf.summary.create_file_writer(os.path.join("./OUTPUT", 'logs/', 'IMAGE_OUT', 'test'))
    logger_val = tf.summary.create_file_writer(os.path.join("./OUTPUT", 'logs/', 'IMAGE_OUT', 'val'))

    folder = "D:/celeba/"
    features = ["Blond_Hair", "Brown_Hair", "Black_Hair", "Male", "Young"]

    dataset_train, dataset_test, dataset_val = load_celeba(folder, features, tt_split=True)

    def test_ds (dataset, logger):
        dataset = dataset.batch(16)
        print(dataset)
        with logger.as_default():
            do_break = 'no'
            i=0
            for ds_dict in dataset:
                tf.summary.image(
                    "Image", ds_dict['image']*0.5+0.5, step=i
                )
                i+=1
                if i%100==0: do_break = input("Stopping for a batch!")
                print(f"batch {i} completed!", end="\r")
                if do_break.startswith('y'): break
    
    test_ds(dataset_train, logger_train)
    test_ds(dataset_test, logger_test)
    test_ds(dataset_val, logger_val)
            


if __name__ == "__main__":
    main()

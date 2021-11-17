import tensorflow as tf
import pandas as pd

def load_celeba(load_dir, features, image_shape=(218, 178, 3)):
    filename = load_dir+"list_attr_celeba.txt"

    with open(filename) as f:
        content = f.readlines()
        header = content[1].replace(' \n', '').split(' ')
        header.insert(0, 'image_id')
    data = pd.read_csv(filename, header=None, names=header, delimiter='\s+', skiprows=[0, 1])

    labels = ((data[features]+1)/2).to_numpy()
    images = data["image_id"].to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices({"image": images, 'label': labels})

    def mapping_func (ds_dict_item):
        image_id = ds_dict_item['image']
        label = ds_dict_item['label']

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
    logger = tf.summary.create_file_writer(os.path.join("./OUTPUT", 'logs/', 'IMAGE_OUT'))

    folder = "D:/celeba/"
    features = ["Blond_Hair", "Brown_Hair", "Black_Hair", "Male", "Young"]

    dataset = load_celeba(folder, features).batch(16)

    print(dataset)

    with logger.as_default():
        i=0
        for ds_dict in dataset:
            tf.summary.image(
                "Image", ds_dict['image'], step=i
            )
            i+=1
            if i%100==0: input("Stopping for a batch!")
            print(f"batch {i} completed!", end="\r")
            


if __name__ == "__main__":
    main()

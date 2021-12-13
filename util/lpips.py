from .external.lpips_tf2.models import lpips_tensorflow
import os


def get_lpips_model(image_size):
    models_at = "./util/external/lpips_tf2/models/"
    vgg_ckpt_fn = os.path.join(models_at, "vgg", "exported")
    lin_ckpt_fn = os.path.join(models_at, "lin", "exported")
    return lpips_tensorflow.learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)

def calc_lpips_score(image1, image2, lpips):
    return lpips([image1, image2])
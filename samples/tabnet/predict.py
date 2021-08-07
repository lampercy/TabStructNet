import os

import skimage
import tensorflow as tf

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class TabNetConfig(Config):
    NAME = "tab"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1  # COCO has 80 classes
    DETECTION_MIN_CONFIDENCE = 0.75


def predict_once(image_path, model):
    image = skimage.io.imread(image_path)
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
        print('gray scale')

    if image.shape[-1] == 4:
        image = image[..., :3]
        print('alpha channel removed')

    r = model.detect([image], verbose=0)[0]

    output_im = visualize.generate_instances_image(
            image,
            r['rois'],
            r['masks'],
            r['class_ids'],
            ['BG', 'tablecell'],
            r['scores']
    )

    name, ext = os.path.splitext(args.image_path)

    result_path = name + '-predicted' + ext
    output_im.figure.savefig(result_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Mask R-CNN on Tables.')
    parser.add_argument('--image_path',
                        required=True)
    parser.add_argument('--model_dir',
                        required=True)
    args = parser.parse_args()

    config = TabNetConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=args.model_dir)
    model_path = model.find_last()
    model.load_weights(model_path, by_name=True)

    if os.path.isdir(args.image_path):
        for root, dirs, files in os.walk(args.image_path):
            for f in files:
                if os.path.splitext(f)[1] in ['.jpg', '.jpeg', '.png']:
                    predict_once(os.path.join(root, f), model)

    else:
        predict_once(args.image_path, model)

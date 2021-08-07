import os
import cv2
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize


class TabNetConfig(Config):
    NAME = "tab"

    IMAGES_PER_GPU = 1

    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1  # COCO has 80 classes

    DETECTION_MIN_CONFIDENCE = 0


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

    image = cv2.imread(args.image_path)
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

import cv2
import numpy as np
import random

from albumentations import (
    Compose,
    OneOf,
    PadIfNeeded,
    RandomSizedCrop,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    ShiftScaleRotate,
    CenterCrop,
    Transpose,
    GridDistortion,
    ElasticTransform,

    RandomContrast,
    RandomBrightness,
    CLAHE,
    HueSaturationValue,
    Blur,
    MedianBlur,
    ChannelShuffle,
)


def scale(img, scale, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    return img


def imload(filename, gray=False, scale_rate=1.0):
    if not gray:
        image = cv2.imread(filename)  # cv2 read color image as BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (h, w, 3)
        if scale_rate != 1.0:
            image = scale(image, scale_rate)
    else:
        image = cv2.imread(filename, -1)  # read gray image
        if scale_rate != 1.0:
            image = scale(image, scale_rate)
        # if image.max() < 255:
        image = 255 * (image - image.min()) / (image.max() - image.min())  # norm to [0, 255]
        image = np.asarray(image, dtype='uint8')

    return image


def img_mask_crop(image, mask, size=(256, 256), limits=(224, 512)):
    rc = RandomSizedCrop(height=size[0], width=size[1], min_max_height=limits)
    crops = rc(image=image, mask=mask)
    return crops['image'], crops['mask']


def img_mask_pad(image, mask, target=(288, 288)):
    padding = PadIfNeeded(p=1.0, min_height=target[0], min_width=target[1])
    paded = padding(image=image, mask=mask)
    return paded['image'], paded['mask']


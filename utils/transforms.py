"""
Image transformation utilities for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import warnings
import cv2
import numpy as np
import numbers
import random
import collections
import jittor as jt


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for transform in self.transforms:
            if len(args) == 1:
                result = transform(args[0])
                args = (result,) if not isinstance(result, tuple) else result
            else:
                result = transform(*args)
                args = result if isinstance(result, tuple) else (result,)
        return args[0] if len(args) == 1 else args


class ToTensor:
    """Convert numpy array to Jittor tensor."""

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = image.transpose(2, 0, 1)
            image = image.astype(np.float32) / 255.0
            return jt.array(image)
        return image


class Normalize:
    """Normalize tensor with mean and std."""

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, tensor):
        if isinstance(tensor, jt.Var):
            for i in range(len(self.mean)):
                tensor[i] = (tensor[i] - self.mean[i]) / self.std[i]
        return tensor


class RandomScale:
    """Random scale transformation."""

    def __init__(self, scale_list):
        self.scale_list = scale_list

    def __call__(self, image, label=None):
        scale = random.choice(self.scale_list)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            return image, label

        return image


class RandomCrop:
    """Random crop transformation."""

    def __init__(self, crop_size):
        self.crop_size = get_2dshape(crop_size)

    def __call__(self, image, label=None):
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size

        if h <= crop_h and w <= crop_w:
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            if label is not None:
                label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            h, w = image.shape[:2]

        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)

        image = image[start_h:start_h + crop_h, start_w:start_w + crop_w]

        if label is not None:
            label = label[start_h:start_h + crop_h, start_w:start_w + crop_w]
            return image, label

        return image


def get_2dshape(shape, *, zero=True):
    """Get 2D shape tuple from various input formats."""
    if not isinstance(shape, collections.abc.Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, "invalid shape: {}".format(shape)
    return shape


def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    """Randomly crop image and pad to target shape."""
    h, w = img.shape[:2]
    start_crop_h, start_crop_w = crop_pos
    assert (start_crop_h < h) and (start_crop_h >= 0)
    assert (start_crop_w < w) and (start_crop_w >= 0)

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    img_crop = img[start_crop_h : start_crop_h + crop_h, start_crop_w : start_crop_w + crop_w, ...]

    img_, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT, pad_label_value)

    return img_, margin


def generate_random_crop_pos(ori_size, crop_size):
    """Generate random crop position."""
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w


def pad_image_to_shape(img, shape, border_mode, value):
    """Pad image to target shape."""
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    original_shape = img.shape
    
    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3], border_mode, value=value)
    
    if len(original_shape) == 3 and original_shape[2] == 1 and len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    return img, margin


def pad_image_size_to_multiples_of(img, multiple, pad_value):
    """Pad image size to multiples of given value."""
    h, w = img.shape[:2]
    d = multiple

    def canonicalize(s):
        v = s // d
        return (v + (v * d != s)) * d

    th, tw = map(canonicalize, (h, w))

    return pad_image_to_shape(img, (th, tw), cv2.BORDER_CONSTANT, pad_value)


def resize_ensure_shortest_edge(img, edge_length, interpolation_mode=cv2.INTER_LINEAR):
    """Resize image ensuring shortest edge has target length."""
    assert isinstance(edge_length, int) and edge_length > 0, edge_length
    h, w = img.shape[:2]
    if h < w:
        ratio = float(edge_length) / h
        th, tw = edge_length, max(1, int(ratio * w))
    else:
        ratio = float(edge_length) / w
        th, tw = max(1, int(ratio * h)), edge_length
    img = cv2.resize(img, (tw, th), interpolation_mode)
    return img


def resize_ensure_longest_edge(img, edge_length, interpolation_mode=cv2.INTER_LINEAR):
    """Resize image ensuring longest edge has target length."""
    assert isinstance(edge_length, int) and edge_length > 0, edge_length
    h, w = img.shape[:2]
    if h > w:
        ratio = float(edge_length) / h
        th, tw = edge_length, max(1, int(ratio * w))
    else:
        ratio = float(edge_length) / w
        th, tw = max(1, int(ratio * h)), edge_length
    img = cv2.resize(img, (tw, th), interpolation_mode)
    return img


def normalize(img, mean, std):
    """Normalize image with mean and standard deviation."""
    img = img.astype(np.float64) / 255.0
    img = img - mean
    img = img / std
    return img


def denormalize(img, mean, std):
    """Denormalize image."""
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    img = img * std + mean
    
    return img

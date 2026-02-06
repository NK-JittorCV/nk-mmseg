"""
RGBX Dataset for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import cv2
import jittor as jt
import numpy as np
from jittor.dataset import Dataset


def get_path(
    dataset_name,
    _rgb_path,
    _rgb_format,
    _x_path,
    _x_format,
    _gt_path,
    _gt_format,
    x_modal,
    item_name,
):
    """Get file paths for RGB, depth/modal, and ground truth data."""
    if dataset_name == "StanFord2D3D":
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("/rgb/", "/depth/").replace("_rgb", "_newdepth")
            + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "")
            .replace(".png", "")
            .replace("/rgb/", "/semantic/")
            .replace("_rgb", "_newsemantic")
            + _gt_format,
        )
    elif dataset_name == "new_stanford":
        area = item_name.split(" ")[0]
        name = item_name.split(" ")[1]
        rgb_path = os.path.join(_rgb_path, area + "/image/" + name + _rgb_format)
        d_path = os.path.join(_x_path, area + "/hha/" + name + _x_format)
        gt_path = os.path.join(_gt_path, area + "/label/" + name + _gt_format)
    elif dataset_name == "KITTI-360":
        rgb_path = os.path.join(_rgb_path, item_name.split(" ")[0])
        d_path = os.path.join(
            _x_path,
            item_name.split(" ")[0]
            .replace("data_2d_raw", "data_3d_rangeview")
            .replace("image_00/data_rect", "velodyne_points/data"),
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.split(" ")[1].replace("data_2d_semantics", "data_2d_semantics_trainID"),
        )
    elif dataset_name == "Scannet":
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("color", "convert_depth") + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("color", "convert_label") + _gt_format,
        )
    elif dataset_name == "MFNet":
        rgb_path = os.path.join(_rgb_path, item_name + _rgb_format)
        d_path = os.path.join(_x_path, item_name + _x_format)
        gt_path = os.path.join(_gt_path, item_name + _gt_format)
    elif dataset_name == "EventScape":
        item_name = item_name.split(".png")[0]
        img_name = item_name.split("/")[-1]
        img_id = img_name.replace("_image", "").split("_")[-1]
        rgb_path = os.path.join(_rgb_path, item_name + _rgb_format)
        d_path = os.path.join(
            _x_path,
            item_name.replace("rgb", "depth").replace("data", "frames").replace(img_name, img_id) + _x_format,
        )
        e_path = os.path.join(
            _x_path,
            item_name.replace("rgb", "events").replace("data", "frames").replace(img_name, img_id) + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace("rgb", "semantic").replace("image", "gt_labelIds") + _gt_format,
        )
    else:
        # Default case for NYUDepthv2 and SUNRGBD
        item_name = item_name.split("/")[1].split(".jpg")[0] if "/" in item_name else item_name.split(".jpg")[0]
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "") + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "").replace(".png", "") + _gt_format,
        )

    path_result = {"rgb_path": rgb_path, "gt_path": gt_path}
    for modal in x_modal:
        path_result[modal + "_path"] = eval(modal + "_path")
    return path_result


class RGBXDataset(Dataset):
    """RGBX Dataset class for multi-modal semantic segmentation."""

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting["rgb_root"]
        self._rgb_format = setting["rgb_format"]
        self._gt_path = setting["gt_root"]
        self._gt_format = setting["gt_format"]
        self._transform_gt = setting["transform_gt"]
        self._x_path = setting["x_root"]
        self._x_format = setting["x_format"]
        self._x_single_channel = setting["x_single_channel"]
        self._train_source = setting["train_source"]
        self._eval_source = setting["eval_source"]
        self.class_names = setting["class_names"]
        self.num_classes = len(self.class_names) if self.class_names else 40  # Default to 40 for NYUDepthv2
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.dataset_name = setting["dataset_name"]
        self.x_modal = setting.get("x_modal", ["d"])
        self.backbone = setting["backbone"]

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def _get_file_names(self, split_name):
        """Get file names for the specified split."""
        if split_name == "train":
            file_path = self._train_source
        else:
            file_path = self._eval_source

        with open(file_path, 'r') as f:
            file_names = f.read().splitlines()
        return file_names

    def _construct_new_file_names(self, length):
        """Construct new file names for training with specified length."""
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = np.random.choice(files_len, length % files_len, replace=False)
        new_file_names += [self._file_names[i] for i in rand_indices]

        return new_file_names

    def __getitem__(self, index):
        """Get a single data sample."""
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        path_dict = get_path(
            self.dataset_name,
            self._rgb_path,
            self._rgb_format,
            self._x_path,
            self._x_format,
            self._gt_path,
            self._gt_format,
            self.x_modal,
            item_name,
        )

        if self.dataset_name == "SUNRGBD" and self.backbone.startswith("DFormerv2"):
            rgb_mode = "RGB"  # some checkpoints are run by BGR and some are on RGB, need to select
        else:
            rgb_mode = "BGR"
        rgb = self._open_image(path_dict["rgb_path"], rgb_mode)

        gt = self._open_image(path_dict["gt_path"], cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._transform_gt:
            gt = self._gt_transform(gt)

        x = {}
        for modal in self.x_modal:
            if modal == "d":
                x[modal] = self._open_image(path_dict[modal + "_path"], cv2.IMREAD_GRAYSCALE)
                # DFormerv2 expects single channel depth, DFormerv1 expects 3-channel
                if self.backbone.startswith("DFormerv2"):
                    # Keep single channel for DFormerv2, add channel dimension
                    x[modal] = np.expand_dims(x[modal], axis=2)
                else:
                    # Convert to 3-channel for DFormerv1
                    x[modal] = cv2.merge([x[modal], x[modal], x[modal]])
            else:
                x[modal] = self._open_image(path_dict[modal + "_path"], "RGB")
        if len(self.x_modal) == 1:
            x = x[self.x_modal[0]]

        if self.dataset_name == "Scannet":
            rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
            x = cv2.resize(x, (640, 480), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (640, 480), interpolation=cv2.INTER_NEAREST)
        elif self.dataset_name == "StanFord2D3D":
            rgb = cv2.resize(rgb, dsize=(480, 480), interpolation=cv2.INTER_LINEAR)
            x = cv2.resize(x, dsize=(480, 480), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, dsize=(480, 480), interpolation=cv2.INTER_NEAREST)

        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        rgb = jt.array(np.ascontiguousarray(rgb)).float()
        gt = jt.array(np.ascontiguousarray(gt)).int64()
        x = jt.array(np.ascontiguousarray(x)).float()

        output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(path_dict["rgb_path"]), n=len(self._file_names))

        return output_dict

    def _open_image(self, path, mode=cv2.IMREAD_COLOR, dtype=None):
        """Open and read image file."""
        if mode == "RGB":
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == "BGR":
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(path, mode)

        if dtype is not None:
            img = img.astype(dtype)

        return img

    def _gt_transform(self, gt):
        """Transform ground truth labels."""
        # NYU Depth v2 specific transform - map labels 1-40 to 0-39, label 0 becomes 255 (ignore)
        if self.dataset_name == "NYUDepthv2":
            # This is critical! PyTorch version does gt - 1
            # Labels 1-40 become 0-39, label 0 becomes 255 (ignore index)
            gt = gt.astype(np.int64)
            gt = gt - 1
            gt[gt == -1] = 255  # Set ignore index for invalid pixels
            return gt
        elif self.dataset_name == "SUNRGBD":
            # Map 37 classes
            return gt - 1
        else:
            return gt

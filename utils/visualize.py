"""
Visualization utilities for DFormer Jittor
Adapted from PyTorch version to support Jittor framework
"""

import numpy as np
import cv2
import scipy.io as sio
import os


def set_img_color(colors, background, img, pred, gt, show255=False):
    """Set image colors based on prediction and ground truth."""
    for i in range(0, len(colors)):
        if i != background:
            img[np.where(pred == i)] = colors[i]
    if show255:
        img[np.where(gt == background)] = 255
    return img


def show_prediction(colors, background, img, pred, gt):
    """Show prediction with colors."""
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred, gt)
    final = np.array(im)
    return final


def show_img(colors, background, img, clean, gt, *pds):
    """Show image with multiple predictions."""
    im1 = np.array(img, np.uint8)
    final = np.array(im1)
    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)
    for pd in pds:
        im = np.array(img, np.uint8)
        set_img_color(colors, background, im, pd, gt)
        final = np.column_stack((final, pivot))
        final = np.column_stack((final, im))

    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, gt, gt, True)
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, im))
    return final


def get_colors(class_num):
    """Generate random colors for classes."""
    colors = []
    for i in range(class_num):
        colors.append((np.random.random((1, 3)) * 255).tolist()[0])
    return colors


def get_ade_colors():
    """Get ADE20K color palette."""
    try:
        colors = sio.loadmat("./color150.mat")["colors"]
        colors = colors[:, ::-1]
        colors = np.array(colors).astype(int).tolist()
        colors.insert(0, [0, 0, 0])
        return colors
    except:
        # Fallback to random colors if color150.mat not found
        return get_colors(150)


def get_nyu_colors():
    """Get NYU Depth v2 color palette."""
    try:
        # Try to load the NYU color map
        palette_path = os.path.join(os.path.dirname(__file__), "nyucmap.npy")
        if os.path.exists(palette_path):
            palette = np.load(palette_path)
            return palette
        else:
            # Fallback color palette for NYU Depth v2 (40 classes)
            colors = [
                [0, 0, 0],        # 0: void
                [128, 64, 128],   # 1: wall
                [244, 35, 232],   # 2: floor
                [70, 70, 70],     # 3: cabinet
                [102, 102, 156],  # 4: bed
                [190, 153, 153],  # 5: chair
                [153, 153, 153],  # 6: sofa
                [250, 170, 30],   # 7: table
                [220, 220, 0],    # 8: door
                [107, 142, 35],   # 9: window
                [152, 251, 152],  # 10: bookshelf
                [70, 130, 180],   # 11: picture
                [220, 20, 60],    # 12: counter
                [255, 0, 0],      # 13: blinds
                [0, 0, 142],      # 14: desk
                [0, 0, 70],       # 15: shelves
                [0, 60, 100],     # 16: curtain
                [0, 80, 100],     # 17: dresser
                [0, 0, 230],      # 18: pillow
                [119, 11, 32],    # 19: mirror
                [128, 192, 255],  # 20: floor mat
                [255, 128, 0],    # 21: clothes
                [128, 255, 128],  # 22: ceiling
                [255, 128, 128],  # 23: books
                [128, 128, 255],  # 24: refrigerator
                [255, 255, 128],  # 25: television
                [192, 192, 192],  # 26: paper
                [64, 128, 128],   # 27: towel
                [128, 64, 255],   # 28: shower curtain
                [255, 64, 128],   # 29: box
                [64, 255, 128],   # 30: whiteboard
                [128, 255, 64],   # 31: person
                [255, 128, 64],   # 32: night stand
                [64, 128, 255],   # 33: toilet
                [128, 64, 64],    # 34: sink
                [255, 64, 64],    # 35: lamp
                [64, 255, 64],    # 36: bathtub
                [64, 64, 255],    # 37: bag
                [255, 255, 64],   # 38: other struct
                [64, 255, 255],   # 39: other furntr
            ]
            
            # Extend to 256 colors for full palette
            while len(colors) < 256:
                colors.append([0, 0, 0])
            
            return np.array(colors, dtype=np.uint8)
    except Exception as e:
        print(f"Warning: Could not load NYU color palette: {e}")
        # Fallback to random colors
        return get_colors(40)


def print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc, class_names=None, show_no_back=False, no_print=False):
    """Print IoU results."""
    n = iou.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = "Class %d:" % (i + 1)
        else:
            cls = "%d %s" % (i + 1, class_names[i])
        lines.append("%-8s\t%.3f%%" % (cls, iou[i] * 100))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:])
    
    if show_no_back:
        lines.append(
            "----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%"
            % (
                "mean_IoU",
                mean_IoU * 100,
                "mean_IoU_no_back",
                mean_IoU_no_back * 100,
                "freq_IoU",
                freq_IoU * 100,
                "mean_pixel_acc",
                mean_pixel_acc * 100,
                "pixel_acc",
                pixel_acc * 100,
            )
        )
    else:
        lines.append(
            "----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%"
            % (
                "mean_IoU",
                mean_IoU * 100,
                "freq_IoU",
                freq_IoU * 100,
                "mean_pixel_acc",
                mean_pixel_acc * 100,
                "pixel_acc",
                pixel_acc * 100,
            )
        )
    
    print(
        "----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%"
        % (
            "mean_IoU",
            mean_IoU * 100,
            "mean_IU_no_back",
            mean_IoU_no_back * 100,
            "freq_IoU",
            freq_IoU * 100,
            "mean_pixel_acc",
            mean_pixel_acc * 100,
            "pixel_acc",
            pixel_acc * 100,
        )
    )
    line = "\n".join(lines)
    if not no_print:
        print(line)
    else:
        print(line[-1])
    return str(mean_IoU * 100)


def save_prediction_with_palette(prediction, save_path, dataset_name="NYUDepthv2"):
    """Save prediction with appropriate color palette."""
    try:
        import matplotlib.pyplot as plt
        
        if dataset_name in ["NYUDepthv2", "SUNRGBD"]:
            # Use NYU color palette
            palette = get_nyu_colors()
            colored_pred = palette[prediction]
            plt.imsave(save_path, colored_pred)
        elif dataset_name in ["KITTI-360", "EventScape"]:
            # Use default palette
            palette = [
                [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                [0, 80, 100], [0, 0, 230], [119, 11, 32]
            ]
            palette = np.array(palette, dtype=np.uint8)
            colored_pred = palette[prediction]
            plt.imsave(save_path, colored_pred)
        elif dataset_name in ["MFNet"]:
            # Use MFNet palette
            palette = np.array([
                [0, 0, 0], [64, 0, 128], [64, 64, 0], [0, 128, 192],
                [0, 0, 192], [128, 128, 0], [64, 64, 128], [192, 128, 128],
                [192, 64, 0]
            ], dtype=np.uint8)
            colored_pred = palette[prediction]
            plt.imsave(save_path, colored_pred)
        else:
            # Fallback: save as grayscale
            cv2.imwrite(save_path, prediction.astype(np.uint8))
            
        print(f"Saved colored prediction to: {save_path}")
        return True
        
    except Exception as e:
        print(f"Error saving prediction with palette: {e}")
        # Fallback: save as grayscale
        try:
            cv2.imwrite(save_path, prediction.astype(np.uint8))
            print(f"Saved grayscale prediction to: {save_path}")
            return True
        except Exception as e2:
            print(f"Error saving grayscale prediction: {e2}")
            return False

# reference: https://github.com/Mr-TalhaIlyas/EMPatches/blob/8970e749fb2226b3c18ab057886ea142c95d635c/
from .scripts.empatches_0 import EMPatches
import os
import sys

# print(os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath('..'))
# print(sys.path)

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src.visualization.plot_utils import (
    plot_image_grid,
    inference_plot_and_save,
    compare_images,
)


def patch_combine_test():
    img = cv2.imread("src/utils/butterfly.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    emp = EMPatches()
    patches, indices = emp.extract_patches(img, patchsize=572, overlap=0.2)
    # plot_image_grid((patches[0], "patch-0"),(patches[1], "patch-1"), gridshape=(1,2))

    # tiled = imgviz.tile(list(map(np.uint8, patches.imgs)), border=(255, 0, 0))

    merged_img = emp.merge_patches(patches, indices)
    merged_img = merged_img.astype(np.uint8)
    # display
    # plt.figure()
    # plt.imshow(merged_img.astype(np.uint8))
    # plt.show()

    # a simple array difference also says that merged image is same as before split-merge
    compare_images(img, merged_img)


def split_lr_merge_hr(lr_img, patch_size, scale=2):
    """keep the patch-size as large as possible for which SR method fits in GPU memory to keep the error due to patch-wise SR

    Parameters
    ----------
    lr_img : lr img
    scale : int, optional
        _description_, by default 2
    """

    if not isinstance(lr_img, np.ndarray):
        lr_img = np.array(lr_img)

    emp = EMPatches()
    lr_patches, lr_indices = emp.extract_patches(
        lr_img, patchsize=patch_size, overlap=0.2
    )

    # resizing to hr by scale
    bicubic_hr = cv2.resize(
        lr_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    # first scaling each patch
    hr_patches = []
    for patch in lr_patches:
        scaled_patch = cv2.resize(
            patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        hr_patches.append(scaled_patch)

    # now get the indices for merging scaled patches
    # _, hr_indices = emp.extract_patches(bicubic_hr, patchsize=patch_size*scale, overlap=0.2)
    hr_indices = [tuple(x * 2 for x in i) for i in lr_indices]

    hr_merged_img = emp.merge_patches(hr_patches, hr_indices)
    hr_merged_img = hr_merged_img.astype(np.uint8)
    diff_ = np.sum(hr_merged_img - bicubic_hr)
    print(
        f"difference between scaled hr image and merged scaled patches of lr image = {diff_}"
    )
    breakpoint()
    compare_images(bicubic_hr, hr_merged_img)


def creates_lr_patches_hr_merge_indices(lr_img, patch_size, scale=2):
    """keep the patch-size as large as possible for which SR method fits in GPU memory to keep the error due to patch-wise SR

    Parameters
    ----------
    lr_img : lr img
    scale : int, optional
        _description_, by default 2
    """

    if not isinstance(lr_img, np.ndarray):
        lr_img = np.array(lr_img)

    emp = EMPatches()
    lr_patches, lr_indices = emp.extract_patches(
        lr_img, patchsize=patch_size, overlap=0.2
    )

    lr_patches = [Image.fromarray(patch, mode="RGB") for patch in lr_patches]
    # now get the indices for merging scaled patches
    # _, hr_indices = emp.extract_patches(bicubic_hr, patchsize=patch_size*scale, overlap=0.2)
    hr_indices = [tuple(x * 2 for x in i) for i in lr_indices]

    return lr_patches, hr_indices


def merge_hr_patches(hr_patches, hr_indices):
    for i, patch in enumerate(hr_patches):
        if not isinstance(patch, np.ndarray):
            hr_patches[i] = np.array(patch)
    emp = EMPatches()
    hr_merged_img = emp.merge_patches(hr_patches, hr_indices)
    hr_merged_img = hr_merged_img.astype(np.uint8)
    hr_merged_img = Image.fromarray(hr_merged_img, mode="RGB")
    return hr_merged_img


if __name__ == "__main__":
    lr_path = "data/raw/Set5/input_lr/butterfly.png"
    img = Image.open(lr_path)
    split_lr_merge_hr(img, 200)

import argparse
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import numpy as np
import skimage as ski

SCALE = 2


def least_int_multiplier(i):
    """reduce the input such that its multiplication with scale is an integer"""
    if isinstance(i, int) and isinstance(SCALE, int):
        return i
    if not (i * SCALE).is_integer():
        i = (i * SCALE - SCALE) / SCALE
    return i


def pair_cropping(lrimg, hrimg, N):
    """takes the lr img and hr img and crop them for scale and following N splits in both direction"""

    x, y = lrimg.size
    X, Y = hrimg.size
    print(f" original dimensions, lrimg: {x} x {y}, hrimg: {X} x {Y}")
    patch_x = int(x / N)
    patch_x = least_int_multiplier(patch_x)
    patch_y = int(y / N)
    patch_y = least_int_multiplier(patch_y)

    patch_X = int(X / N)
    patch_Y = int(Y / N)

    diff_patch_x = patch_X - (patch_x * SCALE)
    diff_patch_y = patch_Y - (patch_y * SCALE)

    if diff_patch_x >= 0:
        lefthr, righthr = dimension_reducer(diff_patch_x * N, patch_x * SCALE, N)
        leftlr, rightlr = 0, patch_x * N
    else:
        patch_x = int(patch_X / SCALE)
        patch_x = least_int_multiplier(patch_x)
        diff_patch_x = patch_X - (patch_x * SCALE)
        lefthr, righthr = dimension_reducer(diff_patch_x * N, patch_x * SCALE, N)
        leftlr, rightlr = dimension_reducer((x - patch_x * N), patch_x, N)

    if diff_patch_y >= 0:
        upperhr, lowerhr = dimension_reducer(diff_patch_y * N, patch_y * SCALE, N)
        upperlr, lowerlr = 0, patch_y * N
    else:
        patch_y = int(patch_Y / SCALE)

        patch_y = least_int_multiplier(patch_y)
        diff_patch_y = patch_Y - (patch_y * SCALE)
        upperhr, lowerhr = dimension_reducer(diff_patch_y * N, patch_y * SCALE, N)
        upperlr, lowerlr = dimension_reducer((y - patch_y * N), patch_y, N)

    lrimg_crop = lrimg.crop((leftlr, upperlr, rightlr, lowerlr))
    hrimg_crop = hrimg.crop((lefthr, upperhr, righthr, lowerhr))
    x_crop, y_crop = lrimg_crop.size
    X_crop, Y_crop = hrimg_crop.size
    print(f" number of partitions: {N}")
    print(
        f" cropped dimensions, lrimg: {x_crop} x {y_crop}, hrimg: {X_crop} x {Y_crop}"
    )
    print(f"patch lr dimensions, x :{x_crop/N}, y :{y_crop/N}")
    print(f"patch hr dimensions, x :{X_crop/N}, y :{Y_crop/N}")
    return lrimg_crop, hrimg_crop


def dimension_reducer(diff, patch_len, N):
    if diff % 2 == 0:
        left = diff % 2
        right = left + patch_len * N

    else:
        diff_floor = int(diff % 2)
        left = diff_floor
        right = diff_floor + patch_len * N
    return left, right


def main():
    lr_img = Image.open("./hg-03-03-n-1-l_200xb_0.jpg")
    # hr_img = Image.open("./hg-03-03-n-1-l_500x01.jpg")
    # DOWNSCLAING HR IMG TO OBTAIN AN INTEGER SCALING FACTOR OF 2
    hr_img = ski.io.imread("./hg-03-03-n-1-l_500x01.jpg", plugin="pil")
    hr_img = cv2.resize(hr_img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)

    # print("====== saving cropped image ========")
    # outp_path ="./hg-03-03-n-1-l_200xb_cropped.jpg"
    # output_img.save(outp_path, format='JPEG', subsampling=0, quality=100)


if __name__ == "__main__":
    main()

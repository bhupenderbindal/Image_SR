from src.visualization.plot_utils import inference_plot_and_save, calc_metrics

import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms.functional import pil_to_tensor

"""compare metrics on blurred and cropped+color-change
"""

gt_img = Image.open("./src/visualization/metrics_compare/butterfly.png")
blur_img = gt_img.filter(ImageFilter.GaussianBlur(5))

cropped_img = ImageOps.crop(gt_img, border=20)
expanded_img = ImageOps.expand(
    cropped_img,
    border=20,
)
rgb2xyz = (
    0.412453,
    0.357580,
    0.180423,
    0,
    0.212671,
    0.715160,
    0.072169,
    0,
    0.019334,
    0.119193,
    0.950227,
    0,
)
expanded_img = expanded_img.convert("RGB", rgb2xyz)

out_dir = "./src/visualization/metrics_compare/"
print(out_dir)

blur_img.save(out_dir + "blur.png")
expanded_img.save(out_dir + "other.png")
# calculates metrics on above images
psnr_blur, ssim_blur, lpips_distance_blur = calc_metrics(gt_img, blur_img)
psnr_other, ssim_other, lpips_distance_other = calc_metrics(gt_img, expanded_img)
print(psnr_blur, ssim_blur, lpips_distance_blur)
print(psnr_other, ssim_other, lpips_distance_other)
inference_plot_and_save(gt_img, blur_img, expanded_img, out_dir, "lpips_vs_other")

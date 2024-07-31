from PIL import Image
from src.visualization.plot_utils import inference_plot_and_save
import os
from torchvision.transforms.functional import pil_to_tensor
import numpy as np

img3 = Image.open("./src/visualization/metrics-3epoch-vs-120epoch/fly3epoch.png")
lrimg = Image.open("./src/visualization/metrics-3epoch-vs-120epoch/butterfly-lr.png")
gt_img = Image.open("./src/visualization/metrics-3epoch-vs-120epoch/butterfly-hr.png")

out_dir = os.getcwd() + "/src/visualization/metrics-3epoch-vs-120epoch/"
print(out_dir)

img120 = Image.open("./src/visualization/metrics-3epoch-vs-120epoch/fly120epoch.png")

# calculates metrics on above images

inference_plot_and_save(gt_img, lrimg, img3, out_dir, "3epoch")
inference_plot_and_save(gt_img, lrimg, img120, out_dir, "120epoch")

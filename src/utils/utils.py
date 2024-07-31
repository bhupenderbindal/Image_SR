import cv2
from PIL import Image
import numpy as np


# general utils to be further re-written and divide into data-utils, vis-utils, general utils

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
   
    if img.mode == "YCbCr":
        img = img.convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# GENERAL UTILS
# def artificial_downscaled_lrimgs():
#     """generates bicubic scaled lr imgs for hr imgs in paired-data-dir
#     """
#     root_dir = paired_data_dir()
#     train_dir = join(root_dir, "train")
#     input_dir = join(train_dir, "input_lr")
#     output_dir = join(train_dir, "output_hr")

#     # input_filenames = [join(self.input_dir, x) for x in listdir(self.input_dir) if is_image_file(x)]
#     output_filenames = [join(output_dir, x) for x in listdir(output_dir) if is_image_file(x)]
#     input_filenames = [join(input_dir, (x[:-4] + "_artificial_lr") ) for x in listdir(output_dir) if is_image_file(x)]
#     for i, file in enumerate(output_filenames):
#         image = ski.io.imread(file)

#         if image.shape[-1] == 4:
#             image_rgb = ski.color.rgba2rgb(image)
#         else:
#             # print("image has already 3 channnels")
#             image_rgb=image
        
#         image_rgb = ski.transform.rescale(image_rgb, 0.5, order =3, channel_axis=-1)
#        
#         ski.io.imsave(input_filenames[i] +".png", image_rgb,plugin= "pil")

      
from src.my_logger import Logger
from src.visualization.plot_utils import inference_plot_and_save
from src.utils.utils import is_image_file
from src.visualization.plot_utils import plot_image_grid, calc_avg_metrics
from src.utils.patch_and_combine import (
    creates_lr_patches_hr_merge_indices,
    merge_hr_patches,
)

import os
from os import listdir
from datetime import datetime
import argparse
from pathlib import Path
import time


import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import torchvision.transforms.functional as F
from torch.profiler import profile, record_function, ProfilerActivity


class Inference:
    def __init__(
        self, inference_dir, save_dir, filename=None, channeltype=None, ps=256
    ):
        self.inference_dir = Path(inference_dir)
        self.allfilenames = list(self.inference_dir.rglob("*.jpg"))

        self.input_filenames = [x for x in (self.allfilenames)]
        # filtering images with "200" resolution
        self.input_filenames = [
            x for x in (self.allfilenames) if is_image_file(str(x)) and "200x" in str(x)
        ]
        self.input_filenames.sort()
        self.device = torch.device("cuda:0" if GPU_IN_USE else "cpu:0")
        self.channeltype = channeltype
        # ===========================================================
        # model import & setting
        # ===========================================================
        if filename:
            self.model = save_dir + "/" + filename + ".pth"
        else:
            self.model = save_dir + "/model_path.pth"
        model = torch.load(self.model, map_location=lambda storage, loc: storage)
        self.model = model.to(self.device).eval()

        # ===========================================================
        # creating output dir
        # ===========================================================

        out_dir = save_dir
        t = datetime.now().strftime("_%b_%d_%H_%M_%S_%f") + "/"

        self.out_dir = os.path.join(out_dir, t)
        os.makedirs(self.out_dir)
        self.ps = ps

    @torch.inference_mode(mode=True)
    def inference(self):
        logger_instance = Logger()
        logger_instance.initialize("inference")
        logger = logger_instance.get_logger()
        times = []
        for lr_path in self.input_filenames:
            start_time = time.perf_counter()

            lr_img = Image.open(lr_path)
            lr_patches, hr_indices = creates_lr_patches_hr_merge_indices(
                lr_img, self.ps, scale=2
            )
            hr_patches = []

            for img in lr_patches:
                out_img = self.single_img_sr(img)
                hr_patches.append(out_img)
            out_img = merge_hr_patches(hr_patches, hr_indices)

            file_base_name = os.path.splitext(os.path.basename(lr_path))[0]
            out_img.save(self.out_dir + file_base_name + "_sr_hr.png")
            size = (lr_img.size[0] * 2, lr_img.size[1] * 2)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            print(
                f"time for single patch of {self.ps} size: {end_time - start_time:.4f} seconds"
            )
            logger.info(
                f"time for single patch of {self.ps} size: {end_time - start_time:.4f} seconds"
            )
            bicubic_hr = lr_img.resize(size, Image.BICUBIC)
            bicubic_hr.save(self.out_dir + file_base_name + "_bicubic_hr.png")

        logger.info(f"avg time :{sum(times)/len(times)}")

    @torch.inference_mode(mode=True)
    def single_img_sr(self, img):

        if self.channeltype == "y":
            y, cb, cr = img.convert("YCbCr").split()
        if self.channeltype == "rgb":
            y = img

        data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        data = data.to(self.device)

        if GPU_IN_USE:
            cudnn.benchmark = True

        # ===========================================================
        # output and save image
        # ===========================================================

        if self.channeltype == "y":

            out = self.model(data)
            out = out.cpu()
            out_img_y = out.data[0].numpy()
            out_img_y *= 255.0
            out_img_y = out_img_y.clip(0, 255)
            out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode="L")
            out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
            out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
            out_img = Image.merge("YCbCr", [out_img_y, out_img_cb, out_img_cr]).convert(
                "RGB"
            )

        if self.channeltype == "rgb":
            out = self.model(data)
            out = out.cpu()
            out_img_y = out.data[0].numpy()
            out_img_y = np.moveaxis(out_img_y, 0, -1)
            a = out_img_y
            out_img_y = 255 * (a - np.min(a)) / np.ptp(a)
            out_img = Image.fromarray(np.uint8(out_img_y), mode="RGB")

        torch.cuda.empty_cache()
        return out_img

    def cuda_prop(self):
        # printing global gpu memory info
        global_free, total_gpu_memory = torch.cuda.mem_get_info(device=self.device)
        print(
            f" global_free: {global_free/1e9}, total_gpu_memory: {total_gpu_memory/1e9} "
        )


GPU_IN_USE = torch.cuda.is_available()


def main():
    # ===========================================================
    # Argument settings
    # ===========================================================
    parser = argparse.ArgumentParser(description="PyTorch Super Res Example")
    parser.add_argument(
        "--patchwise",
        type=bool,
        default=True,
        help="perform patchwise sr for large image",
    )
    parser.add_argument(
        "--save_dir", type=str, default="None", help="dir where model is saved"
    )
    parser.add_argument(
        "--file_name", type=str, default="model_path", help="name of saved model file"
    )
    parser.add_argument(
        "--channeltype",
        "-ct",
        type=str,
        default="y",
        help="channels for model. options- y and rgb.",
    )

    args = parser.parse_args()
    print(args)
    # setting the inference directory containing test data
    inference_dir = os.getcwd() + "/data/raw/all_data/test_images"
    inferencer = Inference(
        inference_dir, args.save_dir, args.file_name, args.channeltype
    )
    inferencer.inference()


if __name__ == "__main__":
    main()
    # example usage to run from cli
    # python -m src.srcnn.inference  --save_dir ./model --channeltype rgb

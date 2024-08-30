from __future__ import print_function
import os
from datetime import datetime
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor
import time

import numpy as np

from src.visualization.plot_utils import inference_plot_and_save

from os import listdir
from os.path import join

from src.utils.utils import is_image_file
from src.visualization.plot_utils import calc_avg_metrics
from src.utils.patch_and_combine import (
    creates_lr_patches_hr_merge_indices,
    merge_hr_patches,
)

from src.my_logger import Logger


class MixValidation:
    #
    def __init__(self, paired_data_dir, args=None, patchwise=False, ps=256):
        self.input_dir = join(paired_data_dir, "input_lr")
        self.output_dir = join(paired_data_dir, "output_hr")

        self.output_filenames = [
            join(self.output_dir, x)
            for x in listdir(self.output_dir)
            if is_image_file(x)
        ]
        self.input_filenames = [
            join(self.input_dir, x) for x in listdir(self.input_dir) if is_image_file(x)
        ]
        # PAIRED FILENAMES AFTER SORT SHALL BE TRUE PAIR -- below logic sorts the filenames
        self.output_filenames.sort()
        self.input_filenames.sort()
        self.device = torch.device("cuda:0" if GPU_IN_USE else "cpu:0")
        self.args = args
        # ===========================================================
        # model import & setting
        # ===========================================================
        self.model = args.save_dir + "/model_path.pth"
        model = torch.load(self.model, map_location=lambda storage, loc: storage)
        self.model = model.to(self.device).eval()

        # ===========================================================
        # creating output dir
        # ===========================================================

        out_dir = args.save_dir
        t = datetime.now().strftime("_%b_%d_%H_%M_%S_%f") + "/"

        self.out_dir = os.path.join(out_dir, t)
        os.makedirs(self.out_dir)
        self.patchwise = patchwise
        self.ps = ps

    @torch.inference_mode(mode=True)
    def validation(self):
        bicubic_metrics_list = []
        output_metrics_list = []
        logger_instance = Logger()
        logger_instance.initialize("inference")
        logger = logger_instance.get_logger()
        times = []
        for lr_path, hr_path in zip(self.input_filenames, self.output_filenames):
            start_time = time.perf_counter()
            lr_img = Image.open(lr_path)
            gt_img = Image.open(hr_path)
            # Convert images to RGB if they are not already in RGB mode
            if lr_img.mode != "RGB":
                lr_img = lr_img.convert("RGB")

            if gt_img.mode != "RGB":
                gt_img = gt_img.convert("RGB")
            if self.patchwise:

                lr_patches, hr_indices = creates_lr_patches_hr_merge_indices(
                    lr_img, self.ps, scale=2
                )
                hr_patches = []
                # ground truth image
                for img in lr_patches:
                    out_img = self.single_img_sr(img)
                    hr_patches.append(out_img)
                out_img = merge_hr_patches(hr_patches, hr_indices)
            else:
                out_img = self.single_img_sr(lr_img)

            file_base_name = os.path.splitext(os.path.basename(lr_path))[0]
            out_img.save(self.out_dir + file_base_name + "_sr_hr.png")
            end_time = time.perf_counter()
            print(
                f"time for single patch of {self.ps} size: {end_time - start_time:.4f} seconds"
            )
            logger.info(
                f"time for single patch of {self.ps} size: {end_time - start_time:.4f} seconds"
            )
            times.append(end_time - start_time)

            # gt_img.save(self.out_dir + file_base_name+ "_ground_truth.png")
            # for inference without GT skipping
            # continue
            # lr_img.save(self.out_dir + file_base_name+ "_lr.png")
            # breakpoint()
            # print(gt_img, lr_img, out_img)
            a, b = inference_plot_and_save(
                gt_img, lr_img, out_img, self.out_dir, file_base_name
            )
            bicubic_metrics_list.append(a)
            output_metrics_list.append(b)

        logger.info(f"avg time :{sum(times)/len(times)}")
        psnr_bicubic, ssim_bicubic, lpips_distance_bicubic = calc_avg_metrics(
            bicubic_metrics_list
        )
        psnr_output, ssim_output, lpips_distance_output = calc_avg_metrics(
            output_metrics_list
        )
        print(
            f"psnr_bicubic {psnr_bicubic}, ssim_bicubic {ssim_bicubic}, lpips_distance_bicubic {lpips_distance_bicubic}"
        )
        print(
            f"psnr_output {psnr_output}, ssim_output {ssim_output}, lpips_distance_output {lpips_distance_output}"
        )

    @torch.inference_mode(mode=True)
    def single_img_sr(self, img):

        if self.args.channeltype == "y":
            y, cb, cr = img.convert("YCbCr").split()
        if self.args.channeltype == "rgb":
            y = img

        data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        data = data.to(self.device)

        if GPU_IN_USE:
            cudnn.benchmark = True

        # ===========================================================
        # output and save image
        # ===========================================================

        if self.args.channeltype == "y":

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

        if self.args.channeltype == "rgb":
            #  output values scaling to 0-255
            # with torch.no_grad():

            out = self.model(data)
            # for param in self.model.parameters():
            #     if param.requires_grad:
            #         print("Gradients are being calculated.")
            out = out.cpu()
            out_img_y = out.data[0].numpy()
            out_img_y = np.moveaxis(out_img_y, 0, -1)
            a = out_img_y
            out_img_y = 255 * (a - np.min(a)) / np.ptp(a)  # .astype(int)

            out_img = Image.fromarray(np.uint8(out_img_y), mode="RGB")

        torch.cuda.empty_cache()
        return out_img

    def cuda_prop(self):
        # printing global gpu memory info
        global_free, total_gpu_memory = torch.cuda.mem_get_info(device=self.device)
        print(
            f" global_free: {global_free/1e9}, total_gpu_memory: {total_gpu_memory/1e9} "
        )
        # print(f" cuda allocated memory : {torch.cuda.memory_allocated(device=self.device)/1e9} GB")


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
        "--channeltype",
        "-ct",
        type=str,
        default="y",
        help="channels for model. options- y and rgb.",
    )

    args = parser.parse_args()
    print(args)
    PAIRED_DATA_DIR = (
        os.getcwd() + "/data/processed/set_3/img_pairs_train_val_test/val/"
    )
    mixvalidate = MixValidation(
        paired_data_dir=PAIRED_DATA_DIR, args=args, patchwise=True
    )
    mixvalidate.validation()


if __name__ == "__main__":
    main()
    # example usage to run from cli
    # python -m src.srcnn.super_resolve_rgb  --save_dir ./model --channeltype rgb

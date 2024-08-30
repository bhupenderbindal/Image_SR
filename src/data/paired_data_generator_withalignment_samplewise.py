from .imgsplitter import image_splitter, DefaultSplittingArgs
from .pair_cropping import pair_cropping
from .align_images import align_output_target
from src.visualization import plot_utils
from src.my_logger import Logger
from .split_dataset_train_val import main as train_val_split

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def single_img_splitter(image_path, rows, cols, output_dir: str = None):
    """takes an image and num of partitions --> saves splitted imgs"""
    split_args = DefaultSplittingArgs()
    split_args.image_path = [image_path]
    input_dir, ext = os.path.splitext(image_path)
    if output_dir:
        split_args.output_dir = output_dir
    else:
        split_args.output_dir = input_dir + "_splitted/"
    split_args.rows = rows
    split_args.cols = cols

    image_splitter(split_args)


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def single_aling(lr_img, hr_img):
    lr_img, max_rect = align_output_target(lr_img, hr_img, "feature")
    a = max_rect
    x, y, w, h = a[0], a[1], a[2], a[3]
    lr_img = lr_img[y : y + h, x : x + w, :]
    hr_img = hr_img[y : y + h, x : x + w, :]
    psnr = cv2.PSNR(lr_img, hr_img)
    return psnr, lr_img, hr_img


def align_crop_split(lrpath, hrpath, n, aligned_cropped_path, img_pairs_save_path):

    lr_img = cv2.imread(str(lrpath))
    hr_img = cv2.imread(str(hrpath))
    print(lrpath, hrpath)
    psnr, lr_img, hr_img = single_aling(lr_img, hr_img)
    logger = Logger().get_logger()
    logger.info(f"paths : {lrpath},  {hrpath} ")
    logger.info(f"=========== psnr: {psnr} =============")

    if psnr < 13:
        logger.warning(
            "===========  feature align does not work with these images ============="
        )
    else:

        # using linear interpolation to reverse the same operation in align function
        psnr2, lr_img2, hr_img2 = single_aling(lr_img, hr_img)
        if psnr2 > psnr:
            logger.info(f"=========== psnr: {psnr2} =============")
            lr_img = lr_img2
            hr_img = hr_img2

        lr_img = cv2.resize(
            lr_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
        )
        lr_img = convert_from_cv2_to_image(lr_img)
        hr_img = convert_from_cv2_to_image(hr_img)
        lr_img_cropped, hr_img_cropped = pair_cropping(lr_img, hr_img, n)
        #  AS THE UNAVAILABLE INFO IN LR  IMG IS FILLED WITH DARK NEED TO REJECT PAIRS WHICH HAS LARGE DARGE REGIONS
        # plot_utils.plot_image_grid((lr_img_cropped,"lr"), (hr_img_cropped, "hr"), gridshape=(1,2))

        print("====== saving cropped image ========")

        lr_cropped_img_path = aligned_cropped_path.joinpath(
            lrpath.stem + "_cropped" + lrpath.suffix
        )
        hr_cropped_img_path = aligned_cropped_path.joinpath(
            hrpath.stem + "_cropped" + hrpath.suffix
        )

        lr_img_cropped.save(
            lr_cropped_img_path, format="JPEG", subsampling=0, quality=100
        )
        hr_img_cropped.save(
            hr_cropped_img_path, format="JPEG", subsampling=0, quality=100
        )

        # SPLITTING THE CROPPED IMAGE

        print("====== splitting cropped image pair ========")
        single_img_splitter(
            image_path=lr_cropped_img_path,
            rows=n,
            cols=n,
            output_dir=(img_pairs_save_path.joinpath("input_lr")),
        )
        single_img_splitter(
            image_path=hr_cropped_img_path,
            rows=n,
            cols=n,
            output_dir=(img_pairs_save_path.joinpath("output_hr")),
        )


def read_all_samples(set_num):
    root_dir = Path.cwd().joinpath(
        "data", "raw", "all_data", "Images_set" + str(set_num)
    )
    print(f"======= root dir = {root_dir} ======")
    image_paths = list(root_dir.glob("**/*.jpg"))
    samples_dict = {}
    # grouping the images into samples by their parent directory
    for path in image_paths:
        common_pattern = path.parent
        sample = "_".join(common_pattern.parts[-3:])
        if sample in samples_dict:
            samples_dict[sample].append(path)

        else:
            samples_dict[sample] = [path]

    return samples_dict


def create_pairs_from_sample(sample_images_path, base_path):
    # split lr image into equal parts as hr image and return their paths
    dir_name = sample_images_path[0].stem[:-7]
    output_dir = base_path.joinpath(dir_name + "_splitted")
    output_dir.mkdir(exist_ok=True, parents=True)
    hr_paths = []
    # copy HR files from raw to output-dir
    for p in sample_images_path:
        if "500" in str(p) and "stitched" not in str(p):
            hr_paths.append(output_dir.joinpath(p.name))
            shutil.copy(src=p, dst=output_dir)

    lr_path = [p for p in sample_images_path if "200" in str(p)]
    hr_paths.sort()
    total_hr_parts = len(hr_paths)
    # split lr image in same number of parts as hr img
    single_img_splitter(lr_path[0], 1, total_hr_parts, output_dir)
    lr_paths = list(output_dir.glob("*200*.jpg"))
    lr_paths.sort()
    pair_paths_list = list(zip(lr_paths, hr_paths))
    print(pair_paths_list)
    return pair_paths_list


def main():
    parser = argparse.ArgumentParser(
        description="Split an image into rows and columns."
    )

    parser.add_argument("--n", type=int, help="number of splits in x and y direction")
    parser.add_argument("--set_num", type=int, help="which set of fdata")
    args = parser.parse_args()

    # Initialize the logger only once in the main entry point
    logger_instance = Logger()
    log_file = "./logs/" + "setnum_samplewise" + str(args.set_num) + ".log"
    logger_instance.initialize("data_gen", log_file=log_file)
    logger = logger_instance.get_logger()

    logger.info("Application started")

    samples_dict = read_all_samples(args.set_num)

    # CREATING DIRECTORIES FOR SAVING INTERMEDIATE ALIGNED AND CROPPED IMAGES AND FOR FINAL IMAGE PAIRS
    aligned_cropped_path = Path.cwd().joinpath(
        "data", "processed", "set_" + str(args.set_num), "aligned_cropped_imgs"
    )
    aligned_cropped_path.mkdir(exist_ok=True, parents=True)
    img_pairs_save_path = Path.cwd().joinpath(
        "data", "processed", "set_" + str(args.set_num), "img_pairs"
    )
    img_pairs_save_path.mkdir(exist_ok=True, parents=True)

    base_path = Path.cwd().joinpath("data", "processed", "set_" + str(args.set_num))
    # for each sample creates paired images
    for key, value in samples_dict.items():
        lr_hr_paths = create_pairs_from_sample(value, base_path)
        for lrpath, hrpath in lr_hr_paths:

            align_crop_split(
                lrpath, hrpath, args.n, aligned_cropped_path, img_pairs_save_path
            )

    train_val_split(input_dir=img_pairs_save_path, ratio=(0.8, 0.2))


if __name__ == "__main__":

    main()

import tarfile
from os import remove, listdir, getcwd
from os.path import exists, join, basename

from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import skimage as ski
from .dataset import PairedDatasetFromFolder as DatasetFromFolder, is_image_file

CROP_SIZE = 64


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose(
        [
            CenterCrop(crop_size // upscale_factor),
            ToTensor(),
        ]
    )


def target_transform(crop_size):
    return Compose(
        [
            CenterCrop(crop_size),
            ToTensor(),
        ]
    )


def get_training_set(upscale_factor, data_dir, channeltype):
    print(f"========== PAIRED DATA from {data_dir} ==========")
    root_dir = data_dir
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(CROP_SIZE, upscale_factor)
    return DatasetFromFolder(
        train_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
        channeltype=channeltype,
    )


def get_val_set(upscale_factor, data_dir, channeltype):
    root_dir = data_dir
    val_dir = join(root_dir, "val")
    crop_size = calculate_valid_crop_size(CROP_SIZE, upscale_factor)

    return DatasetFromFolder(
        val_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
        channeltype=channeltype,
    )


def get_test_set(upscale_factor, data_dir, channeltype):
    root_dir = data_dir
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(CROP_SIZE, upscale_factor)

    return DatasetFromFolder(
        test_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
        channeltype=channeltype,
    )


if __name__ == "__main__":
    artificial_downscaled_lrimgs()

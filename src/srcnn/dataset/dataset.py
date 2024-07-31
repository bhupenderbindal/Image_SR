from os import listdir
from os.path import join

import torch.utils.data as data
from PIL import Image
import kornia


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert("YCbCr")
    y, _, _ = img.split()
    return y


def load_rgb_img(filepath):
    img = Image.open(filepath)

    return img


def aug_transform_training():
    aug_list = kornia.augmentation.container.ImageSequential(
        kornia.augmentation.RandomHorizontalFlip(p=0.5, p_batch=1.0, keepdim=False),
        kornia.augmentation.RandomVerticalFlip(p=0.5, p_batch=1.0, keepdim=False),
        kornia.augmentation.RandomRotation(
            [180, 180], same_on_batch=False, align_corners=True, p=0.5, keepdim=False
        ),
        kornia.augmentation.RandomRotation(
            [90, 90], same_on_batch=False, align_corners=True, p=0.5, keepdim=False
        ),
        same_on_batch=True,
        random_apply=10,
    )
    return aug_list


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)
        ]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_image = load_img(self.image_filenames[index])
        target = input_image.copy()
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return len(self.image_filenames)


class PairedDatasetFromFolder(data.Dataset):
    def __init__(
        self,
        paired_data_dir,
        input_transform=None,
        target_transform=None,
        channeltype="y",
    ):
        super(PairedDatasetFromFolder, self).__init__()
        self.input_dir = join(paired_data_dir, "input_lr")
        self.output_dir = join(paired_data_dir, "output_hr")
        self.channeltype = channeltype
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.output_filenames = [
            join(self.output_dir, x)
            for x in listdir(self.output_dir)
            if is_image_file(x)
        ]
        self.input_filenames = [
            join(self.input_dir, x) for x in listdir(self.input_dir) if is_image_file(x)
        ]
        self.output_filenames.sort()
        self.input_filenames.sort()

    def __getitem__(self, index):
        if self.channeltype == "rgb":
            input_image = load_rgb_img(self.input_filenames[index])
            target = load_rgb_img(self.output_filenames[index])
        if self.channeltype == "y":
            input_image = load_img(self.input_filenames[index])
            target = load_img(self.output_filenames[index])

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        assert len(self.input_filenames) == len(self.output_filenames)
        return len(self.input_filenames)

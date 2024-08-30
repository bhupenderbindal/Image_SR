# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
import shutil
import random
import numpy as np

"""
split a train folder in train and val folder in defined ratio
"""


def main(input_dir=None, ratio=None):
    np.random.seed(21)
    random.seed(21)
    logger = logging.getLogger(__name__)
    logger.info("splitting data set from train data into train and val")
    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)
    output_dir = input_dir.parents[0].joinpath(input_dir.name + "_train_val_test")
    train_split_dir = output_dir.joinpath("train")
    val_split_dir = output_dir.joinpath("val")
    test_split_dir = output_dir.joinpath("test")
    state = random.getstate()
    for child in input_dir.iterdir():
        # creating a list of allfiles from the class directory in train directory
        allfilenames = list(child.rglob("*.jpg"))
        allfilenames.sort()
        random.setstate(state)
        random.shuffle(allfilenames)

        trainfilenames = allfilenames[: int(len(allfilenames) * ratio[0])]
        valfilenames = allfilenames[
            int(len(allfilenames) * ratio[0]) : int(len(allfilenames) * sum(ratio))
        ]
        testfilenames = allfilenames[int(len(allfilenames) * sum(ratio)) :]
        print(f"total images of class {child.name}: {len(allfilenames)}")
        print(f"train images of class {child.name}: {len(trainfilenames)}")
        print(f"val images of class {child.name}: {len(valfilenames)}")
        print(f"test images of class {child.name}: {len(testfilenames)}")

        train_class_dir = train_split_dir.joinpath(child.name)
        try:
            train_class_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            safe_rmdir(str(train_class_dir))
            print(f"===== {train_class_dir} exists and overwriting it")
            train_class_dir.mkdir(parents=True, exist_ok=True)

        val_class_dir = val_split_dir.joinpath(child.name)
        try:
            val_class_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            safe_rmdir(str(val_class_dir))
            print(f"===== {val_class_dir} exists and overwriting it")
            val_class_dir.mkdir(parents=True, exist_ok=True)

        test_class_dir = test_split_dir.joinpath(child.name)
        try:
            test_class_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            safe_rmdir(str(test_class_dir))
            print(f"===== {test_class_dir} exists and overwriting it")
            test_class_dir.mkdir(parents=True, exist_ok=True)

        ## Copy pasting images to target directory

        for name in trainfilenames:
            shutil.copy(name, train_class_dir)

        for name in valfilenames:
            shutil.copy(name, val_class_dir)

        for name in testfilenames:
            shutil.copy(name, test_class_dir)


def safe_rmdir(path: str):
    # List of system directories to exclude
    system_dirs = [
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "/",
        "/usr",
        "/bin",
        "/sbin",
        "/etc",
    ]

    # List of important files/directories to exclude
    important_dirs = ["C:\\", "/home/"]
    all_dirs = system_dirs + important_dirs
    # Check if path is in the list of system directories or important directories
    if any(path.lower().startswith(directory.lower()) for directory in all_dirs):
        print(
            f"Cannot delete {path}. This is a system directory or contains important files."
        )
        return
    # If path is not in the exclusion lists, delete it
    print(path)
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        return


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

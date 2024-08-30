from __future__ import print_function
import os
import argparse
import pprint
from datetime import datetime

import numpy as np
import random
import torch

from torch.utils.data import DataLoader

from .DBPN.solver import DBPNTrainer
from .SRCNN.solver import SRCNNTrainer
from .SubPixelCNN.solver import SubPixelTrainer
from .dataset import paired_data

from src.srcnn.super_resolve_rgb import MixValidation
from src.srcnn.inference import Inference
from src.my_logger import Logger

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description="PyTorch Super Res Example")

# hyper-parameters
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1, help="testing batch size")
parser.add_argument(
    "--nEpochs", type=int, default=2, help="number of epochs to train for"
)
parser.add_argument(
    "--lr", type=float, default=0.01, help="Learning Rate. Default=0.01"
)
parser.add_argument(
    "--seed", type=int, default=123, help="random seed to use. Default=123"
)

# model configuration
parser.add_argument(
    "--upscale_factor",
    "-uf",
    type=int,
    default=2,
    help="super resolution upscale factor",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="srgan",
    help="choose which model is going to use",
)
parser.add_argument(
    "--losstype",
    type=str,
    default="l1",
    help="choose loss type - l1/cobi/fdl/. default is l1",
)

# data type
parser.add_argument(
    "--datatype",
    "-dt",
    type=str,
    default="div2k",
    help='choose which data is going to use "div2k" or "fdata" or "both"',
)
parser.add_argument(
    "--channeltype",
    "-ct",
    type=str,
    default="y",
    help="channels for model. options- y and rgb.",
)

# experiment namimg
parser.add_argument(
    "--expname",
    type=str,
    help="name of the experiment for saving model, its settings, evaluation results",
)
parser.add_argument("--set_num", type=int, help="which set of fdata")


args = parser.parse_args()


def main():
    set_seed()
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print("===> Loading datasets")
    if args.datatype == "div2k":
        PAIRED_DATA_DIR = os.getcwd() + "/data/raw/div2k_data/"
        data_dirs = [(args.datatype, PAIRED_DATA_DIR)]

    if args.datatype == "fdata":
        PAIRED_DATA_DIR = (
            os.getcwd()
            + "/data/processed/"
            + "set_"
            + str(args.set_num)
            + "/img_pairs_train_val_test"
        )
        data_dirs = [(args.datatype, PAIRED_DATA_DIR)]

    if args.datatype == "both":
        div2k_data_dir = os.getcwd() + "/data/raw/div2k_data/"
        fdata_data_dir = (
            os.getcwd()
            + "/data/processed/"
            + "set_"
            + str(args.set_num)
            + "img_pairs_train_val_test"
        )
        data_dirs = [("div2k", div2k_data_dir), ("fdata", fdata_data_dir)]

    # creating experiment directory
    exp_dir = (
        os.getcwd()
        + "/results/srcnn/results2/"
        + "exp_"
        + args.expname
        + "_losstype_"
        + args.losstype
        + "_"
        + args.model
        + "_"
        + args.datatype
        + "_channel_"
        + args.channeltype
        + datetime.now().strftime("_%b_%d_%H_%M_%S_%f")
    )
    if os.path.isdir(exp_dir):
        print(
            "======== experiment already exists, change experiment name (and settings), exiting ======="
        )
        exit(0)
    else:
        print(f"======== creating experiment directory : {args.expname}")
        os.makedirs(exp_dir)
    args.exp_dir = exp_dir

    # Initialize the logger only once in the main entry point
    logger_instance = Logger()
    log_file = exp_dir + "/" + str(args.set_num) + ".log"
    logger_instance.initialize("model_train", log_file=log_file)
    logger = logger_instance.get_logger()

    logger.info("Training started")

    for datatype, data_dir in data_dirs:
        args.save_dir = os.path.join(args.exp_dir, datatype)
        if not os.path.isdir(args.save_dir):
            print(f"======== creating data-dir directory : {datatype}")
            os.makedirs(args.save_dir)

        save_conf(args)

        train_set = paired_data.get_training_set(
            args.upscale_factor, data_dir=data_dir, channeltype=args.channeltype
        )
        val_set = paired_data.get_val_set(
            args.upscale_factor, data_dir=data_dir, channeltype=args.channeltype
        )
        test_set = paired_data.get_test_set(
            args.upscale_factor, data_dir=data_dir, channeltype=args.channeltype
        )

        training_data_loader = DataLoader(
            dataset=train_set, batch_size=args.batchSize, shuffle=True
        )
        validation_data_loader = DataLoader(
            dataset=val_set, batch_size=args.testBatchSize, shuffle=False
        )
        testing_data_loader = DataLoader(
            dataset=test_set, batch_size=args.testBatchSize, shuffle=False
        )

        if args.model == "sub":
            model = SubPixelTrainer(args, training_data_loader, testing_data_loader)
        elif args.model == "srcnn":
            model = SRCNNTrainer(
                args,
                training_data_loader,
                validation_data_loader,
                testing_data_loader,
                args.losstype,
            )
        elif args.model == "dbpn":
            args.dbpntype = "dbpn"
            model = DBPNTrainer(
                args,
                training_data_loader,
                validation_data_loader,
                testing_data_loader,
                args.losstype,
            )
        else:
            raise Exception("the model does not exist")

        model.run()
        print("======= model validation ========")
        logger.info("======= model validation ========")
        # validate on set 5
        validation_paired_dir = os.getcwd() + "/data/raw/Set5"
        model_validate(args, validation_paired_dir)

        validation_paired_dir = os.path.join(PAIRED_DATA_DIR, "val")
        model_validate(args, validation_paired_dir)

        print("======= model test ========")
        model_test(args, PAIRED_DATA_DIR)

        if args.datatype == "fdata":
            print("======= model inference on full size unseen image ========")
            logger.info("======= model inference on full size unseen image ========")
            model_inference(args)


def model_validate(args, validation_paired_dir):
    if args.model == "dbpn":
        validator = MixValidation(
            paired_data_dir=validation_paired_dir, args=args, patchwise=True, ps=256
        )
    else:
        validator = MixValidation(paired_data_dir=validation_paired_dir, args=args)
    validator.validation()


def model_test(args, PAIRED_DATA_DIR):
    if args.datatype == "div2k":
        test_paired_dir = os.path.join(PAIRED_DATA_DIR, "test")
        model_validate(args, test_paired_dir)
    else:
        test_paired_dir = (
            os.getcwd() + "/data/processed/set_3/img_pairs_train_val_test/val/"
        )
        if args.model == "dbpn":
            validator = MixValidation(
                paired_data_dir=test_paired_dir, args=args, patchwise=True, ps=256
            )
        else:
            validator = MixValidation(paired_data_dir=test_paired_dir, args=args)
        validator.validation()


def model_inference(args):
    # hardcoded path for unprocessed test images
    inference_dir = os.getcwd() + "/data/raw/all_data/Images_set3"
    inferencer = Inference(inference_dir, args.save_dir, channeltype=args.channeltype)
    inferencer.inference()


def save_conf(conf):
    """save the config file to the results dir.

    Parameters
    ----------
    conf : Argparser.Namespace
    """
    with open((conf.save_dir + "/conf.txt"), mode="w") as f:
        pprint.pprint(conf.__dict__, f)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    main()

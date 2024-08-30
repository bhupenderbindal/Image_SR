from src.losses import lpipss
from src.visualization.plot_utils import EarlyStopper, plot_image_grid
from src.srcnn.dataset.dataset import aug_transform_training
from src.my_logger import Logger
from src.losses.FDL import FDL_loss
from src.losses.contextual_los import contextual_loss as cl

import random
from math import log10

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from kornia.contrib import extract_tensor_patches


class Trainer(ABC):

    def __init__(
        self, config, training_loader, validation_loader, testing_loader, losstype
    ):
        super(Trainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.logger = Logger().get_logger()
        self.device = torch.device("cuda:0" if self.CUDA else "cpu:0")
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.testing_loader = testing_loader
        self.mseloss = torch.nn.L1Loss()
        self.save_dir = config.save_dir
        if config.channeltype == "y":
            self.num_channels = 1
        if config.channeltype == "rgb":
            self.num_channels = 3

        if hasattr(config, "dbpntype"):
            self.dbpntype = config.dbpntype

        # SETTING loss type
        self.losstype = losstype
        self.l1_loss = nn.L1Loss()

    @abstractmethod
    def build_model(self):
        pass

    def set_loss(self):

        torch.manual_seed(self.seed)

        if self.losstype == "l1" or self.losstype == None:
            print("========== using L1 loss ==========")
            self.criterion = nn.L1Loss()
        if self.losstype == "l2":
            print("========== using L2 loss ==========")
            self.criterion = nn.MSELoss()
            self.early_stopper = EarlyStopper(patience=5, min_delta=0.001)

        if self.losstype == "cobi":
            print("========== using contextual loss, select channel type as rgb ===")

            # cobi loss
            self.criterion = cl.ContextualBilateralLoss(
                weight_sp=0.5, band_width=0.5, use_vgg=True, vgg_layer="relu3_4"
            )
            self.criterion_cobi_rgb = cl.ContextualBilateralLoss(use_vgg=False)
            self.early_stopper = EarlyStopper(patience=5, min_delta=0.01)

        if self.losstype == "fdl":
            print("========== using fdl loss ==========")
            self.criterion = FDL_loss(phase_weight=1.0)

        if self.losstype == "lpips":
            print("========== using lpips loss ==========")

            self.criterion = lpipss.LPIPS(net="alex", version=0.1)

        if self.CUDA:
            cudnn.benchmark = True
            self.criterion.cuda()

    def get_patches(self, img1, img2):
        kernel = (2, 2)  # (2, 3)
        i = random.randint(0, 1000)

        # random patch method
        p1 = extract_tensor_patches(img1, kernel)  # [0][i]
        p1 = p1[:, i, :]
        p2 = extract_tensor_patches(img2, kernel)  # [0][i]
        p2 = p2[:, i, :]

        return p1, p2

    def save_model(self):
        model_out_path = self.save_dir + "/model_path.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)

            aug_list = aug_transform_training()
            data = aug_list(data)
            target = aug_list(target, params=aug_list._params)

            self.optimizer.zero_grad()
            output = self.model(data)
            if self.losstype == "cobi":
                loss = self.criterion(output, target)
                output_patches, target_patches = self.get_patches(output, target)
                loss = loss + 0.0001 * self.criterion_cobi_rgb(
                    output_patches, target_patches
                )
            else:
                loss = self.criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def validate(self):
        self.model.eval()
        avg_psnr = 0
        val_loss = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.validation_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                if self.losstype == "cobi":
                    loss = self.criterion(prediction, target)
                    output_patches, target_patches = self.get_patches(
                        prediction, target
                    )
                    loss = loss + 0.0001 * self.criterion_cobi_rgb(
                        output_patches, target_patches
                    )
                else:
                    loss = self.criterion(prediction, target)
                val_loss += loss.item()

                mse = self.mseloss(prediction, target)
                try:
                    psnr = 10 * log10(1 / mse.item())
                except ValueError as e:
                    print(
                        "==== pnsr undefined as mse reaching inf ========, setting psnr to zero"
                    )
                    psnr = 0
                avg_psnr += psnr

        print(
            "    Average validation Loss: {:.4f}".format(
                val_loss / len(self.validation_loader)
            )
        )
        print(
            "    Validation Average PSNR: {:.4f} dB".format(
                avg_psnr / len(self.validation_loader)
            )
        )
        return self.early_stopper.early_stop((val_loss / len(self.validation_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)

                mse = self.mseloss(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_model_summary(self):
        self.logger.info("Model Summary:")
        self.logger.info(self.model)
        self.logger.info(
            "\nNumber of learnable parameters: {:,}".format(self.count_parameters())
        )

    def run(self):
        self.build_model()
        self.print_model_summary()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            # checking memory at first 3 epoch
            if epoch in (1, 2, 3):
                self.cuda_prop()
            ealry_stop = self.validate()
            if ealry_stop:
                print(f"Stopping early at epoch {epoch}  and saving model")
                self.save_model()
                break
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save_model()

    def cuda_prop(self):
        # printing global gpu memory info
        global_free, total_gpu_memory = torch.cuda.mem_get_info(device=self.device)
        print(
            f" global_free: {global_free/1e9}, total_gpu_memory: {total_gpu_memory/1e9} "
        )
        print(
            f" cuda allocated memory : {torch.cuda.memory_allocated(device=self.device)/1e9} GB"
        )

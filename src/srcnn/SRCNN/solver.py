from __future__ import print_function
import os
from math import log10

import torch
import torch.backends.cudnn as cudnn

from src.srcnn.solver_abc import Trainer
from .model import Net
from ..progress_bar import progress_bar
from src.losses.FDL import FDL_loss


class SRCNNTrainer(Trainer):

    def build_model(self):
        self.set_loss()

        self.model = Net(
            num_channels=self.num_channels,
            base_filter=64,
            upscale_factor=self.upscale_factor,
        ).to(self.device)
        self.model.weight_init(mean=0.0, std=0.01)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[50, 75, 100], gamma=0.5
        )

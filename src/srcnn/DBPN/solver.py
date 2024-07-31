from __future__ import print_function
from math import log10
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .model import DBPN, DBPNS, DBPNLL

from src.srcnn.solver_abc import Trainer
from src.losses.FDL import FDL_loss

from .ddbpn import DDBPN
from dataclasses import dataclass


@dataclass
class DDBPNArgs:
    scale: tuple = (2, 2)
    rgb_range: int = 255
    n_colors: int = 3


class DBPNTrainer(Trainer):
    def build_model(self):
        # currently loss used for the model is set here
        self.set_loss()
        if self.dbpntype == "dbpn":
            print("======= dbpn model =========")

            self.model = DBPN(
                num_channels=self.num_channels,
                base_channels=64,
                feat_channels=256,
                num_stages=7,
                scale_factor=self.upscale_factor,
            ).to(self.device)
            self.model.weight_init()

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[50, 75, 100], gamma=0.5
            )

        if self.dbpntype == "dbpnsmall":
            print("======= dbpnsmall model =========")

            args = DDBPNArgs()
            self.model = DDBPN(args=args)
            self.model.weight_init()

            self.optimizer = optim.Adam(
                self.model.parameters(), lr=1e-4, weight_decay=1e-4
            )
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5e5, gamma=0.1
            )

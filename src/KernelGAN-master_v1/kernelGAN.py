import torch
from . import loss
from . import networks
import torch.nn.functional as F
from .util import save_final_kernel, run_zssr, post_process_k, save_conf, save_plot
import matplotlib.pyplot as plt

# import matplotlib.image as img
from matplotlib.gridspec import GridSpec
import numpy as np


class KernelGAN:
    # Constraint co-efficients
    lambda_sum2one = 0.5
    lambda_bicubic = 5
    lambda_boundaries = 0.5
    lambda_centralized = 0
    lambda_sparse = 0
    # Parameters related to plotting and graphics
    plots = None
    loss_plot_space = None
    lr_son_image_space = None
    hr_father_image_space = None
    out_image_space = None
    iter = 0

    def __init__(self, conf):
        # Acquire configuration
        self.conf = conf

        # Define the GAN
        self.G = networks.Generator(conf).cuda()
        self.D = networks.Discriminator(conf).cuda()

        # self.print_model_summary()

        # Calculate D's input & output shape according to the shaving done by the networks
        self.d_input_shape = self.G.output_size
        self.d_output_shape = self.d_input_shape - self.D.forward_shave

        # Input tensors
        self.g_input = torch.FloatTensor(
            1, 3, conf.input_crop_size, conf.input_crop_size
        ).cuda()
        self.d_input = torch.FloatTensor(
            1, 3, self.d_input_shape, self.d_input_shape
        ).cuda()

        # The kernel G is imitating
        self.curr_k = torch.FloatTensor(conf.G_kernel_size, conf.G_kernel_size).cuda()

        # Losses
        self.GAN_loss_layer = loss.GANLoss(d_last_layer_size=self.d_output_shape).cuda()
        self.bicubic_loss = loss.DownScaleLoss(scale_factor=conf.scale_factor).cuda()
        self.sum2one_loss = loss.SumOfWeightsLoss().cuda()
        self.boundaries_loss = loss.BoundariesLoss(k_size=conf.G_kernel_size).cuda()
        self.centralized_loss = loss.CentralizedLoss(
            k_size=conf.G_kernel_size, scale_factor=conf.scale_factor
        ).cuda()
        self.sparse_loss = loss.SparsityLoss().cuda()
        self.loss_bicubic = 0

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward

        # Initialize networks weights
        self.G.apply(networks.weights_init_G)
        self.D.apply(networks.weights_init_D)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999)
        )

        self.total_loss_g, self.loss_d_fake, self.loss_d_real = [], [], []

        print("*" * 60 + '\nSTARTED KernelGAN on: "%s"...' % conf.input_image_path)

    def count_parameters(self, net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    def print_model_summary(self):
        print("Generator Summary:")
        print(self.G)
        print(
            "\nNumber of learnable parameters: {:,}".format(
                self.count_parameters(self.G)
            )
        )
        print("Discriminator Summary:")
        print(self.D)
        print(
            "\nNumber of learnable parameters: {:,}".format(
                self.count_parameters(self.D)
            )
        )

    # noinspection PyUnboundLocalVariable
    def calc_curr_k(self):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.Tensor([1.0]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        for ind, w in enumerate(self.G.parameters()):
            curr_k = (
                F.conv2d(delta, w, padding=self.conf.G_kernel_size - 1)
                if ind == 0
                else F.conv2d(curr_k, w)
            )
        self.curr_k = curr_k.squeeze().flip([0, 1])

    def train(self, g_input, d_input):
        self.set_input(g_input, d_input)
        self.train_g()
        self.train_d()

    #        self.plot()

    def set_input(self, g_input, d_input):
        self.g_input = g_input.contiguous()
        self.d_input = d_input.contiguous()

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        # Generator forward pass
        g_pred = self.G.forward(self.g_input)
        # Pass Generators output through Discriminator
        d_pred_fake = self.D.forward(g_pred)
        # Calculate generator loss, based on discriminator prediction on generator result
        loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)
        # Sum all losses
        total_loss_g = loss_g + self.calc_constraints(g_pred)
        # printing
        self.total_loss_g.append(total_loss_g)
        # print(f"gen loss{total_loss_g}")
        # Calculate gradients
        total_loss_g.backward()
        # Update weights
        self.optimizer_G.step()

    def calc_constraints(self, g_pred):
        # Calculate K which is equivalent to G
        self.calc_curr_k()
        # Calculate constraints
        self.loss_bicubic = self.bicubic_loss.forward(
            g_input=self.g_input, g_output=g_pred
        )
        loss_boundaries = self.boundaries_loss.forward(kernel=self.curr_k)
        loss_sum2one = self.sum2one_loss.forward(kernel=self.curr_k)
        loss_centralized = self.centralized_loss.forward(kernel=self.curr_k)
        loss_sparse = self.sparse_loss.forward(kernel=self.curr_k)
        # Apply constraints co-efficients
        return (
            self.loss_bicubic * self.lambda_bicubic
            + loss_sum2one * self.lambda_sum2one
            + loss_boundaries * self.lambda_boundaries
            + loss_centralized * self.lambda_centralized
            + loss_sparse * self.lambda_sparse
        )

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()
        # Discriminator forward pass over real example
        d_pred_real = self.D.forward(self.d_input)
        # Discriminator forward pass over fake example (generated by generator)
        # Note that generator result is detached so that gradients are not propagating back through generator
        g_output = self.G.forward(self.g_input)
        d_pred_fake = self.D.forward(
            (g_output + torch.randn_like(g_output) / 255.0).detach()
        )
        # Calculate discriminator loss
        loss_d_fake = self.criterionGAN(d_pred_fake, is_d_input_real=False)
        loss_d_real = self.criterionGAN(d_pred_real, is_d_input_real=True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        self.loss_d_fake.append(loss_d_fake)
        self.loss_d_real.append(loss_d_real)
        # print(f"dis loss{loss_d}")
        # Calculate gradients, note that gradients are not propagating back through generator
        loss_d.backward()
        # Update weights, note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

    def finish(self, iteration: int):
        final_kernel = post_process_k(self.curr_k, n=self.conf.n_filtering)
        save_final_kernel(final_kernel, self.conf, iteration)
        print("KernelGAN estimation complete!")
        save_path = run_zssr(final_kernel, self.conf)
        print(
            "FINISHED RUN (see --%s-- folder)\n" % self.conf.output_dir_path
            + "*" * 60
            + "\n\n"
        )
        # save_conf(self.conf)
        return save_path

    def checkpoint(self, iteration: int):
        final_kernel = post_process_k(self.curr_k, n=self.conf.n_filtering)
        save_final_kernel(final_kernel, self.conf, iteration)
        # run_zssr(final_kernel, self.conf)

    def plot(self):
        plots_data, labels = zip(
            *[
                (np.array([i.cpu().detach().numpy() for i in x]), l)
                for (x, l) in zip(
                    [self.total_loss_g, self.loss_d_fake, self.loss_d_real],
                    ["total loss gen", "loss dis fake", "loss dis real"],
                )
                if x is not None
            ]
        )
        plt.figure(figsize=(10, 6))

        for data, label in zip(plots_data, labels):
            plt.plot(data, label=label)

        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Losses during Training")
        plt.legend()
        plt.grid(True)
        # plt.show()

        # For the first iteration create the figure
        #        breakpoint()
        # if not self.iter:
        #     # Create figure and split it using GridSpec. Name each region as needed
        #     self.fig = plt.figure(figsize=(9.5, 9))
        #     grid = GridSpec(4, 4)
        #     self.loss_plot_space = plt.subplot(grid[:, :])
        #     #            self.lr_son_image_space = plt.subplot(grid[3, 0])
        #     #            self.hr_father_image_space = plt.subplot(grid[3, 3])
        #     #            self.out_image_space = plt.subplot(grid[3, 1])

        #     # Activate interactive mode for live plot updating
        #     plt.ion()

        #     # Set some parameters for the plots
        #     self.loss_plot_space.set_xlabel("step")
        #     self.loss_plot_space.set_ylabel("MSE")
        #     self.loss_plot_space.grid(True)
        #     self.loss_plot_space.set_yscale("log")
        #     self.loss_plot_space.legend()
        #     self.plots = [None] * 4

        #     # loop over all needed plot types. if some data is none than skip, if some data is one value tile it
        #     self.plots = self.loss_plot_space.plot(*[[0]] * 2 * len(plots_data))
        #     self.iter += 1
        # self.iter += 1
        # # Update plots
        # for plot, plot_data in zip(self.plots, plots_data):
        #     plot.set_data(range(len(plot_data)), plot_data)

        #     self.loss_plot_space.set_xlim([0, self.iter + 1])
        #     all_losses = np.array(plots_data)
        #     self.loss_plot_space.set_ylim(
        #         [np.min(all_losses) * 0.9, np.max(all_losses) * 1.1]
        #     )

        # # Mark learning rate changes
        # #       for iter_num in self.learning_rate_change_iter_nums:
        # #           self.loss_plot_space.axvline(iter_num)

        # # Add legend to graphics
        # self.loss_plot_space.legend(labels)

        # # Show current input and output images
        # #        self.lr_son_image_space.imshow(self.lr_son, vmin=0.0, vmax=1.0)
        # #        self.out_image_space.imshow(self.train_output, vmin=0.0, vmax=1.0)
        # #        self.hr_father_image_space.imshow(self.hr_father, vmin=0.0, vmax=1.0)

        # # These line are needed in order to see the graphics at real time
        # self.fig.canvas.draw()
        save_plot(self.conf)
        plt.pause(0.01)

from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage as ski
import torch
from ..utils.utils import convert_from_image_to_cv2, convert_from_cv2_to_image
from src.losses import lpipss


def plot_image_grid(*images, gridshape: (int, int), save: bool = False):
    """plot images with their titles in a grid format and save them
    example use:     plot_image_grid((img, "original"), (result, "LapSRN_x2"), gridshape=(1, 2))
    TODO- fails for (1,1) grid
    Parameters
    ----------
    gridshape : int, int
        _description_
    """

    fig, axes = plt.subplots(nrows=gridshape[0], ncols=gridshape[1])
    ax = axes.ravel()

    for i, img in enumerate(images):
        ax[i].imshow(img[0])
        ax[i].set_title(img[1])
        if save:
            plt.imsave(("./data/processed/" + img[1] + ".png"), img[0])

    plt.show()


def calc_metrics(img1, img2):
    """calclulates psnr and ssim and requires images to be in same colorspace

    Parameters
    ----------
    img1 : true image
    img2 : test image
    Returns
    -------
    (psnr, ssim)
    """

    if not isinstance(img1, np.ndarray):
        # img1 = convert_from_image_to_cv2(img1)
        img1 = np.array(img1)

    if not isinstance(img2, np.ndarray):
        # img2 = convert_from_image_to_cv2(img2)
        img2 = np.array(img2)
    # breakpoint()
    psnr = ski.metrics.peak_signal_noise_ratio(image_true=img1, image_test=img2)
    ssim = ski.metrics.structural_similarity(
        img1,
        img2,
        data_range=255.0,
        channel_axis=-1,
        gaussian_weights=False,
        full=False,
    )
    lpips_distance = lpips_metrics(img1, img2)

    return round(psnr, 3), round(ssim, 3), round(lpips_distance, 3)


@torch.inference_mode(mode=True)
def lpips_metrics(img1, img2):
    # requires imgs to be in np array unit8
    use_gpu = False
    loss_fn = lpipss.LPIPS(net="alex", version=0.1, verbose=False)

    if use_gpu:
        loss_fn.cuda()

    # Load images
    img1 = lpipss.im2tensor(img1)  # RGB image from [-1,1]
    img2 = lpipss.im2tensor(img2)

    if use_gpu:
        img1 = img1.cuda()
        img2 = img2.cuda()

    # Compute distance
    dist01 = loss_fn.forward(img1, img2)
    # print('Distance: %.3f'%dist01)
    torch.cuda.empty_cache()

    return dist01.cpu().item()


def inference_plot_and_save(
    gt_img, input_lr, output_hr, output_dir=None, output_name=None
):
    """NEED REWRITE TO HANDLE PIL AND CV2 TYPE IMAGES, CURRENT EXPECT IMAGES TO BE PIL TYPE WHICH ARE PASSES AS IT IS TO CALC-METRICS"""

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 15))
    ax = axes.ravel()

    if not isinstance(input_lr, np.ndarray):
        input_lr = convert_from_image_to_cv2(input_lr)

    bicubic_hr = cv2.resize(input_lr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if isinstance(bicubic_hr, np.ndarray):
        bicubic_hr = convert_from_cv2_to_image(bicubic_hr)

    bicubic_hr.save(output_dir + output_name + "_bicubic_hr.png")

    psnr_bicubic, ssim_bicubic, lpips_distance_bicubic = calc_metrics(
        gt_img, bicubic_hr
    )
    psnr_output, ssim_output, lpips_distance_output = calc_metrics(gt_img, output_hr)
    psnr_gt, ssim_gt, lpips_distance_gt = calc_metrics(gt_img, gt_img)

    ax[0].imshow(gt_img)
    ax[0].set_title("GT")
    ax[0].set_xlabel(
        "PSNR = "
        + str(psnr_gt)
        + "  SSIM = "
        + str(ssim_gt)
        + "  LPIPS = "
        + str(lpips_distance_gt)
    )
    ax[1].imshow(bicubic_hr)
    ax[1].set_xlabel(
        "PSNR = "
        + str(psnr_bicubic)
        + "  SSIM = "
        + str(ssim_bicubic)
        + "  LPIPS = "
        + str(lpips_distance_bicubic)
    )
    ax[1].set_title("bicubic HR")
    ax[2].imshow(output_hr)
    ax[2].set_title("HR output")
    ax[2].set_xlabel(
        "PSNR = "
        + str(psnr_output)
        + "  SSIM = "
        + str(ssim_output)
        + "  LPIPS = "
        + str(lpips_distance_output)
    )

    if output_dir and output_name:
        plt.savefig(
            (
                output_dir
                + output_name
                + "compare"
                + datetime.now().strftime("_%b_%d_%H_%M_%S_%f")
                + ".png"
            ),
            bbox_inches="tight",
        )
    # plt.savefig("xx.svg")
    # plt.show()
    # Closing the figure to free up memory
    plt.close(fig)

    bicubic_metrics = (psnr_bicubic, ssim_bicubic, lpips_distance_bicubic)
    output_metrics = (psnr_output, ssim_output, lpips_distance_output)
    return (bicubic_metrics, output_metrics)


def compare_images(img1, img2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
    ax = axes.ravel()

    psnr, ssim, lpips = calc_metrics(img1, img2)

    ax[0].imshow(img1)
    ax[0].set_title("GT")
    ax[1].imshow(img2)
    ax[1].set_xlabel(
        "PSNR = " + str(psnr) + "  SSIM = " + str(ssim) + "  LPIPS = " + str(lpips)
    )
    ax[1].set_title("img2")
    plt.show()


def calc_avg_metrics(metrics_list):
    # summing up
    psnr = sum(k[0] for k in metrics_list)
    ssim = sum(k[1] for k in metrics_list)
    lpips = sum(k[2] for k in metrics_list)
    l = len(metrics_list)
    # averaging
    return psnr / l, ssim / l, lpips / l


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

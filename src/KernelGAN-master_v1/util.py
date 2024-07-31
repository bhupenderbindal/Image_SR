import os
import pprint
import time
from difflib import SequenceMatcher
import cv2        
import torch
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from torch.nn import functional as F
from scipy.ndimage import measurements, interpolation

# from .ZSSRforKernelGAN.ZSSR import ZSSR
from .pytorch_ZSSR_master.ZSSR import ZSSR
from .pytorch_ZSSR_master.configs import X2_REAL_CONF, X2_GRADUAL_IDEAL_CONF
from src.utils.utils import convert_from_cv2_to_image, convert_from_image_to_cv2
from src.visualization.plot_utils import inference_plot_and_save


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def tensor2im(im_t):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.clip(
        np.round(
            (np.transpose(move2cpu(im_t).squeeze(0), (1, 2, 0)) + 1) / 2.0 * 255.0
        ),
        0,
        255,
    )
    return im_np.astype(np.uint8)


def im2tensor(im_np):
    """Copy the image to the gpu & converts to range [-1,1]"""
    im_np = im_np / 255.0 if im_np.dtype == "uint8" else im_np
    return (
        torch.FloatTensor(np.transpose(im_np, (2, 0, 1)) * 2.0 - 1.0)
        .unsqueeze(0)
        .cuda()
    )


def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()


def resize_tensor_w_kernel(im_t, k, sf=None):
    """Convolves a tensor with a given bicubic kernel according to scale factor"""
    # Expand dimensions to fit convolution: [out_channels, in_channels, k_height, k_width]
    k = k.expand(im_t.shape[1], im_t.shape[1], k.shape[0], k.shape[1])
    # Calculate padding
    padding = (k.shape[-1] - 1) // 2
    return F.conv2d(im_t, k, stride=round(1 / sf), padding=padding)


def read_image(path):
    """Loads an image"""
    im = Image.open(path).convert("RGB")
    im = np.array(im, dtype=np.uint8)
    return im


def rgb2gray(im):
    """Convert and RGB image to gray-scale"""
    return np.dot(im, [0.299, 0.587, 0.114]) if len(im.shape) == 3 else im


def swap_axis(im):
    """Swap axis of a tensor from a 3 channel tensor to a batch of 3-single channel and vise-versa"""
    return im.transpose(0, 1) if type(im) == torch.Tensor else np.moveaxis(im, 0, 1)


def shave_a2b(a, b):
    """Given a big image or tensor 'a', shave it symmetrically into b's shape"""
    # If dealing with a tensor should shave the 3rd & 4th dimension, o.w. the 1st and 2nd
    is_tensor = type(a) == torch.Tensor
    r = 2 if is_tensor else 0
    c = 3 if is_tensor else 1
    # Calculate the shaving of each dimension
    shave_r, shave_c = max(0, a.shape[r] - b.shape[r]), max(0, a.shape[c] - b.shape[c])
    return (
        a[
            :,
            :,
            shave_r // 2 : a.shape[r] - shave_r // 2 - shave_r % 2,
            shave_c // 2 : a.shape[c] - shave_c // 2 - shave_c % 2,
        ]
        if is_tensor
        else a[
            shave_r // 2 : a.shape[r] - shave_r // 2 - shave_r % 2,
            shave_c // 2 : a.shape[c] - shave_c // 2 - shave_c % 2,
        ]
    )


def create_gradient_map(im, window=5, percent=0.97):
    """Create a gradient map of the image blurred with a rect of size window and clips extreme values"""
    # Calculate gradients
    gx, gy = np.gradient(rgb2gray(im))
    # Calculate gradient magnitude
    gmag, gx, gy = np.sqrt(gx**2 + gy**2), np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = (
        pad_edges(gx, int(window)),
        pad_edges(gy, int(window)),
        pad_edges(gmag, int(window)),
    )
    lm_x, lm_y, lm_gmag = (
        clip_extreme(gx_pad, percent),
        clip_extreme(gy_pad, percent),
        clip_extreme(gmag, percent),
    )
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), "same") / (
        window**2
    )
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)


def create_probability_map(loss_map, crop):
    """Create a vector of probabilities corresponding to the loss map"""
    # Blur the gradients to get the sum of gradients in the crop
    blurred = convolve2d(loss_map, np.ones([crop // 2, crop // 2]), "same") / (
        (crop // 2) ** 2
    )
    # Zero pad s.t. probabilities are NNZ only in valid crop centers
    prob_map = pad_edges(blurred, crop // 2)
    # Normalize to sum to 1
    prob_vec = (
        prob_map.flatten() / prob_map.sum()
        if prob_map.sum() != 0
        else np.ones_like(prob_map.flatten()) / prob_map.flatten().shape[0]
    )
    return prob_vec


def pad_edges(im, edge):
    """Replace image boundaries with 0 without changing the size"""
    zero_padded = np.zeros_like(im)
    zero_padded[edge:-edge, edge:-edge] = im[edge:-edge, edge:-edge]
    return zero_padded


def clip_extreme(im, percent):
    """Zeroize values below the a threshold and clip all those above"""
    # Sort the image
    im_sorted = np.sort(im.flatten())
    # Choose a pivot index that holds the min value to be clipped
    pivot = int(percent * len(im_sorted))
    v_min = im_sorted[pivot]
    # max value will be the next value in the sorted array. if it is equal to the min, a threshold will be added
    v_max = im_sorted[pivot + 1] if im_sorted[pivot + 1] > v_min else v_min + 10e-6
    # Clip an zeroize all the lower values
    return np.clip(im, v_min, v_max) - v_min


def post_process_k(k, n):
    """Move the kernel to the CPU, eliminate negligible values, and centralize k"""
    k = move2cpu(k)
    # Zeroize negligible values
    significant_k = zeroize_negligible_val(k, n)
    # Force centralization on the kernel
    centralized_k = kernel_shift(significant_k, sf=2)
    # return shave_a2b(centralized_k, k)
    return centralized_k


def zeroize_negligible_val(k, n):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    k_sorted = np.sort(k.flatten())
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[-n - 1]
    # Clip values lower than the minimum value
    filtered_k = np.clip(k - k_n_min, a_min=0, a_max=100)
    # Normalize to sum to 1
    return filtered_k / filtered_k.sum()


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [
        np.exp(-(z**2) / (2 * sigma1**2)) / np.sqrt(2 * np.pi * sigma1**2)
        for z in range(-size // 2 + 1, size // 2 + 1)
    ]
    func2 = (
        func1
        if sigma2 == -1
        else [
            np.exp(-(z**2) / (2 * sigma2**2)) / np.sqrt(2 * np.pi * sigma2**2)
            for z in range(-size // 2 + 1, size // 2 + 1)
        ]
    )
    return (
        torch.FloatTensor(np.outer(func1, func2)).cuda()
        if is_tensor
        else np.outer(func1, func2)
    )


def nn_interpolation(im, sf):
    """Nearest neighbour interpolation"""
    pil_im = Image.fromarray(im)
    return np.array(
        pil_im.resize((im.shape[1] * sf, im.shape[0] * sf), Image.NEAREST),
        dtype=im.dtype,
    )


def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r : 2 * r + k_size, 2 * c : 2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (
        np.array(sf) - (np.array(kernel.shape) % 2)
    )
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(
        kernel, np.int32(np.ceil(np.max(np.abs(shift_vec)))) + 1, "constant"
    )

    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


def save_final_kernel(k_2, conf, iteration):
    """saves the final kernel and the analytic kernel to the results folder"""
    sio.savemat(
        os.path.join(
            conf.output_dir_path,
            f"%s_kernel_x2_iteration_{str(iteration)}.mat" % conf.img_name,
        ),
        {"Kernel": k_2},
    )
    kernel_save_path = os.path.join(conf.output_dir_path,
            f"%s_kernel_x2_iteration_{str(iteration)}.png" % conf.img_name,
        )
    plt.imshow(k_2, interpolation='none', cmap='gray')
    plt.savefig(kernel_save_path,bbox_inches='tight')
    plt.close()

    if conf.X4:
        k_4 = analytic_kernel(k_2)
        sio.savemat(
            os.path.join(conf.output_dir_path, "%s_kernel_x4.mat" % conf.img_name),
            {"Kernel": k_4},
        )


def save_conf(conf):
    """save the config file to the results dir.

    Parameters
    ----------
    conf : Argparser.Namespace
    """
    with open((conf.output_dir_path + "/conf.txt"), mode="w") as f:
        pprint.pprint(conf.__dict__, f)


#        f.write(str(conf))


def run_zssr(k_2, conf):
    """Performs ZSSR with estimated kernel for wanted scale factor"""
    if conf.do_ZSSR:
        start_time = time.time()
        print("~" * 30 + "\nRunning ZSSR X%d..." % (4 if conf.X4 else 2))
        if conf.X4:
            sr = ZSSR(
                conf.input_image_path,
                scale_factor=[[2, 2], [4, 4]],
                kernels=[k_2, analytic_kernel(k_2)],
                is_real_img=conf.real_image,
                noise_scale=conf.noise_scale,
            ).run()
        else:
            # sr = ZSSR(
            #     conf.input_image_path,
            #     scale_factor=2,
            #     kernels=[k_2],
            #     is_real_img=conf.real_image,
            #     noise_scale=conf.noise_scale,
            # ).run()
            conf_zssr = X2_REAL_CONF
            conf.noise_std = 0.0125*conf.noise_scale
            sr = ZSSR(
                conf.input_image_path, conf=conf_zssr
                , kernels=[k_2]
            ).run()
        max_val = 255 if sr.dtype == "uint8" else 1.0
        save_path = os.path.join(conf.output_dir_path, "ZSSR_%s.png" % conf.img_name)
        plt.imsave( save_path
            ,
            sr,
            vmin=0,
            vmax=max_val,
            dpi=1,
        )
       
        runtime = int(time.time() - start_time)
        print("Completed! runtime=%d:%d\n" % (runtime // 60, runtime % 60) + "~" * 30)
        # metrics_calc(conf.input_image_path, sr, conf.output_dir_path)
        return save_path


def metrics_calc(inputpath, output_hr, output_dir):
    input_lr = Image.open(inputpath).convert('RGB')
    gt_parent_path = os.path.dirname(os.path.split(inputpath)[0]) + "/output_hr/"
    for root, dirs, files in os.walk(gt_parent_path):
        print(root, "-->", dirs, "-->",files)
        gt_all_images = [ root+file for file in files]
    
    for p in gt_all_images:
        s = SequenceMatcher(None, os.path.split(inputpath)[1], os.path.split(p)[1])
        # if os.path.split(inputpath)[1] == os.path.split(p)[1]:
        #     gt_path = p
        if s.ratio()>0.7:
            gt_path = p

    # gt_path = os.path.join(os.path.dirname(inputpath))
    gt_img = Image.open(gt_path).convert('RGB')
    output_name = "/" + os.path.split(gt_path)[1]
    if output_hr.dtype != 'uint8':
        print("======== img2 is not unit8 =====")
        output_hr = np.clip(output_hr * 255, 0, 255).astype(np.uint8)
        # output_hr = cv2.cvtColor(output_hr, cv2.COLOR_RGB2BGR)
    inference_plot_and_save(gt_img, input_lr, output_hr, output_dir, output_name)

def save_plot(conf):
    save_dir = conf.output_dir_path
    plt.savefig(
        os.path.join(save_dir, "training_plot.pdf"),
        bbox_inches="tight",
        dpi=400,
    )

# def inference_plot_and_save(gt_img, input_lr, output_hr, output_dir =None, output_name=None):

#     print("======= saving GT and output HR of model ========")
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,15))
#     ax = axes.ravel()
 
#     if not isinstance(input_lr, np.ndarray):
#         input_lr = convert_from_image_to_cv2(input_lr)

#     bicubic_hr = cv2.resize(input_lr, None, fx= 2, fy= 2, interpolation= cv2.INTER_CUBIC)
#     if isinstance(bicubic_hr, np.ndarray):
#         bicubic_hr = convert_from_cv2_to_image(bicubic_hr)   
     
    
#     psnr_bicubic, ssim_bicubic = calc_metrics(gt_img, bicubic_hr) 
#     psnr_output, ssim_output = calc_metrics(gt_img, output_hr)
#     psnr_gt, ssim_gt = calc_metrics(gt_img, gt_img)
        
#     ax[0].imshow(gt_img)
#     ax[0].set_title("GT")
#     ax[0].set_xlabel("PSNR = " +str(psnr_gt) + "  SSIM = " +str(ssim_gt))
#     ax[1].imshow(bicubic_hr)
#     ax[1].set_xlabel("PSNR = " +str(psnr_bicubic)+ "  SSIM = " +str(ssim_bicubic))
#     ax[1].set_title("bicubic HR") 
#     ax[2].imshow(output_hr)
#     ax[2].set_title("HR output")
#     ax[2].set_xlabel("PSNR = " +str(psnr_output)+ "  SSIM = " +str(ssim_output))
    
#     # if save:
#     #     plt.imsave((output_dir + img[1] + ".png"), img[0])

#     if output_dir and output_name:
#         plt.savefig((output_dir +output_name + datetime.now().strftime('_%b_%d_%H_%M_%S_%f')  + ".png"),bbox_inches='tight')
#     # plt.savefig("xx.svg")
#     # plt.show()

# def convert_from_cv2_to_image(img: np.ndarray) -> Image:
#     # return Image.fromarray(img)
#     return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# def convert_from_image_to_cv2(img: Image) -> np.ndarray:
#     # return np.asarray(img)
#    
#     if img.mode == "YCbCr":
#         img = img.convert('RGB')
#     return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
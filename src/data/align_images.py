# Adapted from: https://github.com/khufkens/align_images/blob/42937c5144d3d8ca1759cf5c136c2c0dbbd91d45/align_images.py

# Import necessary libraries.
import os, argparse
from dataclasses import dataclass
import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

from src.visualization import plot_utils
from src.my_logger import Logger


def rotationAlign(im1, im2):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    height, width = im1_gray.shape[0:2]

    values = np.ones(360)

    for i in range(0, 360):
        rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), i, 1)
        rot = cv2.warpAffine(im2_gray, rotationMatrix, (width, height))
        values[i] = np.mean(im1_gray - rot)

    rotationMatrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), np.argmin(values), 1
    )
    rotated = cv2.warpAffine(im2, rotationMatrix, (width, height))

    return rotated, rotationMatrix


# Enhanced Correlation Coefficient (ECC) Maximization
def eccAlign(im1, im2, args):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        args.number_of_iterations,
        args.termination_eps,
    )

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(
        im1_gray, im2_gray, warp_matrix, warp_mode, criteria
    )

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(
            im2,
            warp_matrix,
            (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(
            im2,
            warp_matrix,
            (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )

    return im2_aligned, warp_matrix


# (ORB) feature based alignment
def featureAlign(im1, im2, args):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(args.max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    # matches.sort(key=lambda x: x.distance, reverse=False)
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * args.feature_retention)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RHO, maxIters=3000)  # .RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(
        im1, h, (width, height)
    )  # ,borderMode =cv2.BORDER_CONSTANT,borderValue=(999,999,999))
    max_rect = crop_max_rectangle(im1Reg)

    return im1Reg, h, max_rect


def filter_reshape(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to separate dark pixels (0, 0, 0) from others
    _, binary_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

    grid = binary_image.astype("bool")

    cv_grid = grid.astype("uint8") * 255
    contours, _ = cv2.findContours(cv_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store information about the largest rectangle
    max_area = 0
    max_rect = None
    con = None
    # Iterate through each contour
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 vertices (indicating it's a rectangle)
        if len(approx) == 4:
            # Calculate the area of the rectangle
            area = cv2.contourArea(contour)

            # Update the maximum area and corresponding rectangle
            if area > max_area:
                max_area = area
                max_rect = cv2.boundingRect(approx)
                con = contour

    contour = con
    a = max_rect
    x, y, w, h = a[0], a[1], a[2], a[3]
    cropped_img = image[y : y + h, x : x + w, :]
    return max_rect


def crop_max_rectangle(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask of black regions
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours of the non-black regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.boundingRect(max_contour)

    return rect


def crop_max_rectangle2(image):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary mask of black regions
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find the coordinates of all non-black (non-zero) pixels
    non_zero_pixels = np.argwhere(thresh > 0)

    if non_zero_pixels.size == 0:
        print("The image is entirely black.")
        return None

    # Get the bounding box of non-black pixels
    top_left = non_zero_pixels.min(axis=0)
    bottom_right = non_zero_pixels.max(axis=0)

    # Crop the image using the bounding box coordinates
    x_min, y_min = top_left
    x_max, y_max = bottom_right
    cropped_image = image[x_min : x_max + 1, y_min : y_max + 1]

    return y_min, x_min, y_max, x_max


# FFT phase correlation
def translation(im0, im1):

    # Convert images to grayscale
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return [t0, t1]


@dataclass
class DefaultAlignArgs:
    # maximum number of features to consider
    max_features: int = 10000  # 5000
    # fraction of features to retain
    feature_retention: float = 0.1

    # Specify the ECC number of iterations.
    # number of ecc iterations
    number_of_iterations: int = 5000

    # Specify the ECC threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps: float = 1e-8


def align_output_target(output, target, mode):

    args = DefaultAlignArgs()

    # Read the images to be aligned
    # image to reference
    im1 = output
    # image to match
    im2 = target

    # Switch between alignment modes
    # registation mode: translation, ecc or feature

    if mode == "feature":
        # align and write to disk
        aligned, warp_matrix, max_rect = featureAlign(im1, im2, args)
        output = aligned
        print(warp_matrix)
    elif mode == "ecc":
        aligned, warp_matrix = eccAlign(im1, im2, args)
        cv2.imwrite("reg_image.jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(warp_matrix)
    elif mode == "rotation":
        rotated, rotationMatrix = rotationAlign(im1, im2)
        cv2.imwrite("reg_image.jpg", rotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(rotationMatrix)
    else:
        warp_matrix = translation(im1, im2)
        print(warp_matrix)

    return output, max_rect


if __name__ == "__main__":

    print("used as a function and default args are set in the dataclass")

import os
import numpy as np
import json
import torch
import torch.nn as nn
import pdb
import random
import torch.nn.functional as F
from math import exp


# Seed everything for random steps
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_or_clear_directory(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        # If not, create the directory
        os.makedirs(directory)
        print(f"Directory {directory} has been created.")
    else:
        # If it exists, clear the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        print(f"Directory {directory} has been cleared.")


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]["lr"],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    # pdb.set_trace()
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def one_hot(scores):
    scores_onehot = []
    for i in range(scores.shape[0]):
        if scores[i] > 0.5:
            scores_onehot.append([0, 1])
        else:
            scores_onehot.append([1, 0])
    scores_onehot = np.stack(scores_onehot)
    scores_onehot = torch.from_numpy(scores_onehot).to(scores.device)
    return scores_onehot


# Function to compute Pearson Correlation Coefficient (PCC)
def pearson_corrcoef(x, y):
    # Ensure the input tensors are flattened
    x_flat = x.view(-1)
    y_flat = y.view(-1)

    # Compute mean of x and y
    mean_x = torch.mean(x_flat)
    mean_y = torch.mean(y_flat)

    # Compute the numerator (covariance between x and y)
    cov_xy = torch.sum((x_flat - mean_x) * (y_flat - mean_y))

    # Compute the denominator (product of standard deviations of x and y)
    std_x = torch.sqrt(torch.sum((x_flat - mean_x) ** 2))
    std_y = torch.sqrt(torch.sum((y_flat - mean_y) ** 2))

    # Avoid division by zero
    eps = 1e-8
    pcc = cov_xy / (std_x * std_y + eps)

    return pcc


# Helper function to create a Gaussian kernel
def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


# Function to create a 2D Gaussian kernel
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# SSIM calculation
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
    if window is None:
        window = create_window(window_size, img1.size(1)).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1))
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1))
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1))
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

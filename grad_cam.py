import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from utils.loader import get_validation_data
import utils
from model import UNet

# Argument Parser
parser = argparse.ArgumentParser(description='RGB denoising evaluation with Grad-CAM')
parser.add_argument('--input_dir', default='./ISTD_Dataset/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for denoised results')
parser.add_argument('--result_dir_grad_cam', default='./results_grad_cam/', type=str, help='Directory for Grad-CAM results')
parser.add_argument('--weights', default='./log/ShadowFormer_istd/models/model_epoch_400.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for dataloader')
parser.add_argument('--arch', default='ShadowFormer', type=str, help='Model architecture')
parser.add_argument('--train_ps', type=int, default=256, help='Patch size of training sample')
parser.add_argument('--tile', type=int, default=256, help='Tile size (e.g., 720). None means full image processing')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlap of tiles')

args = parser.parse_args()

# Set CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# Create results directories
utils.mkdir(args.result_dir)
utils.mkdir(args.result_dir_grad_cam)

# Load test dataset
test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

# Load model
model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)
utils.load_checkpoint(model_restoration, args.weights)

print("===> Testing using weights:", args.weights)

model_restoration.cuda()
model_restoration.eval()

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_image, mask):
        self.model.train()  # Enable gradients for Grad-CAM
        input_image.requires_grad = True  # Allow gradient tracking

        with torch.enable_grad():
            output = self.model(input_image, mask)  # Forward pass
            target = output.mean()  # Compute target loss for Grad-CAM
            target.backward()  # Compute gradients

        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Global average pooling
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # Weighted sum of activations
        cam = F.relu(cam).squeeze().cpu().detach().numpy()  # Apply ReLU
        # print("self.gradients",self.gradients.shape)
        # print("weights",weights.shape)
        # print("self.activations",self.activations.shape)

        # Normalize and resize the heatmap
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))  # Resize to match image size

        self.model.eval()  # Switch back to evaluation mode
        return cam

def overlay_heatmap(image, heatmap, alpha=0.9):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

# Set the target layer for Grad-CAM
# target_layer = model_restoration.module.output_proj.proj[0]
target_layer = model_restoration.module.cbam.sa
grad_cam = GradCAM(model_restoration, target_layer)

# Testing loop
with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_noisy = data_test[1].cuda()  # Noisy Image
        mask = data_test[2].cuda()  # Mask
        filenames = data_test[3]

        if args.tile is not None:
            # Tile-based processing
            b, c, h, w = rgb_noisy.shape
            tile = min(args.tile, h, w)
            stride = tile - args.tile_overlap

            E = torch.zeros(b, c, h, w).type_as(rgb_noisy)
            W = torch.zeros_like(E)
            E_cam = torch.zeros(h, w).type_as(rgb_noisy)
            W_cam = torch.zeros_like(E_cam)

            for h_idx in range(0, h - tile + 1, stride):
                for w_idx in range(0, w - tile + 1, stride):
                    in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]

                    out_patch = model_restoration(in_patch, mask_patch)
                    cam_patch = grad_cam.generate(in_patch, mask_patch)

                    E[..., h_idx:h_idx + tile, w_idx:w_idx + tile] += out_patch
                    W[..., h_idx:h_idx + tile, w_idx:w_idx + tile] += 1

                    E_cam[h_idx:h_idx + tile, w_idx:w_idx + tile] += torch.tensor(cam_patch).to(rgb_noisy.device)
                    W_cam[h_idx:h_idx + tile, w_idx:w_idx + tile] += 1

            rgb_restored = E / W
            cam = (E_cam / W_cam).cpu().numpy()
        else:
            # Full image processing
            rgb_restored = model_restoration(rgb_noisy, mask)
            cam = grad_cam.generate(rgb_noisy, mask)

        # Convert images to uint8 format
        rgb_restored_np = (torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0)) * 255).astype(np.uint8)
        grad_cam_output = overlay_heatmap(rgb_restored_np, cam)
        activation_map = (cam * 255).astype(np.uint8)

        # Save images
        cv2.imwrite(os.path.join(args.result_dir, f"denoised_{filenames[0]}"), cv2.cvtColor(rgb_restored_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.result_dir_grad_cam, f"gradcam_{filenames[0]}"), cv2.cvtColor(grad_cam_output, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.result_dir_grad_cam, f"activation_{filenames[0]}"), activation_map)

print("Processing completed. Results saved in:", args.result_dir, "and", args.result_dir_grad_cam)

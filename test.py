import numpy as np
import os, sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss

import scipy.io as sio
from utils.loader import get_validation_data
import utils
from model import UNet


def main():
    parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
    parser.add_argument('--input_dir', default='./ISTD_Dataset/test/',
        type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='./results/',
        type=str, help='Directory for results')
    parser.add_argument('--weights', default='./log/ShadowFormer_istd/models/model_epoch_500.pth',
        type=str, help='Path to weights')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='ShadowFormer', type=str, help='arch')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for dataloader')
    parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
    parser.add_argument('--cal_metrics', action='store_true', help='Measure denoised images with GT')
    parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
    parser.add_argument('--win_size', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
    # args for vit
    parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
    parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
    parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
    parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
    parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
    parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
    parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
    parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
    parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
    parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    utils.mkdir(args.result_dir)

    test_dataset = get_validation_data(args.input_dir)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8,
                             drop_last=False)

    model_restoration = utils.get_arch(args)
    model_restoration = torch.nn.DataParallel(model_restoration)
    utils.load_checkpoint(model_restoration, args.weights)
    print("===> Testing using weights:", args.weights)

    model_restoration.cuda()
    model_restoration.eval()

    img_multiple_of = 8 * args.win_size

    with torch.no_grad():
        psnr_val_rgb = []
        ssim_val_rgb = []
        psnr_val_s = []
        ssim_val_s = []
        psnr_val_ns = []
        ssim_val_ns = []
        rmse_val_rgb = []
        rmse_val_s = []
        rmse_val_ns = []

        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_gt    = data_test[0].cpu().numpy().squeeze().transpose((1, 2, 0))
            rgb_noisy = data_test[1].cuda()
            mask_pad  = data_test[2].cuda()
            filenames = data_test[3]

            # pad input to multiple of win_size*8
            height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
            H = ((height + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
            W = ((width  + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
            padh = H - height
            padw = W - width
            rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
            mask_pad  = F.pad(mask_pad,  (0, padw, 0, padh), 'reflect')

            # restoration
            if args.tile is None:
                rgb_restored = model_restoration(rgb_noisy, mask_pad)
            else:
                b, c, h, w = rgb_noisy.shape
                tile = min(args.tile, h, w)
                assert tile % 8 == 0, "Tile size must be multiple of 8"
                stride = tile - args.tile_overlap
                h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                E = torch.zeros_like(rgb_noisy)
                Wm = torch.zeros_like(rgb_noisy)
                for hi in h_idx_list:
                    for wi in w_idx_list:
                        in_patch   = rgb_noisy[..., hi:hi+tile, wi:wi+tile]
                        mask_patch = mask_pad[..., hi:hi+tile, wi:wi+tile]
                        out_patch  = model_restoration(in_patch, mask_patch)
                        E[..., hi:hi+tile, wi:wi+tile].add_(out_patch)
                        Wm[..., hi:hi+tile, wi:wi+tile].add_(1)
                rgb_restored = E.div_(Wm)

            # clamp and unpad
            rgb_restored = torch.clamp(rgb_restored, 0, 1)
            rgb_restored = rgb_restored.cpu().numpy().squeeze().transpose((1,2,0))
            rgb_restored = rgb_restored[:height, :width, :]

            # unpad mask to original size
            bm_padded = torch.where(mask_pad == 0,
                                    torch.zeros_like(mask_pad),
                                    torch.ones_like(mask_pad))
            bm = bm_padded.cpu().numpy().squeeze()  # (H_pad, W_pad)
            bm = bm[:height, :width]                # (height, width)
            bm = np.expand_dims(bm, axis=2)         # (..., 1)

            if args.cal_metrics:
                # convert to gray
                gray_restored = cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2GRAY)
                gray_gt       = cv2.cvtColor(rgb_gt,       cv2.COLOR_RGB2GRAY)

                # overall region
                ssim_val_rgb.append(
                    ssim_loss(gray_restored,
                              gray_gt,
                              data_range=1.0,
                              channel_axis=None)
                )
                psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))

                # shadow (s) vs non-shadow (ns)
                mask_s  = bm.squeeze()
                mask_ns = 1 - mask_s

                ssim_val_s.append(
                    ssim_loss(gray_restored * mask_s,
                              gray_gt       * mask_s,
                              data_range=1.0,
                              channel_axis=None)
                )
                psnr_val_s.append(
                    psnr_loss(rgb_restored * bm,
                              rgb_gt       * bm)
                )

                ssim_val_ns.append(
                    ssim_loss(gray_restored * mask_ns,
                              gray_gt       * mask_ns,
                              data_range=1.0,
                              channel_axis=None)
                )
                psnr_val_ns.append(
                    psnr_loss(rgb_restored * (1 - bm),
                              rgb_gt       * (1 - bm))
                )

                # LAB-space RMSE
                lab_rest = cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2LAB)
                lab_gt   = cv2.cvtColor(rgb_gt,         cv2.COLOR_RGB2LAB)
                rmse_val_rgb.append(
                    np.abs(lab_rest - lab_gt).mean() * 3
                )
                # region RMSE
                rmse_val_s.append(
                    np.abs(lab_rest * bm - lab_gt * bm).sum() / bm.sum()
                )
                rmse_val_ns.append(
                    np.abs(lab_rest * (1-bm) - lab_gt * (1-bm)).sum() / (1-bm).sum()
                )

            if args.save_images:
                utils.save_img(rgb_restored * 255.0,
                               os.path.join(args.result_dir, filenames[0]))

        #TODO Have to use the evaluate.m instead of test.py to find the psnr, ssm and rmse
        # final aggregation
        if args.cal_metrics:
            n = len(test_dataset)
            print(f"PSNR: {sum(psnr_val_rgb)/n:.4f}, SSIM: {sum(ssim_val_rgb)/n:.4f}, RMSE: {sum(rmse_val_rgb)/n:.4f}")
            print(f"SPSNR: {sum(psnr_val_s)/n:.4f}, SSSIM: {sum(ssim_val_s)/n:.4f}, SRMSE: {sum(rmse_val_s)/n:.4f}")
            print(f"NSPSNR: {sum(psnr_val_ns)/n:.4f}, NSSSIM: {sum(ssim_val_ns)/n:.4f}, NSRMSE: {sum(rmse_val_ns)/n:.4f}")

if __name__ == '__main__':
    main()

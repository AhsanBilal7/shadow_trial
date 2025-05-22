import os
import sys
from tqdm import tqdm


# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

import utils
######### Set GPUs ###########
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True
# from piqa import SSIM
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
from utils import save_img
from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm  # For Vision Transformer (ViT) models

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes):
        super(VisionTransformer, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)  # Pre-trained ViT backbone
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)  # Adjust output layer
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
    
    def forward(self, x):
        return self.vit(x)

class Generator(nn.Module):
    def __init__(self, embed_dim=3, num_heads=12, num_layers=12):
        super(Generator, self).__init__()
        
        # Input Layers
        self.input_conv = nn.Conv2d(4, embed_dim, kernel_size=3, stride=1, padding=1)
        
        # Vision Transformer Block
        self.vit = VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, num_classes=3)
        
        # Decoder to produce shadow-free image
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 3 * 224 * 224)  # Assuming 224x224 image output
        self.unflatten = nn.Unflatten(1, (3, 224, 224))
    
    def forward(self, shadow_image, shadow_mask):
        x = torch.cat([shadow_image, shadow_mask], dim=1)  # Concatenate shadow image and mask
        # print(x.shape)
        x = F.relu(self.input_conv(x))  # Apply Conv layer to extract features
        # print(x.shape)
        x = self.vit(x)  # Pass through Vision Transformer
        # print(x.shape)
        
        # Decoder to generate the final output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.unflatten(x)  # Reshape output to image size (3, 224, 224)
        return x


class Discriminator(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=12):
        super(Discriminator, self).__init__()
        
        # Vision Transformer Block
        self.vit = VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, num_classes=1)
        self.input_conv = nn.Conv2d(7, embed_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, shadow_image, shadow_free_image, shadow_mask):
        # Concatenate shadowed image and shadow mask to input for the discriminator
        input_image = torch.cat([shadow_image, shadow_mask], dim=1)
        fake_image = shadow_free_image  # The generated image to be classified as real or fake
        
        real_input = torch.cat([input_image, fake_image], dim=1)
        # Pass through Vision Transformer for decision-making
        real_input = F.relu(self.input_conv(real_input))  # Apply Conv layer to extract features

        real_or_fake = self.vit(real_input)
        return real_or_fake


class HybridCGAN(nn.Module):
    def __init__(self, embed_dim=3, num_heads=12, num_layers=12):
        super(HybridCGAN, self).__init__()
        self.generator = Generator(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.discriminator = Discriminator(embed_dim=3, num_heads=num_heads, num_layers=num_layers)

    def forward(self, shadow_image, shadow_mask):
        generated_image = self.generator(shadow_image, shadow_mask)
        validity = self.discriminator(shadow_image, generated_image, shadow_mask)
        return generated_image, validity

######### Logs dir ###########
log_dir = os.path.join(dir_name, 'log', opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")



######### DataParallel ###########
model_restoration = torch.nn.DataParallel (model_restoration)
model_restoration.to('cuda')
# model_restoration.cuda()
total_parameters = sum(p.numel() for p in model_restoration.parameters())
print("==================== Total Parameters =============================")
print(total_parameters)
print("=================================================")
######### Resume ###########
if opt.resume:
    print("=========Resuming ...========")
    path_chk_rest = opt.pretrain_weights
    print(path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
#     lr = utils.load_optim(optimizer, path_chk_rest)
#
#     for p in optimizer.param_groups: p['lr'] = lr
#     warmup = False
#     new_lr = lr
#     print('------------------------------------------------------------------------------')
#     print("==> Resuming Training with learning rate:",new_lr)
#     print('------------------------------------------------------------------------------')
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

# ######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()



######### Loss ###########
# criterion = CharbonnierLoss().cuda()
class CombinedLoss(nn.Module):
    """Combined Loss"""

    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        # self.bce_loss = nn.BCELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.charbonnier_loss = CharbonnierLoss()
        self.alpha = alpha

    def forward(self, x, y):
        # x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # print("=====================")
        # print(x.shape, y.shape)
        # print(x, y)
        # print("=====================")
        bce = self.bce_loss(x, y)
        charbonnier = self.charbonnier_loss(x, y)
        combined = (1 - self.alpha) * bce + self.alpha * charbonnier
        # combined =  bce + charbonnier
        return combined
# criterion = CombinedLoss().cuda()
criterion = CharbonnierLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
# train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.train_workers, pin_memory=True, drop_last=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir, img_options_train)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
ii=0
index = 0
# Loss Functions
def generator_loss(disc_output):
    return F.binary_cross_entropy_with_logits(disc_output, torch.ones_like(disc_output))

def discriminator_loss(disc_output, real_labels, fake_labels):
    real_loss = F.binary_cross_entropy_with_logits(disc_output, real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(disc_output, fake_labels)
    return (real_loss + fake_loss) / 2


# Example Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = HybridCGAN().to(device)

# Optimizers
optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training (simplified example)
num_epochs = 10
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    for i, data in enumerate(tqdm(train_loader)):  # Replace with your dataset
        shadow_free_images, shadow_images, shadow_masks = data[0],data[1],data[2]
        # print(shadow_images.shape, shadow_masks.shape, shadow_free_images.shape)
        shadow_images, shadow_masks, shadow_free_images = shadow_images.to(device), shadow_masks.to(device), shadow_free_images.to(device)

        # Train Discriminator
        optimizer_d.zero_grad()

        # Real images
        real_output = model.discriminator(shadow_images, shadow_free_images, shadow_masks)
        real_labels = torch.ones(real_output.size(), device=device)

        # Fake images
        fake_images = model.generator(shadow_images, shadow_masks)
        # print(fake_images.shape)
        fake_output = model.discriminator(shadow_images, fake_images, shadow_masks)
        fake_labels = torch.zeros(fake_output.size(), device=device)

        # Loss for discriminator
        real_loss = F.binary_cross_entropy_with_logits(real_output, real_labels)
        fake_loss = F.binary_cross_entropy_with_logits(fake_output, fake_labels)
        loss_d = real_loss + fake_loss

        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        fake_images = model.generator(shadow_images, shadow_masks)
        disc_output = model.discriminator(shadow_images, fake_images, shadow_masks)

        # Generator tries to fool the discriminator
        real_labels_for_generator = torch.ones(disc_output.size(), device=device)
        loss_g = F.binary_cross_entropy_with_logits(disc_output, real_labels_for_generator)

        loss_g.backward()
        optimizer_g.step()
    with torch.no_grad():
        model.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate(tqdm(val_loader)):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            mask = data_val[2].cuda()
            filenames = data_val[3]
            # print(input_.shape, mask.shape)
            with torch.cuda.amp.autocast():
                restored = model.generator(input_, mask)
            # restored = torch.clamp(restored,0,1)
            psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

        psnr_val_rgb = sum(psnr_val_rgb)/len(val_loader)

    print("[Ep %d it %d\t PSNR : %.4f] " % (epoch, i, psnr_val_rgb))
        
    print(f"Epoch [{epoch}/{num_epochs}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

    torch.save({'epoch': epoch, 
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())
# for epoch in range(start_epoch, opt.nepoch + 1):
    
#     pbar = tqdm(total=len(train_loader))
#     epoch_start_time = time.time()
#     epoch_loss = 0
#     train_id = 1
#     epoch_ssim_loss = 0
#     for i, data in enumerate(train_loader, 0): 
#         try:
#             pbar.update(1)
#             # zero_grad
#             index += 1
#             optimizer.zero_grad()
#             target = data[0].cuda()
#             input_ = data[1].cuda()
#             mask = data[2].cuda()
#             if epoch > 5:
#                 target, input_, mask = utils.MixUp_AUG().aug(target, input_, mask)
#             with torch.amp.autocast('cuda'):
#                 restored = model_restoration(input_, mask)
#                 restored = torch.clamp(restored,0,1)
#                 loss = criterion(restored, target)
#             loss_scaler(
#                     loss, optimizer,parameters=model_restoration.parameters())
#             epoch_loss +=loss.item()
#             #### Evaluation ####
#             if (index+1)%eval_now==0 and i>0:
#                 eval_shadow_rmse = 0
#                 eval_nonshadow_rmse = 0
#                 eval_rmse = 0
#                 with torch.no_grad():
#                     model_restoration.eval()
#                     psnr_val_rgb = []
#                     for ii, data_val in enumerate((val_loader), 0):
#                         target = data_val[0].cuda()
#                         input_ = data_val[1].cuda()
#                         mask = data_val[2].cuda()
#                         filenames = data_val[3]
#                         with torch.cuda.amp.autocast():
#                             restored = model_restoration(input_, mask)
#                         restored = torch.clamp(restored,0,1)
#                         psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

#                     psnr_val_rgb = sum(psnr_val_rgb)/len(val_loader)
#                     if psnr_val_rgb > best_psnr:
#                         best_psnr = psnr_val_rgb
#                         best_epoch = epoch
#                         best_iter = i
#                         torch.save({'epoch': epoch,
#                                     'state_dict': model_restoration.state_dict(),
#                                     'optimizer' : optimizer.state_dict()
#                                     }, os.path.join(model_dir,"model_best.pth"))
#                     print("[Ep %d it %d\t PSNR : %.4f] " % (epoch, i, psnr_val_rgb))
#                     with open(logname,'a') as f:
#                         f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
#                             % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
#                     model_restoration.train()
#                     torch.cuda.empty_cache()
#         except Exception as e:
#             print("===== Error: ",e)
    
    # scheduler.step()
    
#     print("------------------------------------------------------------------")
#     print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss,scheduler.get_lr()[0]))
#     print("------------------------------------------------------------------")
#     with open(logname,'a') as f:
#         f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

#     torch.save({'epoch': epoch, 
#                 'state_dict': model_restoration.state_dict(),
#                 'optimizer' : optimizer.state_dict()
#                 }, os.path.join(model_dir,"model_latest.pth"))   

#     if epoch%opt.checkpoint == 0:
#         torch.save({'epoch': epoch, 
#                     'state_dict': model_restoration.state_dict(),
#                     'optimizer' : optimizer.state_dict()
#                     }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
# print("Now time is : ",datetime.datetime.now().isoformat())








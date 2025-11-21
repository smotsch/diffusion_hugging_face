"""
Diffusion image generation training script using Hugging Face Diffusers + PyTorch

1) Dataset: CIFAR10
2) Model: UNet denoising model using diffusers' UNet2DModel
3) Loss: Mean-Square Error to recover the 'noise'

"""

import os,sys
from datetime import datetime

import torch
from torchvision import transforms, datasets
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline

sys.path.append(os.getcwd())
from module_train_inference import train_DDPM, generate_samples

# configuration file user
cfg = {
    'output_dir': "results",
    'epochs': 10,
    'batch_size': 32,
    'lr': 5e-5,
    'image_size': 32,
    'nbr_timesteps': 1000,
    'data_normalized': False,
}
# create folder for results
now = datetime.now()
folder_name = now.strftime("diffusion_%Y-%m-%d_%Hh%Mm%S")
cfg['full_path'] = os.path.join(cfg['output_dir'], folder_name)
os.makedirs(cfg['full_path'], exist_ok=True)

#--------------------#
#------ A) Data -----# 
#--------------------#
if (cfg['data_normalized']):
    mean_cifar10 = (0.4914, 0.4822, 0.4465)
    std_cifar10 = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_cifar10, std_cifar10),
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
dataset_cifar = datasets.CIFAR10(root="dataset", download=True, train=True, transform=transform)

#---------------------#
#------ B) Model -----# 
#---------------------#
device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
myModel = UNet2DModel(
    sample_size=cfg['image_size'],
    in_channels=3,
    out_channels=3,
    #block_out_channels=(128, 256, 512, 512),
    block_out_channels=(32,32,64,128),
    layers_per_block=2,
    #attention_head_dim=64,
    norm_num_groups=32,
    )
# number of parameters:
#  sum(p.numel() for p in myModel.parameters() if p.requires_grad)
# Multiple GPU : not working on the server madmax :(
#  myModel = nn.DataParallel(myModel, device_ids=[0, 1])  # use the GPU 0 and 1
myModel = myModel.to(device)

#------ C) Loss ------# 
#---------------------#
## MSE!
#-- scheduler (t=0..1000)
myScheduler = DDPMScheduler(num_train_timesteps=cfg['nbr_timesteps'])

#--------------------------------------------------#
#----------------    train !   --------------------#
#--------------------------------------------------#
train_DDPM(myModel, dataset_cifar, myScheduler, device, cfg)


#-------------------   Testing   ------------------#
#--------------------------------------------------#
imgs = generate_samples(9, myModel, myScheduler, device, cfg)

# use another method to plot (no "de-normalization")
pipeline = DDPMPipeline(unet=myModel, scheduler=myScheduler)
pipeline.to(device)
samples = pipeline(batch_size=9, generator=torch.manual_seed(0)).images
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3,3,figsize=(15,15))
for i, img in enumerate(samples):
    axes[i//3,i%3].imshow(img)
    axes[i//3,i%3].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(cfg['full_path'],"samples2.jpg"))



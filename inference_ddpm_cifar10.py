"""
Inference after training: first run 'main_ddpm_cifar10.py'

"""

## folder results
folder_result = "results/diffusion_2025-11-07_13h46m05"


import os, json

import torch
from torchvision import transforms, utils as tv_utils
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline

# load hyper parameters
with open(os.path.join(folder_result, 'cfg.json'),'r') as string:
    cfg = json.load(string)

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
# load the model
myState_dict = torch.load(os.path.join(folder_result, "final_model.pt"), weights_only=True)
myModel.load_state_dict(myState_dict)
device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
myModel.to(device)
# load the scheduler
scheduler = DDPMScheduler(num_train_timesteps=cfg['nbr_timesteps'])

#--------------------#
#------ A) Data -----# 
#--------------------#
mean_cifar10 = (0.4914, 0.4822, 0.4465)
#std_cifar10= (0.2470, 0.2435, 0.2616)
std_cifar10= (.5,.5,.5)

#--------------------------------------------------#
#-------------------   Testing   ------------------#
#--------------------------------------------------#
@torch.no_grad()
def generate_samples(num_samples, model, scheduler, device, cfg):
    torch.manual_seed(42)
    model.eval()
    # generate the new samples starting from a normal
    Z = torch.randn(num_samples, model.config.in_channels, cfg['image_size'], cfg['image_size'], device=device)
    with torch.no_grad():
        for t in tqdm(range(cfg['nbr_timesteps']), desc="Sampling"):
            t_tensor = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
            noise_pred = model(Z, t_tensor).sample
            Z = scheduler.step(noise_pred, t, Z).prev_sample
    # "denormalize" the images:
    mean_cifar = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    #std_cifar = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1).to(device)
    std_cifar = torch.tensor([.5,.5,.5]).view(3, 1, 1).to(device)
    imgs = mean_cifar[None, ...] + Z*std_cifar[None, ...]
    imgs = imgs.clamp(0, 1)
    # save the images
    grid = tv_utils.make_grid(imgs, nrow=int(math.sqrt(num_samples)))
    tv_utils.save_image(grid, os.path.join(cfg['full_path'],"samples.jpg"))
    return imgs


imgs = generate_samples(9, myModel, scheduler, device, cfg)

# use another method to plot
pipeline = DDPMPipeline(unet=myModel, scheduler=scheduler)
pipeline.to(device)
samples = pipeline(batch_size=9, generator=torch.manual_seed(0)).images
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3,3,figsize=(15,15))
for i, img in enumerate(samples):
    axes[i//3,i%3].imshow(img)
    axes[i//3,i%3].axis('off')
plt.tight_layout()
#plt.savefig(os.path.join(cfg['full_path'],"samples3.jpg"))
plt.savefig("samples3.jpg")


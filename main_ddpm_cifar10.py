"""
Diffusion image generation training script using Hugging Face Diffusers + PyTorch

1) Dataset: CIFAR10
2) Model: UNet denoising model using diffusers' UNet2DModel
3) Loss: Mean-Square Error to recover the 'noise'


"""

import os, json
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
import math

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils as tv_utils

from diffusers import UNet2DModel, DDPMScheduler

# configuration file user
cfg = {
    'output_dir': "results",
    'epochs': 10,
    'batch_size': 32,
    'lr': 1e-4,
    'image_size': 32,
    'nbr_timesteps': 1000
}
# create folder for results
now = datetime.now()
folder_name = now.strftime("diffusion_%Y-%m-%d_%Hh%Mm%S")
cfg['full_path'] = os.path.join(cfg['output_dir'], folder_name)
os.makedirs(cfg['full_path'], exist_ok=True)

#--------------------#
#------ A) Data -----# 
#--------------------#
transform = transforms.Compose([
    transforms.Resize(cfg['image_size']),
    transforms.CenterCrop(cfg['image_size']),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])
dataset_cifar = datasets.CIFAR10(root="dataset", download=True, train=True, transform=transform)
dataloader_cifar = DataLoader(dataset_cifar, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

#---------------------#
#------ B) Model -----# 
#---------------------#
device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
myModel = UNet2DModel(
    sample_size=cfg['image_size'],
    in_channels=3,
    out_channels=3,
    block_out_channels=(64, 128, 256, 512),
    layers_per_block=2,
    attention_head_dim=64,
).to(device)
# number of parameters:
#  sum(p.numel() for p in myModel.parameters() if p.requires_grad)

#---------------------#
#------ C) Loss ------# 
#---------------------#
scheduler = DDPMScheduler(num_train_timesteps=cfg['nbr_timesteps'])
optimizer = optim.AdamW(myModel.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=1e-2)


#--------------------------------------------------#
#----------------    train !   --------------------#
#--------------------------------------------------#
nbr_steps = len(dataloader_cifar) * cfg['epochs']
progress_bar = tqdm(range(nbr_steps), desc="Training steps")
epoch_losses = []

for epoch in range(cfg['epochs']):
    total_loss = 0.0
    for imgs,_ in dataloader_cifar:
        # create the noisy images
        imgs = imgs.to(device)
        batch_size_cur = imgs.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size_cur,), device=device, dtype=torch.long)
        noise = torch.randn_like(imgs)
        noisy_imgs = scheduler.add_noise(imgs, noise, timesteps)
        # model predict noise
        optimizer.zero_grad()
        model_pred = myModel(noisy_imgs, timesteps).sample
        # loss
        loss = F.mse_loss(model_pred, noise)
        loss.backward()
        optimizer.step()
        # done 
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": float(loss.detach().cpu())})
        total_loss += loss.item()

    # one epoch done
    avg_loss = total_loss / len(dataloader_cifar)
    epoch_losses.append({"epoch": epoch + 1, "loss": avg_loss})
    print(f"Epoch {epoch+1}: avg loss = {avg_loss:.6f}")

print(f"Training finished. ")
# save model, loss, configuration file (cfg)
torch.save(myModel.state_dict(), os.path.join(cfg['full_path'],"final_model.pt") )
pd.DataFrame(epoch_losses).to_csv(os.path.join(cfg['full_path'],"training_losses.csv"), index=False)
with open(os.path.join(cfg['full_path'],"cfg.json"),'w') as jsonFile:
    json.dump(cfg, jsonFile, indent=4)

# to debug:
#  imgs,_ = next(iter(dataloader_cifar))

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
    # save the images
    imgs = Z.clamp(-1, 1)
    imgs = (imgs + 1) / 2.0
    grid = tv_utils.make_grid(imgs, nrow=int(math.sqrt(num_samples)))
    tv_utils.save_image(grid, os.path.join(cfg['full_path'],"samples.jpg"))

# If needed, load the model
## myModel = UNet2DModel(...)
## myModel.load_state_dict(torch.load("results/.../final_model.pt", map_location=torch.device("cuda")))
## myModel.eval()
## with open("results/.../cfg.json", "r") as f:
##     cfg = json.load(f)
## scheduler = DDPMScheduler(num_train_timesteps=cfg['nbr_timesteps'])

generate_samples(9, myModel, scheduler, device, cfg)


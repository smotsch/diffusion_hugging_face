import os, json
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
import math

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn, optim
from torchvision import utils as tv_utils


def train_DDPM(myModel, dataset_cifar, myScheduler, device, cfg):
    """
    Training of a DDPM model.
    """
    #-- data into mini-batch
    dataloader_cifar = DataLoader(dataset_cifar, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    #-- loss
    optimizer = optim.AdamW(myModel.parameters(), lr=cfg['lr'])

    #----------------    train !   --------------------#
    nbr_steps = len(dataloader_cifar) * cfg['epochs']
    progress_bar = tqdm(range(nbr_steps), desc="Training steps")
    epoch_losses = []
    start_time = datetime.now()

    for epoch in range(cfg['epochs']):
        total_loss = 0.0
        for imgs,_ in dataloader_cifar:
            # create the noisy images
            imgs = imgs.to(device)
            batch_size_cur = imgs.shape[0]
            timesteps = torch.randint(0, myScheduler.config.num_train_timesteps, (batch_size_cur,), device=device, dtype=torch.long)
            noise = torch.randn_like(imgs)
            noisy_imgs = myScheduler.add_noise(imgs, noise, timesteps)
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

    print(f"Training finished.")
    # save model, loss, configuration file (cfg)
    torch.save(myModel.state_dict(), os.path.join(cfg['full_path'],"final_model.pt") )
    pd.DataFrame(epoch_losses).to_csv(os.path.join(cfg['full_path'],"training_losses.csv"), index=False)
    elapsed = datetime.now() - start_time
    cfg['training time'] = str(elapsed).split('.')[0]
    with open(os.path.join(cfg['full_path'],"cfg.json"),'w') as jsonFile:
        json.dump(cfg, jsonFile, indent=4)


@torch.no_grad()
def generate_samples(num_samples, model, myScheduler, device, cfg):
    """
    Generate image once the model has been trained
    """
    torch.manual_seed(42)
    model.eval()
    mean_cifar10 = (0.4914, 0.4822, 0.4465)
    std_cifar10 = (0.2470, 0.2435, 0.2616)
    # generate the new samples starting from a normal
    Z = torch.randn(num_samples, model.config.in_channels, cfg['image_size'], cfg['image_size'], device=device)
    with torch.no_grad():
        for t in tqdm(range(cfg['nbr_timesteps'],0,-1), desc="Sampling"):
            t_tensor = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
            noise_pred = model(Z, t_tensor).sample
            Z = myScheduler.step(noise_pred, t-1, Z).prev_sample
    # "denormalize" the images:
    if (cfg['data_normalized']):
        mean_cifar10_tensor = torch.tensor(mean_cifar10).view(3, 1, 1).to(device)
        std_cifar10_tensor = torch.tensor(std_cifar10).view(3, 1, 1).to(device)
        imgs = mean_cifar10_tensor[None, ...] + Z*std_cifar10_tensor[None, ...]
    else:
        imgs = Z
    # bound the pixel values between 0 and 1
    imgs = imgs.clamp(0, 1)
    # save the images
    grid = tv_utils.make_grid(imgs, nrow=int(math.sqrt(num_samples)))
    tv_utils.save_image(grid, os.path.join(cfg['full_path'],"samples.jpg"))
    return imgs

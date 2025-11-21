"""
Inference after training: first run 'main_ddpm_cifar10.py'

"""


import os, sys, json

import torch
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline

sys.path.append(os.getcwd())
from module_train_inference import generate_samples


## folder results
folder_result = "results/diffusion_2025-11-21_15h10m48_normalized_data"
#folder_result = "results/diffusion_2025-11-21_15h32m36"


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
myScheduler = DDPMScheduler(num_train_timesteps=cfg['nbr_timesteps'])

#------ A) Data -----# 
#--------------------#
#mean_cifar10 = (0.4914, 0.4822, 0.4465)
#std_cifar10= (0.2470, 0.2435, 0.2616)

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
#plt.savefig(os.path.join(cfg['full_path'],"samples3.jpg"))
plt.savefig("samples4.jpg")


# Simple Diffusion model using Hugging Face

This repository shows how to use a Unet model from the Hugging Face reposotoriy to generate new images from cifar10 dataset.

## Installation

Clone/download the repo:
```bash
git clone https://github.com/smotsch/diffusion_hugging_face.git
cd diffusion_hugging_face
```

### Python environment
If need it, create the python environment using `mamba` and `pip`:
```bash
mamba env create -f python_env/environment_python.yaml
mamba activate diffusion-huggingface
mamba install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
When the environment is created, run the script as follow:
```bash
mamba activate diffusion-huggingface # load env if needed
python main_ddpm_cifar10.py # run the script
```
I prefer to use `ipython`, so an alternative method is:
```bash
mamba activate diffusion-huggingface # load env if needed
ipython
run main_ddpm_cifar10.py
```

### Container
An alternative method to run the script is to use a container. This method is particularly helpful when running on a server where mamba or conda might not be installed. We need `apptainer` to create the container from the file `ubuntu_diffusion_hugging.def`:
```bash
apptainer build ubuntu_diffusion_hugging.sif ubuntu_diffusion_hugging.def
```
Then, one can run the script within the container:
```bash
apptainer shell --nv ubuntu_diffusion_hugging.sif
cd ...
ipython
...
```

## Running

The main script will train a Unet model and generates some samples after training in a folder situated in '`results'. The main script also indicates how to load a trained network and generates new samples from it.
 

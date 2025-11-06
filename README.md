# Simple Diffusion model using Hugging Face

This repository shows how to use a U-Net model from the Hugging Face repository to generate new images from the CIFAR-10 dataset.

## Installation

Clone or download the repository:
```bash
git clone https://github.com/smotsch/diffusion_hugging_face.git
cd diffusion_hugging_face
```

### Python environment
If needed, create the Python environment using `mamba` and `pip`:
```bash
mamba env create -f python_env/environment_python.yaml
mamba activate diffusion-huggingface
mamba install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
Once the environment is created, run the script as follows:
```bash
mamba activate diffusion-huggingface # load env if needed
python main_ddpm_cifar10.py # run the script
```
If you prefer using `ipython`, you can instead do:
```bash
mamba activate diffusion-huggingface # load env if needed
ipython
run main_ddpm_cifar10.py
```

### Container
Alternatively, you can run the script inside a container. This method is especially useful when working on a server where `mamba` or `conda` are not installed. You'll need `apptainer` to create the container from the definition file ubuntu_diffusion_hugging.def:
```bash
apptainer build ubuntu_diffusion_hugging.sif ubuntu_diffusion_hugging.def
```
Then, run the script inside the container:
```bash
apptainer shell --nv ubuntu_diffusion_hugging.sif
cd ...
ipython
...
```

## Running

The main script trains a U-Net model and generates sample images after training in the `results` folder.
It also shows how to load a trained network and generate new samples from it.

 

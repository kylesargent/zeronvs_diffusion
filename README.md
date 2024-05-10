# Introduction
This is the companion repository to https://github.com/kylesargent/ZeroNVS/. Code in this repo
is for training the underlying 3D-aware diffusion model.

# Setting up the environment
First, set up the environment
```
conda create -n zeronvs_diffusion python=3.10 pip
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```
# Configuration and data format
The zeronvs_diffusion repo uses webdataset to stream and mix large multivew datasets such as CO3D, ACID, and RealEstate10K. 
 
The basic configuration for a single dataset is as follows (see zero123/configs/sd-objaverse-finetune-c_concat-256.yaml for the full context):
```
dataset_config_1:
    dataset_n_shards: 127
    dataset_name: "co3d"
    views_per_scene: 100
    dataset_n_scenes: 18432
    rate: .025
    probability: .34
    compute_nearplane_quantile: False
    dataset_url: null
```
Here `probability` refers to the rate at which the given dataset is sampled from the mixture of datasets (note that the probabilities should sum to 1). The data is expected to be stored in a sharded format. 

I cannot host the full dataset myself for various reasons. [Here](https://drive.google.com/file/d/1Ly29H8vmbZsMxKutcIoyawAk43xb1VA2/view?usp=sharing) is a link to a single example shard. The shards are structured like so:
```
book/197_21268_42838/frame000001.png
book/197_21268_42838/frame000001_depth.png
book/197_21268_42838/frame000001_metadata.json
book/197_21268_42838/frame000002.png
book/197_21268_42838/frame000002_depth.png
book/197_21268_42838/frame000002_metadata.json
.
.
.
bowl/70_5774_13322/frame000001.png
bowl/70_5774_13322/frame000001_depth.png
bowl/70_5774_13322/frame000001_metadata.json
.
.
.
.
.
.
```
# Data preprocessing script
Coming soon.

# Camera conventions.
ZeroNVS uses relative camera poses in the OpenGL camera format (x - right, y - up, z - back) in camera-to-world format. 

# Training command
Use `run_train_local.sh` to train the main model (finetunes from [zero123-xl](https://objaverse.allenai.org/docs/zero123-xl/), which you need to download.)

The original training requires 8 GPUs with at least 40GB memory. Fewer is possible with additional gradient accumulation.
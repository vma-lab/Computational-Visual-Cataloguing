#!/bin/bash

# Exit immediately on error
set -e
# Show the exact command that failed
trap 'echo "❌ Error on line $LINENO: $BASH_COMMAND"; exit 1' ERR
# Print all commands before executing
set -o xtrace

# 0. Remove existing conda environment if it exists
conda remove -n open-mmlab --all -y || true

# 1. Create and activate conda environment with Python 3.7
conda create -n open-mmlab python=3.7 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate open-mmlab

# 2. Install PyTorch 1.8.0 + CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# 3. Fix specific dependency versions for compatibility with MMOCR 0.4.0
pip install yapf==0.30.0
pip install rapidfuzz==1.4.1
pip install ftfy==6.1.1
pip install regex==2022.10.31
pip install num2words==0.5.14

# 4. Vision models
pip install timm==0.4.12

# 5. MMCV and MMDetection (compatible versions)
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install mmdet==2.19.1

# 6. Optional dependencies for augmentation and dataset support
pip install instaboostfast==0.1.2
pip install albumentations==1.3.1

# Optional: Panoptic and LVIS APIs (only if needed)
# pip install git+https://github.com/cocodataset/panopticapi.git
# pip install git+https://github.com/lvis-dataset/lvis-api.git

# 7. MMsegmentation (optional)
pip install mmsegmentation==0.20.2

# 8. MMOCR v0.4.0 (legacy version from PyPI)
pip install mmocr==0.4.0

#-----------
pip install tqdm==4.67.1 

pip install safetensors==0.3.1

pip install sentence-transformers==2.2.2

pip install pandas==1.3.5

pip install seaborn==0.11.2



echo "✅ Environment setup completed successfully"


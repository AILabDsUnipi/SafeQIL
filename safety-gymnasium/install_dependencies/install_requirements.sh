#!/usr/bin/bash

# Run the following in command line
# source anaconda3/bin/activate
# conda create --name safety_gymnasium python=3.8.18 -y
# conda activate safety_gymnasium

# Being in the root directory (~/safety-gymnasium)
cd safety-gymnasium-main
yes | pip install -e .

# Check for NVIDIA GPU presence using lspci
if lspci | grep -i nvidia; then
    echo "GPU detected. Installing PyTorch with CUDA..."
    pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
else
    echo "No GPU detected. Installing CPU-only PyTorch..."
    conda install pytorch=1.13 cpuonly -c pytorch -y
fi

cd ../install_requirements/
yes | pip install -r ./requirements.txt
cd ..

export PYTHONPATH=$PWD:$PYTHONPATH

#!/usr/bin/bash

# Run the following in command line
# source anaconda3/bin/activate
# conda create --name maze_RL_conda_env python=3.6 -y
# conda activate maze_RL_conda_env

yes | pip install matplotlib==3.3.3
yes | pip install numpy==1.19.4
yes | pip install pygame==2.0.0
conda install pandas=1.1.5 -y
yes | pip install PyOpenGL==3.1.5
conda install pyparsing=2.4.7 -y
yes | pip install install pyrr==0.10.3
yes | pip install PyWavefront==1.3.3
conda install PyYAML=5.3.1 -y
conda install pytorch=1.10.2 pytorch-cuda=11.6 -c pytorch -c nvidia -y # or for cpu --> conda install pytorch=1.10.2 cpuonly -c pytorch
conda install tqdm=4.54.1 -y
conda install seaborn=0.11.2 -y
conda install pympler=0.9 -y

export PYTHONPATH=$PWD:$PYTHONPATH
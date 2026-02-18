#!/usr/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Run the following in command line
# source anaconda3/bin/activate
# conda create --name safety_gymnasium python=3.8.18 -y
# conda activate safety_gymnasium

# Being in the root directory (~/code)
cd safety-gymnasium-main
yes | pip install -e .

cd ./../install_dependencies/
yes | pip install -r ./requirements.txt
cd ..


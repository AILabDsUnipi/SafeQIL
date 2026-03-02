# SafeQIL

This is the official repository of:

Learning to maintain safety through expert demonstrations in settings with unknown constraints: A Q-learning perspective.

## Dataset

Coming soon!

## Supplementary Material

You can find the full paper with the supplementary material [here](http://arxiv.org/abs/2602.23816).

## Installation

We recommend to use `conda` to create a virtual environment and install the required packages.

You can follow the commands provided in the file [install_anaconda.sh](code/install_dependencies/install_anaconda.sh) to install conda.

Next, you can follow the commands provided in the file [install_requirements.sh](code/install_dependencies/install_requirements.sh) to activate the conda environment and install the required packages.

## Run

* Edit the file [config_human_alone_test.yaml](code/safeqil_implementation/config/config_human_alone_test.yaml), go to `code/` directory, and run `python safeqil_implementation/human_test.py safeqil_implementation/config/config_human_alone_test.yaml <exp_name>` for human-only game. This mode is suitable for demonstration collection.
* Edit the file [config_safeqil.yaml](code/safeqil_implementation/config/config_safeqil.yaml), go to `code/` directory, and run `python safeqil_implementation/train_sac_agent.py safeqil_implementation/config/config_safeqil.yaml <exp_name>` to train a policy using SafeQIL.
* Edit the file [config_icrl.yaml](code/safeqil_implementation/config/config_icrl.yaml), go to `code/` directory, and run `python safeqil_implementation/train_icrl_agent.py safeqil_implementation/config/config_icrl.yaml <exp_name>` to train a policy using ICRL.
* Edit the file [config_vicrl.yaml](code/safeqil_implementation/config/config_vicrl.yaml), go to `code/` directory, and run `python safeqil_implementation/train_vicrl_agent.py safeqil_implementation/config/config_vicrl.yaml <exp_name>` to train a policy using VICRL.
* Edit the file [config_sacgail.yaml](code/safeqil_implementation/config/config_sacgail.yaml), go to `code/` directory, and run `python safeqil_implementation/train_sac_agent.py safeqil_implementation/config/config_sacgail.yaml <exp_name>` to train a policy using SAC-GAIL.
* Notes before running: 
   * Set the `<exp_name>` as you want.
   * The program will create a `<exp_name>/tmp` and a `<exp_name>/plot` folder in the `code/` folder. The `<exp_name>/tmp` folder contains files with information of the experiment. The `<exp_name>/plot` folder contains figures for that experiment.
   * The program will automatically create an identification number after your name on each folder name created
### Run
* Run `python game/maze3d_human_alone_test.py game/config/config_human_alone_test.yaml <exp_name>` for human-only game. This mode is suitable for demonstration collection.
* Run `python game/maze3d_train_agent-s.py game/config/<config_algo> <exp_name>` for agent(s) training.
* Run `python game/maze3d_test_agent-s.py game/config/<config_algo> <exp_name>` for agent(s) testing.
* Run `python game/maze3d_multi_train_test.py game/config/config_multi_train_test.yaml <exp_name>` for conducting multiple experiments (training and testing).
* Notes before running: 
   * Set the <exp_name>.
   * The program will create a `/tmp` and a `/plot` folder (if they do not exist) in the `results/` folder. The `/tmp` folder contains files with information of the game. The `/plot` folder contains figures for tha game.
   * The program will automatically create an identification number after your name on each folder name created

import os

import numpy as np
import torch
import torch.utils.data
import pandas as pd
import sys

from plot_utils.plot_utils import get_config
from maze3D_new.utils import normalize_features


class exprt_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_paths):

        self.dataset_paths = dataset_paths

        self.data_ids = []
        self.data_buff = {}
        self.trajectory_idx_list = []
        self.timestep_idx_list = []

        print("\nLoad expert datasets ...")
        print("Total number of demonstrators: " + str(len(self.dataset_paths)))

        for path_id, path in enumerate(self.dataset_paths):

            tmp_actions = pd.read_csv(path + '/test_actions.csv')
            if tmp_actions.columns.tolist() != ['X', 'Y']:
                # The column names of the actions dataframe is the first actions.
                # Thus, we need to get them and append them to the dataframe as the
                # first row. Also, we need to rename the columns.
                try:
                    tmp_actions_first_row_as_df = pd.DataFrame([[float(elem) for elem in tmp_actions.columns.tolist()]],
                                                               columns=['X', 'Y'])
                except:
                    tmp_actions_first_row_as_df = pd.DataFrame([[float(elem[0]) for elem in tmp_actions.columns.tolist()]],
                                                               columns=['X', 'Y'])
                tmp_actions.columns = ['X', 'Y']
                tmp_actions = pd.concat([tmp_actions_first_row_as_df, tmp_actions]).reset_index(drop=True)
            # Replace -1 with 2 since -1 is an environment-like action
            # which can be translated to 2 as a class index.
            tmp_actions = tmp_actions.astype(int).replace(-1, 2)

            # We need to drop the states that are terminals since there are no actions that correspond to them.
            tmp_states = pd.read_pickle(path + '/test_logs.pkl')
            tmp_states = tmp_states[tmp_states['done'] == 0.0]
            tmp_states = tmp_states[['ball_pos_x', 'ball_pos_y', 'ball_vel_x', 'ball_vel_y',
                                     'tray_rot_x', 'tray_rot_y', 'tray_rot_vel_x', 'tray_rot_vel_y']].reset_index(drop=True)

            # Check if actions and states have the same length
            assert len(tmp_actions.index) == len(tmp_states.index)

            # Store actions and normalized states
            self.data_buff[path_id] = {'actions': tmp_actions.copy(), 'states': normalize_features(tmp_states.copy())}

            self.trajectory_idx_list += [path_id for _ in range(len(self.data_buff[path_id]['actions'].index))]
            self.timestep_idx_list += [i for i in range(len(self.data_buff[path_id]['actions'].index))]

            print("Total number of samples of demonstrator {}: {}".
                  format(path_id, len(self.data_buff[path_id]['actions'].index)))

        self.total_num_samples = len(self.trajectory_idx_list)

        print("Total number of samples: " + str(self.total_num_samples))

    def __getitem__(self, index):

        trajectory_idx = self.trajectory_idx_list[index]
        timestep_idx = self.timestep_idx_list[index]

        inputs = self.data_buff[trajectory_idx]['states'].loc[timestep_idx].values
        actions = self.data_buff[trajectory_idx]['actions'].loc[timestep_idx][['X', 'Y']].values

        return torch.from_numpy(np.array(inputs, dtype=np.float32)).float(), \
               torch.from_numpy(np.array(actions)).long(), \
               trajectory_idx

    def __len__(self):
        return self.total_num_samples

    def getAll(self):
        """
        Returns all samples in numpy array format
        @return: Tuple of: 1) observations numpy array, 2) actions numpy array
        """

        allInputs = []
        allActions = []
        for index in range(self.total_num_samples):
            trajectory_idx = self.trajectory_idx_list[index]
            timestep_idx = self.timestep_idx_list[index]
            allInputs.append(self.data_buff[trajectory_idx]['states'].loc[timestep_idx].values)
            allActions.append(self.data_buff[trajectory_idx]['actions'].loc[timestep_idx][['X', 'Y']].values)

        return np.array(allInputs, dtype=np.float32), np.array(allActions, dtype=np.long)


if __name__ == "__main__":

    argv_ = sys.argv[1:]
    if os.path.exists(argv_[0]):
        config_path = argv_[0]
    elif os.path.exists('./../../' + argv_[0]):
        config_path = './../../' + argv_[0]
    else:
        print('Cannot find the config file path. Neither {} nor {} exists.'.
              format(argv_[0], './../../' + argv_[0]))
    config = get_config(config_path)
    exprt_dtst = exprt_dataset(config['coGAIL']['expert_dataset_paths'])

    all_inputs, all_actions = exprt_dtst.getAll()
    #print(all_inputs.shape)
    #print(all_actions.shape)

    for idx, sample in enumerate(exprt_dtst):
        inputs_, actions_, trajectory_idx_ = sample

        assert (np.array(inputs_.numpy(), dtype=np.float32) == all_inputs[idx]).all(), \
            "inputs_ {}, all_inputs[idx] {}".format(np.array(inputs_.numpy(), dtype=np.float32), all_inputs[idx])
        assert (np.array(actions_.numpy(), dtype=np.long) == all_actions[idx]).all(), \
            "actions_ {}, all_actions[idx] {}".format(actions_.numpy(), all_actions[idx])

        #print(inputs_)


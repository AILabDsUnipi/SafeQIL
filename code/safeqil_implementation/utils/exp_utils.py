import os
import pickle
import random

import cv2
import numpy as np
import torch as th


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


def print_latest_metrics_from_dict(metrics_dict: dict):
    print()  # just for printing an empty line
    for key, value in metrics_dict.items():
        if len(value) == 0:
            continue  # Ignore empty logs
        print(
            "Avg {}: {}".format(
                key,
                round(float(value[-1]), 2)
            )
        )
    print()  # just for printing another empty line


def get_train_seed(config):
    seed = config['Experiment']['seed']
    if seed == 'None':
        seed = int(np.random.randint(2 ** 32, dtype='int64'))

    return seed


def get_test_seed(config):
    seed = config['Experiment']['test_seed']
    assert isinstance(seed, int)

    return seed


def test_print_logs(
        avg_score,
        avg_steps,
        avg_num_constraint_violations,
        avg_freq_constraint_violations
):

    print('\n##########Average stats for testing##########')
    print(
        f'Avg reward: {round(avg_score, 2)}\n'
        f'Avg number of steps: {round(avg_steps, 2)}'
    )
    for constraint_type in avg_num_constraint_violations:
        print(
            f'Avg number of {constraint_type.replace("cost_", "")} violations: '
            f'{round(avg_num_constraint_violations[constraint_type], 2)}'
        )
        print(
            f'Avg freq of {constraint_type.replace("cost_", "")} violations: '
            f'{round(avg_freq_constraint_violations[constraint_type], 2)}'
        )


def save_demonstrations(demo_dict, file_results_dir, save_pickle_file=True, delete_episode=False):

    # Create the directory to store demonstrations if it does not exist
    demo_dir = os.path.join(file_results_dir, 'demonstrations')
    if os.path.exists(demo_dir) is False:
        os.mkdir(demo_dir)

    # Check the consistency of dict elements
    demo_dict_keys = list(demo_dict.keys())
    first_demo_key = demo_dict_keys[0]
    demo_episode_keys = list(demo_dict[first_demo_key].keys())
    assert len(demo_episode_keys) == 1, f"'demo_episode_keys': {demo_episode_keys}"
    for demo_key in demo_dict.keys():
        assert list(demo_dict[demo_key].keys()) == demo_episode_keys, \
            f"'demo_episode_keys': {demo_episode_keys}, 'demo_dict[demo_key].keys()': {demo_dict[demo_key].keys()}"

    # Get the file name (the same both for pickle and video files)
    demo_file_name = demo_episode_keys[0]

    # Save the demonstrations in pickle file
    demo_file_path = os.path.join(demo_dir, demo_file_name + '.pkl')
    if save_pickle_file is True:
        with open(demo_file_path, 'wb') as pkl_file:
            pickle.dump(demo_dict, pkl_file)

    ## Create and save video
    if 'vision_obs' in demo_dict_keys:
        # Output video parameters
        demo_video_file_path = os.path.join(demo_dir, demo_file_name + '.mp4')
        frame_size = tuple(demo_dict['vision_obs'][demo_file_name]['step_0'].shape[:2])
        fps = 24
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(demo_video_file_path, fourcc, fps, frame_size)
        ## Write each frame in the video
        for step_id in range(len(demo_dict['vision_obs'][demo_file_name].keys())):
            # Get the frame
            frame = demo_dict['vision_obs'][demo_file_name][f'step_{step_id}']
            # Check the frame size
            assert tuple(frame.shape[:2]) == frame_size, \
                f"'frame.shape[:2]': {frame.shape[:2]}, 'frame_size': {frame_size}"
            # Convert from RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Write the frame
            video_writer.write(frame)
        # Release the video writer
        video_writer.release()

    # Delete demonstrations to free up RAM
    if delete_episode is True:
        for demo_key in demo_dict.keys():
            del demo_dict[demo_key][demo_file_name]


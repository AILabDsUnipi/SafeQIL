import safety_gymnasium
import mujoco

import sys
sys.path.append('./../')

from SCOPIL.utils.env_utils import capture_img_from_env_rendering


env_id = 'SafetyPointGoal1Debug-v0'
env = safety_gymnasium.make(env_id, render_mode="human")

obs, info = env.reset(seed=0)

# Change camera view to agent view
env.render()
env.task.viewer.cam.fixedcamid = 3
env.task.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

# Hide menu from rendering
env.task.viewer._hide_menu = True
env.render()

# Get the first image observations and actions after resetting.
img_obs = capture_img_from_env_rendering(env)
actions = env.task.agent.engine.data.ctrl

recorded_data = {"episode_0": {}}
episode_step = 0

while True:

    # Apply an environment step and get the results.
    # No actions needed since the agent is controlled through the keyboard.
    next_obs, reward, cost, terminated, truncated, info = env.step([None, None])

    # Record data as demonstrations
    recorded_data["episode_0"][f"step_{episode_step}"] = {
        "obs": obs.copy(),
        "vision_obs": img_obs.copy(),
        "actions": actions.copy(),
        "reward": reward,
        "cost": cost,
        "terminated": terminated,
        "truncated": truncated,
        "info": info.copy()
    }

    # Get the image observations and actions
    env.render()
    next_img_obs = capture_img_from_env_rendering(env, show_final_image=True)
    next_actions = env.task.agent.engine.data.ctrl

    print()
    print("Reward: ", reward)
    print("Cost: ", cost)
    print("Info: ", info)
    print("Terminated: ", terminated)
    print("Truncated: ", truncated)
    print("Actions", actions)
    print("Observation: ", obs)
    print("Observations shape: ", obs.shape)
    print("Vision observations shape: ", img_obs.shape)

    if terminated or truncated:
        # Store the last observations and actions
        recorded_data["episode_0"][f"step_{episode_step}"] = {
            "obs": next_obs.copy(),
            "vision_obs": next_img_obs.copy(),
            "actions": next_actions.copy()
        }
        break

    # Set the next as current
    obs = next_obs.copy()
    img_obs = next_img_obs.copy()
    actions = next_actions.copy()

    episode_step += 1

env.close()

# render_mode (str): Render mode can be 'human', 'rgb_array', 'depth_array'.
# When 'rbg_array' is selected, the resulted image should have shape [256, 256, 3] for all the following environments.

## Tasks explanation based on: https://github.com/openai/safety-gym, and observation shape.

# Safety{Robot}Goal0-v0: A robot must navigate to a goal.
# Observation shape:
#   Robot=Point: shape: (28,), [0-11](12): Point features, [12-27](16): LiDAR-Goal
#   Robot=Car: shape: (40,), [0-23](24): Point features, [23-39](16): LiDAR-Goal
# max_episode_steps: 1000

# Safety{Robot}Goal1-v0: A robot must navigate to a goal while avoiding hazards.
#                         One vase is present in the scene, but the agent is not penalized for hitting it.
# Safety{Robot}Goal2-v0: A robot must navigate to a goal while avoiding more hazards and vases.
# Observation shape:
#   Robot=Point: shape: (60,), [0-11](12): Point features, [12-27](16): LiDAR-Goal, [27-43](16): LiDAR-Hazards,
#                              [43-59](16): LiDAR-Vases
#   Robot=Car: shape: (72,), [0-23](24): Point features, [23-39](16): LiDAR-Goal, [39-55](16): LiDAR-Hazards,
#                            [55-71](16): LiDAR-Vases
# max_episode_steps: 1000

# Safety{Robot}Button0-v0: A robot must press a goal button. No constraints applied.
# Observation shape:
#   Robot=Point: shape: (44,), [0-11](12): Point features, [12-27](16): LiDAR-Buttons, [28-43](16): LiDAR-GoalButton
#   Robot=Car: shape: (56,), [0-23](24): Point features, [24-39](16): LiDAR-Buttons, [39-55](16): LiDAR-GoalButton
# max_episode_steps: 1000

# Safety{Robot}Button1-v0: A robot must press a goal button while avoiding hazards and gremlins,
#                           and while not pressing any of the wrong buttons.
# Safety{Robot}Button2-v0: A robot must press a goal button while avoiding more hazards and gremlins,
#                           and while not pressing any of the wrong buttons.
# Observation shape:
#   Robot=Point: shape: (76,), [0-11](12): Point features, [12-27](16): LiDAR-Buttons, [28-43](16): LiDAR-GoalButton,
#                              [44-59](16): LiDAR-Hazards, [60-75](16): LiDAR-Gremlins
#   Robot=Car: shape: (88,), [0-23](24): Point features, [24-39](16): LiDAR-Buttons, [39-55](16): LiDAR-GoalButton,
#                            [56-71](16): LiDAR-Hazards, [72-87](16): LiDAR-Gremlins
# max_episode_steps: 1000

# Safety{Robot}Push0-v0: A robot must push a box to a goal.
# Observation shape:
#   Robot=Point: shape: (44,), [0-11](12): Point features, [12-27](16): LiDAR-Goal, [28-43](16): LiDAR-Boxes
#   Robot=Car: shape: (56,), [0-23](24): Point features, [24-39](16): LiDAR-Goal, [39-55](16): LiDAR-Boxes
# max_episode_steps: 1000

# Safety{Robot}Push1-v0: A robot must push a box to a goal while avoiding hazards.
#                         One pillar is present in the scene, but the agent is not penalized for hitting it.
# Safety{Robot}Push2-v0: A robot must push a box to a goal while avoiding more hazards and pillars.
# Observation shape:
#   Robot=Point: shape: (76,), [0-11](12): Point features, [12-27](16): LiDAR-Goal, [28-43](16): LiDAR-Hazards,
#                              [44-59](16): LiDAR-Pillars, [60-75](16): LiDAR-Boxes
#   Robot=Car: shape: (88,), [0-23](24): Point features, [24-39](16): LiDAR-Goal, [39-55](16): LiDAR-Hazards,
#                            [56-71](16): LiDAR-Pillars, [72-87](16): LiDAR-Boxes
# max_episode_steps: 1000

# Safety{Robot}FadingEasy0-v0: Similar to Safety{Robot}Goal0-v0 but the goal fades over time. (Vision)
# Observation shape:
#   Robot=Point: shape: (28,), [0-11](12): Point features, [12-27](16): LiDAR-Goal
#   Robot=Car: shape: (40,), [0-23](24): Point features, [23-39](16): LiDAR-Goal
# max_episode_steps: 1000

# Safety{Robot}FadingEasy1-v0: Similar to Safety{Robot}Goal1-v0 but the goal (only) fades over time. (Vision)
# Safety{Robot}FadingEasy2-v0: Similar to Safety{Robot}Goal2-v0 but the goal and the hazards fades over time. (Vision)
# Observation shape:
#   Robot=Point: shape: (60,), [0-11](12): Point features, [12-27](16): LiDAR-Goal, [27-43](16): LiDAR-Hazards,
#                              [43-59](16): LiDAR-Vases
#   Robot=Car: shape: (72,), [0-23](24): Point features, [23-39](16): LiDAR-Goal, [39-55](16): LiDAR-Hazards,
#                            [55-71](16): LiDAR-Vases
# max_episode_steps: 1000

# Safety{Robot}FadingHard0-v0: Similar to Safety{Robot}FadingEasy0-v0 but the goal fades faster. (Vision)
# Observation shape:
#   Robot=Point: shape: (28,), [0-11](12): Point features, [12-27](16): LiDAR-Goal
#   Robot=Car: shape: (40,), [0-23](24): Point features, [23-39](16): LiDAR-Goal
# max_episode_steps: 1000

# Safety{Robot}FadingHard1-v0: Similar to Safety{Robot}FadingEasy1-v0 but the goal fades faster. (Vision)
# Safety{Robot}FadingHard2-v0: Similar to Safety{Robot}FadingEasy2-v0 but the goal and the hazards fade faster.
#                              Also, the vases fade in contrast to Safety{Robot}FadingEasy2-v0. (Vision)
# Observation shape:
#   Robot=Point: shape: (60,), [0-11](12): Point features, [12-27](16): LiDAR-Goal, [27-43](16): LiDAR-Hazards,
#                              [43-59](16): LiDAR-Vases
#   Robot=Car: shape: (72,), [0-23](24): Point features, [23-39](16): LiDAR-Goal, [39-55](16): LiDAR-Hazards,
#                            [55-71](16): LiDAR-Vases
# max_episode_steps: 1000

# (To make one of the above, make sure to substitute {Robot} for one of Point, Car, Racecar, Doggo, or Ant)

## Tasks explanation based on: https://arxiv.org/pdf/2310.12567, and observation shape.
# Safety{Robot}Circle0-v0: The reward is maximized by moving along the green circle. No constraint applied.
# Safety{Robot}Circle1-v0: The reward is maximized by moving along the green circle and not allowed to enter the
#                          outside of the region defined by the grey areas, i.e., horizontal lines (Sigwalls).
# Safety{Robot}Circle2-v0: The reward is maximized by moving along the green circle and not allowed to enter the
#                          outside of the region defined by the grey areas, i.e., horizontal and vertical lines.
# Observation shape:
#   Robot=Point: shape: (28,), [0-11](12): Point features, [12-27](16): LiDAR-CenterOfAllowedRegion
#   Robot=Car: shape: (40,), [0-23](24): Point features, [23-39](16): LiDAR-CenterOfAllowedRegion
# max_episode_steps: 500

# Safety{Robot}Run0-v0: the robot starts with a random initial direction and a specific initial speed as it embarks
#                       on a journey to reach the opposite side of the map. Not allowed to enter the outside
#                       of the region outside the Sigwalls.
# Observation shape:
#   Robot=Point: shape: (12,), [0-11](12): Point features
#   Robot=Car: shape: (24,), [0-23](24): Point features
# max_episode_steps: 500

# Safety{Robot}BuildingButton0-v0: Requires the agent to operate multiple machines within a construction site
#                                  by touching the blue sign with a 'P'. No constraint applied. (Vision)
# Observation shape:
#   Robot=Point: shape: (44,), [0-11](12): Point features, [12-27](16): LiDAR-Obstacles, [28-43](16): LiDAR-GoalButton
#   Robot=Car: shape: (56,), [0-23](24): Point features, [24-39](16): LiDAR-Obstacles, [39-55](16): LiDAR-GoalButton
# max_episode_steps: 1000

# Safety{Robot}BuildingButton1-v0: Requires the agent to proficiently and accurately operate multiple
#                                  machines within a construction site by touching the blue sign with a 'P',
#                                  while concurrently evading other robots and obstacles present in the area.
#                                  (Vision)
# Safety{Robot}BuildingButton2-v0: Requires the agent to proficiently and accurately operate multiple
#                                  machines within a construction site by touching the blue sign with a 'P',
#                                  while concurrently evading a heightened number of other robots and obstacles
#                                  in the area. (Vision)
#   Robot=Point: shape: (76,), [0-11](12): Point features, [12-27](16): LiDAR-Obstacles, [28-43](16): LiDAR-GoalButton,
#                              [44-59](16): LiDAR-NotAllowedAreas, [60-75](16): LiDAR-Robots
#   Robot=Car: shape: (88,), [0-23](24): Point features, [24-39](16): LiDAR-Obstacles, [39-55](16): LiDAR-GoalButton,
#                            [56-71](16): LiDAR-NotAllowedAreas, [72-87](16): LiDAR-Robots
# max_episode_steps: 1000

# Safety{Robot}BuildingGoal0-v0: Requires the agent to dock at designated positions within a construction
#                                site by touching the blue sign with a 'P'. No constraints applied. (Vision)
# Observation shape:
#   Robot=Point: shape: (28,), [0-11](12): Point features, [12-27](16): LiDAR-Goal
#   Robot=Car: shape: (40,), [0-23](24): Point features, [23-39](16): LiDAR-Goal
# max_episode_steps: 1000

# Safety{Robot}BuildingGoal1-v0: Requires the agent to dock at designated positions within a construction
#                                site by touching the blue sign with a 'P' while ensuring to avoid entry into
#                                hazardous areas. (Vision)
# Safety{Robot}BuildingGoal2-v0: Requires the agent to dock at designated positions within a construction
#                                site by touching the blue sign with a 'P', while ensuring to avoid entry into
#                                hazardous areas and circumventing the site’s exhaust fans. (Vision)
# Observation shape:
#   Robot=Point: shape: (60,), [0-11](12): Point features, [12-27](16): LiDAR-Goal, [27-43](16): LiDAR-NotAllowedAreas,
#                              [43-59](16): LiDAR-ExhaustFans
#   Robot=Car: shape: (72,), [0-23](24): Point features, [23-39](16): LiDAR-Goal, [39-55](16): LiDAR-NotAllowedAreas,
#                            [55-71](16): LiDAR-ExhaustFans
# max_episode_steps: 1000

# Safety{Robot}BuildingPush0-v0: Requires the agent to relocate the box to designated locations
#                                (blue sign with a 'P') within a construction site. No constraints applied. (Vision)
# Observation shape:
#   Robot=Point: shape: (44,), [0-11](12): Point features, [12-27](16): LiDAR-Goal, [28-43](16): LiDAR-Boxes
#   Robot=Car: shape: (56,), [0-23](24): Point features, [24-39](16): LiDAR-Goal, [39-55](16): LiDAR-Boxes
# max_episode_steps: 1000

# Safety{Robot}BuildingPush1-v0: Requires the agent to relocate the box to designated locations
#                                (blue sign with a 'P') within a construction site while avoiding areas demarcated
#                                as restricted. (Vision)
# Safety{Robot}BuildingPush2-v0: Requires the agent to relocate the box to designated locations
#                                (blue sign with a 'P') within a construction site while avoiding numerous
#                                hazardous fuel drums and areas demarcated as restricted. (Vision)
# Observation shape:
#   Robot=Point: shape: (76,), [0-11](12): Point features, [12-27](16): LiDAR-Goal, [28-43](16): LiDAR-NotAllowedAreas,
#                              [44-59](16): LiDAR-FuelDrums, [60-75](16): LiDAR-Boxes
#   Robot=Car: shape: (88,), [0-23](24): Point features, [24-39](16): LiDAR-Goal, [39-55](16): LiDAR-NotAllowedAreas,
#                            [56-71](16): LiDAR-FuelDrums, [72-87](16): LiDAR-Boxes
# max_episode_steps: 1000

# Safety{Robot}Race0-v0: Requires the agent to reach the goal position. No constraints applied. (Vision)
# Observation shape:
#   Robot=Point: shape: (28,), [0-11](12): Point features, [12-27](16): LiDAR-Goal
#   Robot=Car: shape: (40,), [0-23](24): Point features, [23-39](16): LiDAR-Goal
# max_episode_steps: 500

# Safety{Robot}Race1-v0: Requires the agent to reach the goal position while ensuring it avoids straying
#                        into the grass and prevents collisions with roadside objects. (Vision)
# Safety{Robot}Race2-v0: Requires the agent to reach the goal position from a distant starting point while
#                        ensuring it avoids straying into the grass and prevents collisions with roadside objects.
#                        (Vision)
# Observation shape:
#   Robot=Point: shape: (44,), [0-11](12): Point features, [12-27](16): LiDAR-Goal, [28-43](16): LiDAR-Grass
#   Robot=Car: shape: (56,), [0-23](24): Point features, [24-39](16): LiDAR-Goal, [39-55](16): LiDAR-Grass
# max_episode_steps: 500

## Actions space:
# Point/Car: Box(-1.0, 1.0, (2,), float64), that is, two continuous actions both in range [-1, 1].

## Envs supporting 'Debug':
# 'SafetyPoint{Task}Debug-v0', 'SafetyCar{Task}Debug-v0', 'SafetyRacecar{Task}Debug-v0'
# where {Task} should be one of: Goal{0,1,2}, Button{0,1,2}, Push{0,1,2}, Circle{0,1,2}, Run0,
#                                BuildingButton{0,1,2}, BuildingGoal{0,1,2}, BuildingPush{0,1,2},
#                                FadingEasy{0,1,2}, FadingHard{0,1,2}, Race{0,1,2}
#

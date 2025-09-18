import safety_gymnasium
import cv2


# Options (we used):
# - 'SafetyPointGoal1Debug-v0',
# - 'SafetyCarButton1Debug-v0',
# - 'SafetyCarPush2Debug-v0',
# - 'SafetyPointCircle2Debug-v0'
env_id = 'SafetyPointCircle2Debug-v0'
env = safety_gymnasium.make(env_id, render_mode="rgb_array")

obs, info = env.reset(seed=0)

while True:

    act = env.action_space.sample()  # [0, 0]
    obs, reward, cost, terminated, truncated, info = env.step(act)
    vision_obs = env.render()

    print()
    print("Reward: ", reward, type(reward))
    print("Cost: ", cost, type(cost))
    print("Info: ", info, type(info))
    print("Terminated: ", terminated, type(terminated))
    print("Truncated: ", truncated, type(truncated))
    print("Actions", act, type(act))
    print("Observations shape: ", obs.shape, type(obs))
    print("Vision observations shape: ", vision_obs.shape, type(vision_obs))

    # Check vision observation shape
    assert vision_obs.shape == (256, 256, 3), f"Vision observation shape: {vision_obs.shape}"

    # Show vision observations
    bgr_vision_obs = cv2.cvtColor(vision_obs, cv2.COLOR_RGB2BGR)
    cv2.imshow("agent view", bgr_vision_obs)
    cv2.waitKey(100)

    if terminated or truncated:
        break

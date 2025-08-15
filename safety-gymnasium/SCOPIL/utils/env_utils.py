import cv2
import mujoco
import glfw
import numpy as np


def capture_img_from_env_rendering(env, new_image_shape=(256, 256), show_final_image=False):
    # Create an empty image
    img = np.zeros(
        (
            glfw.get_framebuffer_size(env.task.viewer.window)[1],
            glfw.get_framebuffer_size(env.task.viewer.window)[0],
            3,
        ),
        dtype=np.uint8,
    )
    # Get pixel values
    mujoco.mjr_readPixels(img, None, env.task.viewer.viewport, env.task.viewer.con)
    # Flip the image upside down
    img = cv2.flip(img, 0)
    # Check image dimensions
    assert img.shape == (472, 960, 3), f"image shape: {img.shape}"
    img = img[:, 244:-244]  # 960-472=488, 488/2=244
    # Resize the image to 256x256 pixels
    img = cv2.resize(img, new_image_shape)

    if show_final_image is True:
        # RGB to BGR
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Show
        cv2.imshow("agent view", bgr_img)
        cv2.waitKey(100)

    return img


def get_cost_sum_from_info_dict(cost_dict):
    cost_sum = 0
    for cost_type, cost_value in cost_dict.items():
        if 'cost' in cost_type and cost_type != 'cost_sum':
            cost_sum += cost_value

    return cost_sum


def close_env_debug_mode(env):
    # Close it
    env.close()
    # Fix a bug of mujoco which does not check if 'glfw' is not None before calling 'get_current_context'
    if hasattr(env.task, 'viewer') and env.task.viewer is not None:
        def safe_free(self):
            if self.window:
                # Check if 'glfw' is available and not None
                if 'glfw' in globals() and glfw is not None:
                    if glfw.get_current_context() == self.window:
                        glfw.make_context_current(None)
                    glfw.destroy_window(self.window)
                self.window = None

        # Assign the new 'free' method to the viewer
        import types
        env.task.viewer.free = types.MethodType(safe_free, env.task.viewer)


def render(vision_obs):

    # Show vision observations
    bgr_vision_obs = cv2.cvtColor(vision_obs, cv2.COLOR_RGB2BGR)
    cv2.imshow("agent view", bgr_vision_obs)
    cv2.waitKey(10)

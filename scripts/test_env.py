import os

# Set backend before importing anything that may import mujoco.
mujoco_gl = os.environ.get("MUJOCO_GL", "").lower()
has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

if mujoco_gl == "osmesa" or not has_display:
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    render_mode = None
    print("Headless mode enabled.")
    print(f"MUJOCO_GL={os.environ['MUJOCO_GL']}")
else:
    render_mode = "human"
    print(f"GUI mode enabled. MUJOCO_GL={mujoco_gl}")

import sys
import numpy as np

sys.path.insert(0, os.path.abspath("/workspace"))

from rl.envs.g1_balance_env import G1BalanceEnv

env = G1BalanceEnv(render_mode=render_mode)

obs, _ = env.reset()
print("Observation Shape:", obs.shape)

for i in range(1000):
    action = np.zeros(env.action_dim, dtype=np.float32)
    obs, reward, done, trunc, info = env.step(action)
    print(f"Step {i}: Reward={reward:.4f}")

    if done:
        print("G1 fell")
        break

    if trunc:
        print("Episode timed out")
        break

input("Press Enter to close the window.")

env.close()
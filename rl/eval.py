# This file is for running the trained policy without exploration noise, observing behavior.
# Training: consists of sampling and exploration and is noisy.
# Evaluation: consists of deterministic actions and exploitation and is stable.

import os

import cv2
import torch
from tqdm import tqdm

from rl.envs.g1_balance_env import G1BalanceEnv
from rl.ppo.model import ActorCritic

if __name__ == "__main__":
    # Automatically use CUDA or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate the model with the GUI environment (set the rendering mode).
    # Uncomment/comment the lines based on what render_mode you need.
    # env = G1BalanceEnv(render_mode="human")
    env = G1BalanceEnv(render_mode="rgb_array")

    # Get the dimensions from the observation and action tensors.
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_dim

    # Initialize statuses for when the loop ends early.
    terminated_early = False
    termination_reason = None

    # Initialize the model.
    model = ActorCritic(obs_dim, action_dim).to(device)

    # Load the trained model from checkpoints/.
    # Try testing the various iterations from checkpoints/ by changing the model loaded. The
    # differences between each iteration can be seen.
    model.load_state_dict(torch.load("checkpoints/ppo_best.pth", map_location=device))
    model.eval()

    frames = []

    state, _ = env.reset()
    frame = env.render()

    # Add the first frame.
    if frame is not None:
        frames.append(frame)

    # Display a progress bar for evaluation.
    pbar = tqdm(range(1000), desc="Evaluating PPO policy")

    for step in pbar:
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            # Get the deterministic action.
            action_mean, _ = model.actor(state_tensor)

        action = action_mean.cpu().numpy()

        state, reward, done, trunc, _ = env.step(action)

        # Capture the current frame.
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # Show the reward live.
        pbar.set_postfix(reward=f"{reward:.3f}")

        #  Set statuses for episode termination.
        if done:
            terminated_early = True
            termination_reason = "done"
            break
        elif trunc:
            terminated_early = True
            termination_reason = "trunc"

    # Use the statuses to print a statement for the user.
    if terminated_early:
        if termination_reason == "done":
            print(
                f"[INFO] Episode terminated early at step {step}/1000 (done=True: failure or terminal state)."
            )
        elif termination_reason == "trunc":
            print(
                f"[INFO] Episode terminated early at step {step}/1000 (trunc=True: time limit reached.)"
            )
    else:
        print("[INFO] Episode completed full rollout.")

    # Close the environment.
    env.close()

    # Save the video.
    if len(frames) > 0:
        os.makedirs("videos", exist_ok=True)
        h, w, _ = frames[0].shape
        out = cv2.VideoWriter("videos/robot.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        out.release()
    print("Video file saved.")

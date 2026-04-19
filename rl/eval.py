# This file is for running the trained policy without exploration noise, observing behavior.
# Training: consists of sampling and exploration and is noisy.
# Evaluation: consists of deterministic actions and exploitation and is stable.

import torch

from rl.envs.g1_balance_env import G1BalanceEnv
from rl.ppo.model import ActorCritic

if __name__ == "__main__":
    # Evaluate the model with the GUI environment.
    env = G1BalanceEnv(render_mode="human")

    # Get the dimensions from the observation and action tensors.
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_dim

    # Initialize the model.
    model = ActorCritic(obs_dim, action_dim)

    # Load the trained weights.
    model.load_state_dict(torch.load("ppo_model.pth"))
    model.eval()

    state = obs

    for _ in range(1000):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            # Get the deterministic action.
            action_mean, _ = model.actor(state_tensor)

        action = action_mean.cpu().numpy()

        state, reward, done, trunc, _ = env.step(action)

        if done or trunc:
            print("Episode finished.")
            break

    # Close the environment.
    env.close()

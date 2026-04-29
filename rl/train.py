import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

# Import the balance environment class from g1_balance_env.py.
from rl.envs.g1_balance_env import G1BalanceEnv
from rl.ppo.buffer import RolloutBuffer
from rl.ppo.model import ActorCritic
from rl.ppo.trainer import compute_gae, update_ppo

# IMPORTANT NOTES:
# Run this in the terminal before running this file:
# export PYTHONPATH=.
# python3 rl/train.py

# OR
# python -m rl.train

# To view TensorBoard, run this in the terminal:
# tensorboard --logdir runs
# then open http://localhost:6006:

# 1. Run training
# 2. Open TensorBoard in another terminal
# 3. Open browser

# The FCNetwork class is just for debugging/testing purposes. It shows how many input and output
# dimensions we'll need to work with.
# class FCNetwork(nn.Module):
#     '''A Fully Connected Neural Network class.'''
#     def __init__(self, input_dim: int, output_dim: int) -> None:
#         '''Initializer function for the FC network class.

#         Args:
#             input_dim (int): The dimension of the input observation vector.
#             output_dim (int): The dimensin of the output (actions).
#         '''
#         # Initialize the parent nn.Module class for layers and parameters.
#         super().__init__()
#         # The FC feedforward neural network, 128 neurons for applicable layers.
#         # Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
#         self.net = nn.Sequential(
#             # Input layer.
#             nn.Linear(input_dim, 128),
#             # Hidden layers.
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             # Output layer.
#             nn.Linear(128, output_dim)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         '''Function for the forward pass.

#         Args:
#             x (torch.Tensor): Input observation tensor.

#         Returns:
#             torch.Tensor: A forward pass of x through the FC network.
#         '''
#         return self.net(x)


def find_moving_average(data: list[float], window: int = 10) -> npt.NDArray[np.float64]:
    """Function to smooth the episode returns curve.

    Args:
        data (list[float]): The list of episode returns.

    Returns:
        npt.NDArray[np.float64]: A 1D NumPy array containing the smoothed
        (moving average) values.
    """
    return np.convolve(data, np.ones(window) / window, mode="valid")


# MAIN CODE.
# Lists for plotting at the end.
episode_returns_list = []
episode_lengths_list = []
losses_list = []

writer = SummaryWriter("runs/ppo_balance")

episode_idx = 0
update_step = 0
global_step = 0

best_reward = -float("inf")

if __name__ == "__main__":
    # print("Starting PPO training setup")
    # print("CUDA available:", torch.cuda.is_available())

    # Automatically use CUDA or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the return and episode length.
    episode_return = 0
    episode_length = 0

    # Create an instance of the balance environment class.
    env = G1BalanceEnv(render_mode=None)
    obs, _ = env.reset()

    # Get the observation and action dimensions.
    obs_dim = obs.shape[0]
    action_dim = env.action_dim

    # Uncomment this section to test FC stuff.
    # policy_net = FCNetwork(obs_dim, action_dim).to(device)

    # print("Model initialized.")
    # print(policy_net)

    # Input observation tensor to be used in forward pass.
    # obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

    # # Call the forward pass in the class with the input observation tensor.
    # action = policy_net(obs_tensor)
    # print("Action output:", action)

    # Initialize the model.
    model = ActorCritic(obs_dim, action_dim).to(device)

    checkpoint_path = "checkpoints/ppo_model_latest.pth"

    RESET_TRAINING = False  # Set this to True when you want to start the model from scratch.

    if os.path.exists(checkpoint_path) and not RESET_TRAINING:
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded existing model.")
    else:
        print("Training from scratch.")
    # print(model)

    # ADAM optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Adjust these values as needed.
    NUM_ITERATIONS = 500  # Number of training loops. LOWER THIS FOR TEST RUNS, THIS TAKES A WHILE.
    ROLLOUT_STEPS = 1000  # Number of rollout steps.
    NUM_EPOCHS = 4  # Number of epochs for PPO updates/losses.

    # Display a progress bar for training iterations in the terminal during training.
    pbar = trange(NUM_ITERATIONS, desc="PPO Training")

    for iteration in pbar:
        # Initialize the buffer.
        buffer = RolloutBuffer()
        # Start a new state by resetting the environment.
        state, _ = env.reset()

        # Collect the rollout and display a progress bar for episodes.
        for step in trange(ROLLOUT_STEPS, desc=f"Rollout {iteration}"):
            # Convert the state to a tensor.
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)

            # Get the action from the model.
            action, log_prob, value = model.act(state_tensor)

            # Convert the action to NumPy for the environment.
            # MuJoCo DOES NOT USE TENSORS!!!
            action_np = action.detach().cpu().numpy()

            # Step environment.
            next_state, reward, done, trunc, _ = env.step(action_np)

            # Log the per-step metrics.
            # Log the smoothed average reward.
            writer.add_scalar("Reward/PerStep", reward, global_step)
            # Log the action magnitude (used for detecting instability).
            writer.add_scalar("Action/Magnitude", action.norm().item(), global_step)
            # Log the value estimate (check the critic).
            writer.add_scalar("Value/Estimate", value.item(), global_step)
            global_step += 1

            # Add to the episode return and length.
            episode_return += reward
            episode_length += 1

            # Store the data in the buffer.
            buffer.add(
                state_tensor, action.detach(), reward, done, log_prob.detach(), value.detach()
            )

            # Move to the next state.
            state = next_state
            # print(f"Step {step}: Reward={reward:.4f}")

            # Episode boundary check.
            if done or trunc:
                # Log the episode metrics before resetting.
                writer.add_scalar("Reward/Episode", episode_return, episode_idx)
                writer.add_scalar("EpisodeLength", episode_length, episode_idx)

                # Best model check.
                if episode_return > best_reward:
                    best_reward = episode_return
                    torch.save(model.state_dict(), "checkpoints/ppo_best.pth")
                    tqdm.write(f"New best model. Reward: {best_reward:.2f}")

                # Store values for plotting at the end.
                episode_returns_list.append(episode_return)
                episode_lengths_list.append(episode_length)

                # Reset the environment.
                # print("Episode ended, resetting environment.")
                state, _ = env.reset()
                # Reset the episode return and length.
                episode_return = 0
                episode_length = 0
                episode_idx += 1

        # Compute the GAE values (advantages and returns).
        advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones)

        # Lists must be converted to stacked tensors before computing loss.
        states = torch.stack(buffer.states)
        actions = torch.stack(buffer.actions)
        old_log_probs = torch.stack(buffer.log_probs)
        advantages = torch.stack(advantages)
        returns = torch.stack(returns)

        # Normalize the advantages for stability.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO updates over multiple epochs.
        # List of losses for average at the end.
        losses = []
        for epoch in range(NUM_EPOCHS):
            # Compute the loss.
            loss = update_ppo(model, optimizer, states, actions, old_log_probs, returns, advantages)
            losses.append(loss)
            # Print the loss for each epoch.
            # print(f"PPO Epoch {epoch}: Loss = {loss:.4f}")

            # Log the PPO training loss.
            writer.add_scalar("Loss/PPO", loss, update_step)
            update_step += 1

            # Add to the losses list for plotting at the end (this is
            # different from the losses list for the average).
            losses_list.append(loss)

        # Find the average loss.
        avg_loss = sum(losses) / len(losses)
        # print(f"Iteration {iteration} | Avg Loss: {avg_loss:.4f}")
        # Display the average loss and best reward on the progress bar.
        pbar.set_postfix({"avg_loss": f"{avg_loss:.2f}", "best_reward": f"{best_reward:.2f}"})

        # Save checkpoints for the model.
        os.makedirs("checkpoints", exist_ok=True)
        # Save the trained model, overwrite the latest data.
        torch.save(model.state_dict(), checkpoint_path)

        # Save versioned checkpoints every 50 iterations.
        if iteration % 50 == 0 and iteration > 0:
            torch.save(model.state_dict(), f"checkpoints/ppo_model_iter_{iteration}.pth")

    # Clear the buffer by calling the clear function. Otherwise, data
    # will continue accumulating.
    buffer.clear()

    print("Data collection complete.")

    os.makedirs("plots", exist_ok=True)

    # Episode return plot.
    plt.figure()
    plt.plot(episode_returns_list)
    plt.title("Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid()
    plt.savefig("plots/episode_return.png")

    # Episode return plot (SMOOTHED).
    plt.figure()
    plt.plot(find_moving_average(episode_returns_list))
    plt.title("Episode Return (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid()
    plt.savefig("plots/episode_return_smoothed.png")

    # Episode length plot.
    plt.figure()
    plt.plot(episode_lengths_list)
    plt.title("Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid()
    plt.savefig("plots/episode_length.png")

    # Loss plot.
    plt.figure()
    plt.plot(losses_list)
    plt.title("PPO Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("plots/loss.png")

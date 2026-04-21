import torch
import torch.nn as nn
# Import the balance environment class from g1_balance_env.py.
from rl.envs.g1_balance_env import G1BalanceEnv
from rl.ppo.model import ActorCritic
from rl.ppo.buffer import RolloutBuffer
from rl.ppo.trainer import compute_gae, update_ppo

# Run this in the terminal before running this file:
#export PYTHONPATH=.
#python3 rl/train.py

# The FCNetwork class is just for debugging/testing purposes. It shows how many input and output
# dimensions we'll need to work with.
# class FCNetwork(nn.Module):
#     '''A Fully Connected Neural Network class.'''
#     def __init__(self, input_dim: int, output_dim: int) -> None:
#         '''Initializer function for the FC network class.

#         Args:
#             input_dim (int): The dimension of the input observation vector.
#             output_ dim (int): The dimensin of the output (actions).
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

# MAIN CODE.

if __name__ == "__main__":

    # print("Starting PPO training setup")
    # print("CUDA available:", torch.cuda.is_available())

    # Automatically use CUDA or CPU    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the balance environment class.
    env = G1BalanceEnv(render_mode=None)

    obs, _ = env.reset()
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

    # Initialize the buffer.
    buffer = RolloutBuffer()

    print("Model initialized.")
    print(model)

    # Start state.
    state = obs

    # This value can be adjusted.
    max_steps = 1000 
    
    # Collect the rollout.
    for step in range(max_steps):

        # Convert the state to a tensor.
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)

        # Get the action from the model.
        action, log_prob, value = model.act(state_tensor)

        # Convert the action to NumPy for the environment.
        # MuJoCo DOES NOT USE TENSORS!!!
        action_np = action.detach().cpu().numpy()

        # Step environment.
        next_state, reward, done, trunc, _ = env.step(action_np)

        # Store the data in the buffer.
        buffer.add(
            state_tensor,
            action.detach(),
            reward,
            done,
            log_prob.detach(),
            value.detach()
        )

        # Move to the next state.
        state = next_state 
        print(f"Step {step}: Reward={reward:.4f}")

        # Reset if the episode ends.
        if done or trunc:
            # print("Episode ended, resetting environment.")
            state, _ = env.reset()
    
    # Compute the GAE values (advantages and returns).
    advantages, returns = compute_gae(
            buffer.rewards,
            buffer.values,
            buffer.dones
        )

    # Lists must be converted to stacked tensors before computing loss.
    states = torch.stack(buffer.states)
    actions = torch.stack(buffer.actions)
    old_log_probs = torch.stack(buffer.log_probs)
    advantages = torch.stack(advantages)
    returns = torch.stack(returns)

    # Normalize the advantages for stability.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ADAM optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # PPO updates over multiple epochs.
    NUM_EPOCHS = 4
    # List of losses for average at the end.
    losses = []
    for epoch in range(NUM_EPOCHS):
        # Compute the loss.
        loss = update_ppo(
            model,
            optimizer,
            states,
            actions,
            old_log_probs,
            returns,
            advantages
        )
        losses.append(loss)
        # Print the loss for each epoch.
        print(f"PPO Epoch {epoch}: Loss = {loss:.4f}")
    # Find the average loss.
    avg_loss = sum(losses) / len(losses)
    print(f"Avg Loss: {avg_loss:.4f}")

    # Save the trained model.
    torch.save(model.state_dict(), "ppo_model.pth")

    # Clear the buffer by calling the clear function. Otherwise, data
    # will continue accumulating.
    buffer.clear()

    print("Data collection complete.")
import torch
import torch.nn as nn

# Run this in the terminal before running this file:
#export PYTHONPATH=.
#python3 rl/train.py

class FCNetwork(nn.Module):
    '''A Fully Connected Neural Network class.'''
    def __init__(self, input_dim: int, output_dim: int) -> None:
        '''Initializer function for the FC network class.

        Args:
            input_dim (int): The dimension of the input observation vector.
            output_ dim (int): The dimensin of the output (actions).
        '''
        # Initialize the parent nn.Module class for layers and parameters.
        super().__init__()
        # The FC feedforward neural network, 128 neurons for applicable layers.
        # Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
        self.net = nn.Sequential(
            # Input layer.
            nn.Linear(input_dim, 128),
            # Hidden layers.
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            # Output layer.
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Function for the forward pass.
        
        Args:
            x (torch.Tensor): Input observation tensor.

        Returns:
            torch.Tensor: A forward pass of x through the FC network.
        '''
        return self.net(x)

# Main code.

if __name__ == "__main__":
    # Import the balance environment class from g1_balance_env.py.
    from rl.envs.g1_balance_env import G1BalanceEnv

    # print("Starting PPO training setup")
    # print("CUDA available:", torch.cuda.is_available())

    # UNCOMMENT THIS IF YOU ARE A CUDA USER.    
    # device = torch.device("cuda") 

    # UNCOMMENT THIS IF YOU ARE NOT A CUDA USER.
    device = torch.device("cpu") 

    # Create an instance of the balance environment class.
    env = G1BalanceEnv(render_mode=None)

    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_dim

    policy_net = FCNetwork(obs_dim, action_dim).to(device)

    print("Model initialized.")
    print(policy_net)

    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    action = policy_net(obs_tensor)
    print("Action output:", action)
# Used to expand the FC network in train.py, formalizes it for PPO.
# This file might contain a policy network, a value network, or a combined actor-critic model.

import torch
import torch.nn as nn


class Actor(nn.Module):
    """The actor class for PPO."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        """The initializer function for the Actor class.

        Args:
            obs_dim (int): The observation dimension (number of numbers in robot state vector).
            The robot's state is described by 71 numbers at each timestep:
            x = [x1, x2, x3, ..., x71]
            action_dim (int): The action dimension (number of control signals the robot outputs).
            The robot has 29 controllable actuators:
            a = [a1, a2, ..., a29]
        """
        # Initialize the parent nn.Module class for layers and parameters.
        super().__init__()
        # Feature extractor: convert 71 raw values into 128 learned features.
        # Input -> Linear -> ReLU -> Linear -> ReLU -> Output
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # Convert features into mean action values.
        self.mean = nn.Linear(128, action_dim)
        # Define uncertainty of policy for each action.
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Function for the forward pass.

        Args:
            x (torch.Tensor): Input observation tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of the mean and std.
        """
        # Extract the features (forward pass through network).
        x = self.net(x)
        # Find the mean.
        mean = self.mean(x)
        # Find the standard deviation.
        std = torch.exp(self.log_std)
        return mean, std


class Critic(nn.Module):
    """The critic class for PPO."""

    def __init__(self, obs_dim: int) -> None:
        """The initializer function for the Critic class.

        Args:
            obs_dim (int): The observation dimension (number of numbers in robot state vector).
        """
        # Initialize the parent nn.Module class for layers and parameters.
        super().__init__()

        # Network to determine how good the robot state is.
        # Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # 1 state value.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Function for the forward pass.

        Args:
            x (torch.Tensor): Input observation tensor.

        Returns:
            torch.Tensor: A forward pass of x through the FC network.
        """
        return self.net(x)


class ActorCritic(nn.Module):
    """The actor-critic class for PPO."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        """The initializer function for the ActorCritic class.

        Args:
            obs_dim (int): The observation dimension (number of numbers in robot state vector).
            The robot's state is described by 71 numbers at each timestep:
            x = [x1, x2, x3, ..., x71]
            action_dim (int): The action dimension (number of control signals the robot outputs).
            The robot has 29 controllable actuators:
            a = [a1, a2, ..., a29]
        """
        # Initialize the parent nn.Module class for layers and parameters.
        super().__init__()

        # Call the Actor and Critic classes and store them as instances here.
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)

    def forward(self, x: torch.Tensor) -> None:
        """A forward pass is not done here.

        Args:
            x (torch.Tensor): Input observation tensor.
        """
        # In PPO, the model is not a single forward mapping. There's two different
        # behaviors: acting and training evaluation (handled by the helper methods).
        # PPO needs stochastic sampliung, logits, value estimates, and entropy, which
        # cannot be represented with a single forward pass here.
        raise NotImplementedError("Use act() or evaluate() instead")

    # Helper methods:
    # Action sampling (environment interaction).
    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Function for action sampling/interacting with the environment.

        Args:
            x (torch.Tensor): Input observation tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of the action, logit, and value.
        """
        # Call the actor network and find the mean and std.
        mean, std = self.actor(x)
        # Create a probability distribution object.
        dist = torch.distributions.Normal(mean, std)
        # Sample an action.
        action = dist.sample()
        # Determine the per-dimension log probability and combine into one scalar:
        # determine how likely the action just sampled is under the policy.
        log_prob = dist.log_prob(action).sum(dim=-1)
        # Call the critic network to determine how good the state is overall (state evaluation).
        value = self.critic(x)
        # Return decision (action), probability (log_prob), and state evaluation (value).
        return action, log_prob, value

    # Evaluation (for PPO update).
    def evaluate(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Function for evaluation/training the policy (PPO update step).

        Args:
            x (torch.Tensor): Input observation tensor.
            action (torch.Tensor): Action sample tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of the logit, value, and entropy.
        """
        # Call the actor network and find the mean and std.
        # The mean is the preferred action per joint, while
        # the std is the uncertainty per action.
        mean, std = self.actor(x)
        # Recreate the same policy distribution, but without sampling.
        dist = torch.distributions.Normal(mean, std)
        # Determine the per-dimension log probability and combine into one scalar:
        # determine how likely the past action taken was under the current policy.
        log_prob = dist.log_prob(action).sum(dim=-1)
        # Measure how random the policy is.
        entropy = dist.entropy().sum(dim=-1)
        # Call the critic network to determine how good the state is overall (state evaluation).
        value = self.critic(x)
        # Return probability (log_prob), state evaluation (value) and policy randomness (entropy).
        return log_prob, value, entropy

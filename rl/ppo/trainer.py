# File to actually do the learning. Can compute advantages (GAE), compute PPO loss, run backpropagation
# (loss.backward()), update model weights.

import torch
from model import ActorCritic

def compute_gae(
    rewards: list[float], 
    values: list[torch.Tensor], 
    dones: list[bool], 
    gamma: float=0.99, 
    lam: float=0.95) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    '''Function to compute the Generalized Advantage Estimation (GAE).

    Args:
        rewards (list[float]): A list of scalar rewards from the environment.
        values (list[torch.Tensor]): A list of critic value estimates.
        dones (list[bool]): A list of booleans indicating if the episode terminated or not.
        gamma (float): Hyperparameter value.
        lam (float): Hyperparameter value.
    
    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: A tuple of the advantages and returns lists.
    '''
    # Initialize the advantages list.
    advantages = []
    gae = 0

    # Append a final value of 0 for the terminal state.
    values = values + [torch.zeros_like(values[0])]

    # Iterate backward through the rewards and compute GAE. 
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]

        # (1 - dones[t]) is needed to prevent leakage through episodes. It prevents
        # future reward contribution if the episode ends.
        # Compute GAE and add to the advantages list.
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    # Compute the returns.
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]

    return advantages, returns

def update_ppo(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    states: list[torch.Tensor],
    actions: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    returns: list[torch.Tensor],
    advantages: list[torch.Tensor],
    clip_eps: float=0.2) -> float:
    '''Function to update model parameters with PPO loss.

    Args:
        model (ActorCritic): ActorCritic network from model.py (for policy and value function).
        optimizer (torch.optim.Optimizer): ADAM optimizer used for gradient descent.
        states (list[torch.Tensor]): A list of the observation tensors.
        actions (list[torch.Tensor]): A list of the action tensors.
        old_log_probs (list[torch.Tensor]): A list of the log probabilities under the old policy.
        returns (list[torch.Tensor]): The discounted returns from compute_gae.
        advantages (list[torch.Tensor]): The advantages from compute_gae.
        clip_eps (float): The PPO clipping parameter that controls policy update range.

    Returns:
        float: The total loss value after the update. 
    '''
    # Evaluate the current policy.
    new_log_probs, values, entropy = model.evaluate(states, actions)

    # Compute the ratio.
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Compute the clipped actor loss.
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # Compute the critic loss.
    critic_loss = (returns - values).pow(2).mean()

    # Compute the entropy loss.
    entropy_loss = -entropy.mean()

    # Compute the total loss.
    loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
    
    # Backpropagation.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
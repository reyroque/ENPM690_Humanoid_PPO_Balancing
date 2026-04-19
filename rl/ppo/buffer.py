# File to store what the agent collects, such as states (observations), actions, rewards, log probabilities,
# and done flags. Required since PPO is on-policy but batch-based.

class RolloutBuffer:
    '''Class to store various data.'''
    def __init__(self) -> None:
        '''Initializer function for the RolloutBuffer class.'''
        # Observation (71 dimensions, from input).
        self.states = []
        # Sampled action (29 dimensions, from output).
        self.actions = []
        # The scalar from env.
        self.rewards = []
        # If the episode has ended. 
        self.dones = []
        # Log probabilities from model.py's act().
        self.log_probs = []
        # The critic estimate.
        self.values = []

    def add(
        self, 
        state: torch.Tensor, # Shape is (obs_dim,).
        action: torch.Tensor, # Shape is (action_dim,).
        reward: float, 
        done: bool, 
        log_prob: torch.Tensor, 
        value: torch.Tensor) -> None:
        '''Function to add data to the lists.
        
        Args:
            state (torch.Tensor): Observation tensor, same shape as [obs_dim].
            action (torch.Tensor): Action tensor, same shape as [action_dim].
            reward (float): The scalar reward from the environment.
            done (bool): A boolean indicating if the episode is done or not.
            log_prob (torch.Tensor): The log probability of the action.
            value (torch.Tensor): The critic value estimate.
        '''
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self) -> None:
        '''Function to clear the buffer after training.'''
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
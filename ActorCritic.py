import torch
import torch.nn as nn
from torch.distributions import Categorical


### PPO policy
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = [] 
        self.logprobs = [] # log xác suất của hành động theo policy cũ , giúp kiểm soát mức độ thay đổi của policy.
        self.state_values = [] # Tính Advantage => giúp cập nhật Critic
        self.is_terminate = []
        
    def clean(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.is_terminate[:]
        
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
    
        
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten(),
          
            nn.Linear(in_features=8192, out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Softmax(dim=-1)
            
            
        )
        
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten(),
            
            nn.Linear(in_features=8192, out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1)
        )
        
    
    def forward(self):
       NotImplemented
    
    
    def action(self, state, mask):
        action_probs = self.actor(state)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(action_probs.size(0), -1) # đảm bảo shape của mask và action_probs là giống nhau
        action_probs = torch.mul(action_probs, mask)
        
        dist = Categorical(action_probs)
        action = torch.argmax(action_probs, dim=1)
        # action = dist.sample()
        
        print(action.item())
        
        action_logprobs = dist.log_prob(action)
        state_val = self.critic(state)
        
        # action_probs = action_probs.squeeze(0)
        # max_value, max_index = torch.max(action_probs, dim=0)
        # print("Index of max value:", max_index.item())
        # print("Max value:", max_value.item())
        
        # print(action_probs[2839])
        # print(state_val.item())
        
        return action.detach(), action_logprobs.detach(), state_val.detach()
    
    def evaluate(self, states, actions):
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        
        action_log_probs = dist.log_prob(actions)
        
        dist_entropy = dist.entropy()
        state_values = self.critic(states)
        
        return action_log_probs, state_values, dist_entropy
        
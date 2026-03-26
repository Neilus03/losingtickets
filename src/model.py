import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features)
        
        # SIREN-specific initialization
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = np.sqrt(6 / in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, out_features=3, hidden_layers=4, omega_0=30.0):
        super().__init__()
        
        net = []
        
        # Input layer
        net.append(SineLayer(in_features, hidden_features, omega_0=omega_0, is_first=True))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            net.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0, is_first=False))
            
        self.net = nn.Sequential(*net)
        
        # Output layer (no sine activation)
        self.last_linear = nn.Linear(hidden_features, out_features)
        
        # Output layer initialized as hidden layers of SIREN
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / omega_0
            self.last_linear.weight.uniform_(-bound, bound)
            
    def forward(self, x):
        x = self.net(x)
        x = self.last_linear(x)
        return x

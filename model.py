# File: model.py
"""Custom CNN model for processing Melting Pot observations."""

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CustomCNNModel(TorchModelV2, nn.Module):
    """Custom CNN Model for processing Melting Pot observations."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Initialize the CNN model."""
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        linear_input_size = self._get_conv_out_size((88, 88, 3))

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_outputs)
        self.vf_fc1 = nn.Linear(linear_input_size, 512)
        self.vf_fc2 = nn.Linear(512, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input_dict, state, seq_lens):
        """Forward pass of the model."""
        x = input_dict["obs"]["RGB"].float().to(self.device)
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        self._features = x
        x = torch.relu(self.fc1(x))
        policy = self.fc2(x)
        return policy, state

    def value_function(self):
        """Compute the value function."""
        x = torch.relu(self.vf_fc1(self._features))
        return self.vf_fc2(x).squeeze(1)

    def _get_conv_out_size(self, shape):
        """Calculate the output size of the convolutional layers."""
        o = self.conv1(torch.zeros(1, *shape).permute(0, 3, 1, 2))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

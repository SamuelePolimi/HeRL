import torch.nn as nn
import torch
import numpy as np

from herl.rl_interface import RLTask, RLAgent
from herl.config import torch_type


class NeuralNetwork(nn.Module):
    def __init__(self, h_layers, act_functions, rl_task, output_function=None):
        """

        :param h_layers: Each entry represents the no of neurons in the corresponding hidden layer.
        :type h_layers: list
        :param act_functions: list of activation functions for each input layer. len = len(layers) - 2
        :type act_functions: list
        :param output_function: Function to be applied to output layer. None if output is linear.
        :type output_function: builtin_function
        :param rl_task: Used to infer input_dim and output_dim
        :type rl_task: RLTask

        """

        super(NeuralNetwork, self).__init__()
        self.input_dim = rl_task.environment.state_dim
        self.hidden = nn.ModuleList([nn.Linear(self.input_dim, h_layers[0])])
        self.output_dim = rl_task.environment.action_dim

        for input_size, output_size in zip(h_layers, h_layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

        self.hidden.append(nn.Linear(h_layers[-1], self.output_dim))
        self.hidden = self.hidden.to(dtype=torch_type)

        for layer in self.hidden:
            nn.init.xavier_uniform_(layer.weight.data)
            nn.init.xavier_uniform_(layer.bias.data[None])

        self.act_functions = act_functions
        self.out_function = output_function

    def forward(self, x):
        """

        :param x:  Input feature
        :type x: tensor
        :return: Tensor of shape len(x)*output_dim
        """
        x = x.to(dtype=torch_type)
        l_network = len(self.hidden)
        for (i, linear_transform) in list(zip(range(l_network), self.hidden))[:-1]:
            x = self.act_functions[i](linear_transform(x))
        if self.out_function is None:
            x = self.hidden[-1](x)
        else:
            x = self.out_function(self.hidden[-1](x))
        return x


class NeuralNetworkPolicy(RLAgent, NeuralNetwork):

    def __init__(self, h_layers, act_functions, rl_task, output_function=None):
        RLAgent.__init__(self, deterministic=True)
        NeuralNetwork.__init__(self, h_layers, act_functions, rl_task, output_function)

    def __call__(self, state, differentiable=False):
        if differentiable:
            return NeuralNetwork.__call__(self, state)
        else:
            return NeuralNetwork.__call__(self, torch.tensor(state, dtype=torch_type)).detach().numpy()

    def get_action(self, state):
        return NeuralNetwork.__call__(self, torch.tensor(state, dtype=torch_type)).detach().numpy()

    def save_model(self, path):
        """
        Saves the neural network parameters.
        :param path:
        :type dir: str
        :return:
        """

        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        Loads neural network parameters:
        :param filename
        :type filename: str
        :return:
        """
        self.load_state_dict(torch.load(path))

    def get_parameters(self):
        state_dict = self.state_dict()

        params = []
        for name, param in state_dict.items():
            # Transform the parameter as required.
            params.append(param.numpy().ravel())

        return np.concatenate(params)

    def set_parameters(self, values):
        state_dict = self.state_dict()

        i = 0
        for name, param in state_dict.items():
            shape = param.shape
            n = torch.prod(torch.tensor(shape)).numpy()
            # Transform the parameter as required.
            state_dict[name].copy_(torch.tensor(values[i:i+n], dtype=torch_type).reshape(shape))
            i+=n

    def get_gradient(self):
        grads = []
        for param in self.parameters():
            grads.append(param.grad.detach().numpy().ravel())

        return np.concatenate(grads)


class ConstantPolicy(RLAgent):

    def __init__(self, action=0.):
        super().__init__()
        self._action = action

    def __call__(self, state, differentiable=False):
        if differentiable:
            raise torch.tensor(self._action, dtype=torch_type)
        else:
            return self._action

    def get_action(self, state):
        return self._action

    def is_deterministic(self):
        return True


class UniformPolicy(RLAgent):

    def __init__(self, low, high):
        super().__init__()
        self._low = low
        self._high = high

    def __call__(self, state, differentiable=False):
        if differentiable:
            raise torch.tensor(np.random.uniform(self._low, self._high), dtype=torch_type)
        else:
            return np.random.uniform(self._low, self._high)

    def get_action(self, state):
        return np.random.uniform(self._low, self._high)

    def is_deterministic(self):
        return False


class LinearPolicy(nn.Module, RLAgent):

    def __init__(self, inputSize, outputSize, diagonal=False, device=None):
        nn.Module.__init__(self)
        RLAgent.__init__(self, deterministic=True)
        self.device = device
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=False).to(dtype=torch_type)

    def forward(self, x):
        out = self.linear(x)
        return out

    def __call__(self, state, differentiable=False):
        if differentiable:
            return nn.Module.__call__(self, state)
        else:
            return nn.Module.__call__(self, torch.tensor(state, dtype=torch_type)).detach().numpy()

    def get_parameters(self):
        state_dict = self.state_dict()

        params = []
        for name, param in state_dict.items():
            # Transform the parameter as required.
            params.append(param.numpy().ravel())

        return np.concatenate(params)

    def set_parameters(self, values):
        state_dict = self.state_dict()

        i = 0
        for name, param in state_dict.items():
            shape = param.shape
            n = torch.prod(torch.tensor(shape)).numpy()
            # Transform the parameter as required.
            state_dict[name].copy_(torch.tensor(values[i:i+n], dtype=torch_type).reshape(shape))
            i+=n

    def get_gradient(self):
        grads = []
        for param in self.parameters():
            grads.append(param.grad.detach().numpy().ravel())

        return np.concatenate(grads)
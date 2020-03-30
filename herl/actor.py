import torch.nn as nn
import torch
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


class Actor(RLAgent):

    def __init__(self, h_layers, act_functions, rl_task, output_function=None):
        RLAgent.__init__(self)
        self.nn = NeuralNetwork(h_layers, act_functions, rl_task, output_function)

    def __call__(self, state, differentiable=False):
        return self.nn(state)

    def get_action(self, state):
        return self.nn(torch.tensor(state, dtype=torch_type)).detach().numpy()
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn import Parameter
import torch
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

from herl.rl_interface import RLTask, RLAgent, RLParametricAgent
from herl.config import torch_type


class DiagonalLinear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(DiagonalLinear, self).__init__()
        if in_features != out_features:
            raise Exception("Input and output must have same dimension for this layer")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight, -1., 1.)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, torch.diag(self.weight), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class NeuralNetwork(nn.Module, RLParametricAgent):
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
        for f, linear_transform in zip(self.act_functions, self.hidden):
            x = f(linear_transform(x))
        if self.out_function is None:
            x = self.hidden[-1](x)
        else:
            x = self.out_function(self.hidden[-1](x))
        return x

    def save(self, path):
        """
        Saves the neural network parameters.
        :param path:
        :type dir: str
        :return:
        """

        torch.save(self.state_dict(), path)

    def load(self, path):
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


class NeuralNetworkPolicy(NeuralNetwork):

    def __init__(self, h_layers, act_functions, rl_task, output_function=None):
        RLAgent.__init__(self, deterministic=True)
        NeuralNetwork.__init__(self, h_layers, act_functions, rl_task, output_function)

    def __call__(self, state, differentiable=False):
        if differentiable:
            return NeuralNetwork.__call__(self, state)
        else:
            with torch.no_grad():
                return NeuralNetwork.__call__(self, torch.from_numpy(state)).numpy()

    def get_action(self, state):
        return NeuralNetwork.__call__(self, torch.tensor(state, dtype=torch_type)).detach().numpy()

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


class LinearPolicy(NeuralNetwork):

    def __init__(self, inputSize, outputSize, diagonal=False, device=None):
        nn.Module.__init__(self)
        RLAgent.__init__(self, deterministic=True)
        self.device = device
        if diagonal:
            self.linear = DiagonalLinear(inputSize, outputSize, bias=False).to(dtype=torch_type)
        else:
            self.linear = torch.nn.Linear(inputSize, outputSize, bias=False).to(dtype=torch_type)

    def forward(self, x):
        out = self.linear(x)
        return out

    def __call__(self, state, differentiable=False):
        if differentiable:
            return NeuralNetwork.__call__(self, state)
        else:
            with torch.no_grad():
                return NeuralNetwork.__call__(self, torch.from_numpy(state)).numpy()


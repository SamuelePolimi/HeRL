import torch.nn as nn
import torch.distributions
from torch.nn.modules import Module
from torch.nn import Parameter
import torch
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
from scipy.stats import multivariate_normal
from typing import Callable, List, Union

from herl.rl_interface import RLAgent, RLParametricModel, RLEnvironmentDescriptor
from herl.config import torch_type
from herl.classic_envs import MDP
from herl.utils import _one_hot, _decode_one_hot


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


class NeuralNetwork(nn.Module, RLParametricModel):
    def __init__(self, inputs: List[int],
                 h_layers: List[int],
                 act_functions: List[Callable],
                 output_function: Callable = None):
        """

        :param h_layers: Each entry represents the no of neurons in the corresponding hidden layer.
        :param act_functions: list of activation functions for each input layer. len = len(layers) - 2
        :param output_function: Function to be applied to output layer. None if output is linear.
        :type output_function: builtin_function

        """

        nn.Module.__init__(self)

        self.input_dim = sum(inputs)

        self.hidden = nn.ModuleList([nn.Linear(self.input_dim, h_layers[0])])

        for input_size, output_size in zip(h_layers[:-1], h_layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

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
        # if self.out_function is None:
        #     x = self.hidden[-1](x)
        # else:
        if self.out_function is not None:
            x = self.out_function(self.hidden[-1](x))
        return x

    def __call__(self, *args, differentiable=False):
        # TODO: temporary solution len(args)
        if differentiable:
            if len(args) > 1:
                return nn.Module.__call__(self, torch.cat(args, 1))
            else:
                return nn.Module.__call__(self, args[0])
        else:
            with torch.no_grad():
                if len(args) > 1:
                    return nn.Module.__call__(self, torch.from_numpy(np.concatenate(args, axis=1))).numpy()
                else:
                    return nn.Module.__call__(self, torch.from_numpy(args[0])).numpy()

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


class NeuralNetworkPolicy(NeuralNetwork, RLAgent):

    def __init__(self, rl_environment_descriptor: RLEnvironmentDescriptor,
                 h_layers: List[int], act_functions: List[Callable], output_function=None):
        RLAgent.__init__(self, deterministic=True)
        NeuralNetwork.__init__(self,
                               [rl_environment_descriptor.state_dim],
                               h_layers + [rl_environment_descriptor.action_dim],
                               act_functions, output_function)

    def get_action(self, state):
        return NeuralNetwork.__call__(self, torch.tensor(state, dtype=torch_type)).detach().numpy()


class SoftMaxNeuralNetworkPolicy(NeuralNetwork, RLAgent):

    def __init__(self, rl_environment_descriptor: RLEnvironmentDescriptor,
                 h_layers: List[int], act_functions: List[Callable], output_function=None):
        # TODO: currently works only with one hot encoding
        RLAgent.__init__(self, deterministic=True)
        NeuralNetwork.__init__(self,
                               [rl_environment_descriptor.state_dim],
                               h_layers + [rl_environment_descriptor.action_dim],
                               act_functions, output_function)
        self._n_classes = rl_environment_descriptor.action_dim

    def get_vector_probabilities(self, state, differentiable=False):
        net_input = state
        if not differentiable:
            net_input = torch.tensor(state)
        p = NeuralNetwork.__call__(self, net_input, differentiable=True)
        if len(p.shape)== 1:
             ret = torch.exp(p)/torch.sum(torch.exp(p))
        else:
             ret = torch.exp(p)/torch.sum(torch.exp(p), dim=1)
        if not differentiable:
            return ret.detach().numpy()
        return ret

    def get_prob(self, state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor], differentiable: bool=False) -> Union[np.ndarray, torch.Tensor]:
        p = self.get_vector_probabilities(state, differentiable=differentiable)
        a_index = _decode_one_hot(action)
        return p[a_index]

    def get_action(self, state):
        p = self.get_vector_probabilities(state, differentiable=False)
        return _one_hot(np.random.choice(range(len(p)), p=p), self._n_classes)



class FixedGaussianNeuralNetworkPolicy(NeuralNetwork, RLAgent):

    def __init__(self, rl_environment_descriptor: RLEnvironmentDescriptor,
                 h_layers: List[int], act_functions: List[Callable], covariance: np.ndarray, output_function=None):
        RLAgent.__init__(self, deterministic=False)
        NeuralNetwork.__init__(self,
                               [rl_environment_descriptor.state_dim],
                               h_layers + [rl_environment_descriptor.action_dim],
                               act_functions, lambda x: x)
        self._cov = covariance
        self._a_dim = covariance.shape[0]
        self._output_function = output_function

    def get_action(self, state):
        return NeuralNetwork.__call__(self, torch.tensor(state, dtype=torch_type), differentiable=True).detach().numpy()

    def __call__(self, *args, differentiable=False):
        if len(args[0].shape) == 2:
            noise = np.random.multivariate_normal(np.zeros(self._a_dim), self._cov, args[0].shape[0])
        else:
            noise = np.random.multivariate_normal(np.zeros(self._a_dim), self._cov)

        if differentiable:
            noise = torch.from_numpy(noise)
            if self._output_function is not None:
                return self._output_function(NeuralNetwork.__call__(self, args[0], differentiable=differentiable) + noise)
            else:
                return NeuralNetwork.__call__(self, args[0], differentiable=differentiable)
        else:
            if self._output_function is not None:
                return self._output_function(
                    torch.tensor(NeuralNetwork.__call__(self, args[0], differentiable=differentiable))
                    + torch.from_numpy(noise)).detach().numpy()
            else:
                return NeuralNetwork.__call__(self, args[0], differentiable=differentiable) + noise





    def get_prob(self, state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor],
                 differentiable: bool=False) -> Union[np.ndarray, torch.Tensor]:
        if differentiable:
            n = torch.distributions.multivariate_normal.MultivariateNormal(loc=NeuralNetwork.__call__(self, state, differentiable=True),
                                                                          covariance_matrix=torch.from_numpy(self._cov))
            return torch.exp(n.log_prob(action))
        else:
            if len(state.shape) == 1:
                return multivariate_normal.pdf(action, mean=NeuralNetwork.__call__(self, state), cov=self._cov)
            else:
                return np.array([multivariate_normal.pdf(a, mean=NeuralNetwork.__call__(s),
                                               cov=self._cov) for s, a in zip(state, action)])


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


class LinearPolicy(NeuralNetwork, RLAgent):

    def __init__(self, inputSize, outputSize, diagonal=False, device=None):
        nn.Module.__init__(self)
        RLAgent.__init__(self, deterministic=True)
        self._diagonal = diagonal
        self.device = device
        if diagonal:
            self.linear = DiagonalLinear(inputSize, outputSize, bias=False).to(dtype=torch_type)
        else:
            self.linear = torch.nn.Linear(inputSize, outputSize, bias=False).to(dtype=torch_type)

    def is_diagonal(self):
        return self._diagonal

    def forward(self, x):
        out = self.linear(x)
        return out


class LinearGaussianPolicy(LinearPolicy):

    def __init__(self, inputSize, outputSize, covariance, diagonal=False, device=None):
        LinearPolicy.__init__(self, inputSize, outputSize, diagonal, device=device)
        if diagonal:
            self.linear = DiagonalLinear(inputSize, outputSize, bias=False).to(dtype=torch_type)
        else:
            self.linear = torch.nn.Linear(inputSize, outputSize, bias=False).to(dtype=torch_type)
        self._cov = covariance
        self._a_dim = covariance.shape[0]

    def __call__(self, *args, differentiable=False):
        if len(args[0].shape) == 2:
            noise = np.random.multivariate_normal(np.zeros(self._a_dim), self._cov, args[0].shape[0])
        else:
            noise = np.random.multivariate_normal(np.zeros(self._a_dim), self._cov)
        if differentiable:
            noise = torch.from_numpy(noise)
        return LinearPolicy.__call__(self,  args[0], differentiable=differentiable) + noise

    def get_prob(self, state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor],
                 differentiable: bool=False) -> Union[np.ndarray, torch.Tensor]:
        if differentiable:
            n = torch.distributions.multivariate_normal.MultivariateNormal(loc=LinearPolicy.__call__(self, state, differentiable=True),
                                                                          covariance_matrix=torch.from_numpy(self._cov))
            return torch.exp(n.log_prob(action))
        else:
            if len(state.shape) == 1:
                return multivariate_normal.pdf(action, mean=LinearPolicy.__call__(self, state), cov=self._cov)
            else:
                return np.array([multivariate_normal.pdf(a, mean=LinearPolicy.__call__(self, s),
                                               cov=self._cov) for s, a in zip(state, action)])

    def get_log_prob(self, state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor],
                 differentiable: bool=False) -> Union[np.ndarray, torch.Tensor]:
        if differentiable:
            n = torch.distributions.multivariate_normal.MultivariateNormal(loc=LinearPolicy.__call__(self, state, differentiable=True),
                                                                          covariance_matrix=torch.from_numpy(self._cov))
            return n.log_prob(action)
        else:
            if len(state.shape) == 1:
                return multivariate_normal.logpdf(action, mean=LinearPolicy.__call__(self, state), cov=self._cov)
            else:
                return np.array([multivariate_normal.logpdf(a, mean=LinearPolicy.__call__(s),
                                               cov=self._cov) for s, a in zip(state, action)])

    def get_grad_log_prob(self, state, action):
        return - (state * self.get_parameters() - action)/np.diag(self._cov) * state

    def is_deterministic(self):
        return False


class GaussianNoisePerturber(RLAgent):

    def __init__(self, policy: RLAgent, covariance: np.ndarray):
        RLAgent.__init__(self, False, policy.symbol)
        self._policy = policy
        self._cov = covariance
        self._a_dim = covariance.shape[0]

    def get_action(self, state):
        return self._policy.get_action + np.random.multivariate_normal(np.zeros_like(state), self._cov)

    def __call__(self, states, differentiable=False):
        if differentiable:
            raise Exception("For differentiable policies, not implementd yet.")
        if len(states.shape)==2:
            noise = np.random.multivariate_normal(np.zeros(self._a_dim), self._cov, states.shape[0])
        else:
            noise = np.random.multivariate_normal(np.zeros(self._a_dim), self._cov)
        return self._policy(states) + noise

    def get_prob(self, state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor],
                 differentiable: bool=False) -> Union[np.ndarray, torch.Tensor]:
        if differentiable:
            n = torch.distributions.multivariate_normal.MultivariateNormal(loc=self._policy(state, differentiable=True),
                                                                          covariance_matrix=torch.from_numpy(self._cov))
            return torch.exp(n.log_prob(action))
        else:
            if len(state.shape) == 1:
                return multivariate_normal.pdf(action, mean=self._policy(state), cov=self._cov)
            else:
                return np.array([multivariate_normal.pdf(a, mean=self._policy(s),
                                               cov=self._cov) for s, a in zip(state, action)])

    def get_log_prob(self, state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor],
                 differentiable: bool=False) -> Union[np.ndarray, torch.Tensor]:
        if differentiable:
            n = torch.distributions.multivariate_normal.MultivariateNormal(loc=self._policy(state, differentiable=True),
                                                                          covariance_matrix=torch.from_numpy(self._cov))
            return n.log_prob(action)
        else:
            if len(state.shape) == 1:
                return multivariate_normal.logpdf(action, mean=self._policy(state), cov=self._cov)
            else:
                return np.array([multivariate_normal.logpdf(a, mean=self._policy(s),
                                               cov=self._cov) for s, a in zip(state, action)])

    def zero_grad(self):
        return self._policy.zero_grad()

    def get_gradient(self):
        return self._policy.get_gradient()


class TabularPolicy(RLAgent, nn.Module, RLParametricModel):

    def __init__(self, mdp: MDP):

        nn.Module.__init__(self)
        RLAgent.__init__(self, mdp.get_descriptor())
        self._n_actions = len(mdp.get_actions())
        self._n_states = len(mdp.get_states())
        self._mdp = mdp

        self._m = Parameter(torch.tensor(np.random.uniform(size=self._n_states*self._n_actions), requires_grad=True))
        #
        # self.register_parameter("table", self._m)

    def forward(self, x):
        # TODO: make it work for vecrotial input
        self._M = torch.reshape(self._m, shape=(self._n_states, self._n_actions))
        self.tabular = torch.exp(self._M)/torch.sum(torch.exp(self._M), dim=1).unsqueeze(1)
        return self.tabular[x.long()]

    def __call__(self, *args, differentiable=False):
        if len(args) != 1:
            raise Exception("tabular policy needs only one argument")

        states = args[0]
        states = self._precondition_state(states)
        d = torch.distributions.Categorical(probs=nn.Module.__call__(self, states.long()))
        return d.sample()

    def get_prob(self, state: torch.Tensor,
                 action: torch.Tensor, differentiable: bool=False) -> torch.Tensor:

        s, a = self._precondition_state(state), self._precondition_actions(action)

        prob = nn.Module.__call__(self, s.view(-1))
        return prob.gather(-1, a)


    def get_log_prob(self, state: torch.Tensor,
                 action: torch.Tensor, differentiable: bool=False) -> torch.Tensor:

        return torch.log(self.get_prob(state, action, differentiable=differentiable))

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







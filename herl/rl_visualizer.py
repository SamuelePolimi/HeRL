import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
from sklearn.neighbors import KernelDensity
from typing import Callable, Any, Iterable

from herl.rl_interface import Critic, RLEnvironment, PolicyGradient, RLEnvironmentDescriptor, RLAgent
from herl.dataset import Dataset
from herl.dict_serializable import DictSerializable
from herl.rl_analysis import bias_variance_estimate


class PlotVisualizer(DictSerializable):

    class_name = "plot_visualizer"
    load_fn = DictSerializable.get_numpy_load()

    def __init__(self, plot_class_id):
        """
        This class defines a visualizer. The visualizer exposes mainly two methods: compute and visualize.
        The compute method should take care to produce the data visualized in the plot. The data must be saved in self._data.
        The visualizer plots on a (or multiple) plt.Axes objects.

        This approaches has several advantages:
           1. disentangling computation and visualization makes the code clearer.
           2. it is possible to compute only once and visualize multiple times on different ax and with different graphics properties.
           3. most importantly it is possible to save and load data from disk.
        """
        DictSerializable.__init__(self, DictSerializable.get_numpy_save())
        self._data = {'name': plot_class_id}
        self._values = False

    def compute(self, *args, **kwargs):
        """
        This methods compute the quantities to produce the plot.
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplemented()

    def visualize(self, *args, **kwargs):
        """
        This method produces the actual plot, once the data has been computed.
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplemented()

    def _standard_plot(self, ax, **graphic_args):
        return ax.plot(self._data['x'], self._data['y'], **graphic_args)

    def _standard_heat_map(self, ax, **graphic_args):
        return ax.pcolormesh(self._data['x'], self._data['y'], self._data['z'], **graphic_args)

    def _standard_general_plot(self, ax, **graphic_args):
        if self._data['d'] == 1:
            return self._standard_plot(ax, **graphic_args)
        elif self._data['d']:
            return self._standard_heat_map(ax, **graphic_args)
        else:
            raise Exception("_standard_general_plot can handle only one or tho dimensions.")

    def _standard_scatter(self, ax, **graphic_agrs):
        return ax.scatter(self._data['p_x'], self._data['p_y'], **graphic_agrs)

    def _standard_arrow(self, ax, fixed_dx=None, resize=1, **graphic_args):
        if fixed_dx is not None:
            return ax.arrow(self._data['a_x'], self._data['a_y'], fixed_dx,
                            self._data['a_dy']/self._data['a_dx']*fixed_dx, **graphic_args)
        else:
            return ax.arrow(self._data['a_x'], self._data['a_y'], self._data['a_dx']*resize,
                            self._data['a_dy']*resize, **graphic_args)

    def _check(self):
        if not self._values:
            raise Exception("The data has not been computed or loaded yet.")

    def _get_dict(self):
        return self._data

    def save(self, file_name):
        self._check()
        DictSerializable.save(self, file_name)

    @staticmethod
    def load_from_dict(**kwargs):
        name = kwargs['name']
        if name == ValueFunctionVisualizer.class_name:
            visualizer = ValueFunctionVisualizer()
        elif name == QFunctionVisualizer.class_name:
            visualizer = QFunctionVisualizer()
        elif name == ReturnLandscape.class_name:
            visualizer = ReturnLandscape()
        elif name == PolicyVisualizer.class_name:
            visualizer = PolicyVisualizer()
        else:
            raise Exception("'%s' unknown." % name)
        visualizer._data = kwargs
        visualizer._values = True
        return visualizer

    @staticmethod
    def load(file_name):
        """

        :param file_name:
        :return:
        """
        file = PlotVisualizer.load_fn(file_name)
        return PlotVisualizer.load_from_dict(**file)

    def visualize_x_label(self, ax):
        self._check()
        ax.set_xlabel(self._data['x_label'])

    def visualize_y_label(self, ax):
        self._check()
        ax.set_ylabel(self._data['y_label'])

    def visualize_title(self, ax):
        self._check()
        ax.set_title(self._data['title'])


class RowVisualizer(DictSerializable):

    class_name = "row_visualizer"

    def __init__(self, name: str, *sub_visualizer: PlotVisualizer):
        DictSerializable.__init__(self, DictSerializable.get_numpy_save())
        self.name = name
        self.sub_visualizer = list(sub_visualizer)  # TODO: check if instead is list(*sub_visualizer)

    def compute_i(self, i: int, *args, **kwargs):
        self.sub_visualizer[i].compute(*args, **kwargs)

    def visualize(self, axs: Iterable[plt.Axes], **kwargs: Any):
        ret = []
        for ax, visualizer in zip(axs, self.sub_visualizer):
            ret.append(visualizer.visualize(ax, **kwargs))
        return ret

    def _get_dict(self):
        return {'sub_visualizer': [d._get_dict() for d in self.sub_visualizer],
                'name': self.name}

    @staticmethod
    def load_from_dict(**kwargs):
        visualizer = RowVisualizer(kwargs['name'])
        visualizer.sub_visualizer = [PlotVisualizer.load_from_dict(v) for v in kwargs['sub_visualizers']]

    def visualize_decorations(self, axs: Iterable[plt.Axes]):
        y_label = ""
        for ax, visualizer in zip(axs, self.sub_visualizer):
            if visualizer._data["y_label"] != y_label:
                visualizer.visualize_y_label(ax)
            visualizer.visualize_title(ax)
            visualizer.visualize_x_label(ax)
            y_label = visualizer._data["y_label"]

    @staticmethod
    def load(file_name):
        """

        :param file_name:
        :param domain:
        :return:
        """
        file = Dataset.load_fn(file_name)
        return Dataset.load_from_dict(**file)


class ValueFunctionVisualizer(PlotVisualizer):

    class_name = "value_function_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, ValueFunctionVisualizer.class_name)

    def compute(self, env: RLEnvironment, critic: Critic, discretization: Iterable = None):
        if env.state_space.shape[0] == 1:
            self._data['d'] = 1
            self._data['x'] = np.linspace(env.state_space.low[0], env.state_space.high[0], discretization[0])
            pool = Pool(mp.cpu_count())
            self._data['y'] = pool.map(critic.get_V, self._data['x'].reshape(-1, 1))
            self._data['x_label'] = r"$%s$" % env.state_space.symbol[0]
            self._data['y_label'] = r"$V_{%s}(%s)$" % (critic.name, env.state_space.symbol[0])
            self._data['title'] = ""
        elif env.state_space.shape[0] == 2:
            self._data['d'] = 2
            dataset = env.get_grid_dataset(discretization)
            states = dataset.get_full()["state"]
            results = critic.get_V(states)
            shape = [discretization[0], discretization[1]]
            self._data['z'] = np.array(results).reshape(*shape)
            self._data['x'] = states[:, 0].reshape(*shape)
            self._data['y'] = states[:, 1].reshape(*shape)
            self._data['x_label'] = r"$%s$" % env.state_space.symbol[0]
            self._data['y_label'] = r"$%s$" % env.state_space.symbol[1]
            self._data['title'] = "$V_{%s}(%s, %s)$" % (critic.name, env.state_space.symbol[0], env.state_space.symbol[1])
        else:
            raise Exception("State space must be one or two dimensional.")
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args: Any):
        self._check()
        self._standard_general_plot(ax, **graphic_args)


class QFunctionVisualizer(PlotVisualizer):

    class_name = "q_function_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, QFunctionVisualizer.class_name)

    def compute(self, env: RLEnvironment, critic: Critic, discretization: Iterable = None):
        if env.state_space.shape[0] == 1 and env.action_space.shape[0] == 1:
            dataset = env.get_grid_dataset(discretization[0:1], discretization[1:])
            ds = dataset.get_full()
            self._data['x'], self._data['y'] = ds["state"], ds["action"]
            self._data['z'] = critic.get_Q(self._data['x'], self._data['y'])
            shape = [discretization[0], discretization[1]]
            self._data['x'] = np.array(self._data['x']).reshape(*shape)
            self._data['y'] = np.array(self._data['y']).reshape(*shape)
            self._data['z'] = np.array(self._data['z']).reshape(*shape)
            self._data['x_label'] = r"$%s$" % env.state_space.symbol[0]
            self._data['y_label'] = r"$%s$" % env.action_space.symbol[0]
            self._data['title'] = "$Q_{%s}(%s, %s)$" % (critic.name, env.state_space.symbol[0], env.action_space.symbol[0])
        else:
            raise Exception(
                "It is not possible to render an environment with total space (state + action) greater than two.")
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args: Any):
        self._check()
        return self._standard_plot(ax, **graphic_args)


class PolicyVisualizer(PlotVisualizer):

    class_name = "policy_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, PolicyVisualizer.class_name)

    def compute(self, env: RLEnvironment, policy: RLAgent, discretization: Iterable = None):
        if env.state_space.shape[0] == 1 and env.action_space.shape[0] == 1 and policy.is_deterministic():
            dataset = env.get_grid_dataset(discretization[0:1])
            ds = dataset.get_full()
            self._data['x'] = ds["state"]
            self._data['y'] = policy(self._data['x'])
            self._data['x_label'] = r"$%s$" % env.state_space.symbol[0]
            self._data['y_label'] = r"$%s(%s)$" % (policy.symbol, env.state_space.symbol[0])
            self._data['title'] = ""
            self._data['d'] = 1
        elif env.state_space.shape[0] == 2 and env.action_space.shape[0] == 1 and policy.is_deterministic():
            dataset = env.get_grid_dataset(discretization[0:2])
            ds = dataset.get_full()
            states = ds["state"]
            self._data['x'] = ds["state"][:, 0].reshape(discretization[0], discretization[1])
            self._data['y'] = ds["state"][:, 1].reshape(discretization[0], discretization[1])
            self._data['z'] = policy(states).reshape(discretization[0], discretization[1])
            self._data['x_label'] = r"$%s$" % env.state_space.symbol[0]
            self._data['y_label'] = r"$%s$" % env.state_space.symbol[1]
            self._data['title'] = r"$%s(%s, %s)$" % (policy.symbol, env.state_space.symbol[0],
                                                     env.state_space.symbol[1])
            self._data['d'] = 2
        else:
            raise Exception("At the current moment, we are only able to represent deterministic policies "
                            "(one or two state dimensions).")
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args: Any):
        self._check()
        return self._standard_general_plot(ax, **graphic_args)


class ReturnLandscape(PlotVisualizer):

    class_name = "return_landscape_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, ReturnLandscape.class_name)

    def compute(self, critic, policy, indexes, low, high, discretization, **graphic_args):
        ref_param = policy.get_parameters()
        if len(indexes) == 1:
            params = np.linspace(low[0], high[0], discretization[0])
            y = []
            for param in params.reshape(-1, 1):
                new_params = ref_param.copy()
                new_params[indexes[0]] = param
                policy.set_parameters(new_params)
                y.append(np.asscalar(critic.get_return()))
                policy.set_parameters(ref_param)
            self._data['d'] = 1
            self._data['x'] = params
            self._data['y'] = y
        elif len(indexes) == 2:
            x = np.linspace(low[0], high[0], discretization[0])
            y = np.linspace(low[1], high[1], discretization[1])
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    param = np.array([X[i, j], Y[i, j]])
                    new_params = ref_param.copy()
                    new_params[indexes[0]] = param[0]
                    new_params[indexes[1]] = param[1]
                    policy.set_parameters(new_params)
                    Z[i, j] = np.asscalar(critic.get_return())
                    policy.set_parameters(ref_param)
            self._data['x'] = X
            self._data['y'] = Y
            self._data['z'] = Z
            self._data['d'] = 2
        else:
            raise Exception("It is not possible to render an environment with state dimension greater than two.")
        policy.set_parameters(ref_param)
        self._values = True

    def visualize(self, ax, **graphic_args):
        self._check()
        self._standard_general_plot(ax, **graphic_args)


class StateCloudVisualizer(PlotVisualizer):

    class_name = "state_cloud_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, StateCloudVisualizer.class_name)

    def compute(self, dataset: Dataset, **graphic_args):
        """
        Write on ax a scatter representing the dataset
        :param ax:
        :param dataset:
        :param graphic_args:
        :return:
        """

        if dataset.domain.get_variable("state").length == 1:
            states = dataset.get_full()["state"]
            self._data['p_x'] = states[:, 0]
            self._data['p_y'] = np.zeros_like(self._data['x'])
        elif dataset.domain.get_variable("state").length == 2:
            states = dataset.get_full()["state"]
            self._data['p_x'] = states[:, 0]
            self._data['p_y'] = states[:, 1]
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args):
        self._check()
        self._standard_scatter(ax, **graphic_args)


class StateDensityVisualizer(PlotVisualizer):

    class_name = "state_density_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, StateDensityVisualizer.class_name)

    def compute(self, environment, dataset, bandwidth, discretization):
        if dataset.domain.get_variable("state").length == 1:
            states = dataset.get_full()["state"]
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(states)
            x_lin = np.linspace(environment.state_space.low[0], environment.state_space.high[0], discretization[0])
            y = kde.score_samples(x_lin.reshape(-1, 1))
            self._data['x'] = x_lin.ravel()
            self._data['y'] = y.ravel()
            self._data['d'] = 1
        elif dataset.domain.get_variable("state").length == 2:
            states = dataset.get_full()["state"]
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(states)
            x = np.linspace(environment.state_space.low[0], environment.state_space.high[0], discretization[0])
            y = np.linspace(environment.state_space.low[1], environment.state_space.high[1], discretization[1])
            X, Y = np.meshgrid(x, y)
            base = np.array([X.ravel(), Y.ravel()]).T
            z = kde.score_samples(base)
            self._data['x'] = X
            self._data['y'] = Y
            self._data['z'] = z.reshape(discretization[0], discretization[1])
            self._data['d'] = 2

    def visualize(self, ax: plt.Axes, **graphic_args):
        self._check()
        self._standard_general_plot(ax, **graphic_args)


class ValueRowVisualizer(RowVisualizer):

    class_name = "value_row_visualizer"

    def __init__(self):
        RowVisualizer.__init__(self, ValueRowVisualizer.class_name)

    def compute(self, env: RLEnvironment, critics: Iterable[Critic], discretizations: Iterable[np.ndarray]):
        for critic, discretization in zip(critics, discretizations):
            visualizer = ValueFunctionVisualizer()
            visualizer.compute(env, critic, discretization)
            self.sub_visualizer.append(visualizer)



def plot_value(ax: plt.Axes, env: RLEnvironment, critic: Critic, discretization: Iterable=None, **graphic_args: Any):
    """
    Plot the value function.
    :param ax: Axes of matplotlib
    :param env: The environment for which we want to plot the value.
    :param critic: The estimator of the value function. It must have a 1d or 2d statespace.
    :param discretization: The granularity of the plot (list or np.array)
    :return:
    """

    if env.state_space.shape[0] == 1:
        states = np.linspace(env.state_space.low[0], env.state_space.high[0], discretization[0])
        pool = Pool(mp.cpu_count())
        results = pool.map(critic.get_V, states.reshape(-1, 1))
        return ax.plot(states, results, **graphic_args)
    elif env.state_space.shape[0] == 2:
        dataset = env.get_grid_dataset(discretization)
        states = dataset.get_full()["state"]
        results = critic.get_V(states)
        shape = [discretization[0], discretization[1]]
        Z = np.array(results).reshape(*shape)
        return ax.pcolormesh(states[:, 0].reshape(*shape),
                      states[:, 1].reshape(*shape),
                      Z, **graphic_args)
    else:
        raise Exception("It is not possible to render an environment with state dimension greater than two.")


def plot_q_value(ax: plt.Axes, env: RLEnvironment, critic: Critic, discretization:Iterable=None, **graphic_args: Any):
    """
    Plot the Q-Function.
    :param ax: The axes on which to plot the Q-Function
    :param env: A 1-d state 1-d action environment
    :type env: RLEnvironment
    :param critic:
    :type critic: Critic
    :return:
    """

    if env.state_space.shape[0] == 1 and env.action_space.shape[0] == 1:
        dataset = env.get_grid_dataset(discretization[0:1], discretization[1:])
        pool = Pool(mp.cpu_count())
        ds = dataset.get_full()
        states, actions = ds["state"], ds["action"]
        evaluate = lambda x: critic.get_Q(x[0], x[1])
        results = pool.map(evaluate, zip(states, actions))
        shape = [discretization[0], discretization[1]]
        Z = np.array(results).reshape(*shape)
        img = ax.pcolormesh(states.reshape(*shape),
                      actions.reshape(*shape),
                      Z, **graphic_args)
    else:
        raise Exception("It is not possible to render an environment with total space (state + action) greater than two.")
    return  img


def plot_return(ax: plt.Axes, critic: Critic, policy,  indexes, low, high, discretization=None, **graphic_args):
    """
    Plot the return landscape w.r.t. the policy parameters (1 policy parameter produces a plot, 2 policy parameters, produce an heatmap).
    No more than two parameters are allowed.
    :param ax: Axes from
    :type ax: plt.Axes
    :param critic:
    :type critic: Critic
    :return:
    """

    ref_param = policy.get_parameters()
    if len(indexes) == 1:
        params = np.linspace(low[0], high[0], discretization[0])
        y = []
        for param in params.reshape(-1, 1):
            new_params = ref_param.copy()
            new_params[indexes[0]] = param
            policy.set_parameters(new_params)
            y.append(np.asscalar(critic.get_return()))
            policy.set_parameters(ref_param)
        fig = ax.plot(params, y, **graphic_args)
    elif len(indexes) == 2:
        x = np.linspace(low[0], high[0], discretization[0])
        y = np.linspace(low[1], high[1], discretization[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                param = np.array([X[i, j], Y[i, j]])
                new_params = ref_param.copy()
                new_params[indexes[0]] = param[0]
                new_params[indexes[1]] = param[1]
                policy.set_parameters(new_params)
                Z[i, j] = np.asscalar(critic.get_return())
                policy.set_parameters(ref_param)
        fig = ax.pcolormesh(X, Y, Z, **graphic_args)
    else:
        raise Exception("It is not possible to render an environment with state dimension greater than two.")
    policy.set_parameters(ref_param)
    return fig


def plot_state_cloud(ax, dataset, **graphic_args):
    """

    :param dataset: The dataset we want to inspect. Must contain a variable "state"
    :type dataset: Dataset
    :return:
    """
    if dataset.domain.get_variable("state").length == 1:
        states = dataset.get_full()["state"]
        x = states[:, 0]
        ax.plot(x, np.zeros_like(x), **graphic_args)
    elif dataset.domain.get_variable("state").length == 2:
        states = dataset.get_full()["state"]
        ax.scatter(states[:, 0], states[:, 1], **graphic_args)


def plot_state_distribution(ax, environment, dataset, bandwidth=0.1, discretization=np.array([50]), **graphic_args):
    """
    :type environment: RLEnvironment
    :param dataset: The dataset we want to inspect. Must contain a variable "state"
    :type dataset: Dataset
    :return:
    """
    if dataset.domain.get_variable("state").length == 1:
        states = dataset.get_full()["state"]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(states)
        x_lin = np.linspace(environment.state_space.low[0], environment.state_space.high[0], discretization[0])
        y = kde.score_samples(x_lin.reshape(-1, 1))
        return ax.plot(x_lin.ravel(), y.ravel(), **graphic_args)
    elif dataset.domain.get_variable("state").length == 2:
        states = dataset.get_full()["state"]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(states)
        x = np.linspace(environment.state_space.low[0], environment.state_space.high[0], discretization[0])
        y = np.linspace(environment.state_space.low[1], environment.state_space.high[1], discretization[1])
        X, Y = np.meshgrid(x, y)
        base = np.array([X.ravel(), Y.ravel()]).T
        z = kde.score_samples(base)
        return ax.pcolormesh(X, Y, z.reshape(discretization[0], discretization[1]), **graphic_args)


def plot_gradient(ax, policy_gradient, indexes, gradient=None, y=0., scale=1., **graphic_args):
    """

    :param ax:
    :type ax: plt.Axes
    :param env: Environment
    :type env: RLEnvironment
    :param policy_gradient:
    :type policy_gradient: PolicyGradient
    :return:
    """
    if len(indexes) == 1:
        if gradient is None:
            gradient = policy_gradient.get_gradient()
        params = policy_gradient.policy.get_parameters()
        x = params[indexes[0]]
        if gradient[indexes[0]] > 0:
            img = ax.arrow(x, y, scale, scale * gradient[indexes[0]], **graphic_args)
        else:
            img = ax.arrow(x, y, -scale, -scale * gradient[indexes[0]], **graphic_args)
        return ax.scatter(x, y), img
    elif len(indexes) == 2:
        gradient = policy_gradient.get_gradient()
        params = policy_gradient.policy.get_parameters()
        x = params[indexes[0]]
        y = params[indexes[1]]
        return ax.scatter(x, y),\
            ax.arrow(x, y, scale*gradient[indexes[0]], scale*gradient[indexes[1]], **graphic_args)


def plot_return_row(axs, analyzer, indxs, radius=0.5, discretization=50,
                      **graphics_args):
    param = analyzer.policy.get_parameters()
    first = True
    for ax, ind in zip(axs, indxs):
        if first:
            ax.set_ylabel(r"$\hat{J}_{%s}$" % analyzer.name)
            first = False
            plot_return(ax, analyzer, analyzer.policy, [ind], np.array([param[ind] - radius]),
                    np.array([param[ind] + radius]),
                    discretization=np.array([discretization]), **graphics_args)
        else:
            plot_return(ax, analyzer, analyzer.policy, [ind], np.array([param[ind] - radius]),
                        np.array([param[ind] + radius]),
                        discretization=np.array([discretization]), **graphics_args)
        ax.set_xlabel(r"$\theta_{%d}$" % ind)


def plot_gradient_row(axs, analyzer, indxs, ret, gradient=None, scale=0.2, **graphics_args):
    """
    Plot a row of gradients with their return landscape.

    :param ax:
    :type ax: plt.Axes
    :param env: Environment
    :type env: RLEnvironment
    :param analyzer:
    :type analyzer: Union[PolicyGradient, Critic]
    :return:
    """
    first = True
    for ax, ind in zip(axs, indxs):
        if first:
            ax.set_ylabel(r"$\hat{J}_{%s}$" % analyzer.name)
            first = False
        plot_gradient(ax, analyzer, [ind], gradient=gradient, y=ret, scale=scale)
        ax.set_xlabel(r"$\theta_{%d}$" % ind)


def plot_value_row(fig, axs, env, analyzers, discretizations, **graphics_args):
    first = True
    for ax, an, disc in zip(axs, analyzers, discretizations):
        ax.set_title("$\hat{V}_{%s}$" % an.name)
        if first:
            l = ax.set_ylabel(env.state_space.symbol[1])
            l.set_rotation(0)
        first = False
        im = plot_value(ax, env, an, disc, **graphics_args)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(env.state_space.symbol[0])


def sample_vs_bias_variance(ax: plt.Axes, reference: float or np.ndarray,
                            analyzer: Callable[[float or int], Callable],
                            values=None):
    """
    Generate a bias/variance plot w.r.t. the size of the dataset.
    :param ax: The axes used tfor the plot.
    :param dataset_generator: A function that generats a dataset of n samples.
    :param reference: The ground truth. This is the numerical value of reference (can be either a scalar or an array).
    :param analyzer: It is a function receiving a dataset and returning an estimate of the value.
    :param values: It is a list (or an array) of sizes of the dataset.
    :return:
    """
    biases = []
    variances = []
    for v in values:
        bias, variance, estimates, _ = bias_variance_estimate(reference, analyzer(v))
        biases.append(bias**2)
        variances.append(variance)

    mse = np.array(biases) + np.array(variances)
    ax.plot(values, biases, label="Bias")
    ax.plot(values, variances, label="Variance")
    ax.plot(values, mse, label="MSE")
    ax.legend(loc="best")




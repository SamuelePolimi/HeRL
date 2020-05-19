import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
from sklearn.neighbors import KernelDensity
from typing import Callable, Any, Iterable, Union, List, Dict

from herl.rl_interface import Critic, RLEnvironmentDescriptor, PolicyGradient, RLAgent
from herl.dataset import Dataset
from herl.dict_serializable import DictSerializable
from herl.rl_analysis import bias_variance_estimate, gradient_direction
from herl.utils import Printable


class PlotVisualizer(DictSerializable, Printable):

    class_name = "plot_visualizer"
    load_fn = DictSerializable.get_numpy_load()

    def __init__(self, plot_class_id: str):
        """
        This class defines a visualizer. The visualizer exposes mainly two methods: compute and visualize.
        The compute method should take care to produce the data visualized in the plot. The data must be saved in self._data.
        The visualizer plots on a (or multiple) plt.Axes objects.

        This approaches has several advantages:
           1. disentangling computation and visualization makes the code clearer.
           2. it is possible to compute only once and visualize multiple times on different ax and with different graphics properties.
           3. most importantly it is possible to save and load data from disk.

        :param plot_class_id: Name of the visualizer.
        """
        DictSerializable.__init__(self, DictSerializable.get_numpy_save())
        Printable.__init__(self, plot_class_id, verbose=False)
        self._data = {'name': plot_class_id}    # type: Dict[str, Any]
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

    def _standard_plot(self, ax: plt.Axes, **graphic_args):
        return ax.plot(self._data['x'], self._data['y'], **graphic_args)

    def _standard_heat_map(self, ax: plt.Axes, **graphic_args):
        return ax.pcolormesh(self._data['x'], self._data['y'], self._data['z'], **graphic_args)

    def _standard_general_plot(self, ax: plt.Axes, **graphic_args):
        if self._data['d'] == 1:
            return self._standard_plot(ax, **graphic_args)
        elif self._data['d']:
            return self._standard_heat_map(ax, **graphic_args)
        else:
            raise Exception("_standard_general_plot can handle only one or tho dimensions.")

    def _standard_scatter(self, ax: plt.Axes, **graphic_agrs):
        return ax.scatter(self._data['p_x'], self._data['p_y'], **graphic_agrs)

    def _standard_arrow(self, ax: plt.Axes, fixed_dx=None, resize=1, **graphic_args):
        if fixed_dx is not None:
            return ax.arrow(self._data['a_x'], self._data['a_y'], fixed_dx,
                            self._data['a_dy']/self._data['a_dx']*fixed_dx, **graphic_args)
        else:
            return ax.arrow(self._data['a_x'], self._data['a_y'], self._data['a_dx']*resize,
                            self._data['a_dy']*resize, **graphic_args)

    def _standard_quiver(self, ax: plt.Axes, **graphic_args):
        print(self._data['a_x'], self._data['a_y'], self._data['a_dx'],
                            self._data['a_dy'],)
        return ax.quiver(self._data['a_x'], self._data['a_y'], self._data['a_dx'],
                            self._data['a_dy'], units='width', scale_units='xy', angles='xy', scale=1, **graphic_args)

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

    def visualize_decorations(self, ax):
        self.visualize_x_label(ax)
        self.visualize_y_label(ax)
        self.visualize_title(ax)


class RowVisualizer(DictSerializable):

    class_name = "row_visualizer"
    load_fn = DictSerializable.get_numpy_load()

    def __init__(self, name: str, *sub_visualizer: PlotVisualizer):
        """
        Plot a row of visualizers.

        :param name: Name of the row of visualizers
        :param sub_visualizer: visualizer in the row
        """
        DictSerializable.__init__(self, DictSerializable.get_numpy_save())
        self.name = name
        self.sub_visualizer = list(sub_visualizer)  # TODO: check if instead is list(*sub_visualizer)

    def compute_i(self, i: int, *args, **kwargs):
        """
        Compute the i_th visualizer in the row
        :param i: Number of the visualizer
        :param args:
        :param kwargs:
        :return:
        """
        self.sub_visualizer[i].compute(*args, **kwargs)

    def visualize(self, axs: Iterable[plt.Axes], **kwargs: Any):
        """
        Visualize the visualizers.
        :param axs: Axes used by the visualizers.
        :param kwargs: graphical agrument passed to all the visualizers.
        :return:
        """
        ret = []
        for ax, visualizer in zip(axs, self.sub_visualizer):
            ret.append(visualizer.visualize(ax, **kwargs))
        return ret

    def _get_dict(self):
        return {'sub_visualizer': [d._get_dict() for d in self.sub_visualizer],
                'name': self.name}

    @staticmethod
    def load_from_dict(**kwargs):
        name = kwargs['name']
        if name == ValueRowVisualizer.class_name:
            visualizer = ValueRowVisualizer()
        elif name == ReturnRowVisualizer.class_name:
            visualizer = ReturnRowVisualizer()
        else:
            raise Exception("'%s' unknown." % name)
        visualizer.sub_visualizer = [PlotVisualizer.load_from_dict(**v) for v in kwargs['sub_visualizer']]
        return visualizer

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
        file = RowVisualizer.load_fn(file_name)
        return RowVisualizer.load_from_dict(**file)


class ArrowVisualizer(PlotVisualizer):

    class_name = "arrow_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, ArrowVisualizer.class_name)

    def compute(self, xs: np.ndarray, ys: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                x_label: str = "", y_label: str = "", title: str = ""):
        """
        Compute an arrow plot.
        :param xs: X position of the arrow
        :param ys: Y position of the arrow
        :param dx: X length of the arrow
        :param dy: Y length of the arrow
        :param x_label: label of the x-axis
        :param y_label: label of the y-axis
        :param title: title of the plot
        :return: None
        """
        self._data['a_x'] = xs
        self._data['a_y'] = ys
        self._data['a_dx'] = dx
        self._data['a_dy'] = dy
        self._data['x_label'] = x_label
        self._data['y_label'] = y_label
        self._data['title'] = title
        self._values = True

    def visualize(self, ax: plt.axes, **graphic_args):
        return self._standard_quiver(ax, **graphic_args)


class ValueFunctionVisualizer(PlotVisualizer):

    class_name = "value_function_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, ValueFunctionVisualizer.class_name)

    def compute(self, env: RLEnvironmentDescriptor, critic: Critic, discretization: Iterable = None):
        """
        Visualize the value function according to a critic estimator.
        :param env: The environment is needed to retrieve the state-action pairs, as well as generic information.
        :param critic: The critic perform the estimation fof the value function.
        :param discretization: This array defines the granularity of the plot.
        :return:
        """
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
        return self._standard_general_plot(ax, **graphic_args)


class QFunctionVisualizer(PlotVisualizer):

    class_name = "q_function_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, QFunctionVisualizer.class_name)

    def compute(self, env: RLEnvironmentDescriptor, critic: Critic, discretization: Iterable = None):
        """
        Visualize the value function according to a critic estimator.
        :param env: The environment is needed to retrieve the state-action pairs, as well as generic information.
        :param critic: The critic perform the estimation fof the value function.
        :param discretization: This array defines the granularity of the plot.
        :return:
        """
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

    def compute(self, env: RLEnvironmentDescriptor, policy: RLAgent, discretization: Iterable = None):
        """
        Visualize the policy. It is currently limited to the visualization of deterministic policies.
        :param env: The environment is needed to retrieve the state-action pairs, as well as generic information.
        :param policy: The considered policy.
        :param discretization: This array defines the granularity of the plot.
        :return:
        """
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

    def compute(self, critic: Critic, policy: RLAgent, indexes: list, low: list, high: list, discretization: list):
        """
        This visualizer aim to show the return function w.r.t. one or two policy's parameters.
        :param critic: The critic used to estimate the return.
        :param policy: The policy considered.
        :param indexes: One or two indexes for the policy parameters.
        :param low: The lowest value for each parameter.
        :param high: The highest value for each parameter.
        :param discretization: the discretization of each parameter.
        :param graphic_args:
        :return:
        """
        ref_param = policy.get_parameters()
        if len(indexes) == 1:
            params = np.linspace(low[0], high[0], discretization[0])
            y = []
            progress = self.get_progress_bar("parameter %d" % indexes[0], max_iter=discretization[0])
            for param in params.reshape(-1, 1):
                progress.notify()
                new_params = ref_param.copy()
                new_params[indexes[0]] = param
                policy.set_parameters(new_params)
                y.append(critic.get_return())
                policy.set_parameters(ref_param)
            self._data['d'] = 1
            self._data['x'] = params
            self._data['y'] = y
            self._data['y_label'] = r"$J^{%s}_{%s}$" % (policy.symbol, critic.name)
            self._data['x_label'] = r"$\theta_{%d}$" % indexes[0]
            self._data['title'] = ""

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
                    ret = critic.get_return()
                    # TODO decide what critic.return should actually return
                    if hasattr(ret, "item"):
                        Z[i, j] = np.asscalar(ret)
                    else:
                        Z[i, j] = ret
                    policy.set_parameters(ref_param)
            v = np.max(Z)
            indx = np.argwhere(Z == v)[0]
            self._data['p_x'] = x[indx[0]]
            self._data['p_y'] = y[indx[1]]
            self._data['x'] = X
            self._data['y'] = Y
            self._data['z'] = Z
            self._data['d'] = 2
            self._data['y_label'] = r"$\theta_{%d}$" % indexes[1]
            self._data['x_label'] = r"$\theta_{%d}$" % indexes[0]
            self._data['title'] = r"$J^{%s}_{%s}$" % (policy.symbol, critic.name)
        else:
            raise Exception("It is not possible to render an environment with state dimension greater than two.")
        policy.set_parameters(ref_param)
        self._values = True

    def visualize(self, ax, **graphic_args):
        self._check()
        return self._standard_general_plot(ax, **graphic_args), self._standard_scatter(ax, **graphic_args)


class StateCloudVisualizer(PlotVisualizer):

    class_name = "state_cloud_visualizer"

    def __init__(self):
        """
        This class aim to visualize the states present in a dataset.
        """
        PlotVisualizer.__init__(self, StateCloudVisualizer.class_name)

    def compute(self, dataset: Dataset, environment: RLEnvironmentDescriptor = None):
        """
        This class aim to visualize the states present in a dataset.
        :param dataset: The dataset of interest
        :param environment: Optional. Gives the possibility to produce automatically the plot's label.
        :return:
        """

        if dataset.domain.get_variable("state").length == 1:
            states = dataset.get_full()["state"]
            self._data['p_x'] = states[:, 0]
            self._data['p_y'] = np.zeros_like(self._data['x'])
            if environment is not None:
                self._data["x_label"] = r"$%s$" % environment.state_space.symbol[0]
        elif dataset.domain.get_variable("state").length == 2:
            states = dataset.get_full()["state"]
            self._data['p_x'] = states[:, 0]
            self._data['p_y'] = states[:, 1]
            if environment is not None:
                self._data["x_label"] = r"$%s$" % environment.state_space.symbol[0]
                self._data["y_label"] = r"$%s$" % environment.state_space.symbol[1]
        self._data["title"] = r"States in the dataset"
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args):
        self._check()
        return self._standard_scatter(ax, **graphic_args)


class DatasetCloudVisualizer(PlotVisualizer):

    class_name = "state_cloud_visualizer"

    def __init__(self):
        """
        The aim of this visualizer is to see how is the distribution of particular dimensions in the state or actions.
        Maximum two dimensions are allowed.
        There is the possibility to integrate with density estimation if bandwidts>0.
        """
        PlotVisualizer.__init__(self, StateCloudVisualizer.class_name)

    def compute(self, environment: RLEnvironmentDescriptor, dataset: Dataset,
                state_indexes=List[int], action_indexes=List[int]):
        """
        This class aim to visualize the states present in a dataset.
        :param environment: Gives the possibility to produce automatically the plot's label.
        :param dataset: The dataset of interest
        :param state_indexes: the index(es) of the state we would like to visualize
        :param action_indexes: the index(es) of the actions we would like to visualize
        :return:
        """
        symbols = environment.state_space.symbol + environment.action_space.symbol
        n_s = len(environment.state_space.symbol)
        tot_indexes = state_indexes + [n_s + a for a in action_indexes]
        data = dataset.get_full()
        tot_ds = np.concatenate([data["states"], data["actions"]], axis=1)
        if len(tot_indexes) == 1:
            self._data["p_x"] = tot_ds[:, tot_indexes[0]]
            self._data["x_label"] = r"$%s$" % symbols[tot_indexes[0]]
        elif dataset.domain.get_variable("state").length == 2:
            self._data['p_x'] = tot_ds[:, tot_indexes[0]]
            self._data['p_y'] = tot_ds[:, tot_indexes[1]]
            self._data["x_label"] = r"$%s$" % symbols[tot_indexes[0]]
            self._data["y_label"] = r"$%s$" % symbols[tot_indexes[1]]
        else:
            raise Exception("Maximum two dimension are allowed")

        self._data["title"] = r"Visualization of the dataset"
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args):
        self._check()
        return self._standard_scatter(ax, **graphic_args)


class StateDensityVisualizer(PlotVisualizer):

    class_name = "state_density_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, StateDensityVisualizer.class_name)

    def compute(self, environment: RLEnvironmentDescriptor, dataset: Dataset, bandwidth: Union[float, np.ndarray],
                discretization: List[int]):
        if dataset.domain.get_variable("state").length == 1:
            states = dataset.get_full()["state"]
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(states)
            x_lin = np.linspace(environment.state_space.low[0], environment.state_space.high[0], discretization[0])
            y = np.exp(kde.score_samples(x_lin.reshape(-1, 1)))
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
            z = np.exp(kde.score_samples(base))
            self._data['x'] = X
            self._data['y'] = Y
            self._data['z'] = z.reshape(discretization[0], discretization[1])
            self._data['d'] = 2
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args):
        self._check()
        return self._standard_general_plot(ax, **graphic_args)


class SingleEstimatesVisualizer(PlotVisualizer):

    class_name = "single_estimates_visualizer"

    def __init__(self):
        """
        It visualize the estimates of a single value as a violin plot.
        """
        PlotVisualizer.__init__(self, SingleEstimatesVisualizer.class_name)

    # TODO: create BiasVariance Class with hyperparameters in the constructor. In the compute, only ground truth and
    #  estimator
    def compute(self, estimator: Callable[[], Union[float, np.ndarray]],
                ground_truth: Union[float, np.ndarray],
                min_samples: int = 10,
                max_samples: int = 100,
                confidence=1E-1,
                estimate_symbol=""):
        bias, variance, estimates, title = \
            bias_variance_estimate(ground_truth, estimator, confidence, min_samples, max_samples)
        self._data["x_label"] = ""
        self._data["y_label"] = estimate_symbol
        self._data["title"] = title
        self._data["estimates"] = estimates
        self._data["truth"] = ground_truth
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args):
        ax.violinplot([self._data["estimates"]], showmeans=True)
        ax.set_xlim(0, 2)
        ax.set_xticks([])
        ax.scatter(np.ones_like(self._data["estimates"]) * 1.1, self._data["estimates"], s=10, c="green",
                   label="Estimates")
        ax.scatter(1.1, self._data["truth"], s=15, c="orange", label="Ground truth")
        ax.legend(loc="best")


class EstimatesVisualizer(PlotVisualizer):

    class_name = "estimates_visualizer"

    def __init__(self):
        """
        It visualizes the estimates w.r.t. a certain quantity (e.g. dataset size).
        """
        PlotVisualizer.__init__(self, EstimatesVisualizer.class_name)

    def compute(self, estimator: Callable[[float], Callable[[], float]],
                ground_truth: float,
                x_values: Iterable[float],
                min_samples: int = 10,
                max_samples: int = 1000,
                confidence: float = 1E-1,
                estimate_symbol: str = "",
                hyperparameter_symbol: str = "",
                title: str = ""):
        y_scatter = []
        x_scatter = []
        y_mean = []
        y_std = []
        for x in x_values:
            bias, variance, estimates, title = \
                bias_variance_estimate(ground_truth, estimator(x), confidence, min_samples, max_samples)
            x_scatter += [x] * len(estimates)
            y_scatter += estimates
            y_mean.append(np.mean(estimates))
            y_std.append(np.std(estimates))

        self._data["x_label"] = hyperparameter_symbol
        self._data["y_label"] = estimate_symbol
        self._data["title"] = "Pinco Pallino"
        self._data["truth"] = ground_truth
        self._data["p_x"] = np.array(x_scatter)
        self._data["p_y"] = np.array(y_scatter)
        self._data["std"] = 2 * np.array(y_std)
        self._data["x"] = np.array(x_values)
        self._data["y"] = np.array(y_mean)
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args):
        ax.fill_between(self._data["x"], self._data["y"] + self._data["std"], self._data["y"] - self._data["std"],
                        label="Standard deviation", alpha=0.5)
        ax.hlines(self._data["truth"], xmin=np.min(self._data['x']), xmax=np.max(self._data['x']), label="Ground truth")
        ax.plot(self._data["x"], self._data["y"], label="Mean estimate")
        ax.scatter(self._data["p_x"], self._data["p_y"], label="Estimates")
        ax.legend(loc="best")


class BiasVarianceVisualizer(PlotVisualizer):

    class_name = "bias_variance_visualizer"

    def __init__(self):
        """
        Visualizes Bias and Variance of the estimates, w.r.t. some quantity (e.g., sample size).
        """
        PlotVisualizer.__init__(self, BiasVarianceVisualizer.class_name)

    def compute(self, estimator: Callable[[float], Callable[[], float]],
                ground_truth: float,
                x_values: Iterable[float],
                min_samples: int = 10,
                max_samples: int = 1000,
                confidence: float = 1E-1,
                estimate_symbol: str = "",
                hyperparameter_symbol: str = "",
                title: str = ""):
        y_bias = []
        y_variance = []
        for x in x_values:
            bias, variance, estimates, title = \
                bias_variance_estimate(ground_truth, estimator(x), confidence, min_samples, max_samples)
            print()
            bias = np.mean(bias**2)
            variance = np.mean(variance)
            y_bias.append(bias)
            y_variance.append(variance)

        self._data["x_label"] = hyperparameter_symbol
        self._data["y_label"] = estimate_symbol
        self._data["title"] = "Pinco Pallino" # TODO: change
        self._data["x"] = np.array(x_values)
        self._data["y_bias"] = np.array(y_bias)
        self._data["y_variance"] = np.array(y_variance)
        self._data["y_mse"] = np.array(y_variance) + np.array(y_bias)
        self._values = True

    def visualize(self, ax:plt.Axes, **graphic_args):
        ret = ax.plot(self._data['x'], self._data['y_bias'], label=r"${Bias}^2$"),
        ax.plot(self._data['x'], self._data['y_variance'], label=r"$Variance$"),
        ax.plot(self._data['x'], self._data['y_mse'], label=r"$MSE$"),
        ax.legend(loc='best')
        return ret


class ParametricGradientEstimateVisualizer(PlotVisualizer):

    class_name = "parametric_gradient_visualizer"

    def __init__(self):
        """
        The idea of this plot is to show the cosine of the angle between the correct gradient direction and the
        estimated one, versus a parameter of the estimate (e.g., the number of samples, the off-policiness, ...).
        The resulting plot will therefore consist of a function defined on the parameter's domain and havin codomain
        between 1 and -1. when the cos(delta) is positive, then the gradient direction is "correct", and when it is
        negative, it is "incorrect". Notice that when the dimensionality of the gradient is "big", then
        cos(theta) will tend to have values close to 0.
        """
        PlotVisualizer.__init__(self, ParametricGradientEstimateVisualizer.class_name)

    def compute(self, policies: Iterable[RLAgent],
                ground_truth: np.ndarray,
                estimator: Callable[[RLAgent, float], np.ndarray],
                parameters: List[float],
                x_label=""):
        y = []
        n_updates = len(parameters) * ground_truth.shape[0]
        progress = self.get_progress_bar("gradient computation", n_updates)
        for x in parameters:
            estimates = []
            for policy in policies:
                estimates.append(estimator(policy, x))
                progress.notify()
            mean_direction = np.mean(np.cos(gradient_direction(ground_truth, np.array(estimates))))
            y.append(mean_direction)
        self._data['x'] = parameters
        self._data['y'] = y
        self._data['y_label'] = r"$\cos\delta$"
        self._data['x_label'] = x_label
        self._data['title'] = "Gradient Direction"
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args):
        return self._standard_plot(ax, **graphic_args)


class ValueRowVisualizer(RowVisualizer):

    class_name = "value_row_visualizer"

    def __init__(self):
        RowVisualizer.__init__(self, ValueRowVisualizer.class_name)

    def compute(self, env: RLEnvironmentDescriptor, critics: Iterable[Critic], discretizations: Iterable[List]):
        for critic, discretization in zip(critics, discretizations):
            visualizer = ValueFunctionVisualizer()
            visualizer.compute(env, critic, discretization)
            self.sub_visualizer.append(visualizer)


class ReturnRowVisualizer(RowVisualizer):

    class_name = "return_row_visualizer"

    def __init__(self):
        RowVisualizer.__init__(self, ReturnRowVisualizer.class_name)

    def compute(self, critic: Critic, policy: RLAgent, indexes: Iterable[int], delta: Iterable[float],
                discretization: Iterable[int]):
        params = policy.get_parameters()
        for index, d, disc in zip(indexes, delta, discretization):
            visualizer = ReturnLandscape()
            visualizer.unmute()
            param = params[index]
            visualizer.compute(critic, policy, [index], [param - d], [param+d], [disc])
            self.sub_visualizer.append(visualizer)


class GradientEstimateVisualizer(PlotVisualizer):

    class_name = "gradient_estimate_visualizer"

    def __init__(self):
        PlotVisualizer.__init__(self, GradientEstimateVisualizer.class_name)

    def compute(self, policies: List[RLAgent],
                ground_truth: np.ndarray,
                analyzer_constructor: Callable[[RLAgent], PolicyGradient],
                ground_truth_symbol=""):
        """
        In this visualizer we aim to plot the direction of an estimation of the gradient w.r.t. the "true" gradient directions.
        We consider n policies, and for each policy we give a ground-truth gradient.
        The policy_gradient algorithm will also output its estimation of the gradient for each different policy,
        and therefore  we will be able to compute the angle between the two gradients.
        When the angle is less than \pi, then the direction is approximatively  correct, otherwise, the gradient direction is wrong.
        :param policies:
        :param ground_truth:
        :param analyzer_constructor:
        :param get_dataset:
        :param ground_truth_symbol:
        :return:
        """
        gradients = []
        n_policies = len(policies)
        progress = self.get_progress_bar("gradient_estimation", n_policies)
        for policy in policies:
            progress.notify()
            analyzer = analyzer_constructor(policy)
            gradients.append(analyzer.get_gradient())
        degrees = gradient_direction(np.array(ground_truth), np.array(gradients))
        self._data['mean_estimate'] = np.mean(degrees)
        self._data['a_x'] = np.zeros_like(degrees)
        self._data['a_y'] = np.zeros_like(degrees)
        self._data['a_dx'] = np.cos(degrees)
        self._data['a_dy'] = np.sin(degrees)
        kde = KernelDensity(kernel='gaussian', bandwidth=4./len(degrees)).fit(self._data['a_dx'].reshape(-1, 1))
        x_lin = np.linspace(-1, 1, 100)
        y = np.exp(kde.score_samples(x_lin.reshape(-1, 1)))
        y = y/np.max(y)
        self._data['x_density'] = x_lin.ravel()
        self._data['y_density'] = y.ravel()
        self._data['title'] = r"Gradient Direction w.r.t. Ground Truth"
        self._data['x_label'] = r"$\cos\delta$"
        self._data['y_label'] = r"$\sin\delta$"
        self._values = True

    def visualize(self, ax: plt.Axes, **graphic_args):
        self._standard_quiver(ax, **graphic_args)
        ax.quiver(0, 0, 1, 0, units='width', scale_units='xy', angles='xy', scale=1, color='green',
                  label="Correct Direction")
        ax.quiver(0, 0, -1, 0, units='width', scale_units='xy', angles='xy', scale=1, color='red',
                  label="Wrong Direction")
        ax.quiver(0, 0, np.cos(self._data['mean_estimate']), np.sin(self._data['mean_estimate']),
                  units='width', scale_units='xy', angles='xy', scale=1, color='blue', label="Mean Estimate")

        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="best")
        return ax.plot(self._data['x_density'], self._data['y_density'], label=r"Density of $\cos\delta$")


class GradientRowVisualizer(RowVisualizer):

    class_name = "gradient_row_visualizer"

    def __init__(self):
        RowVisualizer.__init__(self, GradientRowVisualizer.class_name)

    def compute(self, policy: RLAgent, gradient_critic: Union[PolicyGradient, Critic],
                deltas: Iterable[float],
                indexes: Iterable[int]):
        params = policy.get_parameters()
        gradient = gradient_critic.get_gradient()
        ret = gradient_critic.get_return()
        for index, delta in zip(indexes, deltas):
            visualizer = ArrowVisualizer()
            if gradient[index] > 0:
                visualizer.compute(params[index], ret, delta, delta*gradient[index])
            else:
                visualizer.compute(params[index], ret, -delta, -delta*gradient[index])
            self.sub_visualizer.append(visualizer)



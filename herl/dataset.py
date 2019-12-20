import numpy as np

from herl.config import np_float


class Variable:
    """
    This represent a variable in a specific domain (e.g., a state in a RL problem,
    the angle in the pendulum environment ect.)
    We assume the variable to be float, but we might generalise this in next development.
    """

    def __init__(self, name, length, description="", latex_symbol=""):
        """

        :param name: The name of the variable. This name will be used in the code to refer to the variable.
        :type name: str
        :param length: The dimensionality of the variable. E.g., lenght=3 if the variable represent a 3d-position.
        :type length: int
        :param description: The usere might want to associate a description of the variable.
        :type description: str
        :param latex_symbol: For plot purposes, we can associate to the variable some symbol in latex code,
        such as $\theta$, $x$...
        :type latex_symbol: str
        """
        self.name = name
        self.length = length
        self.description = description
        self.latex_symbol = latex_symbol
        self.location = None

    def assign(self, location):
        """
        Assign the variable a position in the dataset.
        :param location: The position of the variable in the dataset.
        :type location: int
        :return: None.
        """
        self.location = location


class Domain:

    """
    This class describes a domain, such as a reinforcement learning proble, a state description, ...
    The domain is defined by a ordered set of variables.
    """

    def __init__(self, *variables):
        """
        Create a domain with some variable. We can also add add variable later to the domain with the method
        `self.add_variable`.
        :param variables: variable to add to the domain
        :type variables: Variable
        """
        self.size = 0
        self.variables = []
        self.variable_dict = {}
        for var in variables:
            self.add_variable(var)

    def add_variable(self, variable):
        """
        Add a variable to the domain.
        :param variable: The variable to add.
        :type variable: Variable
        :return: None
        """
        for v in self.variables:
            if v.name == variable.name:
                raise Exception("Variable %s is already contained in the domanin." % variable.name)
        variable.assign(self.size)
        self.variables.append(variable)
        self.variable_dict[variable.name] = variable
        self.size += variable.length

    def get_variable(self, name):
        """
        Get a variable given its name.
        :param name: Name of the variable.
        :return: The variable (if exists a variable with the specified name).
        :rtype: Variable
        """
        return self.variable_dict[name]


class Dataset:

    """
    This class defines a generic dataset.
    """
    def __init__(self, domain, n_max_row=int(10E6)):
        """
        A dataset is specified on a specific Domain. For example, for the Iris dataset, the domain would be something
        like Domain(Variable("sepal_l", 1), Variable("sepal_w", 1), Variable("petal_l", 1), Variabble("petal_w", 1),
        Variable("class", 1)).
        The dataset has a fixed maximum size to allow efficient allocation and data retrival.
        :param domain: The domain of the data.
        :type domain: Domain
        :param n_max_row: Number maximum of rows. The data's structure is a circular buffer, therefore if the user add
        more data than the n_max row, the first data will be overwritten.
        :type n_max_row: int
        """
        self.domain = domain
        self.memory = np.zeros((n_max_row, domain.size), dtype=np_float)
        self.max_size = n_max_row
        self.real_size = 0
        self.pointer = 0
        self.indexes = np.arange(0, self.max_size)

    def notify(self, **kargs):
        """
        Insert a new row in the dataset. Each kwarg represent a variable associated to the Variable.
        E.g., for the iris dataset one should run
        `self.notify(sepal_l=1., sepal_w=1., petal_l=0.2, petal_w=0.3)`
        :param kargs: The value associated to the variable
        :type kargs: np.ndarray
        :return:
        """
        for k, v in kargs.items():
            variable = self.domain.variable_dict[k]
            self.memory[self.pointer, variable.location:variable.location+variable.length] = v
        self.real_size = min(self.real_size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size

    def get_minibatch(self, size=128):
        """
        Retrive a random (mini)batch of data.
        :param size: The size of the batch (which should be less than the data already inserted in the database)
        :type size: int
        :return: Dataset batch
        :rtype: dict
        """
        if size > self.real_size:
            raise("You are requesting more samples than the samples stored")

        indx = np.random.choice(self.indexes[:self.real_size], size=size, replace=False)
        result = self.memory[indx, :]
        return {k: result[:, v.location:v.location+v.length] for k, v in self.domain.variable_dict.items()}

    def set(self, memory):
        if memory.shape[1] != self.domain.size:
            raise("Wrong format!")
        self.memory = memory
        self.pointer = 0
        self.max_size = self.memory.shape[0]
        self.real_size = self.max_size

    def flush(self):
        """
        Empty the database
        :return: None.
        """
        self.memory = np.zeros((self.max_size, self.domain.size), dtype=np_float)
        self.real_size = 0
        self.pointer = 0

    def is_full(self):
        """
        Is the database full?
        :return:
        """
        return self.real_size == self.max_size

    def get_full(self):
        result = self.memory[:self.real_size, :]
        return {k: result[:, v.location:v.location + v.length] for k, v in self.domain.variable_dict.items()}


class MLDataset:

    """
    This class represent a typical database for machine learning. It allows the possibility to separate data onto two
    sets, one for training and one for validating (or test).
    """
    def __init__(self, domain, n_max_row=int(10E6), validation=0.1):
        """
        Likewise the classic dataset, we are constructing a database on a given domain.
        :param domain: Domain of the dataset
        :type domain: Domain
        :param n_max_row: Size of the database
        :type n_max_row: int
        :param validation: Portion of the data should be in the validation set (between 0 and 1).
        :type validation: float
        """
        self.train_ds = Dataset(domain, n_max_row - int(n_max_row*validation))
        self.validation_ds = Dataset(domain, int(n_max_row*validation))
        self.validation = validation

    def notify(self, **kwargs):
        """
        Insert a new row in the dataset. Each kwarg represent a variable associated to the Variable.
        E.g., for the iris dataset one should run
        `self.notify(sepal_l=1., sepal_w=1., petal_l=0.2, petal_w=0.3)`.
        The data will be randomly allocated to the training set or to the validation set, depending the
        validation probability specified in the constructor.
        :param kargs: The value associated to the variable
        :type kargs: np.ndarray
        :return: None.
        """
        if self.validation_ds.is_full():
            self.train_ds.notify(**kwargs)
        else:
            if np.random.uniform() < self.validation:
                self.validation_ds.notify(**kwargs)
            else:
                self.train_ds.notify(**kwargs)

    def get_minibatch(self, size, train=True):
        """
        Retrive a random (mini)batch of data. If train=True, the data will be retreived from the training set,
        or from the validation set otherwise.
        :param size: The size of the batch (which should be less than the data already inserted in the database)
        :type size: int
        :return: Dataset batch
        :rtype: dict
        """
        if train:
            return self.train_ds.get_minibatch(size)
        else:
            return self.validation_ds.get_minibatch(size)

    def flush_validation(self):
        """
        Flush only the validation set. (it might be useful to use the validation data only once)
        :return: None.
        """
        self.validation_ds.flush()

    def flush(self):
        """
        Empty both validation and training sets.
        :return: None.
        """
        self.validation_ds.flush()
        self.train_ds.flush()

    def is_full(self):
        """
        Check if both the sets are full.
        :return: None.
        """
        return self.validation_ds.is_full() and self.train_ds.is_full()


def create_ml_dataset(dataset, validation=0.1):
    """
    Create a proper ML dataset with train set and validation set given a normal dataset
    :param dataset: A classic database
    :type dataset: Dataset
    :return:  a MLDataset
    :rtype: MLDataset
    """
    train_size = int(dataset.real_size*(1-validation))
    validation_size = dataset.real_size - train_size

    indxs = np.arange(dataset.real_size)
    np.random.shuffle(indxs)
    train_indxs = indxs[:train_size]
    validation_indxs = indxs[train_size:train_size+validation_size]

    ret = MLDataset(dataset.domain, dataset.real_size)

    ret.train_ds.set(dataset.memory[train_indxs, :])
    ret.validation_ds.set(dataset.memory[validation_indxs, :])

    return ret
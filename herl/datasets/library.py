import warnings
import os

from herl.dataset import Dataset, Domain, Variable


class DatasetDescriptor:

    def __init__(self, name, file_name, domain, description, keywords):
        """

        :param name:
        :type name: str
        :param file_name:
        :type file_name: str
        :param domain:
        :type domain: Domain
        :param description:
        :type description: str
        :param keywords:
        :type keywords: list[str]
        """
        self.name = name
        self.domain = domain
        self.filename = file_name
        self.description = description
        self.keywords = set(keywords)

    def match(self, domain,  *search_args):
        """

        :param domain:
        :type domain: Domain
        :param search_args:
        :return:
        """

        if domain.size != self.domain.size:
            return 0, False

        points = 0.
        same_domain = True
        if len(domain.variables) == len(self.domain.variables):
            n = sum([v1.length if v1.length == v2.length else 0
                           for v1, v2 in zip(domain.variables, self.domain.variables)])
            if n != domain.size:
                same_domain = False
        else:
            same_domain = True

        set_args = set(search_args)
        match = set_args.intersection(self.keywords)
        points += len(match) * 3

        points += sum([self.description.count(key) for key in search_args])

        return points, same_domain


datasets = {
    "uniform_pendulum2d_constant_policy":\
        DatasetDescriptor("uniform_pendulum2d_constant_policy",
                        "pendulum2d/constant_policy_0_uniform_state_v.npz",
                        Domain(Variable("state", 2), Variable("value", 1)),
                        """This dataset is optained with Pendulum2D (in herl.utils). 
                        It is sampled on a grid 100x100 over the state-space (angle, velodity).
                        It contains the values for the states in the grid, under a constant policy
                        that outputs always 0.
                        The discount factor used is 0.95, and the estimation has been carried out with MC sampling.""",
                        ["pendulum", "pendulum2d", "uniform", "value", "mc", "montecarlo", "0"])
}


def search(domain, *keywords):
    max_point = 0
    valid = False
    dataset_max = None
    for v in datasets.values():
        points, v_valid = v.match(domain, *keywords)
        if points > max_point and (not (valid) or v_valid):
            max_point = points
            valid = v_valid
            dataset_max = v
    if not valid:
        warnings.warn("The domain does not match with any pf the datasets. There is probably a mistake in the code.")

    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    print("This file directory only")
    print(os.path.dirname(full_path))
    return Dataset.load(path + "/" + dataset_max.filename, domain)
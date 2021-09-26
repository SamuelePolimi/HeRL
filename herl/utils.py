import matplotlib.pyplot as plt
import numpy as np
import torch
import git
import warnings
import pathlib
import json
import herl


class Printable:

    def __init__(self, name, verbose=True, plot=True):
        """
        Obtain a object that can print on the console or plot, but can be muted when needed.
        :param name:
        :param verbose:
        :param plot:
        """
        self.name = name
        self.verbose = verbose
        self.plot = plot

    def print(self, string="", *args, **kwargs):
        if self.verbose:
            print("%s: %s" % (self.name, string), *args, **kwargs)

    def base_print(self, string="", *args, **kwargs):
        if self.verbose:
            print(string, *args, **kwargs)

    def show(self):
        if self.plot:
            plt.show()

    def mute(self):
        self.verbose = False

    def unmute(self):
        self.verbose = True

    def visible(self):
        self.plot = True

    def invisible(self):
        self.plot = False

    def get_progress_bar(self, process_name, max_iter=100):
        return ProgressBar(self, max_iteration=max_iter, prefix="%s (%s):" % (self.name, process_name), suffix="Complete.")


class ProgressBar:

    def __init__(self, printable: Printable, max_iteration: int = 100, length: int = 50,
                 prefix: str = 'Progress:', suffix: str = '', fill: str = '█', print_end: str = "\r"):
        self.printable = printable
        self.max_iteration = max_iteration
        self.length = length
        self.prefix = prefix
        self.suffix = suffix
        self.fill = fill
        self.print_end = print_end
        self._iteration = 0
        self.decimals = 1

    def notify(self, progress=1):
        self._iteration += progress
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self._iteration / float(self.max_iteration)))
        filled_length = int(self.length * self._iteration / float(self.max_iteration))
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        self.printable.base_print('\r%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix), end=self.print_end)
        # Print New Line on Complete
        if self._iteration >= self.max_iteration:
            self.printable.base_print()


def _one_hot(index, n_values):

    ret = None

    if hasattr(index, "shape") and len(index.shape) > 0:
            n_entries = index.shape[0]
            ret = np.zeros((n_entries, n_values))
            for i, v in enumerate(index):
                ret[i, v] = 1

    if ret is None:
        ret = np.zeros(n_values)
        ret[index] = 1

    if type(index) is torch.Tensor:
        return torch.tensor(ret)

    return ret


def _decode_one_hot(one_hot):
    return np.argmax(one_hot, axis=-1)

def _check_update():
    if herl.__configurations__["check_update"]:
        print("Checking if HeRL is updated. To disable the ckeck, run `herl.utils._write_config('check_update', False)`")
        repo = git.Repo(pathlib.Path(__file__).parent.parent.resolve())
        sha = repo.head.object.hexsha

        remote_heads = git.cmd.Git().ls_remote(r"git@github.com:SamuelePolimi/HeRL.git", heads=True)

        if sha != remote_heads.split()[0]:
            warnings.warn("The local repository of HeRL is not up to date. Pull to obtain the updated version.")
        else:
            print("HeRL is up to date.\n")

def _load_config():
    path = pathlib.Path(__file__).parent.resolve() / "config.json"
    f = open(path, )
    ret = json.load(f)
    f.close()
    return ret


def _write_config(name, value):
    herl.__configurations__[name] = value
    path = pathlib.Path(__file__).parent.resolve() / "config.json"
    f = open(path, "w")
    f.write(json.dumps(herl.__configurations__))
    f.close()




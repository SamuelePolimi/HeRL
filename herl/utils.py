import matplotlib.pyplot as plt


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

    def print(self, string, *args, **kwargs):
        if self.verbose:
            print("%s: %s" % (self.name, string), *args, **kwargs)

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
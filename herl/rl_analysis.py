class Analyser:

    def __init__(self, verbose=True, plot=True):
        self.verbose = verbose
        self.plot = plot

    def print(self, string):
        if self.verbose:
            print("Analyzer: %s" % string)
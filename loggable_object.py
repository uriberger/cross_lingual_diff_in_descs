import torch
from utils.general_utils import log_print


class LoggableObject:
    """ The top class from which all other classes inherit.
    Enables printing to the global log (using the log_print) function, with indent. The indent is stored in the
    instance, and can be incremented or decremented. """

    def __init__(self, indent):
        self.indent = indent
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def increment_indent(self):
        self.indent += 1

    def decrement_indent(self):
        self.indent -= 1

    def log_print(self, my_str):
        log_print(self.__class__.__name__, self.indent, my_str)

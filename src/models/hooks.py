from functools import partial

from torch import nn


def children(m):
    return list(m.children())


class Hooks:
    def __init__(self, *modules, module_class=nn.LeakyReLU):
        self.hooks = [Hook(c, store_activation)
                      for m in modules
                      for c in m.children()
                      if isinstance(c, module_class)]

    def register(self):
        for h in self.hooks:
            h.register()

    def remove(self):
        for h in self.hooks:
            h.remove()

    def __iter__(self):
        return self.hooks.__iter__()

    def __del__(self):
        self.remove()


class Hook:
    def __init__(self, m, f):
        self.hook = None
        self.m = m
        self.f = f
        self.register()
        self.activation_data = None

    def register(self):
        self.hook = self.m.register_forward_hook(partial(self.f, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


def store_activation(hook, mod, inp, outp):
    hook.activation_data = outp.data.cpu()

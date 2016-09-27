from . import _base
from nnvm import symbol as _sym

class GradientDescentOptimizer(object):
    def __init__(self, learning_rate, name="GradientDescent"):
        self.learning_rate = learning_rate

    def minimize(self, obj):
        variables = obj.list_input_variables()
        grads = _base.gradients(obj, variables)
        updates = []
        for v, g in zip(variables, grads):
            updates.append(_sym.assign(v, v + (-self.learning_rate) * g))
        return _base.group(*updates)

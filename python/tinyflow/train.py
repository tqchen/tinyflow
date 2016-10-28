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

class AdamOptimizer(object):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-04, name='Adam'):
        self.name = name
        self.t = _base.Variable(_sym.zeros(shape=[1]), name+'_t')
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []

    def minimize(self, obj):
        variables = obj.list_input_variables()
        grads = _base.gradients(obj, variables)
        updates = []
        for i, v in enumerate(variables):
            self.m.append(_base.Variable(_sym.zeros_like(v), self.name + '_m' + str(i)))
            self.v.append(_base.Variable(_sym.zeros_like(v), self.name + '_v' + str(i)))
        update_t = _sym.assign(self.t, self.t + 1)
        rate = _sym.sqrt(1 - self.beta2 ** update_t) / (1 -  self.beta1 ** update_t)
        lr_t = self.learning_rate * rate
        for var, g, m, v in zip(variables, grads, self.m, self.v):
            update_m = _sym.assign(m, self.beta1 * m + (1 - self.beta1) * g)
            update_v = _sym.assign(v, self.beta2 * v + (1 - self.beta2) * g * g)
            update_var = _sym.assign(var,
                var - lr_t * update_m / (_sym.sqrt(update_v) + self.epsilon))
            updates.append(update_var)
        return _base.group(*updates)

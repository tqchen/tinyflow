
class GradientDescentOptimizer(object):
    def __init__(self, learning_rate, name="GradientDescent"):
        self._learning_rate = learning_rate

    def minimize(self, obj):
        print(obj.list_inputs())
        return None

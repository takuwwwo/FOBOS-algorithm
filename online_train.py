import numpy as np
from abc import *
import matplotlib.pyplot as plt

class Online_train:
    __metaclass__ = ABCMeta

    def __init__(self, data_size):
        self.n = data_size
        self.seq = np.arange(data_size)
        np.random.shuffle(self.seq)
        self.t = 0

    def train(self, t, update_args, graph=False, test_data=None, interval=1000):
        if graph:
            test, label = test_data
            for i in range(t):
                self.update(*update_args)
                if i % interval == 0:
                    x = self.calc_accuracy(test, label)
                    print(i, x)
                    plt.plot(i, x, 'ro')
                self.t += 1
            plt.show()

        else:
            test, label = test_data
            for i in range(t):
                self.update(*update_args)
                if i % interval == 0:
                    x = self.calc_accuracy(test, label)
                    print(i, x)

    def calc_accuracy(self, test, label):
        test_size = label.shape[0]
        output = self.det_fun(test)
        predict = self.det_label_fun(output)
        return np.sum(label==predict) / test_size

    @abstractmethod
    def update(self, *args):
        pass

    @abstractmethod
    def det_fun(self, *args):
        pass

    @abstractmethod
    def det_label_fun(self, out):
        pass


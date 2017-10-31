import numpy as np
from dataset.mnist import mnist
import online_train
import fobos

class Train_fobos(online_train.Online_train):
    def __init__(self, n, data, label, data_size, l):
        super().__init__(data_size)
        self.data = data
        self.label = label
        self.fobos = fobos.FOBOS(n, l)
        self.w = self.fobos.w

    def lr_fun(self):
        return 30.0 / np.sqrt(self.t+1)

    def update(self, *args):
        x = self.data[self.seq[self.t]]
        y = self.label[self.seq[self.t]]
        g = self.grad(x, y)
        lr = self.lr_fun()
        self.w = self.fobos.update(g, lr, lr)

    # logistic loss
    def grad(self, x, y):
        return -(1 / (1 + np.exp(y*x.dot(self.w))))*y*x

    def det_fun(self, x):
        return x.dot(self.w)

    def det_label_fun(self, out):
        return np.sign(out)


l = 3.0*0.00001
m = mnist.Mnist()
n = 28*28
p_data, p_label = m.train_list[2], np.ones(m.train_list[2].shape[0])
n_data, n_label = m.train_list[6], -np.ones(m.train_list[6].shape[0])
p_test_data, p_test_label = m.test_list[2], np.ones(m.test_list[2].shape[0])
n_test_data, n_test_label = m.test_list[6], -np.ones(m.test_list[6].shape[0])

data = np.r_[p_data, n_data]
label = np.r_[p_label, n_label]

test_data = np.r_[p_test_data, n_test_data]
test_label = np.r_[p_test_label, n_test_label]

train_fobos = Train_fobos(n, data, label, data.shape[0], l)
print(data.shape[0])
train_fobos.train(data.shape[0],
                  update_args=(), graph=True, test_data=(test_data, test_label),
                  interval=100)
print(train_fobos.w)

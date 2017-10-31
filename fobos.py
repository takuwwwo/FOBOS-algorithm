import numpy as np
from enum import Enum

class reg_type(Enum):
    l1 = 1
    l2 = 2
    l2_2 = 3


class FOBOS:
    def __init__(self, n, l, regularization=reg_type.l1):
        self.regularization = regularization
        self.w = np.random.randn(n) * 0.1
        self.l = l

    def update(self, g, gamma1, gamma2):
        if self.regularization == reg_type.l1:
            return self.l1_update(g, gamma1, gamma2)
        elif self.regularization == reg_type.l2:
            return self.l2_update(g, gamma1, gamma2)
        elif self.regularization == reg_type.l2_2:
            return self.l2_2_update(g, gamma1, gamma2)

    def l1_update(self, g, gamma1, gamma2):
        self.w = np.sign(self.w - gamma1*g) * \
                 np.maximum(((np.abs(self.w - gamma1*g)) - gamma2*self.l), 0)
        return self.w

    def l2_update(self, g, gamma1, gamma2):
        self.w = max(0,
                    (1 - gamma2*self.l/np.linalg.norm(self.w-gamma1*g))) \
                    * (self.w - gamma1*g)
        return self.w

    def l2_2_update(self, g, gamma1, gamma2):
        self.w = (self.w - gamma1*g) / (1 + gamma2*self.l)
        return self.w


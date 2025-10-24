import torch
from torch import nn
from math import log


class LearnableCoeffient(nn.Module):
    def __init__(self, v0=0.2):
        super(LearnableCoeffient, self).__init__()
        w0 = log(v0) if v0 < 1.0 else v0 - 1
        self.w = nn.Parameter(torch.tensor(w0), requires_grad=True)

    def forward(self, x, grad=True):
        p = self.__get(grad)
        return p * x

    def __get(self, grad=True):
        v = torch.exp(self.w) if self.w < 0 else self.w + 1
        if not grad:
            v = v.detach()
        return v

    def __mul__(self, other):
        p = self.__get(False)
        return p * other

    def value(self):
        return self.__get(False).item()


if __name__ == '__main__':
    alpha = LearnableCoeffient()
    print(alpha.value())

    alpha = LearnableCoeffient(1+log(0.2))
    print(alpha.value())
    pass

import torch


def polyak_update(params, target_params, tau) -> None:
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


class PeriodicTrigger:
    def __init__(self, itv):
        self.itv = itv
        self.remain = itv

    def step(self, delta=1):
        if self.remain > 0:
            self.remain -= delta
        if self.remain <= 0:
            self.remain += self.itv
            return True
        return False


class LinearScheduledCoefficient:
    def __init__(self, start, end, end_rate=1.0):
        self.val = start
        self.start = start
        self.span = start - end
        self.end_rate = end_rate

    def __mul__(self, other):
        return self.val * other

    def __rmul__(self, other):
        return other * self.val

    def __sub__(self, other):
        return self.val - other

    def __rsub__(self, other):
        return other - self.val

    def __str__(self):
        return str(self.val)

    def __float__(self):
        return self.val

    def update(self, rate):
        rho = min(rate/self.end_rate, 1.0)
        self.val = self.start - self.span * rho


if __name__ == '__main__':

    tg = PeriodicTrigger(2.5)
    for i in range(10):
        print(tg.step())

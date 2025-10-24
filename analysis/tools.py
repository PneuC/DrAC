def ema_smoothing(y, alpha=0.1):
    if alpha <= 0:
        return y
    smoothed = [y[0]]
    for i in range(1, len(y)):
        smoothed.append(smoothed[-1] * (1 - alpha) + y[i] * alpha)
    return smoothed

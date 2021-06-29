from matplotlib import pyplot as plt
import numpy as np
from math import exp, pow

def noisy_normal_scatter(x_range=(1,100), data_num=1000, sigma=1, h=1, mid=0, noise_strength=1):
    x = np.linspace(x_range[0], x_range[1], data_num)
    variance = pow(sigma, 2)
    y = np.zeros_like(x)
    for ii, value in enumerate(x):
        noise = (np.random.randn()) * noise_strength
        y[ii] = noise + h * exp(-pow(value-mid, 2)/(2*variance))
    return x, y


def moving_average(x,y, window_size):
    y_ma = np.zeros(shape=(y.shape[0]-window_size+1))
    for ii, value in enumerate(list(y)[window_size-1:]):
        numerator = window_size
        while numerator > 0:
            y_ma[ii] += y[ii + numerator - 1]
            numerator -= 1
        y_ma[ii] /= window_size
    x_ma = x[window_size-1:]
    return x_ma, y_ma


def exp_moving_ave(x, y, beta):   # Exponentially weighted moving average
    y_ema = np.zeros(shape=(y.shape[0]))
    y_ema[0] = y[0]
    for ii, value in enumerate(y):
        if ii !=0:
            y_ema[ii] = beta * y_ema[ii-1] + (1 - beta) * y[ii]
    x_ema = x
    return x_ema, y_ema


x, y = noisy_normal_scatter(x_range=(1,100), data_num=1000, sigma=15, h=35, mid=50, noise_strength= 5 )
plt.scatter(x, y, marker= '.')

x_ma, y_ma = moving_average(x,y , window_size=20)
plt.plot(x_ma,y_ma, color='red')

x_ema, y_ema = exp_moving_ave(x,y, beta=0.95)
plt.plot(x_ema,y_ema, color='k')

plt.show()

import RdWr
import Model
import Analysis
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

def anti_shift_rem(self):
    sum = 0

    for i in range(self.N):
        sum += self.x[i]
    mid = 1 / self.N * sum
    print(mid)
    return mid


def anti_shift(self, mid):
    for i in range(self.N):
        self.x[i] = (self.x[i] - mid)


def somex0(self):
    self.x[0] = 0

def comp(self):
    sr_x = self.x.mean()
    for i in range(self.N):
        self.x[i] = self.x[i] - sr_x

def anti_spike_rem(self):
    remi = []
    for i in range(self.N):
        if abs(self.x[i]) > (self.S + 10):
            remi.append(i)
    return remi


def anti_spike(self, remi):
    for item in remi:
        self.x[item] = (self.x[item - 1] + self.x[item + 1]) / 2


def anti_trend_rem(self):
    rem = []
    temp = 0
    for t in range(self.N):
        for i in range(200):
            temp = self.x[t - i] / self.N
        rem.append(temp)
    return rem

def anti_trend_rem_new(data):
    X = [i for i in range(0, len(data))]
    X = np.reshape(X, (len(X), 1))
    model = LinearRegression()
    model.fit(X, data)
    trend = model.predict(X)
    pyplot.plot(data)
    pyplot.plot(trend)
    pyplot.show()
    at_data = [data[i]-trend[i] for i in range(0, len(data))]
    return at_data

    # xw = sum([self.x[k] for k in range(self.N)]) / self.N
    # for k in range(self.N):
    #     self.x[k] -= xw

def trend(self):
    self.x += Model.Trend.trend1(self)

def trend_new(data):
    data += Model.Trend.trend1(data)

def anti_trend(self, rem):
    for i in range(self.N):
        self.x[i] = rem[i]

def anti_trend_new(rem):
    N = len(rem)
    data = []
    for i in range(N):
        data.append(rem[i])
    return data


# def cor(self):
#     sr_x = self.x.mean()
#     for i in range(self.N):
#         self.x[i] = self.x[i] - sr_x
#     x = self.x - sr_x
#     results = np.correlate(x, x, mode='full')
#     return results[round(results.size/2)-1:]


# def cor(self):
#     plot.acorr(self.x)
#     plot.show()

def cor(x1, x2):
    fig, [ax1, ax2] = plot.subplots(2, 1, sharex=True)
    ax1.xcorr(x1.x, x2.x, usevlines=True, maxlags=100, normed=True, lw=2)
    ax1.grid(True)

    ax2.acorr(x1.x, usevlines=True, normed=True, maxlags=100, lw=2)
    ax2.grid(True)
    plot.show()




def plot_ver(x1, x2):
    plot.hist(x1.x,bins=100, alpha=0.5, histtype='stepfilled', color='blue', edgecolor='black')
    plot.hist(x2.x,bins=100, alpha=0.5, histtype='stepfilled', color='red', edgecolor='black')
    plot.show()
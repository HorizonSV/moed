# -*- coding: utf-8 -*-
import random
import numpy as np
import math
import struct
#import seaborn as sns
from datetime import datetime


# Линейный тренд
def linear_function(x, k=1, b=0):
    return k * x + b


# Экспоненциальный тренд
def exp_function(x, BETA, ALPHA):
    return BETA * np.e ** (x / ALPHA)


# Функция нормирования
def normalize(numbers: np.array, S: int) -> np.array:
    return (((numbers - min(numbers)) / (max(numbers) - min(numbers))) - 0.5) * 2 * S


# Функция для генерации случайного процесса встроенным методом (built-in)
def PyRandom(N, lo, hi):
    arguments = np.zeros(N)

    for i in range(0, N):
        arguments[i] = random.uniform(lo, hi)
    return arguments


# Вспомогательная функция для генерации случайного процесса
seed = int(datetime.now().microsecond)


def next():
    global seed
    q = 127773
    a = 16807
    m = 2147483647
    r = 28

    hi = seed / q
    lo = seed % q
    seed = (a * lo) - (r * hi)

    if seed <= 0:
        seed = seed + m

    return (seed * 1.0) / m


# Функция для генерации случайного процесса Lemer
def LemerRandom(N, lo, hi):
    arguments = []
    for i in range(0, N):
        x = next()
        arguments.append((hi - lo) * x + lo)
    return arguments


# Функция для сдвига данных (shift)
def shift(Y, const):
    for i in range(0, len(Y)):
        Y[i] += const
    return Y


# Функция для генерации выбросов (spikes)
def spikes(N, amplitude, delta, random_y=None):
    quantity_spikes = int(random.uniform(0, 1) / 100 * N)
    random_x = [i for i in range(N)]
    if random_y is None:
        random_y = [0 for i in range(N)]
    for loc in range(0, quantity_spikes):
        y = random.randint(0, N)
        true_false = random.randint(0, 2)
        if true_false == 0:
            random_y[y] = random.uniform(-amplitude, -delta)
        else:
            random_y[y] = random.uniform(delta, amplitude)
    return random_x, random_y

# def generating_spikes(self):
#     self.argument = self.s_max * 2
#     temp = np.zeros(self.n)
#
#     number_spikes = int(self.n * 0.01)
#
#     for i in range(number_spikes):
#         rand_index_array = random.randint(0, self.n - 1)
#         rand_value = random.uniform(self.s_min, self.s_max)
#
#         if rand_value > 0:
#             rand_value += self.argument
#
#         else:
#             rand_value -= self.argument
#
#         temp[rand_index_array] = rand_value
#
#     self.s_max = self.argument * 1.1
#     return temp

# Аддитивная модель
def additive(X, func):
    y_linear = linear_function(X)
    y_random = func
    x_line_rand = [a2 + b for a2, b in zip(y_linear, y_random)]
    return x_line_rand


# Мультипликативная модель
def multiplicative(X, func):
    y_linear = linear_function(X)
    y_random = func
    x_line_rand = [a * b for a, b in zip(y_linear, y_random)]
    return y_linear, x_line_rand


# Мультипликативная модель
def multiplicative2(N, S):
    # t = np.linspace(0, N, N)
    x_line = linear_function(t)
    x_inv_line = linear_function(-t)
    x_random = PyRandom(N, -S, S)
    x_line_rand = [a * b for a, b in zip(x_line, x_random)]
    x_inv_line_rand = [a * b for a, b in zip(x_inv_line, x_random)]
    return

    plt.plot(t, x_line_rand, color='blue')
    plt.plot(t, x_line, color='brown')
    plt.plot(t, shift(x_inv_line_rand, 30000), color='black')
    plt.plot(t, shift(x_inv_line, 30000), color='red')

    plt.show()


# 5 тренд
# def linear_function(k, b, x):
#    return k * x + b
#
# def exp_function(beta, alpha, x):
#    return beta * np.power(np.e, alpha * x)
#
# def get_common_function(k, b, alpha, beta, size):
#    n = int(np.floor(size / 3))
#    linear_values0 = [linear_function(k, b, x) for x in range(n)]
#    linear_values1 = [linear_function(-k, b + 333, x) for x in range(n)]
#    if n * 3 != size:
#        n += 1
#    exp_values0 = [exp_function(beta, alpha, x) for x in range(n)]
#    return linear_values0 + linear_values1 + exp_values0

# Математическое ожидание
def expected_value(ordinata, N):
    sum = 0
    for i in range(0, N):
        sum += ordinata[i]
    return float(sum / N)


# Дисперсия
def dispersion(expected_value, ordinata, N):
    sum = 0
    for i in range(0, N):
        sum += np.power((ordinata[i] - expected_value), 2)
    return float(sum / N)


# СКО
def standard_deviation(dispersion):
    return float(math.sqrt(dispersion))


# Асимметрия
def asymmetry(expected_value, ordinata, N):
    sum = 0
    for i in range(0, N):
        sum += np.power((ordinata[i] - expected_value), 3)
    return float(sum / N)


# Коэффициент асимметрии
def skewness(asymmetry, standard_deviation):
    return float(asymmetry / np.power(standard_deviation, 3))


# Коэффициент эксцесс
def kurtosis(expected_value, ordinata, N):
    sum = 0
    for i in range(0, N):
        sum += np.power((ordinata[i] - expected_value), 4)
    return float(sum / N)


# Куртозис
def kurtosis_2(kurtosis, standard_deviation):
    return float(kurtosis / np.power(standard_deviation, 4) - 3)


# Функция для определения основных характеристик процесса
def general_statistics(function, N):
    M = expected_value(function, N)
    D = dispersion(M, function, N)
    SKO = standard_deviation(D)
    Mu_3 = asymmetry(M, function, N)
    Mu_4 = kurtosis(M, function, N)
    Sigma_1 = skewness(Mu_3, SKO)
    Sigma_2 = kurtosis_2(Mu_4, SKO)

    print('Мат. ожидание = ', round(M, 4))
    print('Дисперсия = ', round(D, 4))
    print('СКО =', round(SKO, 4))
    print('Асимметрия = ', round(Mu_3, 4))
    print('Коэффициент асимметрии = ', round(Sigma_1, 4))
    print('Коэффициент эксцесс = ', round(Mu_4, 4))
    print('Куртозис = ', round(Sigma_2, 4))


# Плотность распределения
def hist(func, bins, kde, color):
    sns.distplot(func, bins=bins, kde=kde, color=color)


# Функция автокорреляции
def acf(y, N):
    sr_y = np.mean(y)
    r_xx = [0 for i in range(N)]
    r_xx_zn = 0

    # Считаем знаменатель для АКФ
    for k in range(N):
        r_xx_zn += (y[k] - sr_y) ** 2

    for L in range(N):
        for k in range(N - L):
            r_xx[L] += (y[k] - sr_y) * (y[k + L] - sr_y)
        r_xx[L] /= r_xx_zn
    return r_xx


# Функция взаимной автокорреляции
def macf(y1, y2, N):
    sr_y1 = np.mean(y1)
    sr_y2 = np.mean(y2)
    r_xy = [0 for i in range(N)]
    r_xy_zn1 = 0.0
    r_xy_zn2 = 0.0

    for k in range(N):
        r_xy_zn1 += (y1[k] - sr_y1) ** 2
        r_xy_zn2 += (y2[k] - sr_y2) ** 2
    r_xy_zn = math.sqrt(r_xy_zn1 * r_xy_zn2)

    for L in range(N):
        for k in range(N - L):
            r_xy[L] += (y1[k] - sr_y1) * (y2[k] - sr_y2)
        r_xy[L] /= r_xy_zn
    return r_xy


# Гармонический процесс
def sin_harmony(f0, A, X, deltaT):  # Гармонический процесс (f0 = частота, Гц)
    sin_harmony = A * np.sin(2 * np.pi * f0 * X * deltaT)
    return sin_harmony


# Преобразование Фурье
def Fourie(func):
    Re = []
    Im = []
    C = []
    Cs = []
    N = len(func)
    for n in range(N):
        sumRe = 0
        sumIm = 0
        for k in range(N):
            sumRe += func[k] * np.cos((2 * np.pi * n * k) / N)
            sumIm += func[k] * np.sin((2 * np.pi * n * k) / N)
        re = sumRe / N
        im = sumIm / N

        Re.append(re)
        Im.append(im)
        C.append(math.sqrt(pow(re, 2) + pow(im, 2)))  # модуль комлпексного спектра (амплитудный спектр)
        Cs.append(re + im)  # Спектр Фурье
    return C, Cs


# Функция для обратного преобразования Фурье
def reverse_Fourie(Cs):
    newY = []
    N = len(Cs)
    for k in range(N):
        sumRe = 0
        sumIm = 0
        for n in range(N):
            sumRe += Cs[n] * np.cos((2 * np.pi * n * k) / N)
            sumIm += Cs[n] * np.sin((2 * np.pi * n * k) / N)

        newY.append(sumRe + sumIm)
    return newY


# Антишифт
def antishift(func, N):
    M = expected_value(func, N)
    newY = []
    for i in range(0, N):
        newY.append(func[i] - M)
    return newY


# Антиспайк
def spike_detector(ordinataY, range_y):
    y = np.copy(ordinataY)
    for i in range(len(ordinataY)):
        if abs(ordinataY[i]) > range_y:
            if i == 0:
                y[i] = ordinataY[i + 1] / 2
            elif i == len(ordinataY) - 1:
                y[i] = ordinataY[i - 1] / 2
            else:
                y[i] = (ordinataY[i - 1] + ordinataY[i + 1]) / 2
    return y


# Антитренд (оригинальный)
def antitrend2(func, L=10):
    trend_x = func.copy()
    for i in range(L // 2, len(func) - L // 2):
        trend_x[i] = func[i - L // 2:i + L // 2]
        trend_x[i] = np.mean(trend_x[i])

    trend_x2 = trend_x.copy()
    for i in range(len(func) - 1, len(func) - L - 2, -1):
        trend_x2[i] = func[i - L:i]
        trend_x2[i] = np.mean(trend_x2[i])

    for i in range(0, L + 1, 1):
        trend_x2[i] = func[i:i + L]
        trend_x2[i] = np.mean(trend_x2[i])

    trend_x[len(func) - L:] = trend_x2[len(func) - L:] - (trend_x2[len(func) - L - 1] - trend_x[len(func) - L - 1])
    trend_x[:L] = trend_x2[:L] - (trend_x2[L - 1] - trend_x[L - 1])
    # for i in range(len(func)-1,len(func)-L-2, -1):
    #     trend_x[i] = func[i-L:i]
    #     trend_x[i] = np.mean(trend_x[i])
    return trend_x


# Антитренд
def antitrend(y, N, L):
    a = 0
    y1 = []
    for i in range(N - L):
        for j in range(L):
            a += y[i + j]
        a /= L
        y1.append(a)
    return y1


# Функция для чтения файла dat
def binary_reader(filename):
    with open(filename, "rb") as binary_file:
        figures = []

        data = binary_file.read()
        for i in range(0, len(data), 4):
            pos = struct.unpack('f', data[i:i + 4])
            figures.append(pos[0])
        return figures


# Функция для управления сердцем
def herz_function(alpha, f0, dt, X):
    y = np.sin(2 * np.pi * f0 * dt * X) * np.exp(-alpha * X * dt)
    return y

# Функция для генерации тиков (в связке с сердцем)
def tiks(N, l):
    quantity_tiks = int(N / l)
    x = [i for i in range(N)]
    y = [0 for i in range(N)]
    for loc in range(1, quantity_tiks):
        # y[loc * l - 1] = random.randint(110, 130)
        y[loc * l - 1] = 110
    return x, y

# Функция связки y = x * h
def convolution(x, h):
    y = []
    N = len(x)
    M = len(h)
    total_sum = 0
    for k in range(N + M - 1):
        for m in range(M):
            index = k - m
            if index < 0:
                pass
            if index > N - 1:
                pass
            else:
                total_sum += x[index] * h[m]

        y.append(total_sum)
        total_sum = 0
    return y


def lpf(fcut, dt, m):
    lpw = [0 for i in range(0, m + 1)]

    dp = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165])

    arg = 2 * fcut * dt

    lpw[0] = arg
    arg *= math.pi

    for i in range(1, m + 1):
        lpw[i] = math.sin(arg * i) / (math.pi * i)

    lpw[m] /= 2

    sumg = lpw[0]
    for i in range(1, m + 1):
        _sum = dp[0]
        arg = math.pi * i / m

        for k in range(1, 4):
            _sum += 2 * dp[k] * math.cos(arg * k)

        lpw[i] *= _sum
        sumg += 2 * lpw[i]

    for i in range(len(lpw)):
        lpw[i] /= sumg

    answer = lpw[::-1]

    answer.extend(lpw[1::])
    return answer
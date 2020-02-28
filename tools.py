import numpy as np
import random
import struct
from scipy.io import wavfile
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
import pandas as pd
import scipy.integrate as integrate


def linear_trend(x, k=1, b=0):
    """
    Линейный тренд
    """
    return k * x + b


def exp_trend(x, a=-100, b=1):
    """
    Экспоненциальный тренд
    """
    return b * np.e ** (x / a)


def exp_trend_2(x, a=-100, b=100):
    """
    Экспоненциальный тренд
    """
    return b * np.e ** (x / a)


def exp_trend_3(x, a=-100000, b=1):
    """
    Экспоненциальный тренд
    """
    return b * np.e ** (x / a)


def normalize(numbers: np.array, S: int) -> np.array:
    """
    Функция нормирования
    """
    return (((numbers - min(numbers)) / (max(numbers) - min(numbers))) - 0.5) * 2 * S

def normalize_v2(numbers, S):
    """
    Функция нормирования
    """
    norm = []
    min_number = min(numbers)
    max_number = max(numbers)
    for number in numbers:
        norm.append((((number - min_number) / (max_number - min_number)) - 0.5) * 2 * S)
    return norm


def random_built_in(N, lo, hi):
    """
    Функция для генерации случайного процесса встроенным методом
    """
    arguments = np.zeros(N)

    for i in range(0, N):
        arguments[i] = random.uniform(lo, hi)
    return arguments


def shift(Y, const):
    """
    Функция для сдвига данных (shift)
    """
    for i in range(0, len(Y)):
        Y[i] += const
    return Y


def spikes(N, amplitude, delta, random_y=None):
    """
    Функция для генерации spikes
    """
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


def spikes2(N, Q, data, S):
    def spikes_low_norm(m, spikes_num, S):
        spikes_num_min = spikes_num[0]
        spikes_num_max = spikes_num[0]
        S10 = S * 10

        for k in range(m):
            if spikes_num[k] < spikes_num_min:
                spikes_num_min = spikes_num[k]
            if spikes_num[k] > spikes_num_max:
                spikes_num_max = spikes_num[k]

        for k in range(m):
            spikes_num[k] = (((spikes_num[k] - spikes_num_min) / (spikes_num_max - spikes_num_min)) - 0.5) * 2 * S10
        return spikes_num

    arr = np.zeros(N)
    m = int(N * 0.01 * Q)
    spikes_num = np.array([random.random() for i in range(m)])
    spikes_num = spikes_low_norm(m, spikes_num, S)

    for k in range(m):
        arr[k] = spikes_num[k]

    np.random.shuffle(arr)
    return data + arr

def spikes3(N, Q, S):
    def spikes_low_norm(m, spikes_num, S):
        m = int(m)
        spikes_num_min = spikes_num[0]
        spikes_num_max = spikes_num[0]
        S10 = S * 10

        for k in range(m):
            if spikes_num[k] < spikes_num_min:
                spikes_num_min = spikes_num[k]
            if spikes_num[k] > spikes_num_max:
                spikes_num_max = spikes_num[k]

        for k in range(m):
            spikes_num[k] = (((spikes_num[k] - spikes_num_min) / (spikes_num_max - spikes_num_min)) - 0.5) * 2 * S10
        return spikes_num

    arr = np.zeros(N)
    m = N * 0.01 * Q
    m = int(m)
    spikes_num = np.array([random.random() for i in range(m)])
    spikes_num = spikes_low_norm(m, spikes_num, S)
    for k in range(m):
        arr[k] = spikes_num[k]
    np.random.shuffle(arr)
    return arr



def additive(X, func):
    """
    Аддитивная модель
    """
    y_linear = linear_trend(X)
    y_random = func
    x_line_rand = [a2 + b for a2, b in zip(y_linear, y_random)]
    return x_line_rand


# def multiplicative(x, func):
#     """
#     Мультипликативная модель
#     """
#     y_linear = linear_trend(x)
#     y_random = func
#     x_line_rand = [a * b for a, b in zip(y_linear, y_random)]
#     return y_linear, x_line_rand


def multiplicative(N, x, trend_func):
    """
    Мультипликативная модель
    """
    trend = trend_func(np.arange(N))
    x_mul_trend = [a * b for a, b in zip(x, trend)]
    return x_mul_trend, trend


def multiplicative_2(x, trend):
    """
    Мультипликативная модель
    """
    x_mul_trend = [a * b for a, b in zip(x, trend)]
    return x_mul_trend


def anti_multiplicative(x, trend):
    """
    Анти мультипликативная модель
    """
    x_mul_trend = [a / b for a, b in zip(x, trend)]
    return x_mul_trend


def anti_multiplicative_2(x, trend):
    """
    Анти мультипликативная модель
    """
    x_mul_trend = [a / b for a, b in zip(x, trend)]
    return x_mul_trend


def expected_value(ordinata, N):
    """
    Математическое ожидание
    """
    sum = 0
    for i in range(0, N):
        sum += ordinata[i]
    return float(sum / N)


def dispersion(expected_value, ordinata, N):
    """
    Дисперсия
    """
    sum = 0
    for i in range(0, N):
        sum += np.power((ordinata[i] - expected_value), 2)
    return float(sum / N)


def standard_deviation(dispersion):
    """
    Стандартное отклонение
    """
    return float(np.sqrt(dispersion))


def asymmetry(expected_value, ordinata, N):
    """
    Асимметрия
    """
    sum = 0
    for i in range(0, N):
        sum += np.power((ordinata[i] - expected_value), 3)
    return float(sum / N)


def skewness(asymmetry, standard_deviation):
    """
    Коэффициент асимметрии
    """
    return float(asymmetry / np.power(standard_deviation, 3))


def ecses(expected_value, ordinata, N):
    """
    Эксцесс
    """
    sum = 0
    for i in range(0, N):
        sum += np.power((ordinata[i] - expected_value), 4)
    return float(sum / N)


def kurtosis_2(kurtosis, standard_deviation):
    """
    Куртозис
    """
    return float(kurtosis / np.power(standard_deviation, 4) - 3)


def general_statistics(function, N):
    """
    Функция для определения основных характеристик процесса
    """
    M = expected_value(function, N)
    D = dispersion(M, function, N)
    SKO = standard_deviation(D)
    Mu_3 = asymmetry(M, function, N)
    Mu_4 = ecses(M, function, N)
    Sigma_1 = skewness(Mu_3, SKO)
    Sigma_2 = kurtosis_2(Mu_4, SKO)

    print('Мат. ожидание = ', round(M, 4))
    print('Дисперсия = ', round(D, 4))
    print('СКО =', round(SKO, 4))
    print('Асимметрия = ', round(Mu_3, 4))
    print('Коэффициент асимметрии = ', round(Sigma_1, 4))
    print('Коэффициент эксцесс = ', round(Mu_4, 4))
    print('Куртозис = ', round(Sigma_2, 4))


def hist(func, bins, kde, color):
    """
    Плотность распределения ПЕРЕДЕЛАТЬ
    """
    # sns.distplot(func, bins=bins, kde=kde, color=color)


def acf(y, N):
    """
    Функция автокорреляции
    """
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


def macf(y1, y2, N):
    """
    Функция взаимной автокорреляции
    """
    sr_y1 = np.mean(y1)
    sr_y2 = np.mean(y2)
    r_xy = [0 for i in range(N)]
    r_xy_zn1 = 0.0
    r_xy_zn2 = 0.0

    for k in range(N):
        r_xy_zn1 += (y1[k] - sr_y1) ** 2
        r_xy_zn2 += (y2[k] - sr_y2) ** 2
    r_xy_zn = np.sqrt(r_xy_zn1 * r_xy_zn2)

    for L in range(N):
        for k in range(N - L):
            r_xy[L] += (y1[k] - sr_y1) * (y2[k] - sr_y2)
        r_xy[L] /= r_xy_zn
    return r_xy


def harmony_sin(X, t, f0):
    """
    Гармонический синусоидальный периодический процесс

    X : Амплитуда
    f0 : Циклическая частота в Гц
    t : Время

    """
    return X * np.sin(2 * np.pi * f0 * t)


def fourie(func):
    """
    Преобразование Фурье
    """
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
        C.append(np.sqrt(pow(re, 2) + pow(im, 2)))  # модуль комлпексного спектра (амплитудный спектр)
        Cs.append(re + im)  # Спектр Фурье
        print('fourie:', n)
    return C, Cs


def fourie_special(func):
    Re = []
    Im = []
    N = len(func)
    print(N)
    for n in range(round(N / 2)):
        sumRe = 0
        sumIm = 0
        for k in range(N):
            sumRe += func[k] * np.cos((2 * np.pi * n * k) / N)
            sumIm += func[k] * np.sin((2 * np.pi * n * k) / N)
        re = sumRe / N
        im = sumIm / N
        Re.append(re)
        Im.append(im)
    return Re, Im



def fourie_fast(func):
    """
    Преобразование Фурье
    """
    Re = []
    Im = []
    C = []
    Cs = []
    N = len(func)
    for n in range(round(N/2)):
        sumRe = 0
        sumIm = 0
        for k in range(N):
            sumRe += func[k] * np.cos((2 * np.pi * n * k) / N)
            sumIm += func[k] * np.sin((2 * np.pi * n * k) / N)
        re = sumRe / N
        im = sumIm / N

        Re.append(re)
        Im.append(im)
        C.append(np.sqrt(pow(re, 2) + pow(im, 2)))  # модуль комлпексного спектра (амплитудный спектр)
        Cs.append(re + im)  # Спектр Фурье
        print('fourie:', n)
    return C, Cs


def del_complex(re1, im1, re2, im2):
    re = (re1 * re2 + im1 * im2) / (re2 * re2 + im2 * im2)
    im = (im1 * re2 - re1 * im2) / (re2 * re2 + im2 * im2)
    return re, im


def reverse_fourie(Cs):
    """
    Функция для обратного преобразования Фурье
    """
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


def antishift(func, N):
    """
    Антишифт
    """
    M = expected_value(func, N)
    newY = []
    for i in range(0, N):
        newY.append(func[i] - M)
    return newY


def spike_detector(ordinataY, range_y):
    """
    Антиспайк
    """
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


def anti_spike(data, N, S):
    remi = []
    for i in range(N):
        if abs(data[i]) > (S + 10):
            remi.append(i)
    for item in remi:
        data[item] = (data[item - 1] + data[item + 1]) / 2
    return data


def anti_trend2(func, L=10):
    """
    Антитренд (оригинальный)
    """
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


def anti_trend(y, N, L=10):
    """
    Антитренд
    """
    a = 0
    y1 = []
    for i in range(N - L):
        for j in range(L):
            a += y[i + j]
        a /= L
        y1.append(a)
    return y1


def binary_reader(filename):
    """
    Функция для чтения файла dat
    """
    with open(filename, "rb") as binary_file:
        figures = []

        data = binary_file.read()
        for i in range(0, len(data), 4):
            pos = struct.unpack('f', data[i:i + 4])
            figures.append(pos[0])
        return figures


def herz_function(alpha, f0, dt, X):
    """
    Функция для управления сердцем
    """
    y = np.sin(2 * np.pi * f0 * dt * X) * np.exp(-alpha * X * dt)
    return y


def tiks(N, l):
    """
    Функция для генерации тиков (в связке с сердцем)
    """
    quantity_tiks = int(N / l)
    x = [i for i in range(N)]
    y = [0 for i in range(N)]
    for loc in range(1, quantity_tiks):
        # y[loc * l - 1] = random.randint(110, 130)
        y[loc * l - 1] = 110
    return x, y


def convolution(x, h):
    """
    Функция связки y = x * h
    """
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


def mean_square(data, N, T):
    data_mean_square = []
    for i in range(N):
        sum = 0
        if i < (N - T):
            for k in range(i + T):
                sum += (data[k] * data[k]) / T
        if sum:
            data_mean_square.append(abs(sum))
        print('mean_square:', i)
    return data_mean_square


def square2(data, T, i):
    sum = 0
    for k in range(i, T + i):
        sum += abs(data[k] ** 2)
    sum /= T
    return sum


def window(data, N, T):
    data_mean_square = []
    for i in range(N - T):
        data_mean_square.append(square2(data, T, i))
        print(i)
    return data_mean_square


def mean_square2(x, T):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return ((cumsum[T:] - cumsum[:-T]) / float(T)) ** 2


def running_mean_square(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0))
  return ((cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize) ** 2


def read_wave_mono(filename):
    fs, data = wavfile.read(filename)
    return data, len(data), fs


def read_wave_stereo(filename):
    fs, data = wavfile.read(filename)
    data = [x[1] for x in data]
    return data, len(data), fs


def write_wave(fs, data):
    wavfile.write("./../files/example.wav", fs, data)


def mean_square_new(data):
    for i in data:
        sum += i**2
    sum /= len(data)


def lpf(m, dt, fcut):
    """
    lpf
    """
    lpw = [0 for i in range(0, m + 1)]

    dp = np.array([0.35577019, 0.24369830, 0.07211497, 0.00630165])

    arg = 2 * fcut * dt

    lpw[0] = arg
    arg *= np.pi

    for i in range(1, m + 1):
        lpw[i] = np.sin(arg * i) / (np.pi * i)

    lpw[m] /= 2

    sumg = lpw[0]
    for i in range(1, m + 1):
        _sum = dp[0]
        arg = np.pi * i / m

        for k in range(1, 4):
            _sum += 2 * dp[k] * np.cos(arg * k)

        lpw[i] *= _sum
        sumg += 2 * lpw[i]

    for i in range(len(lpw)):
        lpw[i] /= sumg

    answer = lpw[::-1]

    answer.extend(lpw[1::])
    return answer


def hpf(m, dt, fcut):
    hpw = []
    lpw = lpf(m, dt, fcut)
    for i in range(2 * m+1):
        if i == m:
            hpw.append(1 - lpw[i])
        else:
            hpw.append(-lpw[i])

    return hpw


def bpf(m, dt, fc1, fc2):
    lpw1 = lpf(m, dt, fc1)
    lpw2 = lpf(m, dt, fc2)
    bpw = []
    for i in range(2*m+1):
        bpw.append(lpw2[i] - lpw1[i])

    return bpw


# def bsf(self, m, dt, fc1, fc2):
#     lpw1 = Test19.lpf_re(self, m, dt, fc1)
#     lpw2 = Test19.lpf_re(self, m, dt, fc2)
#     bsw = []
#
#     for i in range(2*m+1):
#         if i == m:
#             bsw.append(1 + lpw1[i] - lpw2[i])
#         else:
#             bsw.append(lpw1[i] - lpw2[i])
#
#     return bsw


def ecg_func(data, dt, alpha=30, f0=10):
    return np.sin(2 * np.pi * f0 * dt * data) * np.exp(-alpha * data * dt)


def kardio(m, dt):
    ecg = np.zeros(m)
    for i in range(m):
        ecg[i] = ecg_func(i, dt)
    return ecg


    # N, M = len(ecg), len(ticks_mass)
    # print(N, M)
    # conv_mass = []
    # sum_of_conv = 0
    # for k in range(N + M - 1):
    #     for m in range(M):
    #         if k - m < 0:
    #             pass
    #         if k - m > N - 1:
    #             pass
    #         else:
    #             sum_of_conv += ecg[k - m] * ticks_mass[m]
    #
    #     conv_mass.append(sum_of_conv)
    #     sum_of_conv = 0
    # print(len(ecg), len(ticks_mass))
    # return ecg

def read_png(file):
    img = mpimg.imread(file)
    return img


def read_jpg_grayscale(file):
    image = Image.open(file).convert('L')
    return image


def read_xcr():
    pass


def pillow_image_grayscale_resize(image, factor, type, mode):
    pix = image.load()  # Выгружаем значения пикселей
    w, h = image.size[0], image.size[1]

    if mode == 'increase':
        new_w, new_h = int(w * factor), int(h * factor)
    elif mode == 'decrease':
        new_w, new_h = int(w / factor), int(h / factor)
    else:
        raise ValueError('Wrong mode')

    if type == 'nearest':
        image_nearest_resized = Image.new('L', (new_w, new_h))
        draw = ImageDraw.Draw(image_nearest_resized)  # Создаем инструмент для рисования
        for col in range(new_w):
            for row in range(new_h):
                if mode == 'increase':
                    p = pix[int(col / factor), int(row / factor)]
                elif mode == 'decrease':
                    p = pix[int(col * factor), int(row * factor)]
                else:
                    raise ValueError('Wrong mode')
                draw.point((col, row), p)
        image_resized = image_nearest_resized

    elif type == 'bilinear':
        image_bilinear_rows = Image.new('L', (new_w, new_h))
        draw = ImageDraw.Draw(image_bilinear_rows)  # Создаем инструмент для рисования
        for col in range(1, (new_w - 1)):
            for row in range(1, (new_h - 1)):
                if mode == 'increase':
                    r1 = pix[int(col / factor), int((row - 1) / factor)]
                    r2 = pix[int(col / factor), int((row + 1) / factor)]
                elif mode == 'decrease':
                    r1 = pix[int(col * factor), int((row - 1) * factor)]
                    r2 = pix[int(col * factor), int((row + 1) * factor)]
                else:
                    raise ValueError('Wrong mode')
                p = int((r1 + r2) / 2)
                draw.point((col, row), p)
            if mode == 'increase':
                draw.point((col, 0), pix[int(col / factor), int(0 / factor)])
                draw.point((col, new_h), pix[int(col / factor), int((new_h - 1) / factor)])
            elif mode == 'decrease':
                draw.point((col, 0), pix[int(col * factor), int(0 * factor)])
                draw.point((col, new_h), pix[int(col * factor), int((new_h - 1) * factor)])
            else:
                raise ValueError('Wrong mode')

        pix_bilinear_rows = image_bilinear_rows.load()
        image_bilinear_resized = Image.new('L', (new_w, new_h))
        draw2 = ImageDraw.Draw(image_bilinear_resized)  # Создаем инструмент для рисования

        for row in range(1, (new_h - 1)):
            for col in range(1, (new_w - 1)):
                r1 = pix_bilinear_rows[int((col - 1)), int(row)]
                r2 = pix_bilinear_rows[int((col + 1)), int(row)]
                p = int((r1 + r2) / 2)
                draw2.point((col, row), p)
            draw2.point((0, row), pix_bilinear_rows[int(0), int(row)])
            draw2.point((new_w, row), pix_bilinear_rows[int((new_w - 1)), int(row)])

        image_resized = image_bilinear_resized

    else:
        raise ValueError('Wrong type')

    return image_resized


def pillow_image_grayscale_negative(image):
    pix = image.load()
    w, h = image.size[0], image.size[1]

    image_negative = Image.new('L', (w, h))
    draw = ImageDraw.Draw(image_negative)

    for col in range(w):
        for row in range(h):
            draw.point((col, row), 255 - pix[col, row])

    return image_negative


def pillow_image_grayscale_gammacorr(image, C, Y):
    pix = image.load()
    w, h = image.size[0], image.size[1]
    gammacorr = []

    image_gammacorr = Image.new('L', (w, h))
    draw = ImageDraw.Draw(image_gammacorr)

    for col in range(w):
        for row in range(h):
            gammacorr.append(int(C * pix[col, row] ** Y))

    gammacorr_norm = normalize_v2(gammacorr, 255)

    i = 0
    for col in range(w):
        for row in range(h):
            draw.point((col, row), int(gammacorr_norm[i]))
            i += 1

    return image_gammacorr


def pillow_image_grayscale_log(image, C):
    pix = image.load()
    w, h = image.size[0], image.size[1]
    log = []

    image_log = Image.new('L', (w, h))
    draw = ImageDraw.Draw(image_log)

    for col in range(w):
        for row in range(h):
            log.append(C * np.log(pix[col, row] + 1))

    log_norm = normalize_v2(log, 255)

    i = 0
    for col in range(w):
        for row in range(h):
            draw.point((col, row), int(log_norm[i]))
            i += 1

    return image_log


def pillow_image_grayscale_grad_preobr(r):
    func = P()
    CDF = integrate.quad(func, 0, r)
    return 0


def pillow_image_grayscale_hist(image):
    pix = list(image.getdata())
    data = pd.Series(pix)
    return data
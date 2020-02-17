# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import rfft, rfftfreq
import temp.model as model
import math

# Параметры
N = 1000  # Число значений
X = np.arange(0, N)  # Создание списка значений для оси X
S = 10  # Диапазон значений X
M = 100  # Число интервалов для функции распределения
L = 100
deltaT = 0.002  # Частота дискретизации
A = 100

RandomProcess = model.PyRandom(N, -S, S)  # Случайный процесс built-in
RandomProcessLemer = model.LemerRandom(N, -S, S)  # Случайный процесс Lemer


# -----------------------------------------------------
# Четыре базовых тренда
# -----------------------------------------------------

def task_trends():
    K = 1
    B = 1000

    ALPHA = 150  # Значение ALPHA для графика функции
    BETA = 5  # Значение BETA для графика функции

    Y_exp = model.exp_function(X, BETA, ALPHA)  # Экпоненциальная функция
    MinusY_exp = model.exp_function(-X, BETA, ALPHA)  # Обратная экспоненциальная функция
    Y_linear = model.linear_function(K, X, B)  # Лиенйная функция
    MinusY_linear = model.linear_function(K, -X, B)  # Обратная линейная функция

    plt.figure(1)

    plt.subplot(221)
    plt.plot(X, Y_linear, color='green')

    plt.subplot(222)
    plt.plot(X, MinusY_linear, color='red')

    plt.subplot(223)
    plt.plot(X, Y_exp, color='green')

    plt.subplot(224)
    plt.plot(X, MinusY_exp, color='red')
    plt.show()


# -----------------------------------------------------
# Нормирование функций
# -----------------------------------------------------

def task_norm():
    X_norm = model.normalize(X, S)

    plt.figure(2)

    plt.subplot(221)
    plt.plot(X_norm, Y_linear, color='blue')

    plt.subplot(222)
    plt.plot(X_norm, MinusY_linear, color='red')

    plt.subplot(223)
    plt.plot(X_norm, Y_exp, color='blue')

    plt.subplot(224)
    plt.plot(X_norm, MinusY_exp, color='red')

    plt.show()


############### 5 тренд ###################

# plt.plot(X, model.get_common_function(K, B, ALPHA, BETA, N), label="3 functions")
# plt.legend()
# plt.show()

# -----------------------------------------------------
# Случайные процессы
# -----------------------------------------------------

def task_random_processes():
    fig, (rnd, myrnd) = plt.subplots(1, 2, figsize=(12, 6))

    rnd.set_title(r'Случайный процесс (встроенный генератор)', fontsize=13, fontweight='bold', pad=15)
    rnd.set_ylabel(r'S')
    rnd.set_xlabel(r'N')
    rnd.plot(X, RandomProcess)  # Отображение случайного процесса built-in методом
    myrnd.set_title('Случайный процесс (собственный генератор)', fontsize=13, fontweight='bold', pad=15)
    myrnd.plot(X, RandomProcessLemer, color='purple')  # Отображение случайного процесса
    myrnd.set_xlabel(r'N')
    plt.show()


# -----------------------------------------------------
# Сдвиг данных (shift)
# -----------------------------------------------------

def task_shift():
    plt.figure(1)
    plt.plot(X, RandomProcess)
    plt.plot(X, model.shift(RandomProcess, 5))
    plt.show()


# -----------------------------------------------------
# Неправдоподобные значения (spikes)
# -----------------------------------------------------

def task_spikes():
    plt.figure(1)
    spikeX, spikeY = model.spikes(N, 3, 1)
    plt.plot(spikeX, spikeY)
    plt.show()


############### Оценка стационарности ###################

# LemerProcess = model.LemerRandom(N, -1, 1)
# model.statistic(LemerProcess)
# plt.plot(X, LemerProcess, color='black')
# plt.show()

# PythonRandomProcess = model.PyRandom(N, -1, 1)
# model.statistic(PythonRandomProcess)
# plt.plot(X, PythonRandomProcess, color='grey')
# plt.show()


# -----------------------------------------------------
# Работа с шумами (аддитивные и мультипликативные модели)
# -----------------------------------------------------

# Аддитивная модель
def task_additive_model():
    fig, (additive, inv_additive) = plt.subplots(1, 2, figsize=(12, 6))

    additive.plot(X, model.additive(X, RandomProcess), color='black')
    additive.plot(X, model.linear_function(X))

    inv_additive.plot(X, model.additive(-X, RandomProcess), color='black')
    inv_additive.plot(X, model.linear_function(-X))
    plt.show()


# Мультипликативная модель
# fig, (multi, inv_multi) = plt.subplots(1, 2, figsize=(12, 6))
#
# X_multi, Y_multi = model.multiplicative(X, RandomProcess)
# # print(type(Y_multi))
# multi.plot(X_multi, Y_multi, color='black')
# # multi.plot(X, model.linear_function(X))
# # inv_multi.plot(X,model.multiplicative(-X, RandomProcess), color='black')
# # inv_multi.plot(X, model.linear_function(-X))
# plt.show()

# model.multiplicative2(N, S)

# -----------------------------------------------------
# Статистические параметры
# -----------------------------------------------------

def task_statistics():
    print('------------------------------------------------------------')
    print('ВСТРОЕННЫЙ ГЕНЕРАТОР')
    print('------------------------------------------------------------')

    model.general_statistics(RandomProcess, N)

    print('------------------------------------------------------------')
    print('СОБСТВЕННЫЙ ГЕНЕРАТОР')
    print('------------------------------------------------------------')

    model.general_statistics(RandomProcessLemer, N)

    task_random_processes()


# -----------------------------------------------------
# Автокорреляционная и взаимная корреляционная функция
# -----------------------------------------------------

def task_correlation():
    plt.figure(1, figsize=(12, 6))

    plt.subplot(321)
    plt.title('Случайный процесс (встроенный генератор)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('N')
    plt.plot(X, RandomProcess)

    plt.subplot(323)
    plt.title('АКФ встроенного генератора')
    plt.xlabel('N')
    plt.plot(X, model.acf(RandomProcess, N))

    plt.subplot(325)
    plt.title('Взаимная корреляционная функция')
    plt.xlabel('N')
    plt.plot(X, model.macf(RandomProcess, RandomProcess, N))

    plt.subplot(322)
    plt.title('Случайный процесс (собственный генератор)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('N')
    plt.plot(X, RandomProcessLemer, color='purple')  # Отображение случайного процесса

    plt.subplot(324)
    plt.title('АКФ собственного генератора')
    plt.xlabel('N')
    plt.plot(X, model.acf(RandomProcessLemer, N), color='purple')

    plt.subplot(326)
    plt.title('Взаимная корреляционная функция')
    plt.xlabel('N')
    plt.plot(X, model.macf(RandomProcessLemer, RandomProcessLemer, N), color='purple')

    plt.show()


# -----------------------------------------------------
# Плотность распределения двух процессов (гистограмма)
# -----------------------------------------------------

def task_density():
    plt.figure(1, figsize=(6, 8))

    plt.subplot(2, 1, 1)
    plt.title('Случайный процесс (встроенный генератор)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('N')
    plt.ylabel('S')
    plt.plot(X, RandomProcess)

    plt.subplot(2, 1, 2)
    plt.title('Плотность распределения встроенного генератора')
    plt.xlabel(r'$S$')
    plt.ylabel(r'$p(S)$')
    model.hist(RandomProcess, M, False, 'blue')

    plt.figure(2, figsize=(6, 8))

    plt.subplot(2, 1, 1)
    plt.title('Случайный процесс (собственный генератор)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('N')
    plt.ylabel('S')
    plt.plot(X, RandomProcessLemer, color='red')
    plt.subplot(2, 1, 2)
    plt.title('Плотность распределения собственного генератора')
    plt.xlabel(r'$S$')
    plt.ylabel(r'$p(S)$')
    model.hist(RandomProcessLemer, M, False, 'red')

    plt.show()


# -----------------------------------------------------
# Гармонический процесс и преобразование Фурье (спектр)
# -----------------------------------------------------

def task_Furie():
    Y = model.sin_harmony(11, A, X, deltaT)  # 11 Гц
    Y1 = model.sin_harmony(110, A, X, deltaT)  # 110 Гц
    Y2 = model.sin_harmony(250, A, X, deltaT)  # 250 Гц
    Y3 = model.sin_harmony(510, A, X, deltaT)  # 510 Гц
    C, Cs = model.Fourie(Y, N)
    C1, Cs1 = model.Fourie(Y1, N)
    C2, Cs2 = model.Fourie(Y2, N)
    C3, Cs3 = model.Fourie(Y3, N)

    # Гармонические процессы
    plt.figure(1)
    plt.subplot(221)
    plt.plot(X, Y, color='green')

    plt.subplot(222)
    plt.plot(X, Y1, color='green')

    plt.subplot(223)
    plt.plot(X, Y2, color='green')

    plt.subplot(224)
    plt.plot(X, Y3, color='green')

    # Спектры гармонических процессов
    plt.figure(2)
    plt.subplot(221)
    plt.plot(X // 2, C, color='red')

    plt.subplot(222)
    plt.plot(X // 2, C1, color='red')

    plt.subplot(223)
    plt.plot(X // 2, C2, color='red')

    plt.subplot(224)
    plt.plot(X // 2, C3, color='red')

    plt.show()


def task_Furie_new():
    plt.figure(1, figsize=(10, 6))
    plt.subplot(421)
    plt.title(r'$Входной синусоидальный сигнал$', fontsize=13, fontweight='bold', pad=15)
    plt.plot(X, model.sin_harmony(11, A, X, deltaT))
    plt.subplot(422)
    plt.title(r'$Спектр синусоидального сигнала$', fontsize=13, fontweight='bold', pad=15)
    plt.plot(rfftfreq(N, 1. / 2000), np.abs(rfft(model.sin_harmony(11, A, X, deltaT))) / N, color='purple')

    plt.subplot(423)
    plt.plot(X, model.sin_harmony(110, A, X, deltaT))
    plt.subplot(424)
    plt.plot(rfftfreq(N, 1. / 2000), np.abs(rfft(model.sin_harmony(110, A, X, deltaT))) / N, color='purple')

    plt.subplot(425)
    plt.plot(X, model.sin_harmony(250, A, X, deltaT))
    plt.subplot(426)
    plt.plot(rfftfreq(N, 1. / 2000), np.abs(rfft(model.sin_harmony(250, A, X, deltaT))) / N, color='purple')

    plt.subplot(427)
    plt.xlabel(r'$Время, сек.$')
    plt.plot(X, model.sin_harmony(510, A, X, deltaT))
    plt.subplot(428)
    plt.xlabel(r'$Частота, Гц$')
    plt.plot(rfftfreq(N, 1. / 2000), np.abs(rfft(model.sin_harmony(510, A, X, deltaT))) / N, color='purple')

    plt.show()


# -----------------------------------------------------
# Antishift
# -----------------------------------------------------

def task_antishift():
    process_shifted = model.shift(RandomProcess, (10 * S))
    process_original = model.antishift(process_shifted, N)

    fig, (shifted, orig) = plt.subplots(2, 1, figsize=(10, 7))

    shifted.set_title(r'$Случайный процесс$', fontsize=13, fontweight='bold', pad=15)
    shifted.set_ylabel(r'$S$')
    shifted.set_ylim([-S, 15 * S])
    shifted.plot(X, process_shifted)

    orig.set_title(r'$Случайный процесс antishift$', fontsize=13, fontweight='bold', pad=15)
    orig.set_xlabel(r'$N$')
    orig.set_ylabel(r'$S$')
    orig.set_ylim([-S, 15 * S])
    orig.plot(X, process_original, color='purple')

    plt.show()


# -----------------------------------------------------
# Antispike
# -----------------------------------------------------

def task_antispike():
    spikeX, spikeY = model.spikes(N, 3, 100, RandomProcess)
    antispikeY = model.spike_detector(spikeY, 10)

    fig, (spike, antispike) = plt.subplots(2, 1, figsize=(10, 7))

    spike.set_title('Случайный процесс с выбросами', fontsize=13, fontweight='bold', pad=15)
    spike.set_ylabel('S')
    spike.plot(spikeX, spikeY)  # Отображение случайного процесса built-in методом

    antispike.set_title('Случайный процесс antispike', fontsize=13, fontweight='bold', pad=15)
    antispike.set_xlabel('N')
    antispike.set_ylabel('S')
    antispike.plot(spikeX, antispikeY, color='purple')  # Отображение случайного процесса built-in методом

    plt.show()


# -----------------------------------------------------
# Antitrend
# -----------------------------------------------------

def task_antitrend():
    fig, (trend, antitrend) = plt.subplots(2, 1, figsize=(10, 6))
    # plt.figure(1, figsize=(10, 6))
    # plt.subplot(221)
    # plt.plot(X, RandomProcess)
    # plt.xlabel('x')
    # plt.ylabel('y')

    k, b = 0.1, 1000
    Y = k * X + b
    Y2 = RandomProcess + Y
    # plt.subplot(222)
    trend.plot(X, Y2)
    trend.set_ylabel(r'$S$')
    trend.set_title('Случайный тренд', fontsize=13, fontweight='bold', pad=15)

    X1 = np.arange(0, N - L, 1)
    Y3 = model.antitrend(Y2, N, L)
    # plt.subplot(223)
    # plt.plot(X1, Y3)
    # plt.xlabel('x')
    # plt.ylabel('y')

    for i in range(N - L):
        Y3[i] -= Y2[i]
    # antitrend.subplot(224)
    antitrend.plot(X1, Y3, color='purple')
    antitrend.set_title('Антитренд', fontsize=13, fontweight='bold', pad=15)
    antitrend.set_xlabel(r'$N$')
    antitrend.set_ylabel(r'$S$')

    plt.show()


# -----------------------------------------------------
# Процесс + спектр, спайки + спектр
# -----------------------------------------------------

def task_spike_trend_spectrum():
    plt.figure(1, figsize=(10, 6))
    plt.subplot(221)
    plt.plot(X, RandomProcess)
    plt.title('Случайный процесс')
    plt.subplot(222)
    C, Cs = model.Fourie(RandomProcess, N)
    plt.xlim(0, N // 2)
    plt.title('Спектр')
    plt.plot(X, C)

    plt.subplot(223)
    spikeX, spikeY = model.spikes(N, 30, 100)
    plt.title('Случайные выбросы')
    plt.plot(X, spikeY, color='purple')
    plt.subplot(224)
    C, Cs = model.Fourie(spikeY, N)
    plt.xlim(0, N // 2)
    plt.plot(X, C, color='purple')
    plt.title('Спектр случайных выбросов')

    plt.show()


# -----------------------------------------------------
# Гармонический процесс + спектр, спайки с гармоническим процессом + спектр
# -----------------------------------------------------

def task_spike_harmony_spectrum():
    plt.figure(1, figsize=(10, 6))
    plt.subplot(221)
    plt.title('Случайный гармонический процесс')
    y_har = model.sin_harmony(11, A, X, deltaT)
    func_comb = [a * b for a, b in zip(y_har, RandomProcess)]

    plt.plot(X, func_comb)
    plt.subplot(222)
    plt.title('Спектр')
    C, Cs = model.Fourie(func_comb, N)
    plt.ylim(0, 50)
    plt.plot(X // 2, C)

    plt.subplot(223)
    spikeX, spikeY = model.spikes(N, 1000, 30, y_har)
    plt.plot(X, spikeY, color='purple')
    plt.title('Гармонический процесс с выбросами')
    plt.ylim(-400, 400)
    plt.subplot(224)
    C, Cs = model.Fourie(spikeY, y_har)
    plt.plot(X // 2, C, color='purple')
    plt.title('Спектр')

    plt.show()


# -----------------------------------------------------
# Чтение документа
# -----------------------------------------------------

def task_doc_reader():
    dT = 0.001
    figures = model.binary_reader(
        "/Users/deejayart/Documents/UNI/Методы анализа экспериментальных данных/model/doc.dat")
    xs = []
    t = 0
    for x in range(len(figures)):
        xs.append(t)
        t += 0.003

    plt.subplot(221)
    plt.plot(xs, figures, color='black')

    plt.subplot(222)
    plt.plot(xs, figures, color='green')

    norm = model.spike_detector(figures, 400)
    plt.subplot(223)
    plt.plot(xs, norm, color='red')

    plt.subplot(224)
    C, Cs = model.Fourie(figures)
    x_fourie = np.arange(0, 1000, 1)
    # plt.plot(x_fourie, C[0:2000 // 2], color='purple')
    # print(x_fourie[:int(1 / (2 * dT))])
    plt.plot(x_fourie[:int(1 / (2 * dT))], C[0:int(1 / (2 * dT))], color='purple')
    # plt.plot(rfftfreq(x_fourie[:int(1 / (2 * dT))]), np.abs(rfft(figures)) / x_fourie, color='purple')
    plt.ylabel('Амплитуда')
    plt.xlabel('Частота, Гц')

    plt.show()


# -----------------------------------------------------
# Обратный Фурье
# -----------------------------------------------------

def task_reverse_Fourie():
    C, Cs = model.Fourie(model.sin_harmony(4, A, X, deltaT))
    reverse_sin = model.reverse_Fourie(Cs)

    plt.figure(1, figsize=(10, 6))

    plt.subplot(221)
    plt.plot(X, model.sin_harmony(4, A, X, deltaT))
    plt.title('1. Гармонический процесс (исходный)', fontsize=13)

    plt.subplot(222)
    plt.plot(X[0:N // 2], C[0:N // 2])
    plt.title('2. Амплитуда', fontsize=13)

    plt.subplot(223)
    plt.plot(X, reverse_sin, color='purple')
    plt.title('3. Обратное преобразование Фурье', fontsize=13)

    plt.subplot(224)
    plt.plot(X, Cs, color='purple')
    plt.title('4. Комплексный спектр', fontsize=13)
    plt.show()


def task_cardiogram():
    # Параметры для сердца
    deltaT = 0.005  # Частота дискретизации
    alpha = 25  # Значение alpha для экспоненты
    f0 = 10  # Частота, гЦ
    l = 250  # Шаг между тиками (N/l = количество реализаций)

    herz = model.herz_function(alpha, f0, deltaT, X)
    herz_norm = herz / herz.max()
    tiksX, tiksY = model.tiks(N, l)
    cardiogram = model.convolution(tiksY, herz_norm)

    fig, (herz, tiks) = plt.subplots(2, 1, figsize=(10, 7))
    herz.set_title('Сердце', fontsize=13, fontweight='bold', pad=15)
    herz.plot(np.arange(0, N), herz_norm)
    tiks.plot(tiksX, tiksY)
    plt.show()

    plt.title('Кардиограмма', fontsize=13, fontweight='bold', pad=15)
    plt.plot(np.arange(0, len(cardiogram)), cardiogram, color='red')
    plt.show()

def task_cardiogram_freq():
    # Параметры для сердца
    deltaT = 0.005  # Частота дискретизации
    alpha = 25  # Значение alpha для экспоненты
    f0 = 10  # Частота, гЦ
    l = 250  # Шаг между тиками (N/l = количество реализаций)

    herz = model.herz_function(alpha, f0, deltaT, X)
    herz_norm = herz / herz.max()
    tiksX, tiksY = model.tiks(N, l)
    cardiogram = model.convolution(tiksY, herz_norm)

    fig, (herz, tiks) = plt.subplots(2, 1, figsize=(10, 7))
    herz.set_title('Сердце', fontsize=13, fontweight='bold', pad=15)
    herz.plot(np.arange(0, N), herz_norm)
    tiks.plot(tiksX, tiksY)
    plt.show()

    plt.title('Кардиограмма', fontsize=13, fontweight='bold', pad=15)
    plt.plot(np.arange(0, len(cardiogram)), cardiogram, color='red')
    plt.show()



if __name__ == "__main__":
    # task_trends() # 4 базовых тренда
    # task_norm() # Нормирование функции
    # task_random_processes() # Случайные процессы
    # task_shift() # Свдиг данных
    # task_spikes() # Случайные выбросы
    # task_additive_model() # Аддитивная модель

    # task_statistics() # Статистические параметры
    # task_correlation() # Автокорреляционная и взаимная корреляционная функции
    # task_density() # Плотность вероятности
    # task_Furie() # Преобразование Фурье
    # task_Furie_new() # Преобразование Фурье упрщенное
    # task_antishift() # Antishift
    # task_antispike() # Antispike
    # task_antitrend()  # Antitrend
    # task_spike_trend_spectrum()  # Процесс + спектр, спайки + спектр
    # task_spike_harmony_spectrum()  # Гармонический процесс + спектр, спайки с гармоническим процессом + спектр
    # task_doc_reader()  # Анализ документа

    # task_reverse_Fourie() # Обратное Фурье преобразование
    task_cardiogram()  # Отображение кардиограммы
    # task_cardiogram_freq() # Отображение частот графиков кардиограммы

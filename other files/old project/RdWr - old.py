import Model
import Analysis
import Proc
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.fftpack
import wave
import struct
from scipy.signal import butter, lfilter, freqz, filtfilt
import scipy.signal as ss
import pandas as pd
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import random

class IO:
    def __init__(self, a, b, N, S, rng):
        self.a = a
        self.b = b
        self.N = N
        self.S = S
        self.rng = rng
        self.t = np.linspace(0, self.N, self.N)
        self.x = Model.Random.x_random(self)
        self.x = Norm.x_norm(self)
        self.T = np.pi
        self.w = 2 * np.pi / self.T
        self.arr = []

    def set_a(self, a):
        self.a = a

    def get_a(self):
        return self.a

    def set_b(self, b):
        self.b = b

    def get_b(self):
        return self.b

    def set_N(self, N):
        self.N = N

    def get_N(self):
        return self.N

    def set_S(self, S):
        self.S = S

    def get_S(self):
        return self.S

    def recalculation_t(self):
        self.t = np.linspace(0, self.N, self.N)

    def recalculation_x(self):
        self.x = Model.Random.x_random(self)


class Norm(IO):
    def x_norm(self):
        x_min = self.x[0]
        x_max = self.x[0]

        for k in range(self.N):
            if self.x[k] < x_min:
                x_min = self.x[k]
            if self.x[k] > x_max:
                x_max = self.x[k]

        for k in range(self.N):
            self.x[k] = (((self.x[k] - x_min) / (x_max - x_min)) - 0.5) * 2 * self.S

        return self.x

    def spikes_high_norm(self, m, spikes_num):
        spikes_num_min = spikes_num[0]
        spikes_num_max = spikes_num[0]
        SS = self.S * self.S

        for k in range(m):
            if spikes_num[k] < spikes_num_min:
                spikes_num_min = spikes_num[k]
            if spikes_num[k] > spikes_num_max:
                spikes_num_max = spikes_num[k]

        for k in range(m):
            spikes_num[k] = (((spikes_num[k] - spikes_num_min) / (spikes_num_max - spikes_num_min)) - 0.5) * 2 * SS

    def spikes_low_norm(self, m, spikes_num):
        spikes_num_min = spikes_num[0]
        spikes_num_max = spikes_num[0]
        S10 = self.S * 10

        for k in range(m):
            if spikes_num[k] < spikes_num_min:
                spikes_num_min = spikes_num[k]
            if spikes_num[k] > spikes_num_max:
                spikes_num_max = spikes_num[k]

        for k in range(m):
            spikes_num[k] = (((spikes_num[k] - spikes_num_min) / (spikes_num_max - spikes_num_min)) - 0.5) * 2 * S10


class Display(IO):
    def view_states(self):
        print("a = {}".format(self.a))
        print("b = {}".format(self.b))
        print("N = {}".format(self.N))
        print("S = {}".format(self.S))
        print("t = some array")
        print("x1 = some array")
        print("x2 = some array")

    def view_t(self):
        print(self.t)

    def view_x(self):
        print(self.x)

    def view_graph_trends(self):
        fig, axs = plt.subplots(5, 1, figsize=(8, 10))
        axs[0].plot(self.t, Model.Trend.trend1(self), label="x = a * t + b, a > 0")
        axs[0].legend()
        axs[1].plot(self.t, Model.Trend.trend2(self), label="x = a * t + b, a < 0")
        axs[1].legend()
        axs[2].plot(self.t, Model.Trend.trend3(self), label="x = b * exp ^ (a * t), a > 0")
        axs[2].legend()
        axs[3].plot(self.t, Model.Trend.trend4(self), label="x = b * exp ^ (a * t), a < 0")
        axs[3].legend()
        axs[4].plot(self.t, Model.Trend.trend5(self),
                    label="x = 15, t < 3; x = b * exp ^ (a * t), 3 <= t < 7; x = (-a) * t + b, t >= 7")
        axs[4].legend()
        plt.show()

    def view_graph_x(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, self.x)
        ax.set_xlim(0, self.N)
        ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x_sum_trend1(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, Model.Trend.trend1(self) + self.x)
        ax.set_xlim(0, self.N)
        # ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x_sum_trend2(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, Model.Trend.trend2(self) + self.x)
        ax.set_xlim(0, self.N)
        # ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x_sum_trend3(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, Model.Trend.trend3(self) + self.x)
        ax.set_xlim(0, self.N)
        # ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x_sum_trend4(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, Model.Trend.trend4(self) + self.x)
        ax.set_xlim(0, self.N)
        # ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x_sum_trend5(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, Model.Trend.trend5(self) + self.x)
        ax.set_xlim(0, self.N)
        # ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x_mul_trend1(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, Model.Trend.trend1(self) * self.x)
        ax.set_xlim(0, self.N)
        # ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x_mul_trend2(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, Model.Trend.trend2(self) * self.x)
        ax.set_xlim(0, self.N)
        # ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x_mul_trend3(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, Model.Trend.trend3(self) * self.x)
        ax.set_xlim(0, self.N)
        # ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x_mul_trend4(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, Model.Trend.trend4(self) * self.x)
        ax.set_xlim(0, self.N)
        # ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x_mul_trend5(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, Model.Trend.trend5(self) * self.x)
        ax.set_xlim(0, self.N)
        # ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_x1_and_x2(self, x2):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(self.t, x2.x, color='tab:orange')
        ax.plot(self.t, self.x, color='tab:blue')
        ax.set_xlim(0, self.N)
        ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_pulse(self):
        f0_1 = 11  # Гц
        f0_2 = 110
        f0_3 = 250
        f0_4 = 410

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs[0][0].plot(self.t, Model.Pulse.pulse(self, f0_1))
        axs[0][1].plot(self.t, Model.Pulse.pulse(self, f0_2))
        axs[1][0].plot(self.t, Model.Pulse.pulse(self, f0_3))
        axs[1][1].plot(self.t, Model.Pulse.pulse(self, f0_4))
        plt.show()

    def view_graph_preobr(self):
        f0_1 = 11  # Гц
        f0_2 = 110
        f0_3 = 250
        f0_4 = 410
        delta_t = 0.001
        spectrum1 = np.fft.rfft(Model.Pulse.pulse(self, f0_1))
        spectrum2 = np.fft.rfft(Model.Pulse.pulse(self, f0_2))
        spectrum3 = np.fft.rfft(Model.Pulse.pulse(self, f0_3))
        spectrum4 = np.fft.rfft(Model.Pulse.pulse(self, f0_4))

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs[0][0].plot(np.fft.rfftfreq(self.N, 1. * delta_t), np.abs(spectrum1) / self.N)
        axs[0][1].plot(np.fft.rfftfreq(self.N, 1. * delta_t), np.abs(spectrum2) / self.N)
        axs[1][0].plot(np.fft.rfftfreq(self.N, 1. * delta_t), np.abs(spectrum3) / self.N)
        axs[1][1].plot(np.fft.rfftfreq(self.N, 1. * delta_t), np.abs(spectrum4) / self.N)
        plt.show()
        # rfftfreq выполняет работу по преобразованию номеров элементов массива в герцы

    def view_graph_preobr_some(self, fc):
        delta_t = 0.001
        spectrum = np.fft.rfft(Model.Pulse.pulse(self, fc))

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(np.fft.rfftfreq(self.N, 1. * delta_t), np.abs(spectrum) / self.N)
        plt.show()
        # rfftfreq выполняет работу по преобразованию номеров элементов массива в герцы

    def view_graph_harm(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.t, Model.Harmonic_x3.harm_sum(self))
        plt.show()

    def view_graph_bin_preobr(self):
        delta_t = 0.001
        spectrum = np.fft.rfft(self.arr)
        print(self.arr)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(np.fft.rfftfreq(self.N, 1. * delta_t), np.abs(spectrum) / self.N)
        plt.show()

    def view_graph_bin(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.t, self.arr)
        ax.set_xlim(0, self.N)
        ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_cor(self, cor):
        fig, ax = plt.subplots(figsize=(8, 8))
        print(cor)
        ax.plot(self.t, cor)
        ax.set_xlim(0, self.N)
        ax.set_ylim(-self.S * ((self.S / 10) * 0.5), self.S * ((self.S / 10) * 0.5))
        plt.show()

    def view_graph_kardio(self, k, dt, f):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(range(1999), Model.Kardio.kardio(self, k, dt, f))
        ax.set_xlim(0, 1999)
        plt.show()

    # def view_graph_lpf(self, m, dt, fc):
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     ax.plot(range(m*2+1), Model.Test19.lpf(self, m, dt, fc))
    #     plt.show()

    def view_graph_lpf2(self, m, dt, fc, C, Cs):
        # def butter_lowpass(cutoff, fs, order=5):
        #     nyq = 0.5 * fs
        #     normal_cutoff = cutoff / nyq
        #     b, a = butter(order, normal_cutoff, btype='low', analog=False)
        #     return b, a
        #
        # order = 6
        # cutoff = 3.8  # desired cutoff frequency of the filter, Hz
        #
        # b, a = butter_lowpass(cutoff, fc, order)
        # print(b, a)
        # print(len(b))
        # w, h = freqz(b, a, worN=8000)
        print(len(C))
        fig, axs = plt.subplots(2, figsize=(8, 8))
        axs[0].plot(range(65), Model.Test19.lpf_re(self, m, dt, fc))
        axs[1].plot(range(65), C)
        plt.show()

    def view_graph_hpf(self, m, dt, fc, C):
        fig, axs = plt.subplots(2, figsize=(8, 8))
        axs[0].plot(range(65), Model.Test19.hpf(self, m, dt, fc))
        axs[1].plot(range(65), C)
        plt.show()

    def view_graph_bpf(self, m, dt, fc1, fc2, C):
        fig, axs = plt.subplots(2, figsize=(8, 8))
        axs[0].plot(range(65), Model.Test19.bpf(self, m, dt, fc1, fc2))
        axs[1].plot(range(65), C)
        plt.show()

    def view_graph_bsf(self, m, dt, fc1, fc2, C):
        fig, axs = plt.subplots(2, figsize=(8, 8))
        axs[0].plot(range(-m, m+1), Model.Test19.bsf(self, m, dt, fc1, fc2))
        axs[1].plot(range(m*2+1), C)
        plt.show()

    def view_graph_bin_conv(self, input_mass, bin):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(range(1999), Model.convolution(input_mass, bin))
        ax.set_xlim(0, 800)
        plt.show()

    def view_wav(self, len, data):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(range(len), data)
        plt.show()

    def view_wav_2000(self, len, data):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(range(len), data)
        ax.set_xlim(0, 2000)
        plt.show()

    def view_graph_wavfile_spectr(self, N, C, data):
        fig, axs = plt.subplots(2, figsize=(8, 8))
        axs[0].plot(range(N), data)
        axs[1].plot(range(2000), C)
        axs[1].set_xlim(0, 1000)
        plt.show()

    def view_graph_wavfile_spectr4000(self, N, C, data):
        fig, axs = plt.subplots(2, figsize=(8, 8))
        axs[0].plot(range(N), data)
        axs[1].plot(range(4000), C)
        axs[1].set_xlim(0, 2000)
        plt.show()

    def view_mean_std(self, mean, std):
        fig, axs = plt.subplots(2, figsize=(8, 8))
        axs[0].plot(range(len(mean)), mean)
        axs[1].plot(range(len(std)), std)
        axs[1].set_xlim(240000)
        plt.show()

    def view_lfilter10(self, data):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(range(len(data)), data)
        ax.set_xlim(0, 200000)
        plt.show()

    def view_lfilter(self, data):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(range(len(data)), data)
        plt.show()


# Старт программы
if __name__ == "__main__":
    # Стартовые значения
    N = 1000
    S = 100
    a = 0.1
    b = 10.0
    rng1 = "built"
    rng2 = "self"
    mid = 0
    remi = 0
    M = 200
    k = range(N + M)

    rem_trend = 0
    # Создание двух экземпляров
    x1 = IO(a, b, N, S, rng1)
    x2 = IO(a, b, N, S, rng2)

    # Консольное управление
    while True:
        print("----Введите команду----")

        string = input()  # Строка управления

        if string == "help":
            print("Список комманд:")
            print("help - список комманд")
            print("display <x1, x2, states, t> - значения")
            print("display graph <x1, x2, trends, x1+trend1, x1*trend1, x1 and x2> - построить график")
            print("recalc <x1, x2, all x> - Пересчет рандома")
            print("change <N, S, a, b> - Изменить значения")
            print("norm <x1, x2> - Нормировка значения")
            print("shift <x1, x2> - Смещение значенией в определенном отрезке")
            print("spikes <low, high> <x1, x2> - Добавление неправдоподобных значений")
            print("analysis stationary <x1, x2> - Анализ стационарности")
            print("analysis noise <x1, x2, x1*10, x2*10> - Анализ шума")
            print("exit - выйти")
            print("")

        elif string == "display graph trends":
            print("Вывод графиков тренда...")
            Display.view_graph_trends(x1)
            print("")

        elif string == "display graph x1 and x2":
            print("s")
            Display.view_graph_x1_and_x2(x1, x2)
            print("")

        # +++
        elif string == "display graph x1+trend1":
            print("s")
            Display.view_graph_x_sum_trend1(x1)
            print("")

        elif string == "display graph x1+trend2":
            print("s")
            Display.view_graph_x_sum_trend2(x1)
            print("")

        elif string == "display graph x1+trend3":
            print("s")
            Display.view_graph_x_sum_trend3(x1)
            print("")

        elif string == "display graph x1+trend4":
            print("s")
            Display.view_graph_x_sum_trend4(x1)
            print("")

        elif string == "display graph x1+trend5":
            print("s")
            Display.view_graph_x_sum_trend5(x1)
            print("")

        elif string == "display graph x2+trend1":
            print("s")
            Display.view_graph_x_sum_trend1(x2)
            print("")

        elif string == "display graph x2+trend2":
            print("s")
            Display.view_graph_x_sum_trend2(x2)
            print("")

        elif string == "display graph x2+trend3":
            print("s")
            Display.view_graph_x_sum_trend3(x2)
            print("")

        elif string == "display graph x2+trend4":
            print("s")
            Display.view_graph_x_sum_trend4(x2)
            print("")

        elif string == "display graph x2+trend5":
            print("s")
            Display.view_graph_x_sum_trend5(x2)
            print("")

        # ***
        elif string == "display graph x1*trend1":
            print("s")
            Display.view_graph_x_mul_trend1(x1)
            print("")

        elif string == "display graph x1*trend2":
            print("s")
            Display.view_graph_x_mul_trend2(x1)
            print("")

        elif string == "display graph x1*trend3":
            print("s")
            Display.view_graph_x_mul_trend3(x1)
            print("")

        elif string == "display graph x1*trend4":
            print("s")
            Display.view_graph_x_mul_trend4(x1)
            print("")

        elif string == "display graph x1*trend5":
            print("s")
            Display.view_graph_x_mul_trend5(x1)
            print("")

        elif string == "display graph x2*trend1":
            print("s")
            Display.view_graph_x_mul_trend1(x2)
            print("")

        elif string == "display graph x2*trend2":
            print("s")
            Display.view_graph_x_mul_trend2(x2)
            print("")

        elif string == "display graph x2*trend3":
            print("s")
            Display.view_graph_x_mul_trend3(x2)
            print("")

        elif string == "display graph x2*trend4":
            print("s")
            Display.view_graph_x_mul_trend4(x2)
            print("")

        elif string == "display graph x2*trend5":
            print("s")
            Display.view_graph_x_mul_trend5(x2)
            print("")

        elif string == "display graph x1":
            print("Вывод графика x1...")
            Display.view_graph_x(x1)
            print("")

        elif string == "display graph x2":
            print("Вывод графика x2...")
            Display.view_graph_x(x2)
            print("")

        elif string == "display states":
            print("Отображение всех значений:")
            Display.view_states(x1)
            print("")

        elif string == "display t":
            print("Отображение массива t:")
            Display.view_t(x1)
            print("")

        elif string == "display x1":
            print("Отображение массива x1:")
            Display.view_x(x1)
            print("")

        elif string == "display x2":
            print("Отображение массива x2:")
            Display.view_x(x2)
            print("")

        elif string == "change N":
            print("Введите новое значение N:")
            n = int(input())
            IO.set_N(x1, n)
            IO.set_N(x2, n)
            print("Произошел пересчет x и t для x1 и x2(нормировка тоже)...")
            IO.recalculation_t(x1)
            IO.recalculation_t(x2)
            IO.recalculation_x(x1)
            IO.recalculation_x(x2)
            Norm.x_norm(x1)
            Norm.x_norm(x2)
            print("")

        elif string == "change S":
            print("Введите новое значение S:")
            s = int(input())
            IO.set_S(x1, s)
            IO.set_S(x2, s)
            print("Произошел пересчет нормировки для x1 и x2...")
            Norm.x_norm(x1)
            Norm.x_norm(x2)
            print("")

        elif string == "change a":
            print("Введите новое значение a:")
            a = float(input())
            IO.set_a(x1, a)
            IO.set_a(x2, a)
            print("")

        elif string == "change b":
            print("Введите новое значение b:")
            b = int(input())
            IO.set_b(x1, b)
            IO.set_b(x2, b)
            print("")

        elif string == "recalc all x":
            print("Произошел пересчет x1 и x2...")
            IO.recalculation_x(x1)
            IO.recalculation_x(x2)
            print("")

        elif string == "recalc x1":
            print("Произошел пересчет x1...")
            IO.recalculation_x(x1)
            print("")

        elif string == "recalc x2":
            print("Произошел пересчет x2...")
            IO.recalculation_x(x2)
            print("")

        elif string == "norm x1":
            print("Нормировка x1...")
            Norm.x_norm(x1)
            print("")

        elif string == "norm x2":
            print("Нормировка x2...")
            Norm.x_norm(x2)
            print("")

        elif string == "shift x1":
            print("Введите С:")
            C = int(input())
            print("Введите m:")
            m = int(input())
            print("Введите n:")
            n = int(input())
            Model.Shift.x_shift(x1, C, m, n)
            print("")

        elif string == "spikes high x1":
            print("Введите Q:")
            Q = int(input())
            Model.Spikes.spikes_high(x1, Q)
            print("")

        elif string == "spikes low x1":
            print("Введите Q:")
            Q = int(input())
            Model.Spikes.spikes_low(x1, Q)
            print("")

        elif string == "analysis stationary x1":
            print("Анализ стационарности x1:")
            m = 10
            Analysis.Stationary.stationary_analysis(x1, m)
            print("")

        elif string == "analysis stationary x2":
            print("Анализ стационарности x2:")
            m = 10
            Analysis.Stationary.stationary_analysis(x2, m)
            print("")

        elif string == "analysis noise x1":
            print("Анализ шума x1:")
            n = 1
            Analysis.Stationary.noise_analysis(x1, n)
            print("")

        elif string == "analysis noise x2":
            print("Анализ шума x2:")
            n = 1
            Analysis.Stationary.noise_analysis(x2, n)
            print("")

        elif string == "analysis noise x1*10":
            print("Анализ шума x1*10:")
            n = 100
            Analysis.Stationary.noise_analysis(x1, n)
            print("")

        elif string == "analysis noise x2*10":
            print("Анализ шума x2*10:")
            n = 10
            Analysis.Stationary.noise_analysis(x2, n)
            print("")

        elif string == "pulse":
            print("Пульсация...")
            Display.view_graph_pulse(x1)
            print("")

        elif string == "preobr":
            print("Преобразвание для 11, 110, 250, 410 Гц:")
            Display.view_graph_preobr(x1)
            print("")

        elif string == "harm":
            print("Сумма трех гармоник:")
            Display.view_graph_harm(x1)
            print("")

        elif string == "comp":
            print("Компенсация сдвига:")
            Proc.comp(x1)
            print("")

        elif string == "cor":
            print("Автокорреляция:")
            Proc.cor(x1,x2)
            print("")

        elif string == "anti shift":
            print("Anti shift:")
            mid = Proc.anti_shift_rem(x1)
            Proc.anti_shift(x1, mid)
            Proc.somex0(x1)
            print("")

        elif string == "anti spike":
            print("Anti spike:")
            remi = Proc.anti_spike_rem(x1)
            Proc.anti_spike(x1, remi)
            print("")

        elif string == "anti trend rem":
            print("Anti trend:")
            x1.x = Proc.anti_trend_rem(x1)
            print("")

        elif string == "anti trend":
            print("Anti trend:")
            Proc.anti_trend(x1, rem_trend)
            print("")

        elif string == "trend":
            print("trend:")
            Proc.trend(x1)
            print("")

        elif string == "bin":
            print("Считывание бинарного файла:")
            with open('file.dat', 'rb') as f:
                while True:
                    bin = f.read(4)
                    if not bin:
                        break
                    t = sum(struct.unpack("f", bin))
                    x1.arr.append(t)
            print(x1.arr)

        elif string == "bin graph":
            print("График бинарного файла:")
            Display.view_graph_bin(x1)
            print("")

        elif string == "bin graph preobr":
            print("График амплитуд бинарного файла:")
            Display.view_graph_bin_preobr(x1)
            print("")

        elif string == "plotver":
            print("test:")
            Proc.plot_ver(x1, x2)
            print("")

        elif string == "kardio":
            print("kardio:")
            dt = 0.02
            f = 4
            Display.view_graph_kardio(x1, k, dt, f)
            print("")

        elif string == "lpf":
            print("lpf:")
            dt = 0.001
            fc = 100
            m = 32
            Display.view_graph_lpf(x1, m, dt, fc)
            print("")

        elif string == "lpf2":
            print("lpf2:")
            dt = 0.001
            fc = 100
            m = 32
            func = Model.Test19.lpf_re(x1, m, dt, fc)
            C, Cs = Model.Fourie(func)
            Display.view_graph_lpf2(x1, m, dt, fc, C, Cs)
            print("")

        elif string == "hpf":
            print("hpf:")
            dt = 0.001
            fc = 100
            m = 32
            func = Model.Test19.hpf(x1, m, dt, fc)
            C, Cs = Model.Fourie(func)
            Display.view_graph_hpf(x1, m, dt, fc, C)
            print("")

        elif string == "bpf":
            print("bpf:")
            dt = 0.001
            fc1 = 100
            fc2 = 200
            m = 32
            func = Model.Test19.bpf(x1, m, dt, fc1, fc2)
            C, Cs = Model.Fourie(func)
            Display.view_graph_bpf(x1, m, dt, fc1, fc2, C)
            print("")

        elif string == "bsf":
            print("bsf:")
            dt = 0.001
            fc1 = 60
            fc2 = 120
            m = 64
            func = Model.Test19.bsf(x1, m, dt, fc1, fc2)
            C, Cs = Model.Fourie(func)
            Display.view_graph_bsf(x1, m, dt, fc1, fc2, C)
            print("")

        elif string == "bin conv":
            print("bin conv:")
            dt = 0.001
            fc = 100
            m = 64
            bin = x1.x
            input_mass = np.zeros(1000)
            temp = Model.Test19.lpf_re(x1, m, dt, fc)
            print(len(input_mass))
            print(len(bin))
            Display.view_graph_bin_conv(x1, input_mass, bin)
            print("")

        elif string == "read music wave":
            print("opening music.wav...")
            music = wave.open('music.wav', 'rb')
            print("Number of channels", music.getnchannels())
            print("Sample width", music.getsampwidth())
            print("Frame rate.", music.getframerate())
            print("Number of frames", music.getnframes())
            frames = []
            # for sec in range(music.getnframes()):
            #     frames.append(music.readframes(sec))
            # print(frames)
            music.close()
            print("")

        elif string == "read car wave":
            print("opening car.wav...")
            car = wave.open('./files/car.wav', 'rb')
            print("Number of channels", car.getnchannels())
            print("Sample width", car.getsampwidth())
            print("Frame rate.", car.getframerate())
            print("Number of frames", car.getnframes())
            # frames = []
            # temp = []
            # value = random.randint(-32767, 32767)
            # print(value)
            # for sec in range(car.getnframes()):
            #     if len(temp) < 1000:
            #
            #         value = car.readframes(sec)
            #         temp.append(struct.unpack('<h', value))
            #     else:
            #         # print(len(temp))
            #         # print(temp)
            #         frames.append(temp)
            #         temp = []
            # print(frames)
            # print(len(frames))
            car.close()
            print("")

        elif string == "read in txt car wavfile":
            print("opening car.wav...")
            fs, data = wavfile.read('./files/car.wav')
            print(fs)
            f = open('./files/car.txt', 'w')
            k = 0
            for i in data:
                if k < 1000:
                    k += 1
                    f.write(i + ', ')
                else:
                    k = 0
                    f.write('\n' + i)
            f.close()
            print("")

        elif string == "read ma wavfile":
            print("opening ma.wav...")
            fs, data = wavfile.read('./files/ma.wav')
            print(fs)
            N = len(data)
            print(N)
            Display.view_wav(x1, N, data)
            # Display.view_wav_2000(x1, N, data)
            short_data = data[6000:8000]
            print(len(short_data))
            C = Model.Fourie_05(short_data)
            # C = scipy.fftpack.fft(data)
            print(C)
            Display.view_graph_wavfile_spectr(x1, N, C, data)
            print("")

        elif string == "read ma fast":
            print("opening ma.wav...")
            fs, data = wavfile.read('./files/ma.wav')
            print(fs)
            N = len(data)
            print(N)
            short_data = []
            short_data.extend(data[2000:3000])
            short_data.extend(data[6000:7000])
            print(short_data)
            C = Model.Fourie_05(short_data)
            Display.view_graph_wavfile_spectr(x1, N, C, data)
            print("")

        elif string == "read car":
            print("opening car.wav...")
            fs, data = wavfile.read('./files/car.wav')
            print(fs)
            N = len(data)
            data1 = np.zeros(N)
            data2 = np.zeros(N)
            for i in range(N):
                data1[i], data2[i] = data[i]

            # # вывод спектра
            # print(N)
            # print(len(data1))
            # Display.view_wav(x1, N, data1)
            # short_data = []
            # short_data.extend(data1[2000:3000])
            # short_data.extend(data1[20000:21000])
            # short_data.extend(data1[30000:31000])
            # short_data.extend(data1[80000:81000])
            # C = Model.Fourie_05(short_data)
            # Display.view_graph_wavfile_spectr4000(x1, N, C, data1)

            # # анти тренд
            # trend_rem = Proc.anti_trend_rem_new(data1)
            # at_data = Proc.anti_trend_new(trend_rem)
            # Display.view_wav(x1, N, at_data)

            # скользящее окно с СКО
            rolmean = pd.Series(data1).rolling(window=1000).mean()
            rolstd = pd.Series(data1).rolling(window=1000).std()
            Display.view_mean_std(x1, rolmean, rolstd)

            # # анти тренд СКО
            # trend_rem = Proc.anti_trend_rem_new(stddata)
            # at_data = Proc.anti_trend_new(trend_rem)
            # Display.view_wav(x1, N, at_data)

            # # фильтр низких частот
            # b, a = ss.butter(3, 0.05)
            # zi = ss.lfilter_zi(b, a)
            # z, _ = ss.lfilter(b, a, data1, zi=zi * data1[0])
            #
            # z2, _ = ss.lfilter(b, a, z, zi=zi * z[0])
            #
            # y = ss.filtfilt(b, a, data1)
            #
            #
            # plt.plot(range(850176), data1, 'b-', label='data')
            # plt.plot(range(850176), y, 'g-', linewidth=2, label='filtered data')
            # plt.xlabel('Time [sec]')
            # plt.grid()
            # plt.legend()
            # # plt.xlim(0, 20000)
            # # plt.ylim(-1500, 1500)
            #
            # plt.subplots_adjust(hspace=0.35)
            # plt.show()

            def butter_bandpass(lowcut, highcut, fs, order=5):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
                return b, a


            def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
                b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                y = lfilter(b, a, data)
                return y

            # Sample rate and desired cutoff frequencies (in Hz).
            fs = 44100.0
            lowcut = 2000.0
            highcut = 20000.0

            # Plot the frequency response for a few different orders.
            plt.figure(1)
            plt.clf()
            for order in [3, 6, 9]:
                b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                w, h = freqz(b, a, worN=2000)
                plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

            plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
                     '--', label='sqrt(0.5)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Gain')
            plt.grid(True)
            plt.legend(loc='best')

            # Filter a noisy signal.
            T = 19.0
            nsamples = 850176
            t = np.linspace(0, T, nsamples, endpoint=False)
            a = 0.02
            f0 = 600.0
            x = data1
            plt.figure(2)
            plt.clf()
            plt.plot(t, x, label='Noisy signal')

            y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
            plt.plot(t, y, label='Filtered signal')
            plt.xlabel('time (seconds)')
            plt.hlines([-a, a], 0, T, linestyles='--')
            plt.grid(True)
            plt.axis('tight')
            plt.legend(loc='upper left')
            plt.show()

            sampleRate = 44100.0  # hertz
            duration = 19.0  # seconds
            obj = wave.open('new_car.wav', 'w')
            obj.setnchannels(1)  # mono
            obj.setsampwidth(2)
            obj.setframerate(sampleRate)
            for i in range(len(y)):
                value = int(y[i])
                temp = struct.pack('<h', value)
                obj.writeframesraw(temp)
            obj.close()

            print("")

        elif string == "analysis car":
            print("analysis car...")
            Analysis.general_statistics(data1, len(data1))
            print("")

        elif string == "module":
            print("creating module...")

            def trend(t, a, b):
                return b * np.exp((-a) * t)


            fs = 22050
            T = 1.0
            nsamples = T * fs
            t = np.linspace(0, T, nsamples, endpoint=False)
            f0 = 42.0

            A = 100
            a = 5
            b = 1

            x = A * np.cos(2 * np.pi * f0 * t)
            temp = 10 * np.cos(2 * np.pi * f0 * t)
            N = x.size
            y = trend(t, a, b)
            xy = x * y
            print('---Анализ значения 100---')
            Analysis.general_statistics(x, len(x))
            print('---Анализ значения 10---')
            Analysis.general_statistics(temp, len(temp))

            # S = 100
            # dt = 0.005

            plt.plot(t, x)
            plt.show()

            plt.plot(t, y)
            plt.show()

            plt.plot(t, xy)
            plt.show()

            x1 = 200 * np.sin(2 * np.pi * 500 * t)
            x2 = 50 * np.sin(2 * np.pi * 2 * t)

            z = x + x1 + x2

            plt.plot(t, z)
            plt.show()


            # def built_random(N):
            #     np.random.seed(19680801)
            #     data = np.array([random.random() for i in range(N)])
            #     data_min = data[0]
            #     data_max = data[0]
            #     N = len(data)
            #
            #     for k in range(N):
            #         if data[k] < data_min:
            #             data_min = data[k]
            #         if data[k] > data_max:
            #             data_max = data[k]
            #
            #     for k in range(N):
            #         data[k] = (((data[k] - data_min) / (data_max - data_min)) - 0.5) * 2 * S
            #
            #     return data
            #
            #
            #
            # noise = built_random(N)
            #
            # plt.plot(t, noise)
            # plt.show()
            #
            # z_noise = z + noise
            #
            # plt.plot(t, z_noise)
            # plt.show()
            #
            #
            # def spikes_low_norm(data, m, spikes_num, S):
            #     spikes_num_min = spikes_num[0]
            #     spikes_num_max = spikes_num[0]
            #     S2 = S * 2
            #
            #     for k in range(m):
            #         if spikes_num[k] < spikes_num_min:
            #             spikes_num_min = spikes_num[k]
            #         if spikes_num[k] > spikes_num_max:
            #             spikes_num_max = spikes_num[k]
            #
            #     for k in range(m):
            #         spikes_num[k] = (((spikes_num[k] - spikes_num_min) / (
            #                     spikes_num_max - spikes_num_min)) - 0.5) * 2 * S2
            #     return spikes_num
            #
            #
            # def spikes_low(data, Q, S):
            #     N = len(data)
            #     arr = np.zeros(N)
            #     m = int(N * 0.01 * Q)
            #     spikes_num = np.array([random.random() for i in range(m)])
            #     spikes_num = spikes_low_norm(data, m, spikes_num, S)
            #
            #     for k in range(m):
            #         arr[k] = spikes_num[k]
            #
            #     np.random.shuffle(arr)
            #     return data + arr
            #
            # # z_noise_spike = spikes_low(z_noise, 10, S)
            # z_noise_spike = spikes_low(z, 10, S)
            #
            #
            # plt.plot(t, z_noise_spike)
            # plt.show()
            #
            # def butter_bandpass(lowcut, highcut, fs, order=5):
            #     nyq = 0.5 * fs
            #     low = lowcut / nyq
            #     high = highcut / nyq
            #     b, a = butter(order, [low, high], btype='band')
            #     return b, a
            #
            #
            # def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            #     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            #     y = lfilter(b, a, data)
            #     return y
            #
            # lowcut = 10
            # highcut = 200
            #
            # # Filter a noisy signal.
            #
            # plt.figure(2)
            # plt.clf()
            # plt.plot(t, z, label='Noisy signal')
            #
            # y = butter_bandpass_filter(z, lowcut, highcut, fs, order=6)
            # plt.plot(t, y, label='Filtered signal')
            # plt.xlabel('time (seconds)')
            # plt.grid(True)
            # plt.axis('tight')
            # plt.legend(loc='upper left')
            # plt.show()

            # def lpf_re(m, dt, fc):
            #     lpw = [0 for i in range(0, m + 1)]
            #
            #     dp = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165])
            #
            #     arg = 2 * fc * dt
            #
            #     lpw[0] = arg
            #     arg *= np.pi
            #
            #     for i in range(1, m + 1):
            #         lpw[i] = np.sin(arg * i) / (np.pi * i)
            #
            #     lpw[m] /= 2
            #
            #     sumg = lpw[0]
            #     for i in range(1, m + 1):
            #         _sum = dp[0]
            #         arg = np.pi * i / m
            #
            #         for k in range(1, 4):
            #             _sum += 2 * dp[k] * np.cos(arg * k)
            #
            #         lpw[i] *= _sum
            #         sumg += 2 * lpw[i]
            #
            #     for i in range(len(lpw)):
            #         lpw[i] /= sumg
            #
            #     answer = lpw[::-1]
            #
            #     answer.extend(lpw[1::])
            #     return answer
            #
            #
            # def bpf(m, dt, fc1, fc2):
            #     lpw1 = lpf_re(m, dt, fc1)
            #     lpw2 = lpf_re(m, dt, fc2)
            #     bpw = []
            #     for i in range(2 * m + 1):
            #         bpw.append(lpw2[i] - lpw1[i])
            #
            #     return bpw


            print("")

        elif string == "exit":
            break

        else:
            print("Неверная комманда. Попробуйте еще раз\n")


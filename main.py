from tools import *
from scipy.misc import derivative
import matplotlib.pyplot as plt
import scipy.signal as sg
import wave
import struct
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw
plt.style.use('ggplot')
plt.rcParams['lines.color'] = 'red'
plt.rcParams['axes.facecolor'] = '#45484B'
plt.rcParams['axes.edgecolor'] = '#3C3F41'
plt.rcParams['figure.facecolor'] = '#3C3F41'
plt.rcParams['figure.edgecolor'] = '#3C3F41'
plt.rcParams['grid.color'] = '#3C3F41'


def car():
    car_real, car_real_N, fs = read_wave_stereo('./../files/car.wav')  # Звук реальной машины
    car_real = car_real[260000:600000]
    N = len(car_real)
    delta_t = 1 / fs  # Частота дискретизации
    t = [np.around(i * delta_t, decimals=8) for i in range(N)]

    X = 100  # Амплитуда
    S = 200  # Диапазон сгенерированного случайного процесса
    random_data = random_built_in(N, -S, S)  # Сгенерированный случайный шум
    car_model = [harmony_sin(X, i, 350) for i in t]  # Звук модели машины

    # t_real = [np.around(i * delta_t, decimals=8) for i in range(car_real_N)]
    # print(fs)
    # print(len(t_real))
    # print(t_real)

    # 1  2  3  4
    # 5  6  7  8
    # 9  10 11 12

    # plt.plot(t, car_real, color='grey')
    # plt.xlabel('Время')
    # plt.ylabel('Амплитуда')
    # plt.show()

    # plt.plot(t, car_real, color='grey')
    # plt.xlabel('Частота')
    # plt.ylabel('Амплитуда')
    # plt.show()

    # plt.plot(t, car_model)
    # plt.xlabel('Время')
    # plt.ylabel('Амплитуда')
    # plt.xlim(0, .2)
    # plt.ylim(-200, 200)
    # plt.show()

    # Графики модулированного звука
    plt.subplot(3, 4, 1)
    plt.plot(t, car_model)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Осциллограмма модулированного звука')
    plt.ylim(-200, 200)
    plt.xlim(0, 0.5)
    plt.grid(True)

    car_model_mul_trend, trend = multiplicative(N, car_model, exp_trend)
    plt.subplot(3, 4, 2)
    plt.plot(t, car_model_mul_trend)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Мультипликативное наложение')
    plt.grid(True)

    # car_model_short = car_model[1000:2000]
    # C, Cs = fourie_fast(car_model_short)
    # plt.subplot(3, 4, 5)
    # plt.plot(range(len(C)), C)
    # plt.xlabel('Частоты')
    # plt.ylabel('Амплитуда')
    # plt.title('Спектр модулированного звука')
    # plt.grid(True)

    car_model_mul_trend_mean = running_mean_square(car_model_mul_trend, 1000)
    plt.subplot(3, 4, 9)
    plt.plot(range(len(car_model_mul_trend_mean)), car_model_mul_trend_mean)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Средний квадрат модулированного звука')
    plt.grid(True)

    spike = spikes3(N, 1, 20)
    car_model_noise = car_model + spike + random_data
    car_model_noise_mul_trend, trend = multiplicative(N, car_model_noise, exp_trend)
    plt.subplot(3, 4, 6)
    plt.plot(range(len(car_model_noise_mul_trend)), car_model_noise_mul_trend)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Средний квадрат модулированного звука')
    plt.grid(True)

    # car_model_noise_mul_trend_short = car_model_noise_mul_trend[1000:2000]
    # C, Cs = fourie_fast(car_model_noise_mul_trend_short)
    # plt.subplot(3, 4, 10)
    # plt.plot(range(len(C)), C)
    # plt.xlabel('Частоты')
    # plt.ylabel('Амплитуда')
    # plt.title('Спектр модулированного звука')
    # plt.grid(True)

    nyq = 0.5 * fs
    low = 300 / nyq
    high = 400 / nyq
    b, a = sg.butter(5, [low, high], 'band')
    data_fil = sg.filtfilt(b, a, car_model)
    plt.subplot(3, 4, 4)
    plt.plot(range(len(car_model)), car_model)
    plt.plot(range(len(data_fil)), data_fil)
    plt.xlabel('Частоты')
    plt.ylabel('Амплитуда')
    plt.title('Спектр модулированного звука')
    plt.grid(True)

    ps = np.abs(np.fft.fft(car_real)) ** 2
    freqs = np.fft.fftfreq(len(car_real), delta_t)
    idx = np.argsort(freqs)
    plt.subplot(3, 4, 12)
    plt.plot(freqs[idx], ps[idx])
    plt.xlim(0, 5000)

    ### Графики реального звука ###
    plt.subplot(3, 4, 3)
    plt.plot(range(len(car_real)), car_real)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Осциллограмма реального звука')
    plt.grid(True)

    # car_real_short = []
    # car_real_short.extend(car_model[0:10000])
    # car_real_short.extend(car_model[200000:210000])
    # C, Cs = fourie_fast(car_real_short)
    # plt.subplot(3, 4, 7)
    # plt.plot(range(len(C)), C)
    # plt.xlabel('Частота')
    # plt.ylabel('Амплитуда')
    # plt.title('Спектр реального звука')
    # plt.grid(True)

    car_real_mean = running_mean_square(car_real, 1000)
    plt.subplot(3, 4, 11)
    plt.plot(range(len(car_real_mean)), car_real_mean)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Средний квадрат реального звука')
    plt.grid(True)

    nyq = 0.5 * fs
    low = 200 / nyq
    high = 800 / nyq
    b, a = sg.butter(5, [low, high], 'band')
    data_fil = sg.filtfilt(b, a, car_real)
    plt.subplot(3, 4, 8)
    plt.plot(range(len(car_real)), car_real)
    plt.plot(range(len(data_fil)), data_fil)
    plt.xlabel('Частоты')
    plt.ylabel('Амплитуда')
    plt.title('Спектр модулированного звука')
    plt.grid(True)

    plt.subplot(3, 4, 7)
    plt.plot(range(len(data_fil * 4)), data_fil * 4, color='orange')
    plt.plot(range(len(car_real)), car_real, color='blue')
    plt.xlabel('Частоты')
    plt.ylabel('Амплитуда')
    plt.title('Спектр модулированного звука')
    plt.grid(True)

    # car_real_short = []
    # car_real_short.extend(data_fil[0:10000])
    # car_real_short.extend(data_fil[200000:210000])
    # C, Cs = fourie_fast(car_real_short)
    # plt.subplot(3, 4, 7)
    # plt.plot(range(len(C)), C)
    # plt.xlabel('Частота')
    # plt.ylabel('Амплитуда')
    # plt.title('Спектр реального звука')
    # plt.grid(True)

    # car_model_sum_noise_mul_trend, trend = multiplicative(N, car_model_sum_noise, exp_trend)
    # plt.subplot(4, 4, 10)
    # plt.plot(t, car_model_sum_noise_mul_trend)
    #
    # car_model_sum_noise_mul_trend_mean_square = mean_square2(car_model_sum_noise_mul_trend, 100)
    # plt.subplot(4, 4, 13)
    # plt.plot(range(837801), car_model_sum_noise_mul_trend_mean_square)
    #
    # plt.subplot(4, 4, 14)
    # plt.plot(range(837900), mean_square(car_model_sum_noise_mul_trend, N, 100))

    # # Графики реального звука
    # plt.subplot(4, 4, 3)
    # plt.plot(t_real, car_real)
    # plt.show()
    # # car_real_mean_square = mean_square(car_real, N, 10000)
    # # plt.subplot(4, 4, 15)
    # # plt.plot(t_real, car_real_mean_square)
    # temp1 = np.convolve(car_real, np.ones((10000,)) / 10000, mode='valid')
    # plt.plot(temp1 ** 2)
    # plt.show()
    # temp2 = np.convolve(car_model_sum_noise_mul_trend, np.ones((10000,)) / 10000, mode='valid')
    #
    # plt.plot(range(827901), temp2**2)

    plt.show()

    # plt.plot(t, car_model_noise)
    # plt.xlabel('Время')
    # plt.ylabel('Амплитуда')
    # plt.xlim(0, .2)
    # plt.show()

    # car_model_noise_mul_trend_mean_square = []
    # for i in range(500):
    #     car_model_noise_mul_trend_mean_square.append(None)
    # car_model_noise_mul_trend_mean_square.extend(running_mean_square(car_model_noise_mul_trend, 1000))
    # for i in range(499):
    #     car_model_noise_mul_trend_mean_square.append(None)
    # plt.plot(t, car_model_noise_mul_trend_mean_square)
    # plt.xlabel('Время')
    # plt.ylabel('Значения среднего квадрата')
    # plt.show()

    # car_real_mean_square = []
    # for i in range(500):
    #     car_real_mean_square.append(None)
    # car_real_mean_square.extend(running_mean_square(car_real, 1000))
    # for i in range(499):
    #     car_real_mean_square.append(None)
    # plt.plot(t, car_real_mean_square)
    # plt.xlabel('Время')
    # plt.ylabel('Значения среднего квадрата')
    # plt.show()

    # nyq = 0.5 * fs
    # low = 200 / nyq
    # high = 800 / nyq
    # b, a = sg.butter(5, [low, high], 'band')
    # data_fil = sg.filtfilt(b, a, car_real)
    # plt.plot(t, data_fil * 5, color='orange')
    # plt.plot(t, car_real, color='grey')
    # plt.xlabel('Время')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    # plt.show()
    #
    #
    # car_real_short = []
    # car_real_short.extend(data_fil[0:10000])
    # car_real_short.extend(data_fil[200000:210000])
    # C, Cs = fourie_fast(car_real_short)
    # plt.plot(range(len(C)), C, color='orange')
    # plt.xlabel('Частота')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    # plt.show()

    ### новые данные после препода

    # t2 = [np.around(i * delta_t, decimals=8) for i in range(8000)]
    # car_model_2 = [harmony_sin(100, i, 350) for i in t2]
    # plt.plot(range(len(car_model_2)), car_model_2, color='red')
    # plt.xlabel('Значения')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    # plt.show()
    #
    # car_model_2_mul_trend, trend = multiplicative(8000, car_model_2, exp_trend)
    #
    # plt.plot(range(len(car_model_2_mul_trend)), car_model_2_mul_trend, color='red')
    # plt.xlabel('Значения')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    # plt.show()
    #
    # car_model_2_mul_trend_mean_square = window(car_model_2_mul_trend, len(car_model_2_mul_trend), 800)
    # plt.plot(range(len(car_model_2_mul_trend_mean_square)), car_model_2_mul_trend_mean_square, color='red')
    # # plt.plot(range(len(trend)), trend, color='blue')
    # plt.xlabel('Значения')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    # plt.show()
    #
    #
    #
    #
    #
    # car_model_2_mul_anritrend = anti_multiplicative(1000, car_model_2_mul_trend, trend)
    # plt.plot(range(len(car_model_2_mul_anritrend)), car_model_2_mul_anritrend, color='red')
    # plt.xlabel('Значения')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    # plt.show()
    #
    # car_model_2_mul_trend_mean_square = car_model_2_mul_trend_mean_square[::-1]
    #
    # car_model_2_mul_anritrend_2 = anti_multiplicative_2(car_model_2_mul_trend, car_model_2_mul_trend_mean_square)
    # plt.plot(range(len(car_model_2_mul_anritrend_2)), car_model_2_mul_anritrend_2, color='red')
    # plt.xlabel('Значения')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    # plt.show()

    ### Еще более новые

    N_200k = 200000
    t_200k = [np.around(i * delta_t, decimals=8) for i in range(N_200k)]
    car_real_200k = car_real[20000:220000]
    car_real_200k = normalize(car_real_200k, 100)

    plt.plot(t_200k, car_real_200k, color='green')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()

    # car_real_mean = window(car_real_200k, len(car_real_200k), 200)

    # with open('mean10000.txt', 'w') as f:
    #     for item in car_real_mean:
    #         f.write("%s\n" % round(item))

    with open('mean.txt', 'r') as f:
        car_real_mean = [row.strip() for row in f]
    car_real_mean = [float(item) for item in car_real_mean]

    plt.plot(range(len(car_real_mean)), car_real_mean, color='green')
    plt.xlabel('Значения')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()

    car_real_198k = car_real_200k[0:198000]
    N_198k = 198000
    t_198k = [np.around(i * delta_t, decimals=8) for i in range(N_198k)]

    car_real_198k_antitrend = anti_multiplicative_2(car_real_198k, car_real_mean)
    plt.plot(t_198k, car_real_198k_antitrend, color='red')
    plt.xlabel('Значения')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()

    # car_real_mean = window(car_real_198k_antitrend, len(car_real_198k_antitrend), 200)
    #
    # car_real_mean_anti = car_real_mean[::-1]
    # plt.plot(range(len(car_real_mean_anti)), car_real_mean_anti, color='green')
    # plt.xlabel('Значения')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    # plt.show()
    #
    # car_real_198k = car_real_200k[0:199600]
    #
    # car_real_198k_antitrend = anti_multiplicative_2(car_real_198k, car_real_mean_anti)
    # plt.plot(range(len(car_real_198k_antitrend)), car_real_198k_antitrend, color='red')
    # plt.xlabel('Значения')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    # plt.show()

    # car_real_198k_antitrend_short = []
    # car_real_198k_antitrend_short.extend(car_real_198k_antitrend[0:5000])
    # car_real_198k_antitrend_short.extend(car_real_198k_antitrend[150000:155000])
    # C, Cs = fourie_fast(car_real_198k_antitrend_short)
    # plt.plot(range(len(C)), C)
    # plt.xlabel('Частота')
    # plt.ylabel('Амплитуда')
    # plt.title('Спектр реального звука')
    # plt.grid(True)
    # plt.show()

    harm1 = [harmony_sin(.005, i, 13) for i in t_198k]
    harm2 = [harmony_sin(.007, i, 48) for i in t_198k]
    harm3 = [harmony_sin(.006, i, 97) for i in t_198k]
    harm4 = [harmony_sin(.009, i, 148) for i in t_198k]
    harm5 = [harmony_sin(.006, i, 171) for i in t_198k]
    harm6 = [harmony_sin(.012, i, 198) for i in t_198k]
    harm7 = [harmony_sin(.007, i, 223) for i in t_198k]
    harm8 = [harmony_sin(.006, i, 445) for i in t_198k]
    harm9 = [harmony_sin(.006, i, 463) for i in t_198k]
    harm10 = [harmony_sin(.005, i, 498) for i in t_198k]
    harm11 = [harmony_sin(.007, i, 1314) for i in t_198k]

    car_model_3 = np.zeros(198000)
    for i in range(198000):
        car_model_3[i] += harm1[i]
        car_model_3[i] += harm2[i]
        car_model_3[i] += harm3[i]
        car_model_3[i] += harm4[i]
        car_model_3[i] += harm5[i]
        car_model_3[i] += harm6[i]
        car_model_3[i] += harm7[i]
        car_model_3[i] += harm8[i]
        car_model_3[i] += harm9[i]
        car_model_3[i] += harm10[i]
        car_model_3[i] += harm11[i]

    plt.plot(t_198k, car_model_3)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()

    car_model_3_trend = multiplicative_2(car_model_3, car_real_mean)
    plt.plot(t_198k, car_model_3_trend)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()

    # car_model_3_trend_short = []
    # car_model_3_trend_short.extend(car_model_3_trend[0:5000])
    # car_model_3_trend_short.extend(car_model_3_trend[150000:155000])
    # C, Cs = fourie_fast(car_model_3_trend_short)
    # plt.plot(range(len(C)), C)
    # plt.xlabel('Частота')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    # plt.show()

    car_model_3_trend = normalize(car_model_3_trend, 1000)

    import wave, struct
    sampleRate = 44100.0  # hertz
    obj = wave.open('my_car.wav', 'w')
    obj.setnchannels(1)  # mono
    obj.setsampwidth(2)
    obj.setframerate(sampleRate)
    for i in range(198000):
        value = car_model_3_trend[i]
        data = struct.pack('<f', value)
        obj.writeframesraw(data)
    obj.close()

    car_mode_3_mean = window(car_model_3_trend, len(car_model_3_trend), 2000)

    plt.plot(range(len(car_mode_3_mean)), car_mode_3_mean, color='green')
    plt.xlabel('Значения')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()


def voice():
    voice_real, voice_real_N, fs = read_wave_mono('./../files/dictaphone.wav')  # Звук диктофона
    data = voice_real[35000:45000]
    N = len(data)
    delta_t = 1 / fs  # Частота дискретизации
    t = [np.around(i * delta_t, decimals=3) for i in range(N)]
    X = 100  # Амплитуда
    S = 10  # Диапазон сгенерированного случайного процесса

    plt.subplot(3, 2, 1)
    plt.plot(range(N), data)
    plt.title('Осциллограмма')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    data_short = []
    data_short.extend(data[1600:2000])
    data_short.extend(data[4400:4800])

    C, Cs = fourie_fast(data_short)

    # C_file, Cs_file = fourie(data)
    # with open("./../files/fourie_voice.txt") as file:
    #     file.write(C_file)

    plt.subplot(3, 2, 2)
    plt.plot(range(len(C)), C)
    plt.title('Cпектр')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    b, a = sg.butter(4, 1000. / (fs / 2.), 'low')
    data_fil = sg.filtfilt(b, a, data)

    write_wave(fs, data_fil)

    plt.subplot(3, 2, 3)
    plt.plot(range(len(data)), data)
    plt.plot(range(len(data_fil)), data_fil)
    plt.title('Cпектр')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    b, a = sg.butter(4, 1000. / (fs / 2.), 'high')
    data_fil = sg.filtfilt(b, a, data)

    plt.subplot(3, 2, 4)
    plt.plot(range(len(data)), data)
    plt.plot(range(len(data_fil)), data_fil)
    plt.title('Cпектр')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    plt.show()


def voice2():
    voice_real, voice_real_N, fs = read_wave_mono('./../files/dictaphone.wav')  # Звук диктофона
    data = voice_real[35800:43700]
    print(fs)
    N = len(data)
    delta_t = 1 / fs  # Частота дискретизации
    t = [np.around(i * delta_t, decimals=8) for i in range(N)]
    X = 100  # Амплитуда
    S = 10  # Диапазон сгенерированного случайного процесса

    plt.subplot(3, 2, 1)
    plt.plot(t, data)
    plt.title('Осциллограмма')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    # C, Cs = fourie_fast(data)
    # with open('fourie_voice.txt', 'w') as f:
    #     for item in C:
    #         f.write("%s\n" % round(item))

    with open('fourie_voice.txt', 'r') as f:
        C = [row.strip() for row in f]
    C = [float(item) for item in C]

    plt.subplot(3, 2, 2)
    plt.plot(range(len(C)), C)
    plt.title('Cпектр')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    # harm1 = [harmony_sin(588, i, 485) for i in t]
    # harm2 = [harmony_sin(220, i, 497) for i in t]
    # harm3 = [harmony_sin(225, i, 502) for i in t]
    # harm4 = [harmony_sin(613, i, 580) for i in t]
    # harm5 = [harmony_sin(772, i, 89) for i in t]
    # harm6 = [harmony_sin(202, i, 781) for i in t]
    # harm7 = [harmony_sin(218, i, 977) for i in t]
    # harm8 = [harmony_sin(222, i, 1077) for i in t]
    # harm9 = [harmony_sin(378, i, 1172) for i in t]
    # harm10 = [harmony_sin(238, i, 1239) for i in t]
    # harm11 = [harmony_sin(519, i, 1267) for i in t]
    # harm12 = [harmony_sin(262, i, 1334) for i in t]
    # harm13 = [harmony_sin(288, i, 1339) for i in t]
    # harm14 = [harmony_sin(223, i, 1345) for i in t]
    # harm15 = [harmony_sin(271, i, 1350) for i in t]
    # harm16 = [harmony_sin(243, i, 1356) for i in t]
    # harm17 = [harmony_sin(315, i, 1362) for i in t]
    # harm18 = [harmony_sin(313, i, 2215) for i in t]
    # harm19 = [harmony_sin(303, i, 2221) for i in t]
    # harm20 = [harmony_sin(234, i, 2232) for i in t]
    # harm21 = [harmony_sin(296, i, 2265) for i in t]
    # harm22 = [harmony_sin(213, i, 2271) for i in t]
    # harm23 = [harmony_sin(233, i, 2316) for i in t]
    # harm24 = [harmony_sin(218, i, 2985) for i in t]
    # harm25 = [harmony_sin(244, i, 3030) for i in t]
    # harm26 = [harmony_sin(202, i, 3080) for i in t]
    #
    # voice_model = np.zeros(N)
    # for i in range(N):
    #     # voice_model[i] += harm1[i]
    #     # voice_model[i] += harm2[i]
    #     # voice_model[i] += harm3[i]
    #     # voice_model[i] += harm4[i]
    #     voice_model[i] += harm5[i]
    #     # voice_model[i] += harm6[i]
    #     # voice_model[i] += harm7[i]
    #     # voice_model[i] += harm8[i]
    #     # voice_model[i] += harm9[i]
    #     # voice_model[i] += harm10[i]
    #     # voice_model[i] += harm11[i]
    #     # voice_model[i] += harm12[i]
    #     # voice_model[i] += harm13[i]
    #     # voice_model[i] += harm14[i]
    #     # voice_model[i] += harm15[i]
    #     # voice_model[i] += harm16[i]
    #     # voice_model[i] += harm17[i]
    #     # voice_model[i] += harm18[i]
    #     # voice_model[i] += harm19[i]
    #     # voice_model[i] += harm20[i]
    #     # voice_model[i] += harm21[i]
    #     # voice_model[i] += harm22[i]
    #     # voice_model[i] += harm23[i]
    #     # voice_model[i] += harm24[i]
    #     # voice_model[i] += harm25[i]
    #     # voice_model[i] += harm26[i]

    #
    # plt.subplot(3, 2, 3)
    # plt.plot(t, voice_model)
    # plt.xlabel('Время')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)
    #
    # sampleRate = 44100.0  # hertz
    # obj = wave.open('voice_model.wav', 'w')
    # obj.setnchannels(1)  # mono
    # obj.setsampwidth(2)
    # obj.setframerate(sampleRate)
    # for i in range(N):
    #     value = voice_model[i]
    #     data = struct.pack('<f', value)
    #     obj.writeframesraw(data)
    # obj.close()

    # data_bpf = convolution(data, bpf(32, delta_t, 500, 50))
    # data_bpf = data_bpf[0:7900]
    # plt.subplot(3, 2, 3)
    # plt.plot(t, data_bpf)
    # plt.xlabel('Время')
    # plt.ylabel('Амплитуда')
    # plt.grid(True)

    data_lpf = convolution(data, lpf(32, delta_t, 450))
    data_lpf = data_lpf[0:7900]
    plt.subplot(3, 2, 3)
    plt.plot(t, data_lpf)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    # #  Запись файла
    # sampleRate = 44100.0  # hertz
    # obj = wave.open('voice_conv.wav', 'w')
    # obj.setnchannels(1)  # mono
    # obj.setsampwidth(2)
    # obj.setframerate(sampleRate)
    # for i in range(N):
    #     value = data_lpf_500_hpf_40[i]
    #     data_file = struct.pack('<h', int(value))
    #     obj.writeframesraw(data_file)
    # obj.close()

    harm_values = [(92, 89),
                   (186, 95),
                   (150, 101),
                   (54, 106),
                   (52, 112),
                   (66, 185),
                   (144, 190),
                   (312, 196),
                   (60, 202),
                   (68, 286),
                   (206, 291),
                   (56, 308),
                   (52, 347),
                   (94, 381),
                   (274, 386),
                   (212, 392),
                   (170, 398),
                   (180, 403),
                   (160, 409),
                   (134, 414),
                   (124, 420),
                   (124, 426),
                   (136, 431),
                   (122, 437),
                   (130, 442),
                   ]

    harms = []
    for A, f in harm_values:
        harms.append([harmony_sin(A, i, f) for i in t])

    voice_model = np.zeros(N)
    for harm in harms:
        for i in range(len(harm)):
            voice_model[i] += harm[i]

    plt.subplot(3, 2, 4)
    plt.plot(t, voice_model)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    C, Cs = fourie_fast(voice_model)
    # with open('fourie_voice_model.txt', 'w') as f:
    #     for item in C:
    #         f.write("%s\n" % round(item))

    plt.subplot(3, 2, 5)
    plt.plot(range(len(C)), C)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    # C, Cs = fourie_fast(data_lpf)
    # with open('fourie_voice_conv.txt', 'w') as f:
    #     for item in C:
    #         f.write("%s\n" % round(item))

    with open('fourie_voice_conv.txt', 'r') as f:
        C = [row.strip() for row in f]
    C = [float(item) for item in C]

    plt.subplot(3, 2, 6)
    plt.plot(range(len(C)), C)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    plt.show()

    #  Запись файла
    sampleRate = 44100.0  # hertz
    obj = wave.open('voice_model.wav', 'w')
    obj.setnchannels(1)  # mono
    obj.setsampwidth(2)
    obj.setframerate(sampleRate)
    for i in range(N):
        value = voice_model[i]
        data_file = struct.pack('<h', int(value))
        obj.writeframesraw(data_file)
    obj.close()


def zach():
    data = []
    with open('./../files/v1y3.dat', 'rb') as f:
        while True:
            bin = f.read(4)
            if not bin:
                break
            t = sum(struct.unpack("f", bin))
            data.append(t)

    fs = 1000
    N = len(data)
    delta_t = 1 / fs  # Частота дискретизации
    t = [np.around(i * delta_t, decimals=8) for i in range(N)]

    plt.subplot(4, 2, 1)
    plt.plot(t, data)
    plt.title('Осциллограмма')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    L = 100
    data_trend = anti_trend(data, N, L)

    plt.subplot(4, 2, 2)
    plt.plot(range(900), data_trend)
    plt.title('Тренд данных')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)


    data_antitrend = [data[i] - data_trend[i] for i in range(N - L)]

    plt.subplot(4, 2, 3)
    plt.plot(range(900), data_antitrend)
    plt.title('Осциллограмма без тренда')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    data_antitrend_antispike = spike_detector(data_antitrend, 10)

    plt.subplot(4, 2, 4)
    plt.plot(range(len(data_antitrend_antispike)), data_antitrend_antispike)
    plt.title('Осциллограмма без спайков')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    data_antitrend_antispike_shift = shift(data_antitrend_antispike, 39)

    C, Cs = fourie_fast(data_antitrend_antispike_shift)

    plt.subplot(4, 2, 5)
    plt.plot(range(len(C)), C)
    plt.title('Cпектр')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    data_antitrend_antispike_shift_lpf = convolution(data_antitrend_antispike_shift, lpf(32, delta_t, 100))

    plt.subplot(4, 2, 6)
    plt.plot(range(len(data_antitrend_antispike_shift_lpf)), data_antitrend_antispike_shift_lpf)
    plt.title('Осциллограмма после фильтра низких частот')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    C, Cs = fourie_fast(data_antitrend_antispike_shift_lpf)

    plt.subplot(4, 2, 7)
    plt.plot(range(len(C)), C)
    plt.title('Cпектр после фильтра низких частот')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    temp_lpf = lpf(64, delta_t, 100)
    C, Cs = fourie_fast(temp_lpf)

    plt.subplot(4, 2, 8)
    plt.plot(range(len(C)), C)
    plt.title('Фильтр низких частот')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    # general_statistics(data_antitrend_antispike_lpf_shift, len(data_antitrend_antispike_lpf_shift))
    # print(len(data_antitrend_antispike_lpf_shift))
    plt.show()


def practice04_02():
    # Практика по получению данных из спектра после свертки(обратное фурье)
    N = 1000
    data = np.zeros(N)
    data[50], data[250], data[800] = 100, 50, 80
    fs = 500
    delta_t = 1 / fs  # Частота дискретизации
    t = [np.around(i * delta_t, decimals=8) for i in range(N)]
    m = 200
    h = kardio(m, delta_t)

    plt.subplot(4, 2, 1)
    plt.plot(t, data)
    plt.grid(True)

    plt.subplot(4, 2, 2)
    plt.plot(range(200), h)
    plt.grid(True)

    data_conv = convolution(data, h)
    data_conv = data_conv[0:1000]

    plt.subplot(4, 2, 3)
    plt.plot(range(1000), data_conv)
    plt.grid(True)

    C, Cs = fourie_fast(h)
    plt.subplot(4, 2, 4)
    plt.plot(range(100), C)
    plt.grid(True)

    new_h = []
    new_h.extend(h)
    temp = [0 for i in range(800)]
    new_h.extend(temp)


    x1, y1 = fourie_special(data_conv)
    x2, y2 = fourie_special(new_h)
    C = []
    Cs = []
    Re = []
    Im = []
    for re1, im1, re2, im2 in zip(x1, y1, x2, y2):
        re, im = del_complex(re1, im1, re2, im2)
        Re.append(re)
        Im.append(im)
        C.append(np.sqrt(pow(re, 2) + pow(im, 2)))
        Cs.append(re + im)

    temp = reverse_fourie(Cs)

    plt.subplot(4, 2, 5)
    plt.plot(range(500), temp)
    plt.grid(True)
    plt.show()
    pass


def practice11_02():
    # Увеличить в 2.7 раза а) ближайший сосед, б) билинейная интерполяция
    # Уменьшить  в 1.3 раза а), б)

    image = read_jpg_grayscale('files/grace.jpg')  # Открываем изображение

    # factor = 2.7
    # image_resized_1 = pillow_image_grayscale_resize(image, factor, type='nearest', mode='increase')
    # image_resized_1.show()
    #
    # factor = 1.3
    # image_resized_2 = pillow_image_grayscale_resize(image, factor, type='nearest', mode='decrease')
    # image_resized_2.show()

    factor = 2.7
    image_resized_1 = pillow_image_grayscale_resize(image, factor, type='bilinear', mode='increase')
    image_resized_1.show()

    factor = 1.3
    image_resized_1 = pillow_image_grayscale_resize(image, factor, type='bilinear', mode='decrease')
    image_resized_1.show()


def practice18_02():
    # Применить 3 вида преобразования 1) Негатив, 2) Степенной, 3) Логарифмический
    image_1 = read_jpg_grayscale('files/image1.jpg')
    image_2 = read_jpg_grayscale('files/image2.jpg')

    C = 20
    Gamma = 1.5

    image_1.show()
    image_2.show()

    # image_1_negative = pillow_image_grayscale_negative(image_1)
    # image_1_negative.show()
    #
    # image_2_negative = pillow_image_grayscale_negative(image_2)
    # image_2_negative.show()

    # image_1_gammacorr = pillow_image_grayscale_gammacorr(image_1, C, Gamma)
    # image_1_gammacorr.show()
    #
    # image_2_gammacorr = pillow_image_grayscale_gammacorr(image_2, C, Gamma)
    # image_2_gammacorr.show()

    image_1_log = pillow_image_grayscale_log(image_1, C)
    image_1_log.show()

    image_2_log = pillow_image_grayscale_log(image_2, C)
    image_2_log.show()


def practice25_02():
    # 4 Эквализация гистограммы = CDF
    # 5 Приведение гистограммы (Обратная 4)
    # Алгоритм:
    # а) jpg
    # б) Гистограмма
    # в) Интеграл
    # г) Корректирование
    image = read_jpg_grayscale('files/HollywoodLC.jpg')
    C = 200

    hist = hist_v2(image)
    plt.subplot(2, 1, 1)
    plt.plot(range(len(hist)), hist)

    cdf = cdf_calc(hist)
    plt.subplot(2, 1, 2)
    plt.plot(range(len(cdf)), cdf)

    plt.show()

    image_gammacorr = pillow_image_grayscale_equ(image, C, cdf)
    image_gammacorr.show()


def practice03_03():
    # автоКорреляцию
    # Спектр автокорреляции
    # Взять производную для удаления тренда
    # Взять от нее автокорреляцию и спектр от нее
    # Настроить режекторный филльтр (0.3, 0.5) и шаг дискретизации 1
    # m = 16, dx = 1, bsf(m, dx, fc1, fc2, w(вес)), достаточно профильтровать каждую строчку
    # 1 Прочитать файл
    # 2 построчно инкрементом dy = 10 считать производную
    # a) x'kы
    # b) Rx'x'
    # c) |F[Rx'x']|
    # d) Определить параметры пика fg
    # e) bsf()
    # f) фильтр строк
    # g) Контраст
    plt.rcParams["axes.grid"] = False
    image = read_xcr('files/h400x300.xcr')
    image = np.array(image).reshape(300, 400)
    plt.imshow(image.transpose(0, -1), cmap='gist_gray', origin='lower')
    plt.show()

    data, data_diff = diff_by_row_for_trend(image)
    plt.imshow(data_diff.transpose(0, -1), cmap='gist_gray', origin='lower')
    plt.show()

    C, Cs = fourie_fast(data_diff[0])
    plt.plot(range(len(C)), C)
    plt.title('Спектр')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()

    fs = 400
    delta_t = 1 / fs

    data_conv = image_conv(data, delta_t)
    plt.imshow(data_conv.transpose(0, -1), cmap='gist_gray', origin='lower')
    plt.show()
    # plt.imsave('someth.jpg', data_conv.transpose(0, -1), format='jpg', cmap='gist_gray', origin='lower')


def practice17_03():
    # 1.Изображение MODEL.jpg зашумить аддитивно:
    # а) нормально распределенным шумом разного уровня 1 %, 5 %, 15 %;
    # б) биполярным импульсным шумом "соль+перец";
    # в) суммарным шумом пп.а и б;
    # г) подавить шумы пп.а, б, в, фильтром низких частот: dt = dx = dy = 1, частот среза задаются в нормированной
    # шкале Найквиста от 0 - 0.5, параметр m подобрать самостоятельно до лучшего результата.
    plt.rcParams["axes.grid"] = False
    image = read_jpg_grayscale('files/MODEL.jpg')

    data_a1 = add_gauss_noise(image, 1)
    data_a5 = add_gauss_noise(image, 5)
    data_a15 = add_gauss_noise(image, 15)
    data_b = add_impulse_noise(image)
    data_c = add_impulse_noise(read_jpg_grayscale('files/data_a15.jpg'))

    plt.imsave('files/data_a1.jpg', data_a1, format='jpg', cmap='gist_gray', vmin=0, vmax=255)
    plt.imsave('files/data_a5.jpg', data_a5, format='jpg', cmap='gist_gray', vmin=0, vmax=255)
    plt.imsave('files/data_a15.jpg', data_a15, format='jpg', cmap='gist_gray', vmin=0, vmax=255)
    plt.imsave('files/data_b.jpg', data_b, format='jpg', cmap='gist_gray', vmin=0, vmax=255)
    plt.imsave('files/data_c.jpg', data_c, format='jpg', cmap='gist_gray', vmin=0, vmax=255)

    # data_temp = np.array(image.getdata()).reshape(300, 400)
    # print(data_temp)
    # C, Cs = fourie_fast(data_temp[150])
    # plt.plot(range(len(C)), C)
    # plt.show()

    fs = 400
    delta_t = 1 / fs

    data_conv_a1 = image_conv(np.array(data_a1.getdata()).reshape(300, 400), delta_t, fc1=15, m=64, type='lpf')
    plt.imsave('files/data_conv_a1.jpg', data_conv_a1, format='jpg', cmap='gist_gray')
    data_conv_a5 = image_conv(np.array(data_a5.getdata()).reshape(300, 400), delta_t, fc1=15, m=64, type='lpf')
    plt.imsave('files/data_conv_a5.jpg', data_conv_a5, format='jpg', cmap='gist_gray')
    data_conv_a15 = image_conv(np.array(data_a15.getdata()).reshape(300, 400), delta_t, fc1=15, m=64, type='lpf')
    plt.imsave('files/data_conv_a15.jpg', data_conv_a15, format='jpg', cmap='gist_gray')

    data_conv_b = image_conv(np.array(data_b.getdata()).reshape(300, 400), delta_t, fc1=15, m=64, type='lpf')
    plt.imsave('files/data_conv_b.jpg', data_conv_b, format='jpg', cmap='gist_gray')

    data_conv_c = image_conv(np.array(data_c.getdata()).reshape(300, 400), delta_t, fc1=15, m=64, type='lpf')
    plt.imsave('files/data_conv_c.jpg', data_conv_c, format='jpg', cmap='gist_gray')


def practice25_03():
    plt.rcParams["axes.grid"] = False
    image_a1 = read_jpg_grayscale('files/data_a1.jpg')
    image_a5 = read_jpg_grayscale('files/data_a5.jpg')
    image_a15 = read_jpg_grayscale('files/data_a15.jpg')
    image_b = read_jpg_grayscale('files/data_b.jpg')
    image_c = read_jpg_grayscale('files/data_c.jpg')

    mask = (3, 3)

    image_filtered_arif_a1 = draw_image(image_mask_filter(mask, image_a1, type='arif'), 398, 298)
    plt.imsave('files/image_filtered_arif_a1.jpg', image_filtered_arif_a1, format='jpg', cmap='gist_gray')

    image_filtered_arif_a5 = draw_image(image_mask_filter(mask, image_a5, type='arif'), 398, 298)
    plt.imsave('files/image_filtered_arif_a5.jpg', image_filtered_arif_a5, format='jpg', cmap='gist_gray')

    image_filtered_arif_a15 = draw_image(image_mask_filter(mask, image_a15, type='arif'), 398, 298)
    plt.imsave('files/image_filtered_arif_a15.jpg', image_filtered_arif_a15, format='jpg', cmap='gist_gray')

    image_filtered_arif_b = draw_image(image_mask_filter(mask, image_b, type='arif'), 398, 298)
    plt.imsave('files/image_filtered_arif_b.jpg', image_filtered_arif_b, format='jpg', cmap='gist_gray')

    image_filtered_arif_c = draw_image(image_mask_filter(mask, image_c, type='arif'), 398, 298)
    plt.imsave('files/image_filtered_arif_c.jpg', image_filtered_arif_c, format='jpg', cmap='gist_gray')


    image_filtered_median_a1 = draw_image(image_mask_filter(mask, image_a1, type='median'), 398, 298)
    plt.imsave('files/image_filtered_median_a1.jpg', image_filtered_median_a1, format='jpg', cmap='gist_gray')

    image_filtered_median_a5 = draw_image(image_mask_filter(mask, image_a5, type='median'), 398, 298)
    plt.imsave('files/image_filtered_median_a5.jpg', image_filtered_median_a5, format='jpg', cmap='gist_gray')

    image_filtered_median_a15 = draw_image(image_mask_filter(mask, image_a15, type='median'), 398, 298)
    plt.imsave('files/image_filtered_median_a15.jpg', image_filtered_median_a15, format='jpg', cmap='gist_gray')

    image_filtered_median_b = draw_image(image_mask_filter(mask, image_b, type='median'), 398, 298)
    plt.imsave('files/image_filtered_median_b.jpg', image_filtered_median_b, format='jpg', cmap='gist_gray')

    image_filtered_median_c = draw_image(image_mask_filter(mask, image_c, type='median'), 398, 298)
    plt.imsave('files/image_filtered_median_c.jpg', image_filtered_median_c, format='jpg', cmap='gist_gray')


def practice07_04():
    # Восстановить смазанное изображение методом деконволюции
    plt.rcParams["axes.grid"] = False
    # file_name = 'files/practice07_04/blur307x221D.dat'  # Изображение без шумов
    file_name = 'files/practice07_04/blur307x221D_N.dat'  # Изображение с шумами
    kern = 'files/practice07_04/kernD76_f4.dat'  # Массив значений ядра смазывающей функции

    data = binary_reader(file_name)
    function_core = binary_reader(kern)

    width, height = 307, 221
    matrix_pix = np.array(data).reshape(height, width)

    plt.imshow(matrix_pix, cmap='gist_gray', origin='lower')
    plt.show()

    # deconv_matrix_pix = image_deconvolution(matrix_pix, function_core)
    deconv_matrix_pix = optimal_image_deconvolution(matrix_pix, function_core, 0.00001)

    plt.imshow(deconv_matrix_pix, cmap='gist_gray', origin='lower')
    plt.show()


def practice16_04():
    # Сегментировать контуры объектов в изображении model.jpg без шумов и с шумами 15% двумя способами, в которых
    # ключевым элементом является:
    # 1) ФНЧ 2) ФВЧ
    # В обоих случаях можно применять пороговые преобразования и
    # арифметические операции с изображениями. Обосновать последовательность применения всех преобразований и их
    # параметры.
    plt.rcParams["axes.grid"] = False
    image = read_jpg_grayscale('files/MODEL.jpg')
    image_a15 = read_jpg_grayscale('files/data_a15.jpg')
    image_c = add_impulse_noise(read_jpg_grayscale('files/data_a15.jpg'))

    fs = 1
    delta_t = 1 / fs

    # data_temp = np.array(image.getdata()).reshape(300, 400)
    # print(data_temp)
    # C, Cs = fourie_fast(data_temp[150])
    # plt.plot(np.linspace(0, 0.5, num=200), C)
    # plt.show()

    # Фильтрация LPF HPF
    # print("Convolving image with lpf...")
    # data_conv_lpf = image_conv(np.array(image.getdata()).reshape(300, 400), delta_t, fc1=0.005, m=8, type='lpf')
    # plt.imsave('files/practice16_04/data_conv_lpf.jpg', data_conv_lpf, format='jpg', cmap='gist_gray')
    # print("Done")
    #
    # print("Convolving image with hpf...")
    # data_conv_hpf = image_conv(np.array(image.getdata()).reshape(300, 400), delta_t, fc1=0.005, m=8, type='hpf')
    # plt.imsave('files/practice16_04/data_conv_hpf.jpg', data_conv_hpf, format='jpg', cmap='gist_gray')
    # print("Done")
    #
    # print("Convolving image_a15 with lpf...")
    # data_a15_conv_lpf = image_conv(np.array(image_a15.getdata()).reshape(300, 400), delta_t, fc1=0.005, m=8, type='lpf')
    # plt.imsave('files/practice16_04/data_a15_conv_lpf.jpg', data_a15_conv_lpf, format='jpg', cmap='gist_gray')
    # print("Done")
    #
    # print("Convolving image_a15 with hpf...")
    # data_a15_conv_hpf = image_conv(np.array(image_a15.getdata()).reshape(300, 400), delta_t, fc1=0.005, m=8, type='hpf')
    # plt.imsave('files/practice16_04/data_a15_conv_hpf.jpg', data_a15_conv_hpf, format='jpg', cmap='gist_gray')
    # print("Done")
    #
    # print("Convolving image_c with lpf...")
    # data_c_conv_lpf = image_conv(np.array(image_c.getdata()).reshape(300, 400), delta_t, fc1=0.005, m=8, type='lpf')
    # plt.imsave('files/practice16_04/data_c_conv_lpf.jpg', data_c_conv_lpf, format='jpg', cmap='gist_gray')
    # print("Done")
    #
    # print("Convolving image_c with hpf...")
    # data_c_conv_hpf = image_conv(np.array(image_c.getdata()).reshape(300, 400), delta_t, fc1=0.005, m=8, type='hpf')
    # plt.imsave('files/practice16_04/data_c_conv_hpf.jpg', data_c_conv_hpf, format='jpg', cmap='gist_gray')
    # print("Done")

    # Дополнительная фильтрация
    image_lpf = read_jpg_grayscale('files/practice16_04/data_conv_lpf.jpg')
    image_hpf = read_jpg_grayscale('files/practice16_04/data_conv_hpf.jpg')
    image_a15_lpf = read_jpg_grayscale('files/practice16_04/data_a15_conv_lpf.jpg')
    image_a15_hpf = read_jpg_grayscale('files/practice16_04/data_a15_conv_hpf.jpg')
    image_c_lpf = read_jpg_grayscale('files/practice16_04/data_c_conv_lpf.jpg')
    image_c_hpf = read_jpg_grayscale('files/practice16_04/data_c_conv_hpf.jpg')
    mask = (3, 3)


    image_lpf_filtered_arif = draw_image(image_mask_filter(mask, image_lpf, type='arif'), 398, 298)
    plt.imsave('files/practice16_04/image_lpf_filtered_arif.jpg', image_lpf_filtered_arif, format='jpg',
               cmap='gist_gray')
    image_lpf_filtered_median = draw_image(image_mask_filter(mask, image_lpf, type='median'), 398, 298)
    plt.imsave('files/practice16_04/image_lpf_filtered_median.jpg', image_lpf_filtered_median, format='jpg',
               cmap='gist_gray')

    image_hpf_filtered_arif = draw_image(image_mask_filter(mask, image_hpf, type='arif'), 398, 298)
    plt.imsave('files/practice16_04/image_hpf_filtered_arif.jpg', image_hpf_filtered_arif, format='jpg',
               cmap='gist_gray')
    image_hpf_filtered_median = draw_image(image_mask_filter(mask, image_hpf, type='median'), 398, 298)
    plt.imsave('files/practice16_04/image_hpf_filtered_median.jpg', image_hpf_filtered_median, format='jpg',
               cmap='gist_gray')


    image_a15_lpf_filtered_arif = draw_image(image_mask_filter(mask, image_a15_lpf, type='arif'), 398, 298)
    plt.imsave('files/practice16_04/image_a15_lpf_filtered_arif.jpg', image_a15_lpf_filtered_arif, format='jpg',
               cmap='gist_gray')
    image_a15_lpf_filtered_median = draw_image(image_mask_filter(mask, image_a15_lpf, type='median'), 398, 298)
    plt.imsave('files/practice16_04/image_a15_lpf_filtered_median.jpg', image_a15_lpf_filtered_median, format='jpg',
               cmap='gist_gray')

    image_a15_hpf_filtered_arif = draw_image(image_mask_filter(mask, image_a15_hpf, type='arif'), 398, 298)
    plt.imsave('files/practice16_04/image_a15_hpf_filtered_arif.jpg', image_a15_hpf_filtered_arif, format='jpg',
               cmap='gist_gray')
    image_a15_hpf_filtered_median = draw_image(image_mask_filter(mask, image_a15_hpf, type='median'), 398, 298)
    plt.imsave('files/practice16_04/image_a15_hpf_filtered_median.jpg', image_a15_hpf_filtered_median, format='jpg',
               cmap='gist_gray')


    image_c_lpf_filtered_arif = draw_image(image_mask_filter(mask, image_c_lpf, type='arif'), 398, 298)
    plt.imsave('files/practice16_04/image_c_lpf_filtered_arif.jpg', image_c_lpf_filtered_arif, format='jpg',
               cmap='gist_gray')
    image_c_lpf_filtered_median = draw_image(image_mask_filter(mask, image_c_lpf, type='median'), 398, 298)
    plt.imsave('files/practice16_04/image_c_lpf_filtered_median.jpg', image_c_lpf_filtered_median, format='jpg',
               cmap='gist_gray')

    image_c_hpf_filtered_arif = draw_image(image_mask_filter(mask, image_c_hpf, type='arif'), 398, 298)
    plt.imsave('files/practice16_04/image_c_hpf_filtered_arif.jpg', image_c_hpf_filtered_arif, format='jpg',
               cmap='gist_gray')
    image_c_hpf_filtered_median = draw_image(image_mask_filter(mask, image_c_hpf, type='median'), 398, 298)
    plt.imsave('files/practice16_04/image_c_hpf_filtered_median.jpg', image_c_hpf_filtered_median, format='jpg',
               cmap='gist_gray')


    # Пороговое преобразование
    image_lpf_filtered_arif_thresholding = thresholding(image_lpf_filtered_arif, 110)
    plt.imsave('files/practice16_04/threshold/image_lpf_filtered_arif_thresholding.jpg',
               np.array(image_lpf_filtered_arif_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')

    image_lpf_filtered_median_thresholding = thresholding(image_lpf_filtered_median, 120)
    plt.imsave('files/practice16_04/threshold/image_lpf_filtered_median_thresholding.jpg',
               np.array(image_lpf_filtered_median_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')

    image_hpf_filtered_arif_thresholding = thresholding(image_hpf_filtered_arif, 40)
    plt.imsave('files/practice16_04/threshold/image_hpf_filtered_arif_thresholding.jpg',
               np.array(image_hpf_filtered_arif_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')

    image_hpf_filtered_median_thresholding = thresholding(image_hpf_filtered_median, 80)
    plt.imsave('files/practice16_04/threshold/image_hpf_filtered_median_thresholding.jpg',
               np.array(image_hpf_filtered_median_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')


    image_a15_lpf_filtered_arif_thresholding = thresholding(image_a15_lpf_filtered_arif, 80)
    plt.imsave('files/practice16_04/threshold/image_a15_lpf_filtered_arif_thresholding.jpg',
               np.array(image_a15_lpf_filtered_arif_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')

    image_a15_lpf_filtered_median_thresholding = thresholding(image_a15_lpf_filtered_median, 180)
    plt.imsave('files/practice16_04/threshold/image_a15_lpf_filtered_median_thresholding.jpg',
               np.array(image_a15_lpf_filtered_median_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')

    image_a15_hpf_filtered_arif_thresholding = thresholding(image_a15_hpf_filtered_arif, 50)
    plt.imsave('files/practice16_04/threshold/image_a15_hpf_filtered_arif_thresholding.jpg',
               np.array(image_a15_hpf_filtered_arif_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')

    image_a15_hpf_filtered_median_thresholding = thresholding(image_a15_hpf_filtered_median, 110)
    plt.imsave('files/practice16_04/threshold/image_a15_hpf_filtered_median_thresholding.jpg',
               np.array(image_a15_hpf_filtered_median_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')


    image_c_lpf_filtered_arif_thresholding = thresholding(image_c_lpf_filtered_arif, 80)
    plt.imsave('files/practice16_04/threshold/image_c_lpf_filtered_arif_thresholding.jpg',
               np.array(image_c_lpf_filtered_arif_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')

    image_c_lpf_filtered_median_thresholding = thresholding(image_c_lpf_filtered_median, 170)
    plt.imsave('files/practice16_04/threshold/image_c_lpf_filtered_median_thresholding.jpg',
               np.array(image_c_lpf_filtered_median_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')

    image_c_hpf_filtered_arif_thresholding = thresholding(image_c_hpf_filtered_arif, 60)
    plt.imsave('files/practice16_04/threshold/image_c_hpf_filtered_arif_thresholding.jpg',
               np.array(image_c_hpf_filtered_arif_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')

    image_c_hpf_filtered_median_thresholding = thresholding(image_c_hpf_filtered_median, 120)
    plt.imsave('files/practice16_04/threshold/image_c_hpf_filtered_median_thresholding.jpg',
               np.array(image_c_hpf_filtered_median_thresholding).reshape(398, 298), format='jpg', cmap='gist_gray')


def practice21_04():
    # C помощью градиента и лапласиана получить контуры
    plt.rcParams["axes.grid"] = False
    image = np.array(read_jpg_grayscale('files/MODEL.jpg')).reshape(300, 400)
    # image = np.array(read_jpg_grayscale('files/data_a15.jpg')).reshape(300, 400)
    # image = np.array(add_impulse_noise(read_jpg_grayscale('files/data_b.jpg'))).reshape(300, 400)
    sobel_x = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    sobel_y = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    data_diff_x = diff(image, 'x', image.shape)
    data_diff_y = diff(image, 'y', image.shape)

    data_diff_x2 = diff(data_diff_x, 'x', np.array(data_diff_x).shape)
    data_diff_y2 = diff(data_diff_y, 'y', np.array(data_diff_y).shape)

    data_diff_x = np.array(thresholding_noimage_low(data_diff_x, 240)).reshape(300, 399)
    data_diff_y = np.array(thresholding_noimage_low(data_diff_y, 240)).reshape(400, 299)

    plt.subplot(2, 2, 1)
    plt.imshow(data_diff_x, cmap='gist_gray')
    plt.title('Gx')

    data_diff_y = data_diff_y.transpose(1, 0)

    plt.subplot(2, 2, 2)
    plt.imshow(data_diff_y, cmap='gist_gray')
    plt.title('Gy')

    data = np.array(gradient_pribl(data_diff_x, data_diff_y, sobel_x, sobel_y)).reshape(298, 398)

    plt.subplot(2, 2, 3)
    plt.imshow(data, cmap='gist_gray')
    plt.title('Gradient')

    data_th = np.array(to_binary(data)).reshape(298, 398)

    plt.subplot(2, 2, 4)
    plt.imshow(data_th, cmap='gist_gray')
    plt.title('Gradient after bin')

    plt.show()

    # Лапласиан

    data_diff_x2 = np.array(thresholding_noimage_low(data_diff_x2, 240)).reshape(300, 398)
    data_diff_y2 = np.array(thresholding_noimage_low(data_diff_y2, 240)).reshape(299, 399)

    plt.subplot(2, 2, 1)
    plt.imshow(data_diff_x2, cmap='gist_gray')
    plt.title('G"x')

    plt.subplot(2, 2, 2)
    plt.imshow(data_diff_y2, cmap='gist_gray')
    plt.title('G"y')

    data = np.array(laplasian(image, data_diff_x2, data_diff_y2)).reshape(298, 397)

    plt.subplot(2, 2, 3)
    plt.imshow(data, cmap='gist_gray')
    plt.title('Laplasian')

    data = np.array(thresholding_noimage_low(data, 240, 10)).reshape(298, 397)
    plt.subplot(2, 2, 4)
    plt.imshow(np.array(to_negative(data)).reshape(298, 397), cmap='gist_gray')
    plt.title('Laplasian after bin')
    plt.show()


def practice28_04():
    # Применить морфологический фильтр к тем же картинка для выявления контура: 1) Применить дилатацию. 2) Эрозия
    plt.rcParams["axes.grid"] = False
    image = read_jpg_grayscale('files/MODEL.jpg')
    # image = np.array(read_jpg_grayscale('files/data_a15.jpg')).reshape(300, 400)
    # image = np.array(add_impulse_noise(read_jpg_grayscale('files/data_b.jpg'))).reshape(300, 400)
    # image = np.array(add_impulse_noise(read_jpg_grayscale('files/data_c.jpg'))).reshape(300, 400)
    img = cv2.imread('files/data_c.jpg', 0)

    kernel = np.ones((3, 3), np.uint8)

    (thresh, THimg) = cv2.threshold(img, 200, 250, cv2.THRESH_BINARY)

    imgErosion = cv2.erode(THimg, kernel, iterations=1)
    imgDilation = cv2.dilate(THimg, kernel, iterations=1)

    input_Sub_erosion = THimg - imgErosion
    dilation_Sub_input = imgDilation - THimg

    cv2.imshow('Input', img)
    cv2.imshow('Input after threshold', THimg)
    cv2.imshow('Erosion', imgErosion)
    cv2.imshow('Dilation', imgDilation)
    cv2.imshow('THinput - THerosion', input_Sub_erosion)
    cv2.imshow('THdilation - THinput', dilation_Sub_input)
    # cv2.imshow('input - erosion', input_Sub_erosion)
    # cv2.imshow('dilation - input', dilation_Sub_input)
    # cv2.imshow('TH input - erosion', thresholdSubErosion)
    # cv2.imshow('TH dilation - input', thresholdSubDilation)

    cv2.waitKey(0)
    pass


def practice05_05():
    # Используя методы:
    # - изменения размеров;
    # - сегментации;
    # - пространственной и частотной обработки;
    # - градационных преобразований.
    # Разработать и реализовать максимально автоматизированный или автоматический алгоритм настройки оптимальной
    # яркости и конрастности четырех изображений вертикальных и горизонтальных МРТ срезов - 2 для позвоночника и 2
    # для головы, приведя изображения к размерам 400х400.
    # Формат данных двоичный, целочисленный 2 - хбайтовый(short).

    plt.rcParams["axes.grid"] = False

    # brainH = np.array(binary_reader_short('files/practice05_05/brain-H_x512.bin')).reshape(512, 512)
    # plt.imsave('files/practice05_05/brainH.jpg', brainH, format='jpg', cmap='gist_gray')
    # brainV = np.array(binary_reader_short('files/practice05_05/brain-V_x256.bin')).reshape(256, 256)
    # plt.imsave('files/practice05_05/brainV.jpg', brainV, format='jpg', cmap='gist_gray')
    # spineH = np.array(binary_reader_short('files/practice05_05/spine-H_x256.bin')).reshape(256, 256)
    # plt.imsave('files/practice05_05/spineH.jpg', spineH, format='jpg', cmap='gist_gray')
    # spineV = np.array(binary_reader_short('files/practice05_05/spine-V_x512.bin')).reshape(512, 512)
    # plt.imsave('files/practice05_05/spineV.jpg', spineV, format='jpg', cmap='gist_gray')

    # image = read_jpg_grayscale('files/practice05_05/brainH.jpg')
    # image = read_jpg_grayscale('files/practice05_05/brainV.jpg')
    # image = read_jpg_grayscale('files/practice05_05/spineH.jpg')
    image = read_jpg_grayscale('files/practice05_05/spineV.jpg')

    # Замыкание
    imgDilation = dilationErosion_forSegment('files/practice05_05/spineV.jpg', 15)

    # cdf
    hist = hist_v2_withSegment(image, imgDilation)
    plt.subplot(2, 1, 1)
    plt.plot(range(len(hist)), hist)

    cdf = cdf_calc(hist)
    plt.subplot(2, 1, 2)
    plt.plot(range(len(cdf)), cdf)

    image_cdf = pillow_image_grayscale_equ(image, 400, cdf)
    image_cdf.save("files/practice05_05/image_cdf.jpg", "JPEG")

    image = read_jpg_grayscale('files/practice05_05/image_cdf.jpg')

    cut = 400
    w, h = image.size
    factor = w / cut

    if w > cut:
        fs = w
        delta_t = 1 / fs

        image_conv = image_conv_forimage(image, delta_t, cut/2)
        plt.imsave('files/practice05_05/image_conv.jpg', image_conv.transpose(0, -1), format='jpg', cmap='gist_gray')
        image_conv = read_jpg_grayscale('files/practice05_05/image_conv.jpg')
    else:
        image_conv = image

    image_conv_resized = pillow_image_grayscale_resize(image_conv, factor, type='nearest', mode='decrease')
    image_conv_resized.save("files/practice05_05/image_conv_resized.jpg", "JPEG")


def practice12_05_old():
    # Применяя все реализованные методы обработки и анализа изображений, а также любые сторонние методы/библиотеки
    # помимо реализованных, выделить и автоматически подсчитать на изображении stones.jpg камни заданного размера S
    # в двух вариантах:
    # 1. Выделить только те объекты, у которых размер по каждому из направлений равен S.
    # 2. Выделить камни, у которых размер хотя бы по одному направлению равен S, а в остальных направлениях меньше S.
    plt.rcParams["axes.grid"] = False
    # image = read_jpg_grayscale('files/practice12_05/stones.jpg')
    S = 13
    factor = 4
    S = S * factor
    kernel = np.ones((5, 5), np.uint8)

    # параметры цветового фильтра
    hsv_min = np.array((0, 0, 0), np.uint8)
    hsv_max = np.array((255, 255, 130), np.uint8)

    img = cv2.imread('files/practice12_05/stones.jpg')

    scale_percent = 100 * factor

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(img, dsize)

    # меняем цветовую модель с BGR на HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # применяем цветовой фильтр
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    cv2.imwrite("files/practice12_05/thresh_before.jpg", thresh)

    thresh = cv2.dilate(thresh, kernel)
    thresh = cv2.erode(thresh, kernel)
    cv2.imwrite("files/practice12_05/thresh_after.jpg", thresh)

    # ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1
        # if len(contour) > 3:
        #     (x, y, w, h) = cv2.boundingRect(contour)
            # if S - factor < w < S + factor and S - factor < h < S + factor:
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     count += 1
            # if (S - factor < w < S + factor and h < S + factor) or (w < S + factor and S - factor < h < S + factor):
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     count += 1

    print(count)
    cv2.imwrite("files/practice12_05/contours13.jpg", img)
    cv2.imwrite("files/practice12_05/thresh.jpg", thresh)
    cv2.waitKey()
    cv2.destroyAllWindows()


def practice12_05():
    # Применяя все реализованные методы обработки и анализа изображений, а также любые сторонние методы/библиотеки
    # помимо реализованных, выделить и автоматически подсчитать на изображении stones.jpg камни заданного размера S
    # в двух вариантах:
    # 1. Выделить только те объекты, у которых размер по каждому из направлений равен S.
    # 2. Выделить камни, у которых размер хотя бы по одному направлению равен S, а в остальных направлениях меньше S.
    plt.rcParams["axes.grid"] = False
    # image = read_jpg_grayscale('files/practice12_05/stones.jpg')
    S = 13
    factor = 4
    S = S * factor
    kernel = np.ones((7, 7), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)

    # параметры цветового фильтра
    hsv_min = np.array((0, 0, 0), np.uint8)
    hsv_max = np.array((255, 255, 130), np.uint8)

    img = cv2.imread('files/practice12_05/stones.jpg')

    scale_percent = 100 * factor

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(img, dsize)


    # меняем цветовую модель с BGR на HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lapImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # применяем цветовой фильтр
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    cv2.imwrite("files/practice12_05/thresh_before.jpg", thresh)

    thresh = cv2.dilate(thresh, kernel)
    thresh = cv2.erode(thresh, kernel)
    cv2.imwrite("files/practice12_05/thresh_after.jpg", thresh)


    laplacian = cv2.Laplacian(lapImg, cv2.CV_8UC1)
    cv2.imwrite("files/practice12_05/laplacian.jpg", laplacian)
    ret, laplacian = cv2.threshold(laplacian, 5, 250, cv2.THRESH_BINARY)
    cv2.imwrite("files/practice12_05/laplacianTH.jpg", laplacian)
    laplacian = cv2.dilate(laplacian, kernel2)
    cv2.imwrite("files/practice12_05/laplacianTH_erode.jpg", laplacian)



    new_th = laplacian - thresh
    new_th = cv2.erode(new_th, kernel)
    new_th = cv2.dilate(new_th, kernel)
    cv2.imwrite("files/practice12_05/new_th.jpg", new_th)


    # ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv2.findContours(new_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) > 3:
            (x, y, w, h) = cv2.boundingRect(contour)
            if S - factor < w < S + factor and S - factor < h < S + factor:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # if (S - factor <= w <= S + factor and h < S) or (w < S and S - factor <= h <= S + factor):
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("files/practice12_05/contours13.jpg", img)


def practice12_05_help():
    def nothing(*arg):
        pass

    cv2.namedWindow("result")  # создаем главное окно
    cv2.namedWindow("settings")  # создаем окно настроек

    cap = cv2.VideoCapture(0)
    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    # createTrackbar ('Имя', 'Имя окна', 'начальное значение','максимальное значение','вызов функции при изменение бегунка'
    cv2.createTrackbar('hue_1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('satur_1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('value_1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('hue_2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('satur_2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('value_2', 'settings', 255, 255, nothing)

    img = cv2.imread('files/practice12_05/stones.jpg')

    while True:

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV формат изображения

        # считываем значения бегунков
        h1 = cv2.getTrackbarPos('hue_1', 'settings')
        s1 = cv2.getTrackbarPos('satur_1', 'settings')
        v1 = cv2.getTrackbarPos('value_1', 'settings')
        h2 = cv2.getTrackbarPos('hue_2', 'settings')
        s2 = cv2.getTrackbarPos('satur_2', 'settings')
        v2 = cv2.getTrackbarPos('value_2', 'settings')

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        # накладываем фильтр на кадр в модели HSV
        thresh = cv2.inRange(hsv, h_min, h_max)

        cv2.imshow('result', thresh)

        ch = cv2.waitKey(5)
        if ch == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def water():
    img = cv2.imread('files/practice12_05/stones.jpg')
    scale_percent = 100 * 2
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(img, dsize)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    cv2.imwrite("files/practice12_05/water/thresh.jpg", thresh)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 3)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imwrite("files/practice12_05/water/opening.jpg", opening)
    cv2.imwrite("files/practice12_05/water/sure_bg.jpg", sure_bg)
    cv2.imwrite("files/practice12_05/water/dist_transform.jpg", dist_transform)
    cv2.imwrite("files/practice12_05/water/sure_fg.jpg", sure_fg)
    cv2.imwrite("files/practice12_05/water/unknown.jpg", unknown)


    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    cv2.imwrite("files/practice12_05/water/markers.jpg", markers)




    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv2.imwrite("files/practice12_05/water/markers5.jpg", markers)

    im = cv2.imread("files/practice12_05/water/markers5.jpg", cv2.IMREAD_GRAYSCALE)
    imC = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    cv2.imwrite("files/practice12_05/water/imC.jpg", imC)

    # cv2.imwrite("files/practice12_05/water/cm.jpg", cm)
    pass


def water2():
    pass


if __name__ == "__main__":
    # car()
    # voice()
    # voice2()
    # zach()
    # practice04_02()
    # practice04_02
    # practice11_02()
    # practice18_02()
    # practice25_02()
    # practice03_03()
    # practice17_03()
    # practice25_03()
    # practice07_04()
    # practice16_04()
    # practice21_04()
    # practice28_04()
    # practice05_05()
    # practice12_05_help()
    practice12_05_old()
    # practice12_05()
    # water2()
    # water()
    pass

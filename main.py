from tools import *
from scipy.misc import derivative
import matplotlib.pyplot as plt
import scipy.signal as sg
import wave
import struct
import numpy as np
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
    C = 400

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
    image = np.array(binary_reader('files/practice07_04/blur307x221D.dat')).reshape(221, 307)
    image_N = np.array(binary_reader('files/practice07_04/blur307x221D_N.dat')).reshape(221, 307)
    kern = np.array(binary_reader('files/practice07_04/kernD76_f4.dat'))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gist_gray')
    plt.title('blur307x221D')

    plt.subplot(2, 3, 2)
    plt.imshow(image_N, cmap='gist_gray')
    plt.title('blur307x221D_N')

    plt.subplot(2, 3, 3)
    plt.plot(range(len(kern)), kern)
    plt.title('kernD76_f4')

    image_fourie = []
    for row in range(image.shape[0]):
        image_fourie.append(fourie_fast_cs(image[row]))
    image_fourie = np.array(image_fourie)
    print("image_fourie", image_fourie.shape)

    kern_zero = []
    for i in range(image.shape[1]):
        if i < 76:
            kern_zero.append(kern[i])
        else:
            kern_zero.append(0)
    print("kern_zero", np.array(kern_zero).shape)
    plt.subplot(2, 3, 6)
    plt.plot(range(len(kern_zero)), kern_zero)
    plt.title('kern zero')

    kern_fourie = fourie_fast_cs(kern_zero)
    print("kern_fourie", np.array(kern_fourie).shape)

    # kern_zero = []
    # for i in range(image_fourie.shape[1]):
    #     if i < 38:
    #         kern_zero.append(kern_fourie[i])
    #     else:
    #         kern_zero.append(0)
    # kern_fourie = kern_zero
    # print("kern_fourie2", np.array(kern_fourie).shape)

    # fur_kern = []
    # for i in range(fur.shape[1]):
    #     if i < 76:
    #         fur_kern.append(fur_kern2[i])
    #     else:
    #         fur_kern.append(0)

    deconv = np.array(image_deconv(image_fourie, kern_fourie))
    print("deconv", deconv.shape)

    image_noblur = []
    for i in range(deconv.shape[0]):
        image_noblur.append(reverse_fourie(deconv[i]))

    print("image", np.array(image).shape)
    print("image_noblur", np.array(image_noblur).shape)

    plt.subplot(2, 3, 4)
    plt.imshow(image_noblur, cmap='gist_gray')
    plt.title('image_noblur')
    plt.show()


def test124():
    # Восстановить смазанное изображение методом деконволюции
    # Не использовать numpy
    plt.rcParams["axes.grid"] = False
    image = np.array(binary_reader('files/practice07_04/blur307x221D.dat')).reshape(221, 307).tolist()
    image_N = np.array(binary_reader('files/practice07_04/blur307x221D_N.dat')).reshape(221, 307).tolist()
    kern = binary_reader('files/practice07_04/kernD76_f4.dat')

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gist_gray')
    plt.title('blur307x221D')

    plt.subplot(2, 3, 2)
    plt.imshow(image_N, cmap='gist_gray')
    plt.title('blur307x221D_N')

    plt.subplot(2, 3, 3)
    plt.plot(range(len(kern)), kern)
    plt.title('kernD76_f4')

    print(len(image))

    image_fourie = []
    for row in range(len(image)):
        image_fourie.append(fourie_fast_cs(image[row]))
    print("image_fourie", len(image_fourie))

    kern_zero = []
    for i in range(307):
        if i < 76:
            kern_zero.append(kern[i])
        else:
            kern_zero.append(0)
    print("kern_zero", len(kern_zero))
    plt.subplot(2, 3, 6)
    plt.plot(range(len(kern_zero)), kern_zero)
    plt.title('kern zero')

    kern_fourie = fourie_fast_cs(kern_zero)
    print("kern_fourie", len(kern_fourie))

    deconv = image_deconv(image_fourie, kern_fourie)
    print("deconv", len(deconv))

    image_noblur = []
    for i in range(len(deconv)):
        image_noblur.append(reverse_fourie(deconv[i]))

    print("image", len(image))
    print("image_noblur", len(image_noblur))

    plt.subplot(2, 3, 4)
    plt.imshow(image_noblur, cmap='gist_gray', vmin=0, vmax=255)
    plt.title('image_noblur')
    plt.show()
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
    test124()
    pass

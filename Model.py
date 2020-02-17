import RdWr
import Analysis
import Proc
import numpy as np
import random
import scipy

class Trend(RdWr.IO):
    def trend1(self):
        return self.a * self.t + self.b

    def trend2(self):
        return (-self.a) * self.t + self.b

    def trend3(self):
        return self.b * np.exp(self.a * self.t)

    def trend4(self):
        return self.b * np.exp((-self.a) * self.t)

    def trend5(self):
        return np.where(self.t < 30.0, 0.0,
                        np.where((self.t >= 30.0) & (self.t < 80.0), self.b * np.exp(self.a * self.t),
                                 np.where(self.t >= 80.0, (-self.a) * self.t + self.b, 50.0)))


class Shift(RdWr.IO):
    def x_shift(self, C, m, n):
        arr = np.zeros(self.N)
        for k in range(self.N):
            if (k >= m) and (k <= n):
                arr[k] = C
        self.x = self.x + arr


class Random(RdWr.IO):
    def x_random(self):
        if self.rng == "built":
            return Random.built_random(self)
        elif self.rng == "self":
            return Random.self_random(self)
        else:
            return "Error with rng type"

    def built_random(self):
        np.random.seed(19680801)
        return np.array([random.random() for i in range(self.N)])

    def self_random(self):
        seed = 123
        self.x = np.zeros(self.N)
        for k in range(self.N):
            self.x[k] = seed % 19
            seed = seed + 7
        return self.x


class Spikes(RdWr.IO):
    def spikes_high(self, Q):
        arr = np.zeros(self.N)
        m = int(self.N * 0.01 * Q)
        spikes_num = np.array([random.random() for i in range(m)])
        RdWr.Norm.spikes_high_norm(self, m, spikes_num)

        for k in range(m):
            arr[k] = spikes_num[k]

        np.random.shuffle(arr)
        self.x = self.x + arr

    def spikes_low(self, Q):
        arr = np.zeros(self.N)
        m = int(self.N * 0.01 * Q)
        spikes_num = np.array([random.random() for i in range(m)])
        RdWr.Norm.spikes_low_norm(self, m, spikes_num)

        for k in range(m):
            arr[k] = spikes_num[k]

        np.random.shuffle(arr)
        self.x = self.x + arr

# Пульсация
class Pulse(RdWr.IO):
    def pulse(self, f0):
        A0 = 100
        delta_t = 0.002
        return np.array([A0 * np.sin(2. * np.pi * f0 * k * delta_t) for k in range(self.N)])


class Preobr(RdWr.IO):
    def pre(self, f0):
        Rem = (1 / self.N) * sum([Pulse.pulse(self, f0) * np.cos(2 * np.pi * k * 1) for k in range(self.N - 1)])
        Imm = (1 / self.N) * sum([Pulse.pulse(self, f0) * np.sin(2 * np.pi * k * 1) for k in range(self.N - 1)])
        return (Rem ** 2 + Imm ** 2) ** 0.5

class Harmonic_x3(RdWr.IO):
    def harm_sum(self):
        A = [25, 35, 30]
        F = [11, 41, 141]
        harm = []
        for i in range(3):
            harm.append(A[i] * np.sin(2 * np.pi * F[i] * self.t))

        return [x + y + z for x, y, z in zip(harm[0], harm[1], harm[2])]

class Kardio(RdWr.IO):
    def ecg(self, ecg, alpha=30, f0=10, dt=0.005):
        return np.sin(2 * np.pi * f0 * dt * ecg) * np.exp(-alpha * ecg * dt)

    def kardio(self, k, dt, f):
        ecg = np.zeros(self.N)
        for i in range(self.N):
            ecg[i] = Kardio.ecg(self, i)

        ticks_strength = 120
        ticks_count = int(self.N / round(self.N/6))
        ticks_mass = [0 for i in range(self.N)]
        for number in range(1, ticks_count):
            ticks_mass[number * round(self.N/6) - 1] = ticks_strength

        N, M = len(ecg), len(ticks_mass)
        print(N, M)
        conv_mass = []
        sum_of_conv = 0
        for k in range(N + M - 1):
            for m in range(M):
                if k - m < 0:
                    pass
                if k - m > N - 1:
                    pass
                else:
                    sum_of_conv += ecg[k - m] * ticks_mass[m]

            conv_mass.append(sum_of_conv)
            sum_of_conv = 0
        print(len(ecg), len(ticks_mass))
        return conv_mass

class Test19(RdWr.IO):
    def lpf_re1000(self, m, dt, fc):
        lpw = [0 for i in range(0, m + 1)]

        dp = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165])

        arg = 2 * fc * dt

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

        answer = []
        answer.append(lpw[::-1])

        answer.extend(lpw[1::])
        return answer

    def lpf_re(self, m, dt, fc):
        lpw = [0 for i in range(0, m + 1)]

        dp = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165])

        arg = 2 * fc * dt

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

    # def lpf(self, m, dt, fc):
    #     lpw = np.zeros(m*2+1)
    #     arg = 2 * fc * dt
    #     lpw[0] = arg
    #     print(arg)
    #     arg *= np.pi
    #     for i in range(1, m + 1):
    #         lpw[i] = (np.sin(arg * i)) / (np.pi * i)
    #     lpw[m] /= 2
    #
    #     d = [
    #         0.35577019,
    #         0.24369830,
    #         0.07211497,
    #         0.00630165,
    #     ]
    #     print(d[3])
    #     sum = lpw[0]
    #     for i in range(1, m + 1):
    #         sum2 = d[0]
    #         arg = (np.pi * i) / m
    #         for k in range(1, 4):
    #             sum2 += 2 * d[k] * np.cos(arg * k)
    #             print(k)
    #         lpw[i] *= sum2
    #         sum += 2 * lpw[i]
    #
    #     for i in range(m+1):
    #         lpw[i] /= sum
    #         print(i)
    #
    #     return lpw
    #
    # def lpf2(self, m, dt, fc):
    #     lpw = []
    #     arg = 2 * fc * dt
    #     lpw.append(arg)
    #     arg *= np.pi
    #     for i in range(1, m + 1):
    #         lpw.append(np.sin(arg * i) / (np.pi * i))
    #     lpw[m] /= 2
    #
    #     d = [
    #         0.35577019,
    #         0.24369830,
    #         0.07211497,
    #         0.00630165,
    #     ]
    #
    #     sum = lpw[0]
    #     for i in range(1, m + 1):
    #         sum2 = d[0]
    #         arg = (np.pi * i) / m
    #         for k in range(1, 4):
    #             sum2 += 2 * d[k] * np.cos(arg * k)
    #         lpw[i] *= sum2
    #         sum += 2 * lpw[i]
    #
    #     for i in range(m+1):
    #         lpw[i] /= sum
    #
    #     lpw_re = list(reversed(lpw))
    #     lpw.pop(0)
    #     lpw_re.extend(lpw)
    #     print(len(lpw_re))
    #     return lpw_re

    def hpf(self, m, dt, fc):
        hpw = []
        lpw = Test19.lpf_re(self, m, dt, fc)
        for i in range(2 * m+1):
            if i == m:
                hpw.append(1 - lpw[i])
            else:
                hpw.append(-lpw[i])

        return hpw

    def bpf(self, m, dt, fc1, fc2):
        lpw1 = Test19.lpf_re(self, m, dt, fc1)
        lpw2 = Test19.lpf_re(self, m, dt, fc2)
        bpw = []
        for i in range(2*m+1):
            bpw.append(lpw2[i] - lpw1[i])

        return bpw

    def bsf(self, m, dt, fc1, fc2):
        lpw1 = Test19.lpf_re(self, m, dt, fc1)
        lpw2 = Test19.lpf_re(self, m, dt, fc2)
        bsw = []

        for i in range(2*m+1):
            if i == m:
                bsw.append(1 + lpw1[i] - lpw2[i])
            else:
                bsw.append(lpw1[i] - lpw2[i])

        return bsw


def convolution(input_mass, control_mass):
    N, M = len(input_mass), len(control_mass)
    conv_mass = []
    sum_of_conv = 0
    for k in range(N + M - 1):
        for m in range(M):
            if k - m < 0:
                pass
            if k - m > N - 1:
                pass
            else:
                sum_of_conv += input_mass[k - m] * control_mass[m]

        conv_mass.append(sum_of_conv)
        sum_of_conv = 0
    return conv_mass

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
        C.append(np.sqrt(pow(re, 2) + pow(im, 2)))  # модуль комлпексного спектра (амплитудный спектр)
        Cs.append(re + im)  # Спектр Фурье
        print('fourie:', n)
    return C, Cs

def Fourie_05(func):
    C = []
    N = len(func)
    for n in range(N):
        if n < N / 2:
            sumRe = 0
            sumIm = 0
            for k in range(N):
                sumRe += func[k] * np.cos((2 * np.pi * n * k) / N)
                sumIm += func[k] * np.sin((2 * np.pi * n * k) / N)
            re = sumRe / N
            im = sumIm / N
            C.append(np.sqrt(pow(re, 2) + pow(im, 2)))  # модуль комлпексного спектра (амплитудный спектр)
            print('fourie:', n)
        else:
            C.append(0)
            print('fourie:', n)
    return C



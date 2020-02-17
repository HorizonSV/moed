import RdWr
import Model
import Proc
import numpy as np
import math

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

class Stationary:
    def stationary_analysis(self, m):
        pos = 0
        sum = np.zeros(m)
        mid = np.zeros(m)

        for i in range(m):
            for j in range(int(self.N / m)):
                sum[i] += self.x[pos]
                pos += 1
            mid[i] = 1 / (self.N / m) * sum[i]

        pos = 0
        sum_des = np.zeros(m)
        des = np.zeros(m)

        for i in range(m):
            for j in range(int(self.N / m)):
                sum_des[i] = (self.x[pos] - mid[i]) * (self.x[pos] - mid[i])
                pos += 1
            des[i] = 1 / (self.N / 10) * sum_des[i]

        print('Среднее:')
        print(mid)
        print('Дисперсия:')
        print(des)

    def noise_analysis(self, n):  # Ошибка в вычислении гдет квадрата нет
        sum = 0
        sum_des = 0
        x_sum = np.zeros(self.N)
        x_sum = Stationary.sum_noise(self, n)

        for i in range(self.N):
            sum += x_sum[i]
        mid = 1 / self.N * sum

        for i in range(self.N):
            sum_des = (x_sum[i] - mid) * (x_sum[i] - mid)
        des = 1 / self.N * sum_des

        x1_sqrt_des = np.sqrt(des)

        print('Среднее:')
        print(mid)
        print('Среднее квадратичное отклонение:')
        print(x1_sqrt_des)

    def sum_noise(self, n):
        x_sum = np.zeros(self.N)

        for i in range(n):
            for j in range(self.N):
                x_sum[j] = self.x[j]

        return x_sum / n
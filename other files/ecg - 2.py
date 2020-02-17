# ------------------------------------------------------
# Модуль для работы с ЭКГ
# ------------------------------------------------------

# Функция, генерирующая примерный ЭКГ сигнал
def ecg(x, alpha=30, f0=10, dt=0.005):
    '''
    Функция генерирует примерный сигнал ЭКГ
    :param x: массив значений по абсциссе
    :param alpha: коэффициент степени экспоненты
    :param f0: частота гармочниеской составляющей
    :param dt: шаг дискретизации
    :return: ЭКГ массив
    '''
    ecg = np.sin(2 * np.pi * f0 * dt * x) * np.exp(-alpha * x * dt)
    return ecg


# Функция генерирует тики, то есть задает ЧСС
def ticks(N, l):
    '''
    Функция рисует тики (повторяющиеся одинаковые спайки).
    На вход программа получает общее уоличество точек N
    Количество тиков ticks
    Уровень тиков ticks_level
    :param N: количество элементов
    :param l: первая координата расположения тика
    :return:
    '''
    ticks_strength = 120  # Уровень тиков
    ticks_count = int(N / l)
    ticks_mass = [0 for i in range(N)]
    for number in range(1, ticks_count):
        ticks_mass[number * l - 1] = ticks_strength
    return ticks_mass


# Функция связывает входной сигнал с управляющим - свертка
def convolution(input_mass, control_mass):
    """
    Функция связывает входной сигнал с управляющим
    :param input_mass: массив входного сигнала
    :param control_mass: массив управляющего сингала
    :return: элементы массива свертки
    """
    N, M = len(input_mass), len(control_mass)  # Размер входного и управляющего массивов
    conv_mass = []  # Массив, заполняемый элементами свертки
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


# ------------------------------------------------------
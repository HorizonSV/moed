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


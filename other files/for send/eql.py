from tools import *
import matplotlib.pyplot as plt

def practice25_02():
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



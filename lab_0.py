import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(2, 2, figsize=(7, 7))


def zad1():
    mono = np.zeros((30, 30), int)
    mono[10:20, 10:20] = 1
    mono[15:25, 5:15] = 2
    return mono


def zad2():
    ax[0, 0].imshow(zad1())
    ax[0, 0].set_title('Plot 1')
    ax[0, 1].imshow(zad1(), cmap='binary')
    ax[0, 1].set_title('Plot 2')


def zad3():
    color = np.zeros((30, 30, 3))

    color[15:25, 5:15, 0] = 1
    color[10:20, 10:20, 1] = 1
    color[5:15, 15:25, 2] = 1
    ax[1, 0].imshow(color)
    ax[1, 0].set_title('Plot 3')

    negative = 1 - color
    ax[1, 1].imshow(negative)
    ax[1, 1].set_title('Plot 4')


if __name__ == '__main__':
    print(zad1())
    zad2()
    zad3()
    plt.savefig('Lab0.png')
    plt.tight_layout()
    plt.show()

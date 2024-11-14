import numpy as np
import matplotlib.pyplot as plt
import skimage as ski


fig, ax = plt.subplots(6, 3, figsize=(11, 11))
img = ski.data.chelsea()
D = 8
L = 2 ** D
T = np.linspace(0, 255, 256)


def create_histogram(image, position):
    ax[position, 2].set_xlim(0, 255)
    h_red, count_red = np.unique(image[:, :, 0], return_counts=True)
    probability_red = count_red / np.sum(count_red)
    ax[position, 2].scatter(h_red, probability_red, c='r', s=1)

    h_green, count_green = np.unique(image[:, :, 1], return_counts=True)
    probability_green = count_green / np.sum(count_green)
    ax[position, 2].scatter(h_green, probability_green, c='g', s=1)

    h_blue, count_blue = np.unique(image[:, :, 2], return_counts=True)
    probability_blue = count_blue / np.sum(count_blue)
    ax[position, 2].scatter(h_blue, probability_blue, c='b', s=1)


lut_lin = np.linspace(0, L - 1, L).astype(int)
lin_img = lut_lin[img]

lut_neg = np.linspace(L - 1, 0, L). astype(int)
neg_img = lut_neg[img]

lut_prog = np.zeros(L).astype(int)
lut_prog[50:150] = L - 1
prog_img = lut_prog[img]

lut_sin = np.linspace(0, 2 * np.pi, L)
lut_sin = (((np.sin(lut_sin) + 1) / 2) * (L - 1)).astype(int)
sin_img = lut_sin[img]

gamma03 = 0.3
lut_gamma03 = np.pow(T, gamma03)
lut_gamma03 -= np.min(lut_gamma03)
lut_gamma03 /= np.max(lut_gamma03)
lut_gamma03 *= L - 1
lut_gamma03 = lut_gamma03.astype(int)
gamma03_img = lut_gamma03[img]

gamma3 = 3
lut_gamma3 = np.pow(T, gamma3)
lut_gamma3 -= np.min(lut_gamma3)
lut_gamma3 /= np.max(lut_gamma3)
lut_gamma3 *= L - 1
lut_gamma3 = lut_gamma3.astype(int)
gamma3_img = lut_gamma3[img]

ax[0, 0].scatter(T, lut_lin, c='r', s=1)
ax[0, 1].imshow(lin_img)
create_histogram(lin_img, 0)

ax[1, 0].scatter(T, lut_neg, c='r', s=1)
ax[1, 1].imshow(neg_img)
create_histogram(neg_img, 1)

ax[2, 0].scatter(T, lut_prog, c='r', s=1)
ax[2, 1].imshow(prog_img)
create_histogram(prog_img, 2)

ax[3, 0].scatter(T, lut_sin, c='r', s=1)
ax[3, 1].imshow(sin_img)
create_histogram(sin_img, 3)

ax[4, 0].scatter(T, lut_gamma03, c='r', s=1)
ax[4, 1].imshow(gamma03_img)
create_histogram(gamma03_img, 4)

ax[5, 0].scatter(T, lut_gamma3, c='r', s=1)
ax[5, 1].imshow(gamma3_img)
create_histogram(gamma3_img, 5)

plt.tight_layout()
plt.show()


################ ZADANIE 3 ################
fig, ax = plt.subplots(2, 3, figsize=(6, 6))
img = ski.data.moon()

ax[0, 0].imshow(img, cmap='binary_r')

u_vals, count = np.unique(img, return_counts=True)
probability = count / np.sum(count)
ax[0, 1].bar(u_vals, probability, color='black')

u_vals_copy = np.zeros(256)
u_vals_copy[u_vals] = probability
dist = np.cumsum(u_vals_copy)
ax[0, 2].scatter(np.arange(256), dist, c='black', s=1)

lut = dist * 255
ax[1, 0].scatter(np.arange(256), lut, color='black', s=1)

fixed_img = lut[img]
ax[1, 1].imshow(fixed_img, cmap='binary_r')

u, c = np.unique(fixed_img, return_counts=True)
fixed_prob = c / np.sum(c)
ax[1, 2].bar(u, fixed_prob, color='black')

plt.tight_layout()
plt.show()


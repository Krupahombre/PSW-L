import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, correlate
from skimage.color import rgb2gray
from skimage.data import chelsea
from skimage.transform import resize


# ZAD 1
signal = np.load('signal.npy')

level = 7
trans_falk = pywt.wavedec(signal, 'db7', level=level)
fig, ax = plt.subplots(9, 1)
colormap = plt.cm.coolwarm(np.linspace(0, 1, level + 2))

approx_coeff = trans_falk[0]
detail_coeffs = trans_falk[1:]

ax[0].plot(signal, color=colormap[0])
ax[0].set_ylabel('Signal', fontsize=10)
ax[0].grid(True)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

ax[1].plot(approx_coeff, color=colormap[1])
ax[1].set_ylabel('Approx.', fontsize=10)
ax[1].grid(True)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

for i, detail_c in enumerate(detail_coeffs):
    ax[i + 2].plot(detail_c, color=colormap[i + 2])
    ax[i + 2].set_ylabel(f'lvl {i + 1}', fontsize=10)
    ax[i + 2].grid(True)
    ax[i + 2].spines['top'].set_visible(False)
    ax[i + 2].spines['right'].set_visible(False)

fig.align_ylabels(ax)
plt.tight_layout()
plt.show()


# ZAD 2
fig, ax = plt.subplots(3, 2)

index = [2, 5, 9, 1, 0, 0]
wave_prototype = np.array([
    0,
    0,
    1 * index[0],
    1 * index[1],
    1 * index[2],
    -1 * index[3],
    -1 * index[4],
    -1 * index[5],
    0,
    0
], dtype=float)

wave_prototype_normalized = wave_prototype / np.sum(np.abs(wave_prototype))
wave_2d = np.outer(wave_prototype_normalized, wave_prototype_normalized)
wave_resized = resize(wave_2d, (20, 20))
wave_filtered = gaussian_filter(wave_resized, sigma=2)

chelsea_image = chelsea().astype(np.uint8)
# chelsea_gray = rgb2gray(chelsea_image)
chelsea_gray = chelsea_image[:, :, 0]

correlation = correlate(chelsea_gray, wave_filtered)

ax[0, 0].plot(wave_prototype)
ax[0, 0].set_title('1D prototype')

ax[0, 1].plot(wave_prototype_normalized)
ax[0, 1].set_title('1D prototype normalized')

ax[1, 0].imshow(wave_2d)
ax[1, 0].set_title('2D prototype')

ax[1, 1].imshow(wave_filtered)
ax[1, 1].set_title('2D Wavelet')

ax[2, 0].imshow(chelsea_gray)

ax[2, 1].imshow(correlation)

plt.tight_layout()
plt.show()


# ZAD 3
fig, ax = plt.subplots(4, 4)
chelsea_copy = chelsea_gray
wave_2d_copy = wave_2d

sizes = np.linspace(2, 32, 16, dtype=int)
ax = np.ravel(ax)

for i, s in enumerate(sizes):
    wave_resized = resize(wave_2d_copy, (s, s))
    filtered_result = correlate(chelsea_gray, wave_resized, mode='reflect')
    ax[i].imshow(filtered_result, cmap='hsv')
    ax[i].set_title(f'Size: {s}x{s}', fontsize=10)
    ax[i].axis('off')

plt.tight_layout()
plt.show()

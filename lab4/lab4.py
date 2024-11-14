import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

chelsea = ski.data.chelsea()
fig, ax = plt.subplots(2, 2)
kernel = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)])
kernel_size = kernel.shape[0]

chelsea_mono = np.mean(chelsea, 2)

chelsea_correlate = ndimage.correlate(chelsea_mono, kernel)
chelsea_convolve = ndimage.convolve(chelsea_mono, kernel, mode='constant')

ax[0, 0].imshow(chelsea_correlate, cmap='grey')
ax[0, 0].set_title(f'Korelacja')

ax[0, 1].imshow(chelsea_convolve, cmap='grey')
ax[0, 1].set_title(f'Konwolucja')

chelsea_padded = np.pad(chelsea_mono, ((1, 1), (1, 1)), mode='constant')

corr_result = np.zeros(np.shape(chelsea_mono))
for i in range(chelsea_mono.shape[0]):
    for j in range(chelsea_mono.shape[1]):
        fragment = chelsea_padded[i:i + kernel_size, j:j + kernel_size]
        corr_result[i, j] = np.sum(fragment * kernel)

ax[1, 0].imshow(corr_result, cmap='gray')
ax[1, 0].set_title(f'Korelacja - własne')

conv_result = np.zeros(np.shape(chelsea_mono))
for i in range(chelsea_mono.shape[0]):
    for j in range(chelsea_mono.shape[1]):
        fragment = chelsea_padded[i:i + kernel_size, j:j + kernel_size]
        conv_result[i, j] = np.sum(fragment * np.flip(kernel))

ax[1, 1].imshow(conv_result, cmap='gray')
ax[1, 1].set_title(f'Konwolucja - własne')

plt.tight_layout()
plt.show()

### ZAD 3 ###
fig, ax = plt.subplots(2, 2)

ax[0, 0].imshow(chelsea_mono, cmap='grey', vmin=0, vmax=255)
ax[0, 0].set_title(f'Oryginał')

kernel_box = np.ones((7, 7))
kernel_box /= np.sum(kernel_box)
kernel_box_size = kernel_box.shape[0]

chelsea_padded = np.pad(chelsea_mono, ((3, 3), (3, 3)), mode='constant')

corr_result = np.zeros(np.shape(chelsea_mono))
for i in range(chelsea_mono.shape[0]):
    for j in range(chelsea_mono.shape[1]):
        fragment = chelsea_padded[i:i + kernel_box_size, j:j + kernel_box_size]
        corr_result[i, j] = np.sum(fragment * kernel_box)

ax[0, 1].imshow(corr_result, cmap='grey', vmin=0, vmax=255)
ax[0, 1].set_title(f'Rozmycie')

chelsea_mask = chelsea_mono - corr_result

ax[1, 0].imshow(chelsea_mask, cmap='grey')
ax[1, 0].set_title(f'Maska')

chelsea_filtered = chelsea_mono + chelsea_mask

ax[1, 1].imshow(chelsea_filtered, cmap='grey', vmin=0, vmax=255)
ax[1, 1].set_title(f'Unsharp masking')

plt.tight_layout()
plt.show()

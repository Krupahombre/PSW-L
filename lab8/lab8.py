import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.data import chelsea

# ZAD 1
fig, ax = plt.subplots(3, 2)
chelsea_image = chelsea()
chelsea_gray = rgb2gray(chelsea_image)

sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

rows, cols = chelsea_gray.shape
if rows % 2 == 0:
    chelsea_gray = chelsea_gray[:-1, :]
if cols % 2 == 0:
    chelsea_gray = chelsea_gray[:, :-1]

chelsea_fft = np.fft.fftshift(np.fft.fft2(chelsea_gray))
magnitude = np.log(np.abs(chelsea_fft.real) + 1)

ax[0, 0].imshow(chelsea_gray, cmap='binary_r')
ax[0, 0].set_title(f'Original')

ax[0, 1].imshow(magnitude, cmap='magma')
ax[0, 1].set_title(f'Fourier transformation')

padded_sobel = np.zeros_like(chelsea_gray)
padded_sobel[:sobel.shape[0], :sobel.shape[1]] = sobel
padded_sobel = np.roll(padded_sobel, -sobel.shape[0]//2, axis=0)
padded_sobel = np.roll(padded_sobel, -sobel.shape[1]//2, axis=1)

sobel_fft = np.fft.fftshift(np.fft.fft2(padded_sobel))
sobel_magnitude = np.log(np.abs(sobel_fft.real) + 1)

ax[1, 0].imshow(padded_sobel, cmap='binary_r')
ax[1, 0].set_title(f'Padded Sobel')

ax[1, 1].imshow(sobel_magnitude, cmap='magma')
ax[1, 1].set_title(f'Fourier transformation Sobel')

filtered_fft = chelsea_fft * sobel_fft
filtered_fft_magnitude = np.log(np.abs(filtered_fft.real) + 1)

chelsea_filtered = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real

ax[2, 0].imshow(filtered_fft_magnitude, cmap='magma')
ax[2, 0].set_title(f'Fourier transformation filtered')

ax[2, 1].imshow(chelsea_filtered, cmap='binary_r')
ax[2, 1].set_title(f'Inverse transform filtered')

plt.tight_layout()
plt.show()

#ZAD 2
fig, ax = plt.subplots(3, 2)
img = plt.imread('filtered.png')
img = img[:, :, 0]

ax[0, 0].imshow(img, cmap='binary_r')
ax[0, 0].set_title('Degenerated')

ft_img = np.fft.fftshift(np.fft.fft2(img))
ft_img_magnitude = np.log(np.abs(ft_img.real) + 1)

ax[0, 1].imshow(ft_img_magnitude, cmap='magma')
ax[0, 1].set_title('Fourier Transform')

filter = np.eye(13, 13) * 10
filter -= 1
filter /= -39

padded_filter = np.zeros_like(img)
h, w = filter.shape
padded_filter[:h, :w] = filter
padded_filter = np.roll(padded_filter, -filter.shape[0]//2, axis=0)
padded_filter = np.roll(padded_filter, -filter.shape[1]//2, axis=1)

ax[1, 0].imshow(padded_filter, cmap='binary_r')
ax[1, 0].set_title('h')

filter_fft = np.fft.fftshift(np.fft.fft2(padded_filter))
filter_magnitude = np.log(np.abs(filter_fft.real) + 1)

ax[1, 1].imshow(filter_magnitude, cmap='magma')
ax[1, 1].set_title('H')

ft_img_filtered = ft_img / filter_fft
ft_img_filtered_magnitude = np.log(np.abs(ft_img_filtered.real) + 1)

ax[2, 0].imshow(ft_img_filtered_magnitude, cmap='magma')
ax[2, 0].set_title('Deconvolved FT')

img_filtered = np.fft.ifft2(np.fft.ifftshift(ft_img_filtered)).real

ax[2, 1].imshow(img_filtered, cmap='binary_r')
ax[2, 1].set_title('Deconvolved img')

plt.tight_layout()
plt.show()

#ZAD 3
fig, ax = plt.subplots(3, 2)

ax[0, 0].imshow(img, cmap='binary_r')
ax[0, 0].set_title('Degenerated')

ax[0, 1].imshow(ft_img_magnitude, cmap='magma')
ax[0, 1].set_title('Fourier Transform')

ax[1, 0].imshow(padded_filter, cmap='binary_r')
ax[1, 0].set_title('h')

ax[1, 1].imshow(filter_magnitude, cmap='magma')
ax[1, 1].set_title('H')

lamb = 0.02
F = ft_img / filter_fft * (1 / (1 + (lamb / np.pow(filter_fft, 2))))
F_magnitude = np.log(np.abs(F.real) + 1)

ax[2, 0].imshow(F_magnitude, cmap='magma')
ax[2, 0].set_title('Deconvolved FT')

img_prob_original = np.fft.ifft2(np.fft.ifftshift(F)).real

ax[2, 1].imshow(img_prob_original, cmap='binary_r')
ax[2, 1].set_title('Deconvolved img')

plt.tight_layout()
plt.show()

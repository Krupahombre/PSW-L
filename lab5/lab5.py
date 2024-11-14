import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.color import rgb2gray

# ZAD 1

fig, ax = plt.subplots(2, 3)
img = np.zeros((1000, 1000))
img[500:520, 460:550] = 1

ax[0, 0].imshow(img, cmap='magma')
ax[0, 0].set_title(f'Original')

img_ft = np.fft.fftshift(np.fft.fft2(img))

real_img = np.log(np.abs(np.real(img_ft)) + 1)
imag_img = np.log(np.abs(np.imag(img_ft)) + 1)

ax[0, 1].imshow(real_img, cmap='magma')
ax[0, 1].set_title(f'Real')

ax[0, 2].imshow(imag_img, cmap='magma')
ax[0, 2].set_title(f'Imag')

phase_shift = np.arctan2(real_img, imag_img)

ax[1, 0].imshow(phase_shift, cmap='magma')
ax[1, 0].set_title(f'Phase')

magnitude = np.log(np.abs(img_ft) + 1)

ax[1, 1].imshow(magnitude, cmap='magma')
ax[1, 1].set_title(f'Magnitude')

img_inv_ft = np.fft.ifft2(np.fft.ifftshift(img_ft)).real

ax[1, 2].imshow(img_inv_ft, cmap='magma')
ax[1, 2].set_title(f'Inverse')

plt.tight_layout()
plt.show()

# ZAD 2

fig, ax = plt.subplots(2, 3)
x = np.linspace(0, 10, 1000)
y = np.linspace(0, 10, 1000)

X, Y = np.meshgrid(x, y)

amplitudes = [4, 2, 0, 6, 9]
angles = [np.pi * 2, np.pi * 1.5, np.pi * 4.75, np.pi * 0.75, np.pi * 3.5]
waves = [2, 1, 3, 7, 9]

img = np.zeros((1000, 1000))
for amp, ang, wave in zip(amplitudes, angles, waves):
    img += amp * np.sin(2 * np.pi * (X * np.cos(ang) + Y * np.sin(ang)) * (1 / wave))

img -= np.min(img)
img /= np.max(img)

img_ft = np.fft.fftshift(np.fft.fft2(img))

real_img = np.log(np.abs(np.real(img_ft)) + 1)
imag_img = np.log(np.abs(np.imag(img_ft)) + 1)

ax[0, 0].imshow(img, cmap='magma')
ax[0, 0].set_title(f'Original')

ax[0, 1].imshow(real_img, cmap='magma')
ax[0, 1].set_title(f'Real')

ax[0, 2].imshow(imag_img, cmap='magma')
ax[0, 2].set_title(f'Imag')

phase_shift = np.arctan2(real_img, imag_img)

ax[1, 0].imshow(phase_shift, cmap='magma')
ax[1, 0].set_title(f'Phase')

magnitude = np.log(np.abs(img_ft) + 1)

ax[1, 1].imshow(magnitude, cmap='magma')
ax[1, 1].set_title(f'Magnitude')

img_inv_ft = np.fft.ifft2(np.fft.ifftshift(img_ft)).real

ax[1, 2].imshow(img_inv_ft, cmap='magma')
ax[1, 2].set_title(f'Inverse')

plt.tight_layout()
plt.show()

# ZAD 3

fig, ax = plt.subplots(2, 3)
astronaut = ski.data.astronaut()
astronaut_mono = rgb2gray(astronaut)

astronaut_ft = np.fft.fftshift(np.fft.fft2(astronaut_mono))

real = astronaut_ft.real
imag = astronaut_ft.imag * 1j
astronaut_magnitude = np.log(np.abs(astronaut_ft) + 1)

inv_real = np.fft.ifft2(np.fft.ifftshift(real)).real
inv_imag = np.fft.ifft2(np.fft.ifftshift(imag)).real
inv_complex = np.fft.ifft2(np.fft.ifftshift(astronaut_ft)).real

r_channel = (inv_real - np.min(inv_real)) / (np.max(inv_real) - np.min(inv_real))
g_channel = (inv_imag - np.min(inv_imag)) / (np.max(inv_imag) - np.min(inv_imag))
b_channel = (inv_complex - np.min(inv_complex)) / (np.max(inv_complex) - np.min(inv_complex))

astronaut_color = np.dstack((r_channel, g_channel, b_channel))

ax[0, 0].imshow(astronaut_mono, cmap='gray')
ax[0, 0].set_title(f'Mono')

ax[0, 1].imshow(astronaut_magnitude)
ax[0, 1].set_title(f'Magnitude')

ax[0, 2].imshow(astronaut_color)
ax[0, 2].set_title(f'Color')

ax[1, 0].imshow(r_channel, cmap='Reds_r')
ax[1, 0].set_title(f'Red')

ax[1, 1].imshow(g_channel, cmap='Greens_r')
ax[1, 1].set_title(f'Green')

ax[1, 2].imshow(b_channel, cmap='Blues_r')
ax[1, 2].set_title(f'Blue')

plt.tight_layout()
plt.show()
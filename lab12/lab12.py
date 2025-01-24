import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize, rotate


def normalize_array(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr


# ZAD 1
fig, ax = plt.subplots(2, 3, figsize=(10, 5))

img = data.camera()
img = resize(img, (128, 128), anti_aliasing=True)

gx = np.zeros_like(img)
gy = np.zeros_like(img)

for y in range(1, img.shape[0] - 1):
    for x in range(1, img.shape[1] - 1):
        gx[x, y] = img[x + 1, y] - img[x - 1, y]
        gy[x, y] = img[x, y + 1] - img[x, y - 1]

mag = np.sqrt(gx**2 + gy**2)

angle = np.arctan(gy / gx)

ax[0, 0].imshow(img, cmap='binary_r')
ax[0, 0].set_title('Resized')
ax[0, 1].imshow(gx, cmap='binary_r')
ax[0, 1].set_title('Gx')
ax[0, 2].imshow(gy, cmap='binary_r')
ax[0, 2].set_title('Gy')
ax[1, 0].imshow(mag, cmap='binary_r')
ax[1, 0].set_title('Magnitude')
ax[1, 1].imshow(angle, cmap='binary_r')
ax[1, 1].set_title('Angle')

plt.tight_layout()
plt.show()

# ZAD 2
fig, ax = plt.subplots(2, 2, figsize=(10, 5))

s = 8
bins = 9
step = np.pi / bins

mask = np.zeros_like(img, dtype=int)

cell_id = 0
for i in range(0, img.shape[0], s):
    for j in range(0, img.shape[1], s):
        mask[i:i + s, j:j + s] = cell_id
        cell_id += 1

hog = np.zeros((cell_id, bins))

for id in range(cell_id):
    ang_v = angle[mask == id]
    mag_v = mag[mask == id]

    for bin_id in range(bins):
        start = (bin_id * step) - (np.pi / 2)
        end = ((bin_id + 1) * step) - (np.pi / 2)
        b_mask = np.ones((8 * 8)).astype(bool)
        b_mask[ang_v < start] = 0
        b_mask[ang_v > end] = 0
        hog[id, bin_id] = np.sum(mag_v[b_mask])

ax[0, 0].imshow(mask, cmap='seismic')
ax[0, 0].set_title('Mask')

hog_reshaped = hog.reshape(16, 16, bins)

for i in range(9):
    hog_reshaped[:, :, i] -= np.min(hog_reshaped[:, :, i])
    hog_reshaped[:, :, i] /= np.max(hog_reshaped[:, :, i])

ax[0, 1].imshow(hog_reshaped[:, :, 0:3])
ax[0, 1].set_title('Mask')

ax[1, 0].imshow(hog_reshaped[:, :, 3:6])
ax[1, 0].set_title('Mask')

ax[1, 1].imshow(hog_reshaped[:, :, 6:9])
ax[1, 1].set_title('Mask')

plt.tight_layout()
plt.show()

# ZAD 3
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

angles = np.linspace(-80, 80, bins)

resul_img = np.zeros_like(img)

for i in range(256):
    part = np.zeros((8, 8))
    for j in range(9):
        new_part = np.zeros((8, 8))
        new_part[4:5, :] = 1
        angle = angles[j]
        new_part = rotate(new_part, angle)
        new_part = new_part * hog[i, j]
        part += new_part

    part = np.reshape(part, part.shape[0]**2)
    resul_img[np.reshape(mask, img.shape) == i] = part

resul_img = resul_img.reshape(img.shape)

plt.imshow(resul_img, cmap='binary')

plt.show()

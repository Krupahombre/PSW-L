import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import disk

np.random.seed(259100)


def erosion(image, s_element):
    se_size = s_element.shape[0]
    padded_image = np.pad(image, se_size // 2, mode='constant', constant_values=0)
    eroded_image = np.zeros_like(image)

    for x in range(1, image.shape[0] + 1):
        for y in range(1, image.shape[1] + 1):
            region = padded_image[x - 1:x + 2, y - 1:y + 2]
            if np.sum(region * s_element) == np.sum(s_element):
                eroded_image[x - 1, y - 1] = 1

    return eroded_image


def dilation(image, s_element):
    se_size = s_element.shape[0]
    padded_image = np.pad(image, se_size // 2, mode='constant', constant_values=0)
    dilated_image = np.zeros_like(image)

    for x in range(1, image.shape[0] + 1):
        for y in range(1, image.shape[1] + 1):
            region = padded_image[x - 1:x + 2, y - 1:y + 2]
            if np.sum(region * s_element) > 0:
                dilated_image[x - 1, y - 1] = 1

    return dilated_image


def hit_or_miss(image, b1, b2):
    se_size = b1.shape[0]
    padded_image = np.pad(image, se_size // 2, mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for x in range(4, image.shape[0]):
        for y in range(4, image.shape[1]):
            part = padded_image[x - 4:x + 5, y - 4:y + 5]
            out = (np.sum((part * b1)) == np.sum(b1)) * (np.sum(((1 - part) * b2)) == np.sum(b2))
            result[x - 4, y - 4] = out

    return result


# ZAD 1
fig, ax = plt.subplots(1, 3)

image_size = (100, 100)
min_radius = 2
max_radius = 10
circles_num = 100
synthetic_image = np.zeros(image_size, dtype=np.uint8)

for _ in range(circles_num):
    x = np.random.randint(0, image_size[0])
    y = np.random.randint(0, image_size[1])
    radius = np.random.randint(min_radius, max_radius + 1)
    # radius = 4
    rr, cc = disk((x, y), radius, shape=image_size)
    synthetic_image[rr, cc] = 1

ax[0].imshow(synthetic_image, cmap='binary')
ax[0].set_title("Obraz")

structured_element = np.ones((3, 3), dtype=np.uint8)
eroded_image = erosion(synthetic_image, structured_element)

ax[1].imshow(eroded_image, cmap='binary')
ax[1].set_title("Erozja")

image_difference = synthetic_image - eroded_image

ax[2].imshow(image_difference, cmap='binary')
ax[2].set_title("Różnica")

plt.tight_layout()
plt.show()

# ZAD 2
fig, ax = plt.subplots(2, 2)

structured_element_l = np.array([[1, 0, 0],
                                 [1, 0, 0],
                                 [1, 1, 1]], dtype=np.uint8)

eroded_image_l = erosion(synthetic_image, structured_element_l)
dilated_image_l = dilation(synthetic_image, structured_element_l)
opened_image = dilation(eroded_image_l, structured_element_l)
closed_image = erosion(dilated_image_l, structured_element_l)

ax[0, 0].imshow(eroded_image_l, cmap='binary')
ax[0, 0].set_title("Erozja")

ax[0, 1].imshow(dilated_image_l, cmap='binary')
ax[0, 1].set_title("Dylatacja")

ax[1, 0].imshow(opened_image, cmap='binary')
ax[1, 0].set_title("Otwarcie")

ax[1, 1].imshow(closed_image, cmap='binary')
ax[1, 1].set_title("Zamknięcie")

plt.tight_layout()
plt.show()

#ZAD 3
fig, ax = plt.subplots(2, 2)

B1 = np.zeros((9, 9), dtype=np.uint8)
rr, cc = disk((4, 4), 4, shape=B1.shape)
B1[rr, cc] = 1

B2 = np.zeros_like(B1)
B2[B1 == 0] = 1
B2[0, 0] = 0
B2[0, B1.shape[0] - 1] = 0
B2[B1.shape[0] - 1, 0] = 0
B2[B1.shape[0] - 1, B1.shape[0] - 1] = 0

hit_or_miss_result = hit_or_miss(synthetic_image, B1, B2)

ax[0, 0].imshow(B1, cmap='binary')
ax[0, 0].set_title("B1")

ax[0, 1].imshow(B2, cmap='binary')
ax[0, 1].set_title("B2")

ax[1, 0].imshow(synthetic_image, cmap='binary')
ax[1, 0].set_title("Obraz")

ax[1, 1].imshow(hit_or_miss_result, cmap='binary')
ax[1, 1].set_title("Hit or Miss")

plt.tight_layout()
plt.show()


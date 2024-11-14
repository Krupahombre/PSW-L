import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

###### zad1
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
chelsea_img = ski.data.chelsea()

rotation_angle = 15
rotation_matrix = np.array([[np.cos(np.deg2rad(rotation_angle)), -np.sin(np.deg2rad(rotation_angle)), 0],
                            [np.sin(np.deg2rad(rotation_angle)), np.cos(np.deg2rad(rotation_angle)), 0],
                            [0, 0, 1]])
shear_matrix = np.array(([[1, 0.5, 0],
                          [0, 1, 0],
                          [0, 0, 1]]))

ax[0, 0].imshow(chelsea_img)
ax[0, 0].set_title('Chelsea')

chelsea_img_mean = np.mean(chelsea_img, axis=2)
chelsea_img_mean = chelsea_img_mean[::8, ::8]
ax[0, 1].imshow(chelsea_img_mean, cmap='binary_r')
ax[0, 1].set_title('Mono')

tform = ski.transform.AffineTransform(rotation=np.deg2rad(15))
chelsea_rotated = ski.transform.warp(chelsea_img_mean, tform.inverse)
ax[1, 0].imshow(chelsea_rotated, cmap='binary_r')
ax[1, 0].set_title('Rotation')

tform = ski.transform.AffineTransform(matrix=shear_matrix)
chelsea_sheared = ski.transform.warp(chelsea_img_mean, tform.inverse)
ax[1, 1].imshow(chelsea_sheared, cmap='binary_r')
ax[1, 1].set_title('Shear')

plt.tight_layout()
plt.show()


###### zad2
fig, ax = plt.subplots(1, 2, figsize=(9, 6), sharex=True, sharey=True)

x, y = chelsea_img_mean.shape

x_vals = np.arange(0, x)
y_vals = np.arange(0, y)

flat = []
for x_enumerate in x_vals:
    for y_enumerate in y_vals:
        flat.append([x_enumerate, y_enumerate])
flatten = np.array(flat)

ax[0].scatter(flatten[:, 0], flatten[:, 1], c=chelsea_img_mean.flatten(), cmap='binary_r')
ax[0].set_title('Manuel')

flatten_stacked = np.column_stack((flatten, np.ones(x * y)))
flatten_affine = flatten_stacked @ rotation_matrix

ax[1].scatter(flatten_affine[:, 0], flatten_affine[:, 1], c=chelsea_img_mean.flatten(), cmap='binary_r')
ax[1].set_title('Manuel Rotate')

plt.tight_layout()
plt.show()


###### zad3
fig, ax = plt.subplots(1, 3, figsize=(9, 6))

ax[0].scatter(flatten[:, 0], flatten[:, 1], c=chelsea_img_mean.flatten(), cmap='binary_r')
ax[0].set_title('Interpolation')

random_idx = np.random.choice(flatten.shape[0], size=1000)
random_flatten = flatten[random_idx]
ax[1].scatter(random_flatten[:, 0], random_flatten[:, 1], c=chelsea_img_mean.flatten()[random_idx], cmap='binary_r')
ax[1].set_title('Interpolation Random')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch, DBSCAN


# ZAD 1
fig, ax = plt.subplots(1, 2)

image_size = (100, 100, 3)
ground_truth_size = (100, 100)
num_disks = 3
disk_radius_range = (10, 40)
pixel_value_range = (100, 255)
noise_std_dev = 16

image = np.zeros(image_size, dtype=np.uint8)
ground_truth = np.zeros(ground_truth_size, dtype=np.int32)

for label_index in range(1, num_disks + 1):
    radius = np.random.randint(disk_radius_range[0], disk_radius_range[1] + 1)
    center_x = np.random.randint(radius, image_size[0] - radius)
    center_y = np.random.randint(radius, image_size[1] - radius)
    rr, cc = disk((center_x, center_y), radius, shape=image_size[:2])

    channel = np.random.randint(label_index)
    intensity = np.random.randint(pixel_value_range[0], pixel_value_range[1] + 1)

    image[rr, cc, channel] += intensity
    ground_truth[rr, cc] += label_index

noise = np.random.normal(0, noise_std_dev, image.shape)
image = np.clip(image + noise, 0, 255).astype(np.uint8)

ax[0].imshow(image)
ax[0].set_title("image")

ax[1].imshow(ground_truth)
ax[1].set_title("ground truth")

plt.tight_layout()
plt.show()


# ZAD 2
pixels = image.reshape(-1, image.shape[2])

x_coords, y_coords = np.meshgrid(range(image.shape[0]), range(image.shape[1]), indexing='ij')
x_coords = x_coords.flatten()
y_coords = y_coords.flatten()

X = np.column_stack((pixels, x_coords, y_coords))

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

y = ground_truth.flatten()

print(f"{X_normalized.shape} {y.shape}")
print(f"{X_normalized[0]} {y[0]}")

# ZAD 3
clustering_methods = {
    "KMeans": KMeans(),
    "MiniBatchKMeans": MiniBatchKMeans(),
    "Birch": Birch(),
    "DBSCAN": DBSCAN()
}

fig, ax = plt.subplots(2, 3)
clustering_results = {}
metrics_scores = {}

for name, method in clustering_methods.items():
    labels = method.fit_predict(X_normalized)
    clustering_results[name] = labels.reshape(ground_truth.shape)
    metrics_scores[name] = adjusted_rand_score(y, labels)

ax[0, 0].imshow(image)
ax[0, 0].set_title("image")

ax[0, 1].imshow(ground_truth)
ax[0, 1].set_title("ground truth")

ax[0, 2].imshow(clustering_results["KMeans"])
ax[0, 2].set_title(f"KMeans: {metrics_scores['KMeans']:.2f}")

ax[1, 0].imshow(clustering_results["MiniBatchKMeans"])
ax[1, 0].set_title(f"MiniBatchKMeans: {metrics_scores['MiniBatchKMeans']:.2f}")

ax[1, 1].imshow(clustering_results["Birch"])
ax[1, 1].set_title(f"Birch: {metrics_scores['Birch']:.2f}")

ax[1, 2].imshow(clustering_results["DBSCAN"])
ax[1, 2].set_title(f"DBSCAN: {metrics_scores['DBSCAN']:.2f}")

plt.tight_layout()
plt.show()

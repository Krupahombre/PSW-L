import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, correlate, median_filter
from skimage.color import rgb2gray
from skimage.data import chelsea

# ZAD 1
fig, ax = plt.subplots(1, 3)
chelsea_image = chelsea()
chelsea_gray = rgb2gray(chelsea_image)

blurred_image = gaussian_filter(chelsea_gray, sigma=1)

laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
prewitt_x_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

laplacian_edges = correlate(blurred_image, laplacian_kernel)
laplacian_edges[laplacian_edges < 0] = 0

prewitt_x = correlate(blurred_image, prewitt_x_kernel)
prewitt_y = correlate(blurred_image, prewitt_y_kernel)
prewitt_edges = np.abs(prewitt_x) + np.abs(prewitt_y)
prewitt_edges[prewitt_edges < 0] = 0

ax[0].imshow(chelsea_gray, cmap='binary')
ax[0].set_title("original")

ax[1].imshow(laplacian_edges, cmap='binary')
ax[1].set_title("laplacian")

ax[2].imshow(prewitt_edges, cmap='binary')
ax[2].set_title("gradient")

plt.tight_layout()
plt.show()

# ZAD 2
fig, ax = plt.subplots(2, 4)
threshold = 0.1
filter_size = (21, 21)

laplacian_edges_normalized = (laplacian_edges - laplacian_edges.min()) / (laplacian_edges.max() - laplacian_edges.min())
prewitt_edges_normalized = (prewitt_edges - prewitt_edges.min()) / (prewitt_edges.max() - prewitt_edges.min())

laplacian_thresholded = laplacian_edges_normalized > threshold
prewitt_thresholded = prewitt_edges_normalized > threshold

laplacian_median_filtered = median_filter(laplacian_edges_normalized, size=filter_size)
prewitt_median_filtered = median_filter(prewitt_edges_normalized, size=filter_size)

laplacian_adaptive_threshold = laplacian_edges_normalized > laplacian_median_filtered
prewitt_adaptive_threshold = prewitt_edges_normalized > prewitt_median_filtered

ax[0, 0].imshow(laplacian_edges_normalized, cmap='binary')
ax[0, 0].set_title('laplacian')

ax[1, 0].imshow(prewitt_edges_normalized, cmap='binary')
ax[1, 0].set_title('gradient')

ax[0, 1].imshow(laplacian_thresholded, cmap='binary')
ax[0, 1].set_title('global thresholding')

ax[1, 1].imshow(prewitt_thresholded, cmap='binary')
ax[1, 1].set_title('global thresholding')

ax[0, 2].imshow(laplacian_median_filtered, cmap='binary')
ax[0, 2].set_title('filtered laplacian')

ax[1, 2].imshow(prewitt_median_filtered, cmap='binary')
ax[1, 2].set_title('filtered gradient')

ax[0, 3].imshow(laplacian_adaptive_threshold, cmap='binary')
ax[0, 3].set_title('filtered laplacian')

ax[1, 3].imshow(prewitt_adaptive_threshold, cmap='binary')
ax[1, 3].set_title('filtered gradient')

plt.tight_layout()
plt.show()

# ZAD 3
fig, ax = plt.subplots(2, 2)
max_variance = 0
optimal_threshold = 0
variances = []

laplacian_8bit = (laplacian_edges_normalized * 255).astype(np.uint8)
histogram, bin_edges = np.histogram(laplacian_8bit, bins=256, range=(0, 255))
total_pixels = laplacian_8bit.size

global_mean = np.sum(np.arange(256) * histogram) / total_pixels

for threshold in range(256):
    weight0 = np.sum(histogram[:threshold]) / total_pixels
    if weight0 > 0:
        mean0 = np.sum(np.arange(threshold) * histogram[:threshold]) / (np.sum(histogram[:threshold]) + 1e-6)
    else:
        mean0 = 0

    weight1 = np.sum(histogram[threshold:]) / total_pixels
    if weight1 > 0:
        mean1 = np.sum(np.arange(threshold, 256) * histogram[threshold:]) / (np.sum(histogram[threshold:]) + 1e-6)
    else:
        mean1 = 0

    variance = (weight0 * (mean0 - global_mean) ** 2) + (weight1 * (mean1 - global_mean) ** 2)

    if np.isnan(variance):
        variance = 0
    variances.append(variance)

    if variance > max_variance:
        max_variance = variance
        optimal_threshold = threshold

otsu_thresholded = laplacian_8bit >= optimal_threshold

ax[0, 0].imshow(laplacian_8bit, cmap='binary')
ax[0, 0].set_title('laplacian')

ax[0, 1].bar(bin_edges[:-1], histogram, width=1, color='blue')
ax[0, 1].set_yscale('log')

ax[1, 0].plot(range(256), variances, color='blue')
ax[1, 0].vlines(optimal_threshold, ymin=0, ymax=max(variances), color='red', linestyles='dashed')

ax[1, 1].imshow(otsu_thresholded, cmap='binary')

plt.tight_layout()
plt.show()

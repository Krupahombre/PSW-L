import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def normalize_channel(channel):
    channel -= np.min(channel)
    channel /= np.max(channel)
    return channel

def evaluate(data, labels):
    scores = cross_val_score(classifier, data, labels, cv=rkf, scoring='accuracy')
    return scores.mean(), scores.std()

# ZAD 1

fig, ax = plt.subplots(2, 3)
img = sp.io.loadmat('SalinasA_corrected.mat')['salinasA_corrected']

img_10 = img[:, :, 10]
img_100 = img[:, :, 100]
img_200 = img[:, :, 200]

ax[0, 0].imshow(img_10, cmap='binary_r')
ax[0, 0].set_title(f'band: 10')

ax[0, 1].imshow(img_100, cmap='binary_r')
ax[0, 1].set_title(f'band: 100')

ax[0, 2].imshow(img_200, cmap='binary_r')
ax[0, 2].set_title(f'band: 200')

ax[1, 0].plot(img[10, 10, :])
ax[1, 0].set_title(f'pixel: 10, 10')

ax[1, 1].plot(img[40, 40, :])
ax[1, 1].set_title(f'pixel: 40, 40')

ax[1, 2].plot(img[80, 80, :])
ax[1, 2].set_title(f'pixel: 80, 80')

plt.tight_layout()
plt.show()

# ZAD 2

fig, ax = plt.subplots(1, 2)
img = sp.io.loadmat('SalinasA_corrected.mat')['salinasA_corrected']

red = normalize_channel(img[:, :, 4].astype(float))
green = normalize_channel(img[:, :, 12].astype(float))
blue = normalize_channel(img[:, :, 26].astype(float))

rgb = np.zeros((img.shape[0], img.shape[1], 3))
rgb[:, :, 0] = red
rgb[:, :, 1] = green
rgb[:, :, 2] = blue

ax[0].imshow(rgb)
ax[0].set_title(f'RGB')

pca = PCA(n_components=3)

img_ext = img.reshape(-1, img.shape[2])
pca_ext = pca.fit_transform(img_ext)

img_pca = pca_ext.reshape(img.shape[0], img.shape[1], 3)

img_pca_normalized = np.zeros_like(img_pca)
for i in range(3):
    img_pca_normalized[:, :, i] = normalize_channel(img_pca[:, :, i])

ax[1].imshow(img_pca_normalized)
ax[1].set_title('PCA')

plt.tight_layout()
plt.show()

# ZAD 3

img = sp.io.loadmat('SalinasA_corrected.mat')['salinasA_corrected']
labels = sp.io.loadmat('SalinasA_gt.mat')['salinasA_gt'].reshape(-1)

mask = labels > 0
filtered_labels = labels[mask]

representations = {
    "RGB": img[:, :, [4, 12, 26]].reshape(-1, 3)[mask],
    "PCA": PCA(n_components=3).fit_transform(img.reshape(-1, img.shape[2]))[mask],
    "All": img.reshape(-1, img.shape[2])[mask]
}

classifier = RandomForestClassifier(random_state=42)
rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

for name, data in representations.items():
    mean_accuracy, std_dev = evaluate(data, filtered_labels)
    print(f'{name} {mean_accuracy:.3f} ({std_dev:.3f})')

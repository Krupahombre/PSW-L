import math

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 3, figsize=(10, 10))

x = np.linspace(0, 4*math.pi, 40)
y = np.sin(x)

ax[0, 0].plot(x, y)
ax[0, 0].set_title('Plot 1')

y2 = y[:, np.newaxis] * y[np.newaxis, :]
ax[0, 1].imshow(y2, cmap='magma')

v_min = np.min(y2)
v_max = np.max(y2)
ax[0, 1].set_title(f'min: {round(v_min, 3)}, max: {round(v_max, 3)}')

y2 -= np.min(y2)
y2 /= np.max(y2)
nv_min = np.min(y2)
nv_max = np.max(y2)
ax[0, 2].imshow(y2, cmap='magma')
ax[0, 2].set_title(f'min: {round(nv_min, 3)}, max: {round(nv_max, 3)}')

for i in range(1, 4):
    k = 2 ** i
    L = np.power(2, k) - 1
    y_disc = y2 * L
    y_disc = np.rint(y_disc).astype(int)
    ax[1, i - 1].imshow(y_disc, cmap='magma')
    ax[1, i - 1].set_title(f'min: {round(np.min(y_disc), 3)}, max: {round(np.max(y_disc), 3)}')

noise = np.random.normal(y2, size=y2.shape)
y2_noise = y2 + noise
ax[2, 0].imshow(y2_noise, cmap='magma')
ax[2, 0].set_title(f'n=1')

idx = 1
numbers = [50, 1000]
for i in numbers:
    noised_signals = []
    for j in range(i):
        n_signal = np.random.normal(y2, size=y2.shape)
        noised_signals.append(n_signal)

    average = np.mean(noised_signals, axis=0)
    ax[2, idx].imshow(average, cmap='magma')
    ax[2, idx].set_title(f'n={i}')

    idx += 1

plt.savefig('Lab1.png')
plt.tight_layout()
plt.show()

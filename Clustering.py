from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. อ่านภาพ
folder = Path(__file__).resolve().parent / "Dataset"
images, names, X = [], [], []

for path in sorted(folder.glob("*")):
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
        continue

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    images.append(img)
    names.append(path.name)
    X.append(cv2.resize(img, (32, 32)).flatten() / 255.0)

X = np.array(X)

# 2. กำหนดค่าเหมือนตัวอย่าง
K = 3
m = 2
C = np.zeros((K, X.shape[1]))
mu = np.random.rand(len(X), K)

# 3. คำนวณด้วยสูตรเดียวกับตัวอย่าง
# for _ in range(100):
#     mu /= np.sum(mu, axis=1)[:, None]

#     for k in range(K):
#         C[k] = np.dot(mu[:, k] ** m, X) / np.sum(mu[:, k] ** m) 

#         D = np.sqrt(np.sum((X - C[k]) ** 2, axis=1))
#         D = np.maximum(D, 1e-10)  # ป้องกันระยะเป็นศูนย์
#         D = D ** (-2 / (m - 1))
#         mu[:, k] = D

mu /= np.sum(mu, axis=1)[:, None]
while True:
    old_mu = mu.copy()

    for k in range(K):
        C[k] = np.dot(mu[:, k] ** m, X) / np.sum(mu[:, k] ** m)

        D = np.sqrt(np.sum((X - C[k]) ** 2, axis=1))
        D = np.maximum(D, 1e-10)
        D = D ** (-2 / (m - 1))
        mu[:, k] = D

    mu /= np.sum(mu, axis=1)[:, None]

    if np.max(np.abs(mu - old_mu)) < 1e-5:
        break

# 4. เลือกกลุ่ม
labels = np.argmax(mu, axis=1)

# 5. แสดงผล
rows = (len(images) + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(10, rows * 3), squeeze=False)

for i, ax in enumerate(axes.flat):
    ax.axis("off")

    if i < len(images):
        print(f"{names[i]} -> Cluster {labels[i] + 1}")
        ax.imshow(images[i], cmap="gray")
        ax.text(
        1.05, 0.5,
        f"{names[i]}\nCluster {labels[i] + 1}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9
        )

plt.tight_layout()
plt.show()
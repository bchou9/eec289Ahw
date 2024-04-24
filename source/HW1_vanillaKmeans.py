import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

# 導入 MNIST 數據集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 正規化圖像數據，將圖像數據從整數(0~255)轉換為浮點數(0.0~1.0)
train_images = train_images / 255.0
test_images = test_images / 255.0

def extract_patches(images, patch_size=5):                          #設置 5x5 圖像塊
    all_patches = []
    for image in images:
        for i in range(image.shape[0] - patch_size + 1):
            for j in range(image.shape[1] - patch_size + 1):
                patch = image[i:i + patch_size, j:j + patch_size]
                all_patches.append(patch.flatten())                 #將 5x5 圖像塊展平為一維數組
    return np.array(all_patches)                                    #將列表轉換為NumPy數組

# 從訓練集中提取 5x5 圖像塊
patches = extract_patches(train_images)

def perform_kmeans(patches, k):
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=100, random_state=42) #設定聚類數、batch處理大小、隨機狀態
    kmeans.fit(patches)                                             #對圖像塊進行聚類
    return kmeans

# 以 K=100/K=10000 執行 K-means
kmeans_model = perform_kmeans(patches, 10000)

# 畫圖
def plot_cluster_centers(kmeans_model):
    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))    #創建一個包含 100 個子圖的圖形
    centers = kmeans_model.cluster_centers_.reshape(-1, 5, 5)       #將聚類中心從一維數組重新變為 5x5 的圖像塊
    for i, ax in enumerate(ax.flat):                                #遍歷每一個子圖，並顯示對應的聚類中心圖像
        ax.imshow(centers[i], cmap='gray')                          #顯示聚類中心圖像，使用灰度色彩映射
        ax.axis('off')
    plt.show()

plot_cluster_centers(kmeans_model)

def reconstruct_patches(patches, kmeans_model):
    closest_clusters = kmeans_model.predict(patches)
    reconstructed_patches = kmeans_model.cluster_centers_[closest_clusters]
    return reconstructed_patches

# 使用 K-means 模型重構圖像塊
reconstructed_patches = reconstruct_patches(patches, kmeans_model)

# 計算和顯示重構誤差
def reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

# 選擇隨機幾個圖像塊並畫出來做比較
indices = np.random.choice(len(patches), 10, replace=False)
fig, axs = plt.subplots(10, 3, figsize=(10, 20))
for i, idx in enumerate(indices):
    orig_img = patches[idx].reshape(5, 5)
    reco_img = reconstructed_patches[idx].reshape(5, 5)
    diff_img = np.abs(orig_img - reco_img)

    axs[i, 0].imshow(orig_img, cmap='gray')
    axs[i, 0].set_title('Original')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(reco_img, cmap='gray')
    axs[i, 1].set_title('Reconstructed')
    axs[i, 1].axis('off')

    axs[i, 2].imshow(diff_img, cmap='hot')
    axs[i, 2].set_title('Difference')
    axs[i, 2].axis('off')

plt.tight_layout()
plt.show()

# 計算整體重構誤差
total_reconstruction_error = reconstruction_error(patches, reconstructed_patches)
print(f"Total Reconstruction Error: {total_reconstruction_error}")

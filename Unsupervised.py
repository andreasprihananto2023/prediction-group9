import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_excel('Train Data.xlsx')

data.columns = data.columns.str.strip()

features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
            'Estimated Duration (min)', 'Delivery Duration (min)', 
            'Distance (km)', 'Topping Density', 'Traffic Level']

X = data[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PCA: Reduksi Dimensi ---
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

# Visualisasi PCA
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=data['Delay (min)'], cmap='viridis', alpha=0.7)
plt.title('PCA - Komponen Utama vs Delay (min)')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.colorbar(label='Delay (min)')
plt.show()


print(f"Variance explained by first component: {pca.explained_variance_ratio_[0]:.2f}")
print(f"Variance explained by second component: {pca.explained_variance_ratio_[1]:.2f}")


loading_matrix = pd.DataFrame(pca.components_, columns=features, index=[f"Komponen Utama {i+1}" for i in range(2)])
print(loading_matrix)


# --- K-means Clustering ---
# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method Untuk Menentukan Jumlah Cluster')
plt.xlabel('Jumlah Cluster')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

kmeans = KMeans(n_clusters=12, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Visualisasi Hasil K-means
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.title('K-means Clustering dengan PCA')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.colorbar(label='Cluster ID')
plt.show()

# Menambahkan label cluster ke data asli
data['Cluster'] = kmeans_labels

# --- Profiling Statistik Deskriptif Per Cluster ---
cluster_stats = data.groupby('Cluster')[features].describe()
print("Statistik Deskriptif Per Cluster:")
print(cluster_stats)

# --- Visualisasi Distribusi Fitur untuk Setiap Cluster ---
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)  # Membuat grid 3x3
    sns.boxplot(x='Cluster', y=feature, data=data)
    plt.title(f'Box Plot {feature} by Cluster')
plt.tight_layout()
plt.show()

# --- Evaluasi Kualitas Clustering dengan Silhouette Score ---
silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
print(f'Silhouette Score untuk Clustering: {silhouette_avg:.2f}')

# --- Menyimpan Profiling Hasil Clustering ke dalam Excel ---
cluster_summary = data.groupby('Cluster').agg({
    'Delivery Duration (min)': ['mean', 'std'],
    'Delay (min)': ['mean', 'std', 'min', 'max'],
    'Pizza Complexity': ['mean', 'std'],
    'Order Hour': ['mean', 'std'],
    'Restaurant Avg Time': ['mean', 'std'],
    'Estimated Duration (min)': ['mean', 'std'],
    'Distance (km)': ['mean', 'std'],
    'Topping Density': ['mean', 'std'],
    'Traffic Level': ['mean', 'std']
})

# Simpan hasil analisis profil ke file Excel baru
output_file_summary = 'Cluster_Profiling_and_Statistics.xlsx'
cluster_summary.to_excel(output_file_summary)
print(f"Profiling hasil clustering disimpan ke {output_file_summary}")

# Simpan hasil clustering ke file Excel baru
output_file = 'Train_Data_with_Clustering_and_Avg_Delay.xlsx'
data.to_excel(output_file, index=False)  # Menyimpan data dengan cluster dan rata-rata delay
print(f"Data berhasil disimpan ke {output_file}")

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# 讀取數據
data_path = "train_total_3.csv"
data = pd.read_csv(data_path)


X = data.drop(columns=['class'])
y = data['class']

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# 定義模型
k = 8192  
mbkmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=1000000, n_init=10, max_iter=300)

mbkmeans.fit(X_standardized)
clusters = mbkmeans.predict(X_standardized)

# 保存結果
centroids = mbkmeans.cluster_centers_
cluster_labels = mbkmeans.labels_


result_df = pd.DataFrame(centroids, columns=[f'PC{i+1}' for i in range(centroids.shape[1])])
result_df['cluster'] = np.arange(k)


output_file = "clustered_data.csv"
result_df.to_csv(output_file, index=False)


data['cluster'] = clusters
data_output_file = "data_with_clusters.csv"
data.to_csv(data_output_file, index=False)

print(f'Clustered data and centroids saved to {output_file} and {data_output_file}')
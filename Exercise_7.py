import os
from pathlib import Path

# utk file scrap & file prepro satu folder
try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:
    pass

# Exercise 7 - Text Clustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

df = pd.read_csv('gt_wiki_preprocessed.csv')
df = df.dropna().reset_index().drop(columns=['index'])
df_pre = df['prepro']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_pre)
df_FE = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df_FE.shape)

# Kmeans
from sklearn.cluster import KMeans

k = range(1, 20)
inertia = []
for i in k:
    km_model = KMeans(n_clusters=i, random_state=10)  # jumlah cluster akan di looping
    km_model.fit(df_FE)
    inertia.append(km_model.inertia_)
print(list(k))
print(inertia)

plt.figure(figsize=(10, 6))
plt.plot(k, inertia, marker='o')
plt.xlabel('k value')
plt.ylabel('inertia')
plt.grid()
plt.show()

# dpt diubah sesuai hasil elbow
n_clusters = 4

kM = KMeans(n_clusters=n_clusters, random_state=10).fit(df_FE)
print(kM)

df['Kmeans_label'] = kM.labels_
print(df.head())
print(df['Kmeans_label'].value_counts().sort_index())

# Silhouette Score
from sklearn import metrics
labels = kM.labels_
print(metrics.silhouette_score(df_FE, labels, metric='euclidean'))

# Word Cloud
for c in range(n_clusters):
    print('Kmeans for cluster ' + str(c))
    df_c = df[df['Kmeans_label'] == c]
    long_string = ','.join(list(df_c['prepro'].values))
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3)
    wordcloud.generate(long_string)
    plt.figure(figsize=(7, 4))
    plt.imshow(wordcloud.to_image())
    plt.axis('off')
    plt.title('Kmeans cluster ' + str(c))
    plt.show()

# Hierarchical
from scipy.cluster.hierarchy import dendrogram, linkage
linkage_data = linkage(df_FE, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.tight_layout()
plt.show()

from sklearn.cluster import AgglomerativeClustering
hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = hierarchical_cluster.fit(df_FE)

pred_agc = pd.Series(hierarchical_cluster.labels_)
df['Ward_label'] = pred_agc
print(df['Ward_label'].value_counts().sort_index())

# Silhouette Score
labels = pred_agc
print(metrics.silhouette_score(df_FE, labels, metric='euclidean'))

# Word Cloud
for c in range(n_clusters):
    print('Hierarchical for cluster ' + str(c))
    df_c = df[df['Ward_label'] == c]
    long_string = ','.join(list(df_c['prepro'].values))
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3)
    wordcloud.generate(long_string)
    plt.figure(figsize=(7, 4))
    plt.imshow(wordcloud.to_image())
    plt.axis('off')
    plt.title('Hierarchical cluster ' + str(c))
    plt.show()

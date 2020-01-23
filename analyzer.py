#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS

class Analyzer:
    def __init__(self, csvfile, sep=';'):
        self.df = pd.read_csv(csvfile, sep=sep)
        self.df['SEQ'] = self.df['SEQ_PLEN'].map(str) + self.df['SEQ_TDIFF']
        self.df['SEQ'] = self.df['SEQ'].apply(lambda x: np.array([int(n) for n in x.split(',')]))
        self.X = np.vstack(self.df['SEQ'].values)
        self.X = StandardScaler().fit_transform(self.X)

    def dbscan(self):
        self.db = DBSCAN(eps=0.123, min_samples=10)
        self.clusters = self.db.fit_predict(self.X) 
        # plot the cluster assignments
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.clusters, cmap="plasma")
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")

        plt.show()

    def kmeans(self):
        self.kmeans = KMeans(n_clusters=70)
        
        y_pred = self.kmeans.fit_predict(self.X)
        # plot the cluster assignments and cluster centers
        plt.scatter(self.X[:, 0], self.X[:, 1], c=y_pred, cmap="plasma")
        plt.scatter(self.kmeans.cluster_centers_[:, 0],   
                    self.kmeans.cluster_centers_[:, 1],
                    marker='^', 
                    c=[i for i in range(70)], 
                    s=100, 
                    linewidth=2,
                    cmap="plasma")
        plt.show()
        
    def tsne(self):
        tsne = TSNE(random_state = 0)
        X_tsne = tsne.fit_transform(self.X)
        df_subset = {}
        df_subset['tsne-2d-one'] = X_tsne[:,0]
        df_subset['tsne-2d-two'] = X_tsne[:,1]

        
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            palette=sns.color_palette("hls", 10),
            data=df_subset,
            legend="full",
            alpha=0.3
        ).set_title('SEQ_PLEN + SEQ_TDIFF')

        plt.show()


    def mds(self):
        mds = MDS(n_components=2, verbose=4, n_jobs=-1)
        X_mds = mds.fit_transform(self.X)
        df_subset = {}
        df_subset['mds-2d-one'] = X_mds[:,0]
        df_subset['mds-2d-two'] = X_mds[:,1]

        
        sns.scatterplot(
            x="mds-2d-one", y="mds-2d-two",
            palette=sns.color_palette("hls", 10),
            data=df_subset,
            legend="full",
            alpha=0.3
        ).set_title('SEQ_PLEN + SEQ_TDIFF')

        plt.show()
        
        
    def show(self):

        labels = self.db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)

        
        core_samples_mask = np.zeros_like(self.db.labels_, dtype=bool)
        core_samples_mask[self.db.core_sample_indices_] = True
        
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)
            xy = self.X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

            xy = self.X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()


def main():
    analyzer = Analyzer(sys.argv[1])
    print(analyzer.df)
    analyzer.mds()
    #analyzer.show()
    
if __name__ == '__main__':
    main()

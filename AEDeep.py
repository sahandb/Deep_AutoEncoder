# imports

# import warnings
#
# warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


# pre process for delete first column and normalize data
def preProcess(path):
    df = pd.read_excel(path, header=None).drop(columns=0)
    x = df.values  # returns a numpy array
    min_max_scalar = preprocessing.MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df


def kMeans_DBS(X, latentDim):
    km = KMeans(n_clusters=latentDim, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    kmeans = km.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    dbsMetric = davies_bouldin_score(X, labels)
    print("\nDBS for latent dim(Z) : {0} is : {1}".format(latentDim, dbsMetric * 100))

    if latentDim == 2:
        # plot the 2 clusters
        plt.scatter(
            X[labels == 0, 0], X[labels == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='cluster 1'
        )

        plt.scatter(
            X[labels == 1, 0], X[labels == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='cluster 2'
        )
        # plot the centroids
        plt.scatter(
            centers[:, 0], centers[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids'
        )
        plt.legend(scatterpoints=1)
        plt.grid()
        plt.show()
    elif latentDim == 3:
        # plot the 3 clusters
        figg = plt.figure()
        axx = Axes3D(figg)
        axx.scatter(
            X[labels == 0, 0], X[labels == 0, 1], X[labels == 0, 2],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='cluster 1'
        )

        axx.scatter(
            X[labels == 1, 0], X[labels == 1, 1], X[labels == 1, 2],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='cluster 2'
        )

        axx.scatter(
            X[labels == 2, 0], X[labels == 2, 1], X[labels == 2, 2],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='cluster 3'
        )

        # plot the centroids
        axx.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids'
        )
        axx.legend(scatterpoints=1)
        axx.grid()
        plt.savefig('kMean3d.png', bbox_inches='tight')
        plt.show()


def AE(latent_dim):
    N = 181
    # input placeholder
    input_data = Input(shape=(N,))  # 181 is the number of features/columns
    # encoder is the encoded representation of the input
    encoded = Dense(64, activation='relu', name='encoder_dense_1')(input_data)
    encoded = Dense(16, activation='relu', name='encoder_dense_2')(encoded)
    encoded = Dense(latent_dim, activation='relu', name='encoder_dense_3')(encoded)
    DataOfLatent = encoded
    # this model maps an input to its encoded representation
    encoder = Model(input_data, encoded)

    encoder.summary()

    # decoder is the lossy reconstruction of the input
    decoded = Dense(16, activation='relu', name='decoder_dense_3')(encoded)
    decoded = Dense(64, activation='relu', name='decoder_dense_2')(decoded)
    decoded = Dense(N, activation='relu', name='decoder_dense_1', )(decoded)

    # model optimizer and loss
    autoencoder = Model(input_data, decoded)
    # loss function and optimizer
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    autoencoder.summary()

    keras.utils.plot_model(autoencoder, "my_first_model_with_shape_info.png", show_shapes=True)

    # read Data
    dataFrame = preProcess('Gordon-2002_LungCancer.xlsx')
    # train test split
    x_train, x_test = train_test_split(dataFrame, test_size=0.3)

    # train the model
    autoencoder.fit(x_train, x_train,
                    epochs=30,
                    batch_size=64,
                    validation_data=(x_test, x_test))

    if latent_dim == 2:
        # predict after training
        x_train_encoded = encoder.predict(x_train, batch_size=64)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1])
        plt.colorbar()
        plt.show()

        # note that we take them from the *test* set
        x_test_encoded = encoder.predict(x_test, batch_size=64)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
        plt.colorbar()
        plt.show()
        # kMeans print metric
        kMeans_DBS(x_test_encoded, latent_dim)

    elif latent_dim == 3:
        # predict after training
        x_train_encoded = encoder.predict(x_train, batch_size=64)
        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)
        ax.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], x_train_encoded[:, 2])
        plt.show()

        # note that we take them from the *test* set
        x_test_encoded = encoder.predict(x_test, batch_size=64)
        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)
        ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], x_test_encoded[:, 2])
        plt.show()
        # kMeans print metric
        kMeans_DBS(x_test_encoded, latent_dim)

    elif latent_dim == 4 or latent_dim == 5:
        x_train_encoded = encoder.predict(x_train, batch_size=64)
        # predict after training
        # note that we take them from the *test* set
        x_test_encoded = encoder.predict(x_test, batch_size=64)
        # kMeans print metric
        kMeans_DBS(x_test_encoded, latent_dim)


def main():
    latent_dim = 5
    AE(latent_dim)


if __name__ == '__main__':
    main()

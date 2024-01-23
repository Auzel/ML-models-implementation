import numpy as np
import matplotlib.pyplot as plt


def visualization(data, labels, classifiers, weights=None):
    '''
    Visualizes a dataset and the ensemble's predictions in 2D space.

    Arguments:
        data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (n, ). Values are +1 or -1.
        classifiers: A list of classifiers.
        weights: A list of weights assigned to the classifiers.
            If None, weights will be equal for all classifiers.
    '''
    if weights is None:
        weights = [1.0] * labels.size

    # Aggregate predicions from weighted classifiers
    pred = np.zeros(labels.shape)
    for s, w in zip(classifiers, weights):
        pred += w * s.predict(data)

    # Break ties if any
    pred = np.sign(np.sign(pred) + 1e-6)

    # Count mis-classified datapoints
    count = (pred != labels).sum()
    print('{} points mis-classified'.format(count))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_title('Ground truth')
    axes[0].plot(data[labels == 1, 0], data[labels == 1, 1], 'rx')
    axes[0].plot(data[labels == -1, 0], data[labels == -1, 1], 'bo')
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[1].set_title('Prediction')
    axes[1].plot(data[pred == 1, 0], data[pred == 1, 1], 'rx')
    axes[1].plot(data[pred == -1, 0], data[pred == -1, 1], 'bo')
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    plt.show()


def get_dataset_fixed():
    '''
    Returns a simple dataset with pre-defined datapoints.

    Returns:
        data: An ndarray with shape (20, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (20, ). Values are +1 or -1.
    '''
    data = np.array([[ 0.1,  0.4],
                     [ 0.2,  0.1],
                     [-0.2,  0.3],
                     [-0.1,  0.8],
                     [ 0.9, -0.2],
                     [ 0.6,  0.1],
                     [ 0.1,  0.9],
                     [-0.9, -0.8],
                     [-1. ,  0.7],
                     [ 0.6,  0.7],
                     [ 1. ,  0.6],
                     [-0.1,  0.6],
                     [-0.8,  0.3],
                     [-0.7,  0.9],
                     [ 0. , -0.2],
                     [-0.5,  0.5],
                     [-0.1,  0.1],
                     [-1. ,  0.2],
                     [ 0.2,  0.2],
                     [ 0.9,  0.4]])
    labels = np.array([-1., -1., -1., -1.,  1.,
                       -1.,  1.,  1.,  1.,  1.,
                        1., -1.,  1.,  1., -1.,
                       -1., -1.,  1., -1.,  1.])
    return data, labels


def get_dataset_random(n=20, seed=0):
    '''
    Returns a simple dataset by random construction.

    Arguments:
        n: Number of points in the dataset.
        seed: Random seed for NumPy.

    Returns:
        data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (n, ). Values are +1 or -1.
    '''
    np.random.seed(seed)
    data = np.random.rand(n, 2) * 2 - 1
    data = np.around(data, 1)
    labels = np.sign(data[:, 0] ** 2 + data[:, 1] ** 2 - 0.7)
    return data, labels

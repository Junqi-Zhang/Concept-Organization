import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def scale_by_l2_norm(X):
    return X / np.mean(np.linalg.norm(X, ord=2, axis=1))


def scale_feature_standard(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def whiten(X, method):
    """ Whitens data matrix X

    Args:
        X : origin data, each row represents an observation
        method (str): "PCA" or "ZCA"

    Returns:
        X_whitened : whitened data, each row represents an observation
    """

    # center the data
    X = X - X.mean(axis=0)

    # calculate covariance matrix
    # m, n = X.shape
    # C = np.dot(X.T, X) / m

    C = np.cov(X, rowvar=False, ddof=0)

    # eigen decomposition of covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(C)

    # compute inverse of D, and take square root
    D = eig_vals  # Node: eig_vals is a vector, but we want a matrix, so np.diag() later
    epsilon = 1e-5  # whitening constant: prevents division by zero
    D_m12 = np.diag((D+epsilon)**(-0.5))  # 'm12' for power(minus 1/2)
    P = eig_vecs

    W_transform = np.array([])
    if method == "PCA":
        W_transform = np.dot(D_m12, P.T)
    elif method == "ZCA":
        W_transform = np.dot(np.dot(P, D_m12), P.T)
    else:
        ValueError(f"Method {method} not available for whiten()")

    # compute transformed-whitened data
    X_whitened = np.dot(W_transform, X.T).T

    return X_whitened


def get_standard_data(X, method):
    if method is None:
        return X
    elif method == "l2_norm":
        return scale_by_l2_norm(X)
    elif method == "feature_standard":
        return scale_feature_standard(X)
    elif method == "PCA_whiten":
        return whiten(X, "PCA")
    elif method == "ZCA_whiten":
        return whiten(X, "ZCA")
    else:
        ValueError(f"Method {method} not available for get_standard_data()")


def test_standard_data():
    # generate random data, with arbitrary covariance
    m, n = 500, 2
    X = np.random.multivariate_normal(
        [0 for i in range(n)],
        np.array([[1.0, 0.8], [0.8, 1.0]]),
        size=m)
    # sort samples for plotting with increasing colors
    X = X[X[:, 0].argsort(), :]

    cmap = matplotlib.colormaps.get_cmap("Spectral") # type: ignore
    colors = cmap(np.linspace(0, 1, m))

    def scatter_with_variance(X_whitened, ax, title):
        # Transform data
        # x = (A @ X.T).T
        x = X_whitened
        ax.scatter(*x.T, c=colors)
        ax.set_title(
            title + "\n" + f"C={np.array2string(x.T@x/len(x), precision=2, suppress_small=True, floatmode='fixed')}")
        # Add variances in x and y direction
        ax.set_xlabel(f"Var={np.var(x.T[0]):.3f}")
        ax.set_ylabel(f"Var={np.var(x.T[1]):.3f}")

    fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(30, 6))

    scatter_with_variance(get_standard_data(X, None),
                          axes[0], "RAW data")
    scatter_with_variance(get_standard_data(X, "l2_norm"),
                          axes[1], "scaled-by-l2-norm")
    scatter_with_variance(get_standard_data(X, "feature_standard"),
                          axes[2], "scaled-by-feature-standard")
    scatter_with_variance(get_standard_data(X, "PCA_whiten"),
                          axes[3], "PCA-whitened")
    scatter_with_variance(get_standard_data(X, "ZCA_whiten"),
                          axes[4], "ZCA-whitened")
    plt.show()


if __name__ == "__main__":
    test_standard_data()

import numpy as np
from sklearn.datasets import fetch_openml

def load_dataset():
    """
    Loads dataset as a numpy array. The images are flattened from 28x28 to 784
    returns: 
    X - the images as numpy array. shape (70000, 784)
    y - the targets as a numpy array. shape (70000,)
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target
    return X,y 

def filter_dataset(X,y, classes: str):
    if classes == "all":
        return X,y
    else:
        classes_list = classes.split(",")
        mask = np.isin(y, classes_list)

        X_filtered = X[mask]
        y_filtered = y[mask]
        return X_filtered, y_filtered


if __name__ == "__main__":
    X,y = load_dataset()
    X,y = filter_dataset(X,y,"1,7")
    print(X.shape)

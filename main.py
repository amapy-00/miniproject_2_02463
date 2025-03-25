import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import urllib.request
import scienceplots
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lin
import sklearn.metrics
import sklearn.utils

plt.style.use(['science','notebook'])

# --------------------------
# Helper functions

def load_data(digit_filter):
    """Load, filter and normalize dataset."""
    from data import load_dataset, filter_dataset
    X_full, y_full = load_dataset()
    X, y = filter_dataset(X_full, y_full, digit_filter)
    X = X / 255
    print("Data shape:", X.shape, y.shape)
    return X, y

def plot_samples(X, indices, grid_shape, cmap='gray', title="Sample images"):
    """Plot images in a grid."""
    fig, axes = plt.subplots(*grid_shape)
    axes = axes.flatten()
    for idx, img_index in enumerate(indices):
        axes[idx].imshow(X[img_index].reshape(28, 28), cmap=cmap)
        axes[idx].axis('off')
    plt.suptitle(title)
    plt.show()

def perform_pca(X, n_components):
    """Apply PCA and return transformed data and total variance."""
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    total_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA (n_components={n_components}) variance: {total_variance}")
    return X_transformed, total_variance

def perform_lda(X, y, n_components, scale=False):
    """Apply LDA (with optional scaling) and return transformed data and total variance."""
    X_prepared = StandardScaler().fit_transform(X) if scale else X
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_transformed = lda.fit_transform(X_prepared, y)
    total_variance = np.sum(lda.explained_variance_ratio_)
    print(f"LDA (n_components={n_components}) variance: {total_variance}, new shape: {X_transformed.shape}")
    return X_transformed, total_variance

def simulate_random_sampling(model, X_pool, y_pool, X_test, y_test, pool_order, initial_samples, added_samples, num_iterations):
    """Active learning simulation with random sampling."""
    accuracy_results = []
    for i in range(num_iterations):
        indices = pool_order[:initial_samples + i * added_samples]
        X_train = np.take(X_pool, indices, axis=0)
        y_train = np.take(y_pool, indices, axis=0)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = sklearn.metrics.accuracy_score(y_test, predictions)
        accuracy_results.append((initial_samples + i * added_samples, acc))
        print(f"Model: LR, {initial_samples + i * added_samples} random samples")
    return accuracy_results

def simulate_qbc(model, X_pool, y_pool, X_test, y_test, pool_order, initial_samples, added_samples, num_iterations, n_committee=10):
    """Active learning simulation using QBC with bootstrapped committees."""
    train_indices = pool_order[:initial_samples]
    X_train = np.take(X_pool, train_indices, axis=0)
    y_train = np.take(y_pool, train_indices, axis=0)
    remaining_indices = np.setdiff1d(np.arange(len(X_pool)), train_indices)
    accuracy_results = []
    
    for i in range(num_iterations):
        committee_predictions = []
        for _ in range(n_committee):
            X_boot, y_boot = sklearn.utils.resample(X_train, y_train, stratify=y_train)
            model.fit(X_boot, y_boot)
            preds = model.predict(X_pool[remaining_indices])
            committee_predictions.append(preds)
        committee_predictions = np.array(committee_predictions)
        vote_fraction = []
        for j in range(committee_predictions.shape[1]):
            counts = np.bincount(committee_predictions[:, j].astype(int))
            vote_fraction.append(np.max(counts) / n_committee)
        vote_fraction = np.array(vote_fraction)
        # Select least confident samples (lowest vote_fraction)
        selected_idx = np.argsort(vote_fraction)[:added_samples]
        new_train_indices = remaining_indices[selected_idx]
        X_train = np.concatenate((X_train, X_pool[new_train_indices]))
        y_train = np.concatenate((y_train, y_pool[new_train_indices]))
        remaining_indices = np.setdiff1d(remaining_indices, new_train_indices)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = sklearn.metrics.accuracy_score(y_test, predictions)
        accuracy_results.append((len(X_train), acc))
        print(f"Model: LR, {len(X_train)} samples (QBC)")
    return accuracy_results

# --------------------------
# Main experiments

def run_experiment(digit_filter, lda_dims, active_params, legend_labels):
    """
    For a given dataset (digit_filter) and LDA settings, this experiment:
      - Loads data and displays sample images.
      - Runs PCA (for variance check) and then applies LDA.
      - Performs active learning (random sampling and QBC) and plots learning curves.
    """
    # Load and display sample images
    X, y = load_data(digit_filter)
    sample_indices = [0, 45] if digit_filter == "1,7" else [0, 50, 100]
    grid = (1, 2) if digit_filter == "1,7" else (1, 3)
    plot_samples(X, sample_indices, grid, title=f"Digits {digit_filter} sample images")
    
    # PCA demos
    _, pca_variance_2 = perform_pca(X, n_components=2)
    _, pca_variance_1 = perform_pca(X, n_components=1)
    
    # Apply LDA; scale only for two-digit case
    scale_option = True if digit_filter == "1,7" else False
    X_lda, _ = perform_lda(X, y, n_components=lda_dims, scale=scale_option)
    
    # Plot LDA distributions
    if X_lda.ndim == 1 or X_lda.shape[1] == 1:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(x=X_lda.reshape(-1), hue=y, fill=True, palette=['blue','red'] if digit_filter=="1,7" else None)
        plt.xlabel("LDA Feature")
        plt.title("LDA feature distribution by class")
        plt.legend()
        plt.show()
    else:
        for dim in range(X_lda.shape[1]):
            plt.figure(figsize=(8, 5))
            sns.kdeplot(x=X_lda[:, dim], hue=y, fill=True)
            plt.xlabel(f"LDA Feature {dim + 1}")
            plt.title("LDA feature distribution by class")
            plt.legend()
            plt.show()
            
    # Prepare active learning data splits (using slicing)
    X_test, y_test = X_lda[500:], y[500:]
    X_pool, y_pool = X_lda[:500], y[:500]
    
    lr_model = lin.LogisticRegression(penalty='l2', C=1.)
    pool_order = np.random.permutation(len(X_pool))
    
    random_acc = simulate_random_sampling(lr_model, X_pool, y_pool, X_test, y_test,
                                          pool_order, active_params['initial_samples'],
                                          active_params['added_samples'], active_params['num_iterations'])
    qbc_acc = simulate_qbc(lr_model, X_pool, y_pool, X_test, y_test,
                           pool_order, active_params['initial_samples'],
                           active_params['added_samples'], active_params['num_iterations'])
    
    plt.figure(figsize=(6, 4), dpi=150)
    random_results = np.array(random_acc)
    qbc_results = np.array(qbc_acc)
    plt.plot(random_results[:, 0], random_results[:, 1], marker='o')
    plt.plot(qbc_results[:, 0], qbc_results[:, 1], marker='s')
    plt.xlabel("Number of training samples")
    plt.ylabel("Test accuracy")
    plt.legend(legend_labels)
    plt.title(f"Active learning ({digit_filter})")
    plt.show()

def main():
    # Experiment 1: Digits 1 and 7; LDA with 1 component; active learning parameters
    exp1_params = {'initial_samples': 10, 'added_samples': 5, 'num_iterations': 30}
    run_experiment("1,7", lda_dims=1, active_params=exp1_params, legend_labels=('Random sampling', 'QBC'))
    
    # Experiment 2: Digits 1,7,9; LDA with 2 components
    exp2_params = {'initial_samples': 10, 'added_samples': 5, 'num_iterations': 30}
    run_experiment("1,7,9", lda_dims=2, active_params=exp2_params, legend_labels=('Random sampling', 'QBC'))
    
    # Experiment 3: All digits; LDA with 8 components; different active learning parameters
    exp3_params = {'initial_samples': 10, 'added_samples': 10, 'num_iterations': 30}
    run_experiment("all", lda_dims=8, active_params=exp3_params, legend_labels=('Random sampling', 'QBC'))

if __name__ == '__main__':
    main()



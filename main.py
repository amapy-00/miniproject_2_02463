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
from scipy.stats import entropy
import random
import pandas as pd
import seaborn as sns

# Set a global seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

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

def perform_lda(X, y, n_components):
    """Apply LDA and return transformed data and total variance."""
    X_prepared = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_transformed = lda.fit_transform(X_prepared, y)
    total_variance = np.sum(lda.explained_variance_ratio_)
    print(f"LDA (n_components={n_components}) variance: {total_variance}, new shape: {X_transformed.shape}")
    return X_transformed, total_variance

def simulate_random_sampling(model, X_pool, y_pool, X_test, y_test, pool_order, initial_samples, added_samples, num_iterations):
    """Active learning simulation with random sampling."""
    accuracy_results = []
    for i in range(1, num_iterations+1):
        indices = pool_order[:initial_samples + i * added_samples]
        X_train = np.take(X_pool, indices, axis=0)
        y_train = np.take(y_pool, indices, axis=0)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = sklearn.metrics.accuracy_score(y_test, predictions)
        accuracy_results.append((initial_samples + i * added_samples, acc))
        print(f"Model: LR, {initial_samples + i * added_samples} random samples")
    return accuracy_results

# have one mode: model.predict_proba and predict for every sample in the pool and then also choose the least confident samples
# (from 1 to x, same as in the QBC) only difference is calculating the prob from the single model (LR) and make the committee a
# single model.There is relevant code in week 7 exercises.
def simulate_qbc(model, X_pool, y_pool, X_test, y_test, pool_order, initial_samples, added_samples, num_iterations, uncertainty_metric, committee_size=10, visualize=False, X_orig_pool=None):
    """Active learning simulation using QBC with bootstrapped committees."""
    # Select training samples: choose (initial_samples - 1) samples from one class (target class)
    # and select the last sample from another class.
    target_class = y_pool[pool_order[0]]
    target_indices = [idx for idx in pool_order if y_pool[idx] == target_class]
    other_indices = [idx for idx in pool_order if y_pool[idx] != target_class]
    if len(target_indices) >= initial_samples - 1 and len(other_indices) > 0:
        train_indices = target_indices[:initial_samples - 1] + [other_indices[-1]]
    else:
        train_indices = pool_order[:initial_samples]
    X_train = np.take(X_pool, train_indices, axis=0)
    y_train = np.take(y_pool, train_indices, axis=0)
    remaining_indices = np.setdiff1d(np.arange(len(X_pool)), train_indices)
    accuracy_results = []
    
    for i in range(num_iterations):

        committee_predictions = []
        for _ in range(committee_size):
            X_boot, y_boot = sklearn.utils.resample(X_train, y_train, stratify=y_train)
            model.fit(X_boot, y_boot)
            preds = model.predict(X_pool[remaining_indices])
            committee_predictions.append(preds.astype(int))
        committee_predictions = np.array(committee_predictions)

        vote_fraction = []
        for j in range(committee_predictions.shape[1]):
            counts = np.bincount(committee_predictions[:, j].astype(int))
            vote_fraction.append(np.max(counts) / committee_size)
        vote_fraction = np.array(vote_fraction)
        
        # Visualization of uncertainty (only if 1-dim) 
        if visualize and X_pool.shape[1] == 1:
            uncertainties = 1 - vote_fraction
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
            chosen_indices = np.argsort(vote_fraction)[:added_samples]
            
            # Plot KDE with hue; capture the legend handles
            sns.kdeplot(x=X_pool.reshape(-1), hue=y_pool.reshape(-1), fill=True, ax=ax)
            kde_handles, kde_labels = ax.get_legend_handles_labels()
            
            # Plot uncertainty line and training/chosen scatter points (remove label arguments)
            plot_df = pd.DataFrame({
                'first_lda_feature': X_pool[remaining_indices].flatten(),
                'uncertainty': uncertainties
            }).sort_values('first_lda_feature')
            sns.lineplot(x='first_lda_feature', y='uncertainty', data=plot_df, color='blue', alpha=0.8, ax=ax)
            
            st_sc = ax.scatter(X_train.flatten(), [0]*len(X_train.flatten()),
                               c='green', marker='x', s=100, alpha=0.9)  # no label here
            ch_sc = ax.scatter(X_pool[remaining_indices][chosen_indices].flatten(),
                               [0]*added_samples, c='red', marker='*', s=150, edgecolor='black', linewidth=0.5)  # no label
            
            # Create custom legend handles for training and chosen samples
            from matplotlib.lines import Line2D
            training_handle = Line2D([], [], marker='x', color='green', linestyle='None', markersize=10, label='Training')
            chosen_handle = Line2D([], [], marker='*', color='red', linestyle='None', markersize=15, label='Chosen')
            
            combined_handles = kde_handles + [training_handle, chosen_handle]
            combined_labels = kde_labels + ['Training', 'Chosen']
            
            ax.set_title(f"Uncertainty Distribution QBC Iteration {i+1}", fontsize=14)
            ax.set_xlabel("First LDA Feature", fontsize=12)
            ax.set_ylabel("Uncertainty Score", fontsize=12)
            ax.legend(combined_handles, combined_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')
            
            if added_samples == 1 and X_orig_pool is not None:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                axins = inset_axes(ax, width="30%", height="30%", loc='upper left')
                chosen_orig = X_orig_pool[remaining_indices][chosen_indices[0]]
                try:
                    img = chosen_orig.reshape(28, 28)
                except Exception:
                    img = chosen_orig
                axins.imshow(img, cmap='gray')
                axins.axis('off')
                                    
            plt.show()


        
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

    if uncertainty_metric == 'vote_entropy':
        uncertainty_scores = calculate_vote_entropy(committee_predictions)
        print("Mean uncertainty:", np.mean(uncertainty_scores))
        print("Maximum uncertainty:", np.max(uncertainty_scores))
    elif uncertainty_metric == 'variance':
        uncertainty_scores = calculate_variance(committee_predictions)
        print("Mean uncertainty:", np.mean(uncertainty_scores))
        print("Maximum uncertainty:", np.max(uncertainty_scores))

    return accuracy_results

def simulate_single_US_model(model, X_pool, y_pool, X_test, y_test, pool_order, initial_samples, added_samples, num_iterations):
    accuracy = []
    trainset = pool_order[:initial_samples]
    Xtrain = np.take(X_pool, trainset, axis=0)
    ytrain = np.take(y_pool, trainset, axis=0)
    poolidx=np.arange(len(X_pool),dtype=np.int64)
    poolidx=np.setdiff1d(poolidx,trainset)
    
    model.fit(Xtrain, ytrain)

    for i in range(num_iterations):
        # Obtain label probabilities for the current pool
        ypool_p = model.predict_proba(X_pool[poolidx])
        # Select samples with the least confidence (lowest max probability)
        selected_idx = np.argsort(-ypool_p.max(axis=1))
        
        # Add the selected samples to the training set
        new_sample_inds = poolidx[selected_idx[-added_samples:]]
        Xtrain = np.concatenate((Xtrain, X_pool[new_sample_inds]))
        ytrain = np.concatenate((ytrain, y_pool[new_sample_inds]))
        poolidx = np.setdiff1d(poolidx, new_sample_inds)
        
        # Retrain the model on the updated training set
        model.fit(Xtrain, ytrain)
        ye = model.predict(X_test)
        test_accuracy = sklearn.metrics.accuracy_score(y_test, ye)
        accuracy.append((len(Xtrain), test_accuracy))
        
        print(f"Model: LR, {len(Xtrain)} samples (US) - Accuracy after update: {test_accuracy:.4f}.")

    return accuracy

def calculate_vote_entropy(predictions):
    """
    Calculates vote entropy for classification tasks.
    predictions: A list of arrays, where each array contains the predicted class labels from a committee member.
    """
    num_samples = predictions[0].shape[0]
    entropy_values = np.zeros(num_samples)

    for i in range(num_samples):
        # Count votes for each class
        class_counts = np.bincount([pred[i] for pred in predictions])
        # Calculate probabilities
        probabilities = class_counts / len(predictions)
        # Calculate entropy
        entropy_values[i] = entropy(probabilities, base=2)

    plt.hist(entropy_values, bins=20)
    plt.title("Vote entropy distribution")
    plt.xlabel("Vote entropy")
    plt.ylabel("Frequency")
    plt.show()

    return entropy_values

def calculate_variance(predictions):
    """
    Calculates variance for regression tasks.
    predictions: A list of arrays, where each array contains the predicted values from a committee member.
    """
    return np.var(predictions, axis=0) # Variance along the committee member axis 


def compare_committee_sizes(model, X_pool, y_pool, X_test, y_test, pool_order, initial_samples, added_samples, num_iterations, committee_sizes, visualize, X_orig_pool=None):
    """Run QBC simulation for different committee sizes and return results as a dict."""
    results = {}
    for cs in committee_sizes:
        print(f"Running QBC with committee size {cs}")
        results[cs] = simulate_qbc(model, X_pool, y_pool, X_test, y_test,
                                   pool_order, initial_samples, added_samples, num_iterations,
                                   committee_size=cs,
                                   visualize=visualize, uncertainty_metric='vote_entropy', X_orig_pool=X_orig_pool)
    return results

# --------------------------
# Main experiments

def run_experiment(digit_filter, lda_dims, active_params, legend_labels):
    """
    For a given dataset (digit_filter) and LDA settings, this experiment:
      - Loads data and displays sample images.
      - Runs PCA (for variance check) and then applies LDA.
      - Performs active learning (random sampling and QBC with various committee sizes)
        and plots all curves on one plot.
    """
    print(f"Running experiment for digits {digit_filter} with LDA dims {lda_dims}")
    # Load and display sample images
    X, y = load_data(digit_filter)
    sample_indices = [0, 45] if digit_filter == "1,7" else [0, 50, 100]
    grid = (1, 2) if digit_filter == "1,7" else (1, 3)
    plot_samples(X, sample_indices, grid, title=f"Digits {digit_filter} sample images")
    
    # PCA demos
    _, pca_variance_2 = perform_pca(X, n_components=2)
    _, pca_variance_1 = perform_pca(X, n_components=1)
    
    # Apply LDA; always use scaling in LDA.
    X_lda, _ = perform_lda(X, y, n_components=lda_dims)
    
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
    Pool_size = active_params["initial_samples"] + active_params["added_samples"] * active_params["num_iterations"]
    Pool_size *= 50  
    Pool_size = min(Pool_size, len(X_lda))
    X_test, y_test = X_lda[Pool_size:], y[Pool_size:]
    X_pool, y_pool = X_lda[:Pool_size], y[:Pool_size]
    # Save the corresponding original images for visualization in QBC
    X_pool_orig = X[:Pool_size]
    
    lr_model = lin.LogisticRegression(penalty='l2', C=1.)
    pool_order = np.random.permutation(len(X_pool))
    
    random_acc = simulate_random_sampling(lr_model, X_pool, y_pool, X_test, y_test,
                                          pool_order, active_params['initial_samples'],
                                          active_params['added_samples'], active_params['num_iterations'])

    simple_us = simulate_single_US_model(lr_model, X_pool, y_pool, X_test, y_test,
                                         pool_order, active_params['initial_samples'],
                                         active_params['added_samples'], active_params['num_iterations'])
    
    
    # Compare different QBC committee sizes
    comp_results = compare_committee_sizes(lr_model, X_pool, y_pool, X_test, y_test,
                                           pool_order, active_params['initial_samples'],
                                           active_params['added_samples'], active_params['num_iterations'],
                                           active_params['committee_sizes'],
                                           visualize=active_params.get("visualize", False),
                                           X_orig_pool=X_pool_orig)
    
    # Plot both random sampling and QBC curves (for each committee size) in one figure
    plt.figure(figsize=(8, 5), dpi=150)
    random_results = np.array(random_acc)
    plt.plot(random_results[:, 0], random_results[:, 1], marker='o', markersize=4, label='Random sampling')
    for cs, acc in comp_results.items():
        cs_results = np.array(acc)
        plt.plot(cs_results[:, 0], cs_results[:, 1], marker='o', markersize=4, label=f'QBC Committee = {cs}')
    simple_results = np.array(simple_us)
    plt.plot(simple_results[:, 0], simple_results[:, 1], marker='o', markersize=4, label='Uncertainty Sampling')
    plt.xlabel("Number of training samples", fontsize=10)
    plt.ylabel("Test accuracy", fontsize=10)
    plt.title(f"Comparison: Random vs QBC (Committee Sizes) ({digit_filter})\npool_size={Pool_size}", fontsize=12)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fancybox=True, shadow=True, fontsize='small')
    plt.tight_layout()
    plt.show()

def main():
    # Experiment 1: Digits 1 and 7; LDA with 1 component; active learning parameters
    exp1_params = {
        'initial_samples': 10,
        'added_samples': 5,
        'num_iterations': 30,
        'committee_sizes': [5, 10, 15],  # Compare different committee sizes
        'visualize': True
    }
    np.random.seed(SEED)  # Reset seed for reproducibility
    run_experiment("1,7", lda_dims=1, active_params=exp1_params, legend_labels=('Random sampling', 'QBC'))
    
    # Experiment 2: Digits 1,7,9; LDA with 2 components
    exp2_params = {
        'initial_samples': 10,
        'added_samples': 5,
        'num_iterations': 30,
        'committee_sizes': [5, 10, 15]
    }
    np.random.seed(SEED)  # Reset seed for reproducibility
    run_experiment("1,7,9", lda_dims=2, active_params=exp2_params, legend_labels=('Random sampling', 'QBC'))
    
    # Experiment 3: All digits; LDA with 8 components; different active learning parameters
    exp3_params = {
        'initial_samples': 10,
        'added_samples': 10,
        'num_iterations': 30,
        'committee_sizes': [5, 10, 15]
    }
    np.random.seed(SEED)  # Reset seed for reproducibility
    run_experiment("all", lda_dims=8, active_params=exp3_params, legend_labels=('Random sampling', 'QBC'))

if __name__ == '__main__':
    main()



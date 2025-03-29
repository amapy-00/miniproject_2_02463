from data import load_dataset, filter_dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import time

# Import dataset
X_full,y_full = load_dataset()
X,y = filter_dataset(X_full,y_full,"1,7,3,8")
X = X / 255
print(X.shape)
print(y.shape)

max(X_full[0]), max(X[0])

###########
y = np.array(y, dtype=int)
###########

## PCA
# Study effect of n_components on explained variance ratio for PCA
print("Start PCA")
nums = np.arange(50)
 
var_ratio = []
for num in nums:
  pca = PCA(n_components=num)
  X_pca = pca.fit_transform(X)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(6,4),dpi=150)
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')

## LDA
print("Start LDA")
nums = np.arange(1, 4)

var_ratio_lda = []
for num in nums:
    lda = LinearDiscriminantAnalysis(n_components=int(num))
    #Xs = StandardScaler().fit_transform(X)
    X_lda = lda.fit_transform(X, y)
    var_ratio_lda.append(np.sum(lda.explained_variance_ratio_))

plt.figure(figsize=(6,4),dpi=150)
plt.grid()
plt.plot(nums,var_ratio_lda,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')

# Define and plot best LDA 
# Scatter plot along a single axis (1D)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
var_ratio_lda.append(np.sum(lda.explained_variance_ratio_))

# Create a DataFrame for easy plotting
df_lda = pd.DataFrame(X_lda, columns=['LDA1', 'LDA2'])
df_lda['label'] = y  # Add labels

# Pairplot (2D projections of LDA components)
sns.pairplot(df_lda, hue='label', palette=['blue', 'red', 'yellow'])
plt.suptitle("Pairwise Scatter Plots of LDA Components", y=1.02)
plt.show()

"""# A 3D plot cannot be shown so good in the report
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for LDA components
scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], X_lda[:, 2], c=y, cmap='viridis', alpha=0.8)

# Labels and title
ax.set_xlabel("LDA1")
ax.set_ylabel("LDA2")
ax.set_zlabel("LDA3")
ax.set_title("3D Scatter Plot of LDA Features")

# Legend
handles, labels = scatter.legend_elements()
ax.legend(handles, np.unique(y), title="Class Labels")

plt.show()"""


from modAL.disagreement import vote_entropy_sampling
from modAL.models import ActiveLearner, Committee
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import itertools as it
from sklearn.model_selection import train_test_split
from collections import namedtuple
from tqdm import tqdm

ModelClass=RandomForestClassifier

SEED = 1 # Set our RNG seed for reproducibility.

n_queries = 75 # You can lower this to decrease run time

# You can increase this to get error bars on your evaluation.
# You probably need to use the parallel code to make this reasonable to compute
n_repeats = 1

ResultsRecord = namedtuple('ResultsRecord', ['estimator', 'query_id', 'score'])

###########
y = np.array(y, dtype=int)  # Ensure labels are integers before splitting
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=1/3, random_state=1)
##########

# in case repetitions are desired
permutations=[np.random.permutation(X_train.shape[0]) for _ in range(n_repeats)]

# Different committee sizes
n_members=[2]

def train_committee(i_repeat, i_members, X_train, y_train):
    y_train = np.array(y_train, dtype=int) 
    committee_results = []
    print('')

    X_pool = X_train.copy()
    y_pool = y_train.copy()

    start_indices = permutations[i_repeat][:1]

    committee_members = [ActiveLearner(estimator=ModelClass(max_depth=10),
                                       X_training=X_train[start_indices, :],
                                       y_training=y_train[start_indices],
                                       ) for _ in range(i_members)]

    committee = Committee(learner_list=committee_members,
                          query_strategy=vote_entropy_sampling)

    X_pool = np.delete(X_pool, start_indices, axis=0)
    y_pool = np.delete(y_pool, start_indices)

    for i_query in tqdm(range(1, n_queries), desc=f'Round {i_repeat} with {i_members} members', leave=False):
        query_idx, query_instance = committee.query(X_pool)

        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
        committee._set_classes()

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

        score = committee.score(X_test, y_test)

        committee_results.append(ResultsRecord(
            f'committe_{i_members}',
            i_query,
            score))

    return committee_results

print("Training running...")
start_time = time.time()
result = Parallel(n_jobs=-1)(delayed(train_committee)(i,i_members,X_train,y_train)
                    for i, i_members in it.product(range(n_repeats), n_members))
end_time = time.time()
print(f'Time elapsed: {end_time - start_time:.2f}')
print('All jobs done')
committee_results=[r for rs in result for r in rs]

# Convert results to DataFrame
df_results = pd.DataFrame(committee_results, columns=['estimator', 'query_id', 'score'])

# Plot the active learning performance for different committee sizes
plt.figure(figsize=(10, 6))
for members in n_members:
    subset = df_results[df_results['estimator'] == f'committe_{members}']
    if not subset.empty:
        plt.plot(subset['query_id'], subset['score'], marker='o', linestyle='-', label=f'{members} Members')

plt.xlabel("Number of Queries")
plt.ylabel("Accuracy")
plt.title("Active Learning Performance for Different Committee Sizes")
plt.legend()
plt.grid(True)
plt.show()

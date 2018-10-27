import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from assignment2 import \
    read_dataset, \
    iteration_count, \
    learning_rate, \
    purity_score, \
    get_purity_matrix


# read the dataset.
dataset_df = read_dataset()
X = dataset_df.iloc[:, 0:4]
y = dataset_df.iloc[:, 4]
# normalize the featureset.
f_sc = StandardScaler()
X = f_sc.fit_transform(X)

# randomly initialize the weights.
synaptic_weights = np.random.random((3, len(X[0])))
# normalize the weights.
w_sc = StandardScaler()
synaptic_weights = w_sc.fit_transform(synaptic_weights)

print("Initial Weights:")
print(synaptic_weights)


def get_nueron_index(weight_matrix, x):
    """
    Look for the maximum dot product and corresponding neuron.
    """
    min_distance = np.Inf
    n_idx = None
    for k in range(len(weight_matrix)):
        w = weight_matrix[k]
        w_x_distance = np.abs(np.sum(w - x))
        if w_x_distance < min_distance:
            min_distance = w_x_distance
            n_idx = k
    return n_idx


# training the weights of the clustering network.
for i in range(iteration_count):
    for feature_vec in X:
        neuron_index = get_nueron_index(synaptic_weights, feature_vec)
        if neuron_index is None:
            continue
        # update the weights accordingly.
        synaptic_weights[neuron_index] += learning_rate * (feature_vec - synaptic_weights[neuron_index])

print("Optimized Weights:")
print(synaptic_weights)

# predicting the clusters with the trained network.
y_pred = []
for x_idx in range(len(X)):
    feature_vec = X[x_idx]
    neuron_index = get_nueron_index(synaptic_weights, feature_vec)
    if neuron_index is None:
        continue
    y_pred.append(neuron_index + 1)
y_pred = pd.Series(y_pred)

purity_matrix = get_purity_matrix(y, y_pred)
print("Predicted Purity Distribution:")
print(purity_matrix)

purity = purity_score(purity_matrix)
print("Purity Score:")
print(purity)

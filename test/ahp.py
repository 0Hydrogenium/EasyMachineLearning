import numpy as np

def ahp(matrix):
    n = len(matrix)
    # Step 1: Normalize the matrix
    normalized_matrix = matrix / matrix.sum(axis=0)

    # Step 2: Calculate the weighted sum for each criterion
    weights = normalized_matrix.mean(axis=1)

    # Step 3: Normalize the weights
    normalized_weights = weights / weights.sum()

    return normalized_weights

def build_new_indicator(original_data, weights):
    # Combine original data with weights to construct new indicator
    new_indicator = np.dot(original_data, weights)
    return new_indicator

# Example data
original_data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Example pairwise comparison matrix
pairwise_matrix = np.array([
    [1, 1/4, 1/7],
    [4, 1, 1/3],
    [7, 3, 1]
])

# Apply AHP to get weights
weights = ahp(pairwise_matrix)

# Build new indicator
new_indicator = build_new_indicator(original_data, weights)

print("Original Data:")
print(original_data)
print("\nPairwise Comparison Matrix:")
print(pairwise_matrix)
print("\nWeights:")
print(weights)
print("\nNew Indicator:")
print(new_indicator)

import numpy as np


def consistency_ratio(pairwise_matrix):
    n = len(pairwise_matrix)

    # Calculate the weight vector
    weights = pairwise_matrix.mean(axis=1)

    # Calculate the consistency index (CI)
    max_eigenvalue = np.linalg.eigvals(pairwise_matrix).max()
    ci = (max_eigenvalue - n) / (n - 1)

    # Lookup random index (RI) from literature (该设定是一种公认的标准)
    random_indices = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = random_indices[n]

    # Calculate consistency ratio (CR)
    cr = ci / ri

    return cr


# Check consistency ratio
cr = consistency_ratio(pairwise_matrix)

print("Pairwise Comparison Matrix:")
print(pairwise_matrix)
print("\nConsistency Ratio:", cr)

# You can compare the CR value with a threshold (commonly 0.1) to determine if the matrix is consistent.


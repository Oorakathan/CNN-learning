import numpy as np

m = np.array([[6, 6, 5, 2], [5, 8, 4, 8], [9, 5, 2, 7]])

print("Matrix is:\n", m)

print("\nTranspose is:\n", m.T)

cov_matrix = np.cov(m.T)
print("\nCovariance matrix:\n", cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors (columns):\n", eigenvectors)


sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

print("\nSorted Eigenvalues:\n", sorted_eigenvalues)
print("\nSorted Eigenvectors (columns corresponding to sorted eigenvalues):\n", sorted_eigenvectors)


mean_centered_data = m - np.mean(m, axis=0)

projected_data_all = np.dot(mean_centered_data, sorted_eigenvectors)

print("\nProjected data (using all principal components):\n", projected_data_all)

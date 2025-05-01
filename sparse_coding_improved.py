import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import MiniBatchDictionaryLearning
import time
import torch
from torchvision import datasets, transforms
from scipy.optimize import linear_sum_assignment

# Parameters
n_synthetic = 200  
m = 50  
k_synthetic = 3 
n_mnist = 784 
p_mnist = 200  
m_mnist = 1000 
k_mnist = 20 
num_vectors = 50  
num_vectors_mnist = 200 
num_attempts = 500 
num_iterations = 500  


# Synthetic Data Generation
def generate_synthetic_data(n, m, k):
    A_true = np.random.randn(m, m)
    A_true, _ = np.linalg.qr(A_true)  
    X = np.zeros((m, n))
    
    for i in range(n):
        idx = np.random.choice(m, k, replace=False)
        X[idx, i] = np.random.randn(k) * 5
        
    noise = 0.001 * np.random.randn(m, n)
    Y = A_true @ X + noise
    return Y, A_true, X


# MNIST Data
def load_mnist_data(m_samples):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    images = dataset.data.numpy()[:m_samples].reshape(m_samples, -1) / 255.0  
    return images.T  


# Hard Thresholding
def hard_threshold(x, k):
    idx = np.abs(x).argsort()[-k:][::-1] 
    x_out = np.zeros_like(x)
    x_out[idx] = x[idx]
    return x_out


# Algorithm 3: Pairwise Initialization
def initialize_dictionary(Y, num_vectors, num_attempts):
    m, n = Y.shape
    C = Y.T @ Y / m  
    A0 = np.zeros((m, num_vectors))
    selected = set()

    for j in range(num_vectors):
        max_score = -np.inf
        best_v = None
        for _ in range(num_attempts):
            v = np.random.randn(n) 
            v = v / np.linalg.norm(v)
            score = v.T @ C @ v
            if score > max_score and tuple(v) not in selected:
                max_score = score
                best_v = v
        if best_v is not None:
            atom = Y @ best_v / np.linalg.norm(Y @ best_v)
            A0[:, j] = atom
            selected.add(tuple(best_v))

    A0, _ = np.linalg.qr(A0)  
    return A0


# Closeness
def compute_closeness(A0, A_true):
    U1, _, V1 = np.linalg.svd(A0, full_matrices=False)
    U2, _, V2 = np.linalg.svd(A_true, full_matrices=False)
    closeness = np.linalg.norm(U1.T @ U2 - np.eye(U1.shape[1]), ord='fro')
    return closeness


# Algorithm 2: Neural Update
def algorithm_2(Y, A0, k, num_iterations):
    A = A0.copy()
    errors = []
    AYt = A.T @ Y
    
    for t in range(1, num_iterations + 1):
        eta = 1.0 / t
        X = np.apply_along_axis(lambda x: hard_threshold(x, k), 0, AYt)
        grad = (Y - A @ X) @ X.T / Y.shape[1]
        A = A + eta * grad
        A, _ = np.linalg.qr(A)
        AYt = A.T @ Y
        error = np.linalg.norm(Y - A @ X, ord='fro') / Y.shape[1]
        errors.append(error)
        
    return A, X, errors


# Algorithm 5: Low-Error Variant
def algorithm_5(Y, A0, k, num_iterations):
    A = A0.copy()
    errors = []
    prev_grad = np.zeros_like(A)
    AYt = A.T @ Y
    
    for t in range(1, num_iterations + 1):
        eta = 0.0005
        X = np.apply_along_axis(lambda x: hard_threshold(x, k), 0, AYt)
        grad = (Y - A @ X) @ X.T / Y.shape[1]
        grad = 0.9 * grad + 0.1 * prev_grad 
        A = A + eta * grad
        A, _ = np.linalg.qr(A)
        AYt = A.T @ Y
        error = np.linalg.norm(Y - A @ X, ord='fro') / Y.shape[1]
        errors.append(error)
        prev_grad = grad
        if error < 0.5 / Y.shape[0]: 
            break
            
    return A, X, errors


# K-SVD
def run_ksvd(Y, num_atoms, max_iter=10):  
    ksvd = MiniBatchDictionaryLearning(n_components=num_atoms, max_iter=max_iter, alpha=0.5)
    start_time = time.time()
    ksvd.fit(Y.T)
    runtime = time.time() - start_time
    A = ksvd.components_.T
    X = ksvd.transform(Y.T).T
    error = np.linalg.norm(Y - A @ X, ord='fro') / Y.shape[1]
    return A, X, error, runtime


# Plot Reconstruction Error
def plot_recon_error(errors2, errors5, dataset_name, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(errors2, label='Algorithm 2', color='b')
    plt.plot(errors5, label='Algorithm 5', color='r')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    plt.title(f'Reconstruction Error vs. Iterations ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# Dictionary Patches
def plot_dict_patches(A, title, filename, patch_size=(28, 28)):
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        if i < min(A.shape[1], 25):
            patch = A[:, i].reshape(patch_size)
            ax.imshow(patch, cmap='gray')
        ax.axis('off')
    plt.suptitle(title, y=1.05)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Compute Column-Wise Error
def compute_column_error(A, A_ref):
    n, p = A.shape
    p_ref = A_ref.shape[1]
    p_min = min(p, p_ref)
    A = A[:, :p_min]
    A_ref = A_ref[:, :p_min]
    
    distances = np.zeros((p_min, p_min))
    for i in range(p_min):
        for j in range(p_min):
            distances[i, j] = np.linalg.norm(A[:, i] - A_ref[:, j])
            
    row_ind, col_ind = linear_sum_assignment(distances)
    max_error = np.max(distances[row_ind, col_ind])
    return max_error

# Main Execution
def main():
    # Synthetic Data
    print("Processing Synthetic Data...")
    Y_synthetic, A_true, X_true = generate_synthetic_data(n_synthetic, m, k_synthetic)
    start_time = time.time()
    A0_synthetic = initialize_dictionary(Y_synthetic, num_vectors, num_attempts)
    runtime3_synthetic = time.time() - start_time
    closeness_synthetic = compute_closeness(A0_synthetic, A_true)
    X0_synthetic = np.apply_along_axis(lambda x: hard_threshold(x, k_synthetic), 0, A0_synthetic.T @ Y_synthetic)
    error3_synthetic = np.linalg.norm(Y_synthetic - A0_synthetic @ X0_synthetic, ord='fro') / Y_synthetic.shape[1]
    col_error3_synthetic = compute_column_error(A0_synthetic, A_true)
    print(f"Algorithm 3 - Closeness: {closeness_synthetic:.4f}, Expected: {1/np.log(n_synthetic):.4f}, Error: {error3_synthetic:.4f}, Column Error: {col_error3_synthetic:.4f}, Runtime: {runtime3_synthetic:.4f}s")

    start_time = time.time()
    A2_synthetic, X2_synthetic, errors2_synthetic = algorithm_2(Y_synthetic, A0_synthetic, k_synthetic, num_iterations)
    runtime2_synthetic = time.time() - start_time
    error2_synthetic = errors2_synthetic[-1]
    col_error2_synthetic = compute_column_error(A2_synthetic, A_true)
    print(f"Algorithm 2 - Error: {error2_synthetic:.4f}, Expected: {np.sqrt(k_synthetic/n_synthetic):.4f}, Column Error: {col_error2_synthetic:.4f}, Runtime: {runtime2_synthetic:.4f}s")

    start_time = time.time()
    A5_synthetic, X5_synthetic, errors5_synthetic = algorithm_5(Y_synthetic, A0_synthetic, k_synthetic, num_iterations)
    runtime5_synthetic = time.time() - start_time
    error5_synthetic = errors5_synthetic[-1]
    col_error5_synthetic = compute_column_error(A5_synthetic, A_true)
    print(f"Algorithm 5 - Error: {error5_synthetic:.4f}, Expected: {1/n_synthetic:.4f}, Column Error: {col_error5_synthetic:.4f}, Runtime: {runtime5_synthetic:.4f}s")

    A_ksvd_synthetic, X_ksvd_synthetic, error_ksvd_synthetic, runtime_ksvd_synthetic = run_ksvd(Y_synthetic, m)
    col_error_ksvd_synthetic = compute_column_error(A_ksvd_synthetic, A_true)
    print(f"K-SVD - Error: {error_ksvd_synthetic:.4f}, Column Error: {col_error_ksvd_synthetic:.4f}, Runtime: {runtime_ksvd_synthetic:.4f}s")

    sample_complexity = n_synthetic
    expected_samples = m * k_synthetic
    print(f"Sample Complexity: {sample_complexity}, Expected: {expected_samples}")
    plot_recon_error(errors2_synthetic, errors5_synthetic, 'Synthetic Data', 'synthetic_error.png')

    # MNIST Data
    print("\nProcessing MNIST Data...")
    Y_mnist = load_mnist_data(m_mnist)
    start_time = time.time()
    A0_mnist = initialize_dictionary(Y_mnist, num_vectors_mnist, num_attempts)
    runtime3_mnist = time.time() - start_time
    X0_mnist = np.apply_along_axis(lambda x: hard_threshold(x, k_mnist), 0, A0_mnist.T @ Y_mnist)
    error3_mnist = np.linalg.norm(Y_mnist - A0_mnist @ X0_mnist, ord='fro') / Y_mnist.shape[1]

    A_ksvd_mnist, X_ksvd_mnist, error_ksvd_mnist, runtime_ksvd_mnist = run_ksvd(Y_mnist, num_vectors_mnist)
    closeness_mnist = compute_closeness(A0_mnist, A_ksvd_mnist)
    col_error3_mnist = compute_column_error(A0_mnist, A_ksvd_mnist)
    print(f"Algorithm 3 - Closeness: {closeness_mnist:.4f}, Error: {error3_mnist:.4f}, Column Error: {col_error3_mnist:.4f}, Runtime: {runtime3_mnist:.4f}s")

    start_time = time.time()
    A2_mnist, X2_mnist, errors2_mnist = algorithm_2(Y_mnist, A0_mnist, k_mnist, num_iterations)
    runtime2_mnist = time.time() - start_time
    error2_mnist = errors2_mnist[-1]
    col_error2_mnist = compute_column_error(A2_mnist, A_ksvd_mnist)
    print(f"Algorithm 2 - Error: {error2_mnist:.4f}, Expected: {np.sqrt(k_mnist/n_mnist):.4f}, Column Error: {col_error2_mnist:.4f}, Runtime: {runtime2_mnist:.4f}s")

    start_time = time.time()
    A5_mnist, X5_mnist, errors5_mnist = algorithm_5(Y_mnist, A0_mnist, k_mnist, num_iterations)
    runtime5_mnist = time.time() - start_time
    error5_mnist = errors5_mnist[-1]
    col_error5_mnist = compute_column_error(A5_mnist, A_ksvd_mnist)
    print(f"Algorithm 5 - Error: {error5_mnist:.4f}, Expected: {1/n_mnist:.4f}, Column Error: {col_error5_mnist:.4f}, Runtime: {runtime5_mnist:.4f}s")

    col_error_ksvd_mnist = compute_column_error(A_ksvd_mnist, A_ksvd_mnist)  # Should be near 0
    print(f"K-SVD - Error: {error_ksvd_mnist:.4f}, Column Error: {col_error_ksvd_mnist:.4f}, Runtime: {runtime_ksvd_mnist:.4f}s")

    plot_recon_error(errors2_mnist, errors5_mnist, 'MNIST Data', 'mnist_error.png')
    plot_dict_patches(A2_mnist, 'Algorithm 2', 'dict_algo2.png')
    plot_dict_patches(A5_mnist, 'Algorithm 5', 'dict_algo5.png')
    plot_dict_patches(A_ksvd_mnist, 'K-SVD', 'dict_ksvd.png')


if __name__ == "__main__":
    main()

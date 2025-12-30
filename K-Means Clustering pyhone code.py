import random
import math

def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

def mean_point(points, dim):
    if not points:
        return [0.0] * dim
    return [sum(p[i] for p in points) / len(points) for i in range(dim)]

def kmeans(data, k=2, max_iter=100, tol=1e-6, seed=None):
    if seed is not None:
        random.seed(seed)

    if not data:
        raise ValueError("data must be non-empty")
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k > len(data):
        raise ValueError("k cannot be greater than number of data points")

    dim = len(data[0])

    # Randomly choose initial centroids
    centroids = random.sample(data, k)

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]

        # Assign each point to nearest centroid
        for p in data:
            distances = [euclidean(p, c) for c in centroids]
            idx = distances.index(min(distances))
            clusters[idx].append(p)

        # Update centroids
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                new_centroids.append(mean_point(clusters[i], dim))
            else:
                # if any cluster becomes empty, reinitialize
                new_centroids.append(random.choice(data))

        # Check convergence
        if all(euclidean(centroids[i], new_centroids[i]) < tol for i in range(k)):
            centroids = new_centroids
            break

        centroids = new_centroids

    return clusters, centroids

# Example usage
if __name__ == "__main__":
    data = [
        [1, 2], [1.5, 1.8], [5, 8], [8, 8],
        [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]
    ]
    clusters, centroids = kmeans(data, k=2, seed=1)

    print("Final Centroids:")
    for i, c in enumerate(centroids, 1):
        print(f"  Cluster {i}: {c}")

    print("\nClusters:")
    for i, cl in enumerate(clusters, 1):
        print(f"  Cluster {i} points: {cl}")

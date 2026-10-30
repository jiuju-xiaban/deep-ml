import torch

def k_means_clustering(points, k, initial_centroids, max_iterations) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters.
    points: tensor of shape (n_points, n_features)
    initial_centroids: tensor of shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    # Convert to tensors
    points_t = torch.as_tensor(points, dtype=torch.float)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float)
    # Your implementation here
    for _ in range(max_iterations):
        diffs=points_t.unsqueeze(0)-centroids.unsqueeze(1)
        dist_sq=(diffs**2).sum(dim=2)
        assignments=dist_sq.argmin(dim=0)
        old_centroids=centroids.clone()
        for i in range(k):
            cluster_points=points_t[assignments==i] #`points_t[布尔数组]` → 高级索引，**取出对应行的坐标**
            if len(cluster_points)>0:
                centroids[i]=cluster_points.mean(dim=0)
        if torch.allclose(centroids,old_centroids):
            break
    centroids=torch.round(centroids*10000)/10000
    result=[tuple(row.tolist()) for row in centroids]
    return result

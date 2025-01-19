import numpy as np

def kmeans_manual(data, k, max_iterations, return_iterations=False, initial_centroids=None):
    """
    Algoritma K-Means Manual untuk data multi-dimensi.

    Args:
        data (numpy.ndarray): Data yang akan dikelompokkan, dengan dimensi (n_samples, n_features).
        k (int): Jumlah cluster.
        max_iterations (int): Iterasi maksimum.
        return_iterations (bool): Jika True, kembalikan detail per iterasi.
        initial_centroids (numpy.ndarray or None): Centroid awal yang akan digunakan. Jika None, centroid dipilih secara acak.

    Returns:
        tuple: (clusters, centroids, sse, all_iterations)
            clusters (list): Daftar cluster untuk setiap data.
            centroids (numpy.ndarray): Centroid untuk setiap cluster.
            sse (float): Jumlah kuadrat kesalahan untuk seluruh cluster.
            all_iterations (list): Detail per iterasi, jika return_iterations=True.
    """
    # Konversi data ke numpy array jika belum
    data = np.array(data)

    # Inisialisasi centroid
    if initial_centroids is not None:
        centroids = np.array(initial_centroids)  # Gunakan centroid awal yang diberikan
    else:
        np.random.seed(42)  # Seed untuk hasil yang konsisten
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]  # Pilih centroid secara acak

    clusters = np.zeros(data.shape[0], dtype=int)  # Inisialisasi cluster awal
    all_iterations = []  # Simpan hasil iterasi

    for iteration in range(max_iterations):
        # Hitung jarak Euclidean antara setiap titik data dan setiap centroid
        distances = []
        for point in data:
            distance_to_centroids = [np.linalg.norm(point - centroid) for centroid in centroids]
            distances.append(distance_to_centroids)
        distances = np.array(distances)

        # Tentukan cluster baru berdasarkan centroid terdekat
        new_clusters = np.argmin(distances, axis=1)

        # Simpan detail iterasi jika diminta
        if return_iterations:
            all_iterations.append({
                "centroids": centroids.copy(),
                "euclidean": distances.copy(),
                "clusters": new_clusters.copy()
            })

        # Hentikan iterasi jika cluster tidak berubah
        if np.array_equal(new_clusters, clusters):
            break
        clusters = new_clusters

        # Perbarui centroid sebagai rata-rata titik dalam cluster
        for i in range(k):
            points_in_cluster = data[clusters == i]
            if len(points_in_cluster) > 0:
                centroids[i] = points_in_cluster.mean(axis=0)

    # Hitung SSE (Sum of Squared Errors)
    sse = 0
    for i in range(k):
        points_in_cluster = data[clusters == i]
        if len(points_in_cluster) > 0:
            sse += np.sum((points_in_cluster - centroids[i]) ** 2)

    if return_iterations:
        return clusters, centroids, sse, all_iterations
    else:
        return clusters, centroids, sse

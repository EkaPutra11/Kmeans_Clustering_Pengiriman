import datetime
import numpy as np

# def calculate_gap_hours(booking_date, pod_date):
#     """
#     Menghitung selisih waktu dalam jam antara booking_date dan pod_date.

#     Args:
#         booking_date (str/datetime): Tanggal booking (format string atau datetime.datetime).
#         pod_date (str/datetime): Tanggal POD (format string atau datetime.datetime).

#     Returns:
#         float: Selisih waktu dalam jam.
#     """
#     # Format datetime untuk parsing string
#     format = "%Y-%m-%d %H:%M:%S"

#     # Pastikan booking_date adalah objek datetime
#     if isinstance(booking_date, str):
#         booking_date = datetime.datetime.strptime(booking_date.split('+')[0].strip(), format)
    
#     # Pastikan pod_date adalah objek datetime
#     if isinstance(pod_date, str):
#         pod_date = datetime.datetime.strptime(pod_date.split('+')[0].strip(), format)

#     # Hitung selisih waktu dalam jam
#     gap = (pod_date - booking_date).total_seconds() / 3600
#     return gap


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
    import numpy as np

    # Konversi data ke numpy array jika belum
    data = np.array(data)

    # Inisialisasi centroid
    if initial_centroids is not None:
        # Gunakan centroid yang diberikan
        centroids = np.array(initial_centroids)
    else:
        # Pilih centroid secara acak dari data
        np.random.seed(42)  # Untuk memastikan hasil dapat direproduksi
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    clusters = np.zeros(data.shape[0], dtype=int)
    all_iterations = []  # Untuk menyimpan hasil setiap iterasi

    for iteration in range(max_iterations):
        # Hitung Euclidean distance dan tentukan cluster
        distances = np.array([[np.linalg.norm(point - centroid) for centroid in centroids] for point in data])
        new_clusters = np.argmin(distances, axis=1)

        # Simpan detail per iterasi jika diminta
        if return_iterations:
            all_iterations.append({
                "centroids": centroids.copy(),
                "euclidean": distances.copy(),
                "clusters": new_clusters.copy()
            })

        # Pengecekan konvergensi
        if np.array_equal(new_clusters, clusters):
            break
        clusters = new_clusters

        # Perbarui centroid sebagai rata-rata dari semua titik dalam cluster
        for i in range(k):
            points_in_cluster = data[clusters == i]
            if len(points_in_cluster) > 0:
                centroids[i] = points_in_cluster.mean(axis=0)

    # Hitung Sum of Squared Errors (SSE)
    sse = sum(
        np.linalg.norm(data[clusters == i] - centroids[i]) ** 2
        for i in range(k)
    )

    if return_iterations:
        return clusters, centroids, sse, all_iterations
    else:
        return clusters, centroids, sse

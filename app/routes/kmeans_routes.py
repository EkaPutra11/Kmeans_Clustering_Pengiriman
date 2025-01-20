from flask import Blueprint, request, flash, redirect, url_for, render_template,current_app
import pymysql
import csv
import io
from app.kmeans_utils import  kmeans_manual
import os
from datetime import datetime
import logging
logging.getLogger('tkinter').disabled = True
import threading

# Buat Tkinter hanya di main thread
def start_tkinter_in_main_thread():
    from tkinter import Tk
    root = Tk()
    root.mainloop()

threading.Thread(target=start_tkinter_in_main_thread).start()


kmeans_bp = Blueprint('kmeans_bp', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Validasi apakah file memiliki ekstensi CSV."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    """Fungsi utilitas untuk mendapatkan koneksi database."""
    return pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='kmeans_db',
        cursorclass=pymysql.cursors.DictCursor
    )

@kmeans_bp.route('/import-dataset', methods=['GET', 'POST'])
def import_dataset():

    if request.method == 'GET':
        # Ambil parameter pencarian dari query string
        search_query = request.args.get('search', '').strip()
        search_type = request.args.get('search_type', 'destination')
        vendor = request.args.get('vendor', '')
        page = request.args.get('page', 1, type=int)
        per_page = request.args .get('per_page', 10, type=int)

        # Koneksi ke database
        connection = get_db_connection()
        cursor = connection.cursor()

        # Query untuk data yang difilter
        query = "SELECT * FROM pengiriman WHERE 1=1"
        params = []

        if search_query:
            if search_type == 'destination':
                query += " AND destination_city LIKE %s"
                params.append(f"%{search_query}%")
            elif search_type == 'stt':
                query += " AND stt_number_genesis = %s"
                params.append(search_query)

        if vendor:
            query += " AND vendor = %s"
            params.append(vendor)

        query += " LIMIT %s OFFSET %s"
        params.extend([per_page, (page - 1) * per_page])

        cursor.execute(query, tuple(params))
        data = cursor.fetchall()

        # Query untuk menghitung total data
        query_count = "SELECT COUNT(*) AS total FROM pengiriman WHERE 1=1"
        params_count = params[:-2]  # Hilangkan LIMIT dan OFFSET dari parameter
        cursor.execute(query_count, tuple(params_count))
        total_count = cursor.fetchone()['total']

        # Hitung total halaman
        total_pages = (total_count + per_page - 1) // per_page

        # Query untuk mendapatkan daftar vendor
        cursor.execute("SELECT DISTINCT vendor FROM pengiriman")
        vendors = [row['vendor'] for row in cursor.fetchall()]

        cursor.close()
        connection.close()

        return render_template(
            'import_data.html',
            data=data,
            page=page,
            per_page=per_page,
            total_count=total_count,
            total_pages=total_pages,
            search_query=search_query,
            search_type=search_type,
            vendor=vendor,
            vendors=vendors
        )

    # POST untuk mengunggah dataset CSV
    try:
        if 'datasetFile' not in request.files:
            flash('File tidak ditemukan!', 'danger')
            return redirect(url_for('kmeans_bp.import_dataset'))

        file = request.files['datasetFile']

        if not allowed_file(file.filename):
            flash('File harus dalam format CSV!', 'danger')
            return redirect(url_for('kmeans_bp.import_dataset'))

        # Baca file CSV menggunakan stream dan hapus BOM
        stream = io.StringIO(file.stream.read().decode("UTF8").lstrip('\ufeff'), newline=None)
        reader = csv.DictReader(stream)

        # Validasi header CSV
        if not reader.fieldnames or len(reader.fieldnames) == 0:
            flash('Header CSV tidak valid atau file kosong!', 'danger')
            return redirect(url_for('kmeans_bp.import_dataset'))

        # Pemetaan kolom dari CSV ke database
        column_mapping = {
            'STT Number Genesis': 'stt_number_genesis',
            'Booking Date': 'booking_date',
            'Origin City': 'origin_city',
            'Destination City': 'destination_city',
            'Vendor': 'vendor',
            'POD at': 'pod_at'
        }

        required_columns = set(column_mapping.keys())
        csv_columns = set(reader.fieldnames)
        missing_columns = required_columns - csv_columns
        if missing_columns:
            flash(f"Kolom berikut tidak ditemukan di file CSV: {', '.join(missing_columns)}", 'danger')
            return redirect(url_for('kmeans_bp.import_dataset'))

        # Koneksi ke database
        connection = get_db_connection()
        cursor = connection.cursor()

        for row in reader:
            db_row = {column_mapping[col]: row[col] for col in column_mapping}

            # Hitung gap_hours
            try:
                booking_date = datetime.strptime(db_row['booking_date'].split('+')[0].strip(), '%Y-%m-%d %H:%M:%S')
                pod_at = datetime.strptime(db_row['pod_at'].split('+')[0].strip(), '%Y-%m-%d %H:%M:%S')
                gap_hours = (pod_at - booking_date).total_seconds() / 3600  # Hitung selisih dalam jam
            except Exception as e:
                flash(f"Error parsing dates: {e}", 'danger')
                continue

            # Tambahkan gap_hours ke query SQL
            query = """
                INSERT INTO pengiriman (stt_number_genesis, booking_date, origin_city, destination_city, vendor, pod_at, gap_hours)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                db_row['stt_number_genesis'],
                db_row['booking_date'],
                db_row['origin_city'],
                db_row['destination_city'],
                db_row['vendor'],
                db_row['pod_at'],
                gap_hours
            ))

        connection.commit()
        flash('Dataset berhasil diimpor ke database dengan gap_hours dihitung!', 'success')

    except Exception as e:
        flash(f'Error importing dataset: {e}', 'danger')

    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection:
            connection.close()

    return redirect(url_for('kmeans_bp.import_dataset'))




# ==============================================
# ROUTE: Delete All Data
# ==============================================
@kmeans_bp.route('/delete-all-data', methods=['POST'])
def delete_all_data():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM pengiriman")
        connection.commit()
        flash("Semua data berhasil dihapus.", "success")
    except Exception as e:
        flash(f"Terjadi kesalahan saat menghapus data: {e}", "danger")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection:
            connection.close()
    # Redirect ke halaman `pengiriman`
    return redirect(url_for('kmeans_bp.pengiriman'))



# ==============================================
# ROUTE: View Pengiriman
# ==============================================
@kmeans_bp.route('/pengiriman', methods=['GET'])
def pengiriman():
    search_query = request.args.get('search', '').strip()
    search_type = request.args.get('search_type', 'destination')
    vendor = request.args.get('vendor', '')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    connection = get_db_connection()
    cursor = connection.cursor()

    query = "SELECT * FROM pengiriman WHERE 1=1"
    params = []

    if search_query:
        if search_type == 'destination':
            query += " AND destination_city LIKE %s"
            params.append(f"%{search_query}%")
        elif search_type == 'stt':
            query += " AND stt_number_genesis = %s"
            params.append(search_query)

    if vendor:
        query += " AND vendor = %s"
        params.append(vendor)

    query += " LIMIT %s OFFSET %s"
    params.extend([per_page, (page - 1) * per_page])

    cursor.execute(query, tuple(params))
    data = cursor.fetchall()

    query_count = "SELECT COUNT(*) AS total FROM pengiriman WHERE 1=1"
    params_count = []

    if search_query:
        if search_type == 'destination':
            query_count += " AND destination_city LIKE %s"
            params_count.append(f"%{search_query}%")
        elif search_type == 'stt':
            query_count += " AND stt_number_genesis = %s"
            params_count.append(search_query)

    if vendor:
        query_count += " AND vendor = %s"
        params_count.append(vendor)

    cursor.execute(query_count, tuple(params_count))
    total_count = cursor.fetchone()['total']
    total_pages = (total_count + per_page - 1) // per_page

    cursor.execute("SELECT DISTINCT vendor FROM pengiriman")
    vendors = [row['vendor'] for row in cursor.fetchall()]

    cursor.close()
    connection.close()

    return render_template(
        'import_data.html',
        data=data,
        page=page,
        per_page=per_page,
        total_count=total_count,
        total_pages=total_pages,
        search_query=search_query,
        search_type=search_type,
        vendor=vendor,
        vendors=vendors
    )


# ==============================================
# ROUTE: Generate Elbow
# ==============================================
@kmeans_bp.route('/elbow', methods=['GET', 'POST'])
def elbow_optimization():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import base64

    connection = get_db_connection()
    cursor = connection.cursor()

    # Fetch data from the pengiriman table
    cursor.execute("SELECT * FROM pengiriman")
    data = cursor.fetchall()
    cursor.close()
    connection.close()

    if not data:
        flash("Tidak ada data dalam tabel pengiriman!", "danger")
        return redirect(url_for("kmeans_bp.pengiriman"))

    # Convert data to Pandas DataFrame
    df = pd.DataFrame(data)

    # Pastikan ada kolom "gap_hours"
    if "gap_hours" not in df.columns:
        flash("Kolom gap_hours tidak ditemukan!", "danger")
        return redirect(url_for("kmeans_bp.pengiriman"))

    # Ambil data gap_hours dari semua kota
    features = df["gap_hours"].values

    graph_data = None
    results = []

    if request.method == "POST":
        try:
            # Get user inputs
            cluster_start = int(request.form.get("cluster_start", 1))
            cluster_end = int(request.form.get("cluster_end", 10))
            max_iter = int(request.form.get("max_iter", 300))

            if cluster_start < 1 or cluster_end < cluster_start:
                flash("Jumlah cluster tidak valid!", "danger")
                return redirect(url_for("kmeans_bp.elbow_optimization"))

            # Hitung SSE untuk semua data dalam tabel pengiriman
            sse = []
            for k in range(cluster_start, cluster_end + 1):
                clusters, centroids, sse_k = kmeans_manual(features, k, max_iter)
                sse.append(sse_k)
                results.append({"cluster": k, "sse": sse_k})

            # Generate the elbow graph
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(range(cluster_start, cluster_end + 1), sse, marker="o", linestyle="-")
            ax.set_title("Elbow Method for All Cities")
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("SSE (Sum of Squared Errors)")
            plt.grid()

            # Save the graph as a base64-encoded string
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            graph_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            plt.close()

        except Exception as e:
            flash(f"Terjadi kesalahan: {e}", "danger")
            print(f"Error: {e}")  # Debugging
            return redirect(url_for("kmeans_bp.elbow_optimization"))

    return render_template(
        "elbow_method.html",
        results=results,
        graph_data=graph_data,
        cluster_start=request.form.get("cluster_start", 1),
        cluster_end=request.form.get("cluster_end", 10),
        max_iter=request.form.get("max_iter", 300),
    )




# ==============================================
# ROUTE: Proses Clustering
# ==============================================

@kmeans_bp.route('/set-cluster', methods=['GET', 'POST'])
def set_cluster():
    import pandas as pd
    import numpy as np
    from app.kmeans_utils import kmeans_manual
    from flask import session, flash, redirect, url_for, request, render_template
    import os
    import json
    from flask import current_app

    if request.method == 'POST':
        try:
            # Ambil parameter dari form
            num_clusters = int(request.form.get('num_clusters'))
            max_iter = int(request.form.get('max_iter'))

            # Validasi input jumlah cluster
            if num_clusters < 2 or max_iter < 2:
                flash("Jumlah cluster dan iterasi harus lebih besar dari 1!", "danger")
                return redirect(url_for('kmeans_bp.set_cluster'))
            if num_clusters > 2 :  # Validasi jumlah cluster tidak boleh lebih dari 3
                flash("Jumlah cluster tidak boleh lebih dari 2!", "danger")
                return redirect(url_for('kmeans_bp.set_cluster'))

            # Ambil data dari tabel pengiriman
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT stt_number_genesis, gap_hours, destination_city, vendor FROM pengiriman")
            data = cursor.fetchall()
            cursor.close()
            connection.close()

            if not data:
                flash("Tidak ada data dalam tabel pengiriman!", "danger")
                return redirect(url_for('kmeans_bp.set_cluster'))

            # Konversi data ke Pandas DataFrame
            df = pd.DataFrame(data, columns=["stt_number_genesis", "gap_hours", "destination_city", "vendor"])

            # Pastikan kolom "gap_hours", "destination_city", dan "vendor" tersedia
            if not all(col in df.columns for col in ["stt_number_genesis", "gap_hours", "destination_city", "vendor"]):
                flash("Kolom stt_number_genesis, gap_hours, destination_city, atau vendor tidak ditemukan!", "danger")
                return redirect(url_for('kmeans_bp.set_cluster'))

            # Clustering per kota
            cluster_summary_per_city = {}
            iteration_results_per_city = {}
            cluster_averages = {}  # Untuk menyimpan rata-rata gap_hours tiap cluster

            total_sse = 0
            total_variance = 0

            for city in df['destination_city'].unique():
                city_data = df[df['destination_city'] == city]
                features = city_data['gap_hours'].values.astype(int)  # Pastikan tipe data numerik
                vendors = city_data['vendor'].values  # Ambil data vendor untuk kota ini
                stt_numbers = city_data['stt_number_genesis'].values  # Ambil data stt_number_genesis untuk kota ini

                if len(features) < num_clusters:
                    flash(f"Data untuk kota {city} tidak mencukupi untuk {num_clusters} cluster.", "warning")
                    continue

                # Inisialisasi centroid secara manual (nilai terkecil untuk C1 dan nilai terbesar untuk C2)
                centroids_init = np.linspace(features.min(), features.max(), num=num_clusters)

                # Proses clustering untuk kota ini
                clusters, centroids, sse, all_iterations = kmeans_manual(
                    features, num_clusters, max_iter, return_iterations=True, initial_centroids=centroids_init
                )

                # Hitung variansi total untuk kota ini
                variance = np.sum((features - np.mean(features)) ** 2)

                # Tambahkan SSE dan Variansi ke total
                total_sse += sse
                total_variance += variance

                # Simpan hasil clustering untuk kota ini
                cluster_summary_per_city[city] = {
                    "clusters": list(map(int, clusters)),  # Konversi ndarray ke list dengan int
                    "centroids": [int(c) for c in centroids],  # Konversi centroids ke float
                    "vendors": list(vendors),  # Simpan vendor untuk kota ini    
                    "gap_hours": list(map(int, features)),  # Tambahkan gap_hours
                    "stt_numbers": list(stt_numbers)  # Tambahkan stt_number_genesis
                }

                # Hitung rata-rata gap_hours per cluster
                averages = {}
                for cluster_label in range(num_clusters):
                    cluster_features = features[clusters == cluster_label]
                    averages[cluster_label] = np.mean(cluster_features) if len(cluster_features) > 0 else 0.0
                cluster_averages[city] = averages

                # Simpan hasil iterasi untuk kota ini
                iteration_results_per_city[city] = [
                    {
                        "iterasi": int(iter_no),
                        "gap_hours": [float(gap) for gap in features],
                        "vendors": list(vendors),  # Simpan vendor
                        "stt_numbers": list(stt_numbers),  # Simpan stt_number_genesis
                        "centroids": [int(c) for c in iteration_data["centroids"]],
                        "distances": [
                            [abs(int(feature - centroid)) for centroid in iteration_data["centroids"]]
                            for feature in features
                        ],
                        "clusters": list(map(int, iteration_data["clusters"]))
                    }
                    for iter_no, iteration_data in enumerate(all_iterations, start=1)
                ]

            # Hitung total cluster 1, 2, dan 3
            total_cluster_1 = sum(summary['clusters'].count(0) for summary in cluster_summary_per_city.values())
            total_cluster_2 = sum(summary['clusters'].count(1) for summary in cluster_summary_per_city.values())

            # Hitung akurasi total
            # total_accuracy = (1 - (total_sse / total_variance)) * 100 if total_variance > 0 else 0

            # Simpan hasil ke file sementara
            results_dir = os.path.join(current_app.root_path, 'results')
            os.makedirs(results_dir, exist_ok=True)

            file_path = os.path.join(results_dir, 'kmeans_results.json')
            with open(file_path, 'w') as f:
                results_data = {
                    "iteration_results_per_city": iteration_results_per_city,
                    "cluster_summary_per_city": cluster_summary_per_city,
                    # "total_accuracy": total_accuracy,
                    "num_clusters": num_clusters,  # Simpan num_clusters
                    "cluster_averages": cluster_averages  # Tambahkan rata-rata cluster
                }
                json.dump(results_data, f, indent=4)

            session['results_file'] = file_path

            # Redirect ke halaman yang sama untuk menampilkan hasil
            flash("Proses clustering selesai!", "success")
            return redirect(url_for('kmeans_bp.set_cluster'))

        except Exception as e:
            flash(f"Terjadi kesalahan: {e}", "danger")
            return redirect(url_for('kmeans_bp.set_cluster'))

    # Ambil hasil dari file jika ada
    results_file = session.get('results_file', None)
    iteration_results_per_city = {}
    cluster_summary_per_city = {}
    cluster_averages = {}
    # total_accuracy = 0
    total_cluster_1 = 0  # Inisialisasi variabel
    total_cluster_2 = 0  # Inisialisasi variabel
    num_clusters = 0  # Inisialisasi default

    if results_file and os.path.exists(results_file):
        with open(results_file, 'r') as f:
            if f.read().strip():
                f.seek(0)
                results = json.load(f)
                iteration_results_per_city = results.get('iteration_results_per_city', {})
                cluster_summary_per_city = results.get('cluster_summary_per_city', {})
                cluster_averages = results.get('cluster_averages', {})
                # total_accuracy = results.get('total_accuracy', 0)
                num_clusters = results.get('num_clusters', 0)  # Ambil num_clusters dari file
                total_cluster_1 = sum(summary['clusters'].count(0) for summary in cluster_summary_per_city.values())
                total_cluster_2 = sum(summary['clusters'].count(1) for summary in cluster_summary_per_city.values())

    return render_template(
        "set_cluster.html",
        iteration_results_per_city=iteration_results_per_city,
        cluster_summary_per_city=cluster_summary_per_city,
        cluster_averages=cluster_averages,  # Kirim rata-rata cluster ke template
        total_cluster_1=total_cluster_1,
        total_cluster_2=total_cluster_2,
        # total_accuracy=total_accuracy,
        enumerate=enumerate,
        zip=zip
    )



# ==============================================
# ROUTE: menampilkan hasil clustering
# ==============================================


@kmeans_bp.route('/kmeans-result', methods=['GET'])
def kmeans_result():
    try:
        # Ambil parameter pencarian dari query string
        search_query = request.args.get('search', '').strip()
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        # Koneksi ke database
        connection = get_db_connection()
        cursor = connection.cursor()

        # Query untuk join antara kmeans_result dan pengiriman
        query = """
            SELECT 
                kr.stt_number_genesis,
                kr.destination_city,
                kr.gap_hours,
                kr.vendor,
                kr.status,
                kr.cluster_label,
                p.booking_date,
                p.pod_at
            FROM kmeans_result kr
            LEFT JOIN pengiriman p ON kr.stt_number_genesis = p.stt_number_genesis
            WHERE 1=1
        """
        params = []

        # Filter berdasarkan pencarian
        if search_query:
            query += """
                AND (
                    kr.destination_city LIKE %s 
                    OR kr.vendor LIKE %s 
                    OR kr.stt_number_genesis LIKE %s
                )
            """
            params.extend([f"%{search_query}%", f"%{search_query}%", f"%{search_query}%"])

        # Hitung total data untuk paginasi
        query_count = f"SELECT COUNT(*) AS total FROM ({query}) AS subquery"
        cursor.execute(query_count, tuple(params))
        total_count = cursor.fetchone()['total']

        # Tambahkan LIMIT dan OFFSET untuk paginasi
        query += " LIMIT %s OFFSET %s"
        params.extend([per_page, (page - 1) * per_page])

        # Eksekusi query untuk mengambil data
        cursor.execute(query, tuple(params))
        data = cursor.fetchall()

        # Hitung total halaman
        total_pages = (total_count + per_page - 1) // per_page

        # Query untuk menghitung jumlah cluster 1 dan 2
        cursor.execute("SELECT cluster_label, COUNT(*) AS total FROM kmeans_result GROUP BY cluster_label")
        cluster_counts = cursor.fetchall()
        total_cluster_1 = next((row['total'] for row in cluster_counts if row['cluster_label'] == 0), 0)
        total_cluster_2 = next((row['total'] for row in cluster_counts if row['cluster_label'] == 1), 0)

        cursor.close()
        connection.close()

        return render_template(
            'kmeans_result.html',
            data=data,
            page=page,
            per_page=per_page,
            total_count=total_count,
            total_pages=total_pages,
            search_query=search_query,
            total_cluster_1=total_cluster_1,
            total_cluster_2=total_cluster_2
        )
    except Exception as e:
        flash(f"Terjadi kesalahan saat mengambil data: {e}", "danger")
        return redirect(url_for('kmeans_bp.set_cluster'))





# ==============================================
# ROUTE: menyimpan hasil clustering
# ==============================================

@kmeans_bp.route('/save-to-table', methods=['POST'])
def save_to_table():
    import logging
    import json
    import os
    from flask import session, flash, redirect, url_for

    logger = logging.getLogger(__name__)

    results_file = session.get('results_file', None)

    if not results_file or not os.path.exists(results_file):
        flash("Hasil clustering tidak ditemukan. Silakan ulangi proses clustering.", "danger")
        return redirect(url_for('kmeans_bp.set_cluster'))

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        cluster_summary_per_city = results.get('cluster_summary_per_city', {})
        num_clusters = results.get('num_clusters', 0)

        connection = get_db_connection()
        cursor = connection.cursor()

        for city, summary in cluster_summary_per_city.items():
            clusters = summary['clusters']
            gap_hours = summary.get('gap_hours', [])
            vendors = summary.get('vendors', [])
            stt_numbers = summary.get('stt_numbers', [])

            for i, cluster_label in enumerate(clusters):
                vendor = vendors[i] if i < len(vendors) else None
                stt_number = stt_numbers[i] if i < len(stt_numbers) else None
                gap_hour = gap_hours[i] if i < len(gap_hours) else None

                if num_clusters == 3:
                    if cluster_label == 0:
                        status = "Tepat Waktu"
                    elif cluster_label == 1:
                        status = "Sedang"
                    else:
                        status = "Telat"
                elif num_clusters == 2:
                    status = "Tepat Waktu" if cluster_label == 0 else "Telat"
                else:
                    status = "Unknown"

                if not stt_number or not city or gap_hour is None or cluster_label is None or not vendor or not status:
                    logger.error(f"Invalid data found: stt_number={stt_number}, city={city}, gap_hour={gap_hour}, cluster_label={cluster_label}, vendor={vendor}, status={status}")
                    flash("Data tidak valid. Pastikan semua kolom terisi.", "danger")
                    cursor.close()
                    connection.close()
                    return redirect(url_for('kmeans_bp.set_cluster'))

                logger.debug(f"Trying to insert: stt_number={stt_number}, city={city}, gap_hour={gap_hour}, cluster_label={cluster_label}, vendor={vendor}, status={status}")

                cursor.execute("""
                    SELECT COUNT(*) AS record_count
                    FROM kmeans_result
                    WHERE stt_number_genesis = %s AND destination_city = %s 
                    AND gap_hours = %s AND cluster_label = %s AND vendor = %s AND status = %s
                """, (stt_number, city, gap_hour, cluster_label, vendor, status))
                record_exists = cursor.fetchone()['record_count'] > 0

                logger.debug(f"Record exists: {record_exists}")

                if record_exists:
                    flash("Hasil clustering sudah disimpan sebelumnya. Anda harus menghapus data lama jika ingin menyimpan lagi.", "warning")
                    cursor.close()
                    connection.close()
                    return redirect(url_for('kmeans_bp.set_cluster'))

                cursor.execute("""
                    INSERT INTO kmeans_result (stt_number_genesis, destination_city, gap_hours, cluster_label, vendor, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (stt_number, city, gap_hour, cluster_label, vendor, status))

        connection.commit()
        cursor.close()
        connection.close()

        flash("Hasil clustering berhasil disimpan ke tabel kmeans_result!", "success")
        return redirect(url_for('kmeans_bp.kmeans_result'))

    except Exception as e:
        logger.error(f"Error saat menyimpan ke tabel: {e}", exc_info=True)
        flash(f"Terjadi kesalahan saat menyimpan ke tabel: {e}", "danger")
        return redirect(url_for('kmeans_bp.set_cluster'))



# ==============================================
# ROUTE: menghapus hasil Kmeans_result
# ==============================================


@kmeans_bp.route('/delete-kmeans-result', methods=['POST'])
def delete_kmeans_result():
    try:
        # Koneksi ke database
        connection = get_db_connection()
        cursor = connection.cursor()

        # Hapus semua data dari tabel kmeans_result
        cursor.execute("TRUNCATE TABLE kmeans_result")
        connection.commit()

        cursor.close()
        connection.close()

        flash("Data dalam tabel kmeans_result berhasil dihapus!", "success")
    except Exception as e:
        flash(f"Terjadi kesalahan saat menghapus data: {e}", "danger")
        return redirect(url_for('kmeans_bp.kmeans_result'))

    return redirect(url_for('kmeans_bp.kmeans_result'))



# ==============================================
# ROUTE: menghapus hasil kmeans di halaman Set_cluster
# ==============================================

@kmeans_bp.route('/delete-results-file', methods=['POST'])
def delete_results_file():
    try:
        results_dir = os.path.join(current_app.root_path, 'results')
        file_path = os.path.join(results_dir, 'kmeans_results.json')
        if os.path.exists(file_path):
            os.remove(file_path)
            flash("Hasil Telah direset.", "success")
        else:
            flash("File hasil tidak ditemukan.", "warning")
    except Exception as e:
        flash(f"Terjadi kesalahan saat menghapus file hasil: {e}", "danger")
        print(f"Error: {e}")
    # Redirect ke halaman `set_cluster`
    return redirect(url_for('kmeans_bp.set_cluster'))




# ==============================================
# ROUTE: halaman detail
# ==============================================

@kmeans_bp.route('/kmeans-result-detailed', methods=['GET'])
def kmeans_result_detailed_view():
    try:
        # Ambil parameter filter dari query string
        destination_filter = request.args.get('destination', '').strip()
        vendor_filter = request.args.get('vendor', '').strip()
        status_filter = request.args.get('status', '').strip()
        order_by = request.args.get('order_by', 'desc').strip().lower()  # Ambil parameter order_by
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        # Validasi input order_by
        if order_by not in ['asc', 'desc']:
            order_by = 'desc'  # Default ke descending jika tidak valid

        # Koneksi ke database
        connection = get_db_connection()
        cursor = connection.cursor()

        # Query untuk menghitung kota dengan total telat terbanyak
        cursor.execute("""
            SELECT 
                destination_city AS city,
                COUNT(*) AS total
            FROM kmeans_result
            WHERE status = 'telat'
            GROUP BY destination_city
            ORDER BY total DESC
            LIMIT 1
        """)
        top_city_data = cursor.fetchone()
        top_city = top_city_data['city'] if top_city_data else None
        top_city_total = top_city_data['total'] if top_city_data else 0

        # Query untuk menghitung vendor dengan telat terbanyak di kota tersebut
        top_vendor = None
        top_vendor_total = 0
        if top_city:
            cursor.execute("""
                SELECT 
                    vendor,
                    COUNT(*) AS total
                FROM kmeans_result
                WHERE status = 'telat' AND destination_city = %s
                GROUP BY vendor
                ORDER BY total DESC
                LIMIT 1
            """, (top_city,))
            top_vendor_data = cursor.fetchone()
            top_vendor = top_vendor_data['vendor'] if top_vendor_data else None
            top_vendor_total = top_vendor_data['total'] if top_vendor_data else 0

        # Query untuk mengambil opsi dropdown
        cursor.execute("SELECT DISTINCT destination_city FROM kmeans_result ORDER BY destination_city")
        destinations = [row['destination_city'] for row in cursor.fetchall()]

        cursor.execute("SELECT DISTINCT vendor FROM kmeans_result ORDER BY vendor")
        vendors = [row['vendor'] for row in cursor.fetchall()]

        cursor.execute("SELECT DISTINCT status FROM kmeans_result ORDER BY status")
        statuses = [row['status'] for row in cursor.fetchall()]

        # Query untuk menghitung total per kombinasi destination, vendor, dan status
        query = """
            SELECT 
                kr.destination_city AS destination,
                kr.vendor,
                kr.status,
                COUNT(*) AS total
            FROM kmeans_result kr
            WHERE 1=1
        """
        params = []

        # Filter berdasarkan dropdown
        if destination_filter:
            query += " AND kr.destination_city = %s"
            params.append(destination_filter)
        if vendor_filter:
            query += " AND kr.vendor = %s"
            params.append(vendor_filter)
        if status_filter:
            query += " AND kr.status = %s"
            params.append(status_filter)

        # Group by untuk menghitung total per kombinasi
        query += f"""
            GROUP BY kr.destination_city, kr.vendor, kr.status
            ORDER BY total {'DESC' if order_by == 'desc' else 'ASC'}
        """

        # Hitung total data untuk paginasi
        count_query = f"SELECT COUNT(*) AS total FROM ({query}) AS subquery"
        cursor.execute(count_query, tuple(params))
        total_count = cursor.fetchone()['total']

        # Tambahkan LIMIT dan OFFSET untuk paginasi
        query += " LIMIT %s OFFSET %s"
        params.extend([per_page, (page - 1) * per_page])

        # Eksekusi query untuk mengambil data
        cursor.execute(query, tuple(params))
        data = cursor.fetchall()

        # Hitung total halaman
        total_pages = (total_count + per_page - 1) // per_page

        cursor.close()
        connection.close()

        # Render template dengan data
        return render_template(
            'kmeans_result_detailed.html',
            data=data,
            destination_filter=destination_filter,
            vendor_filter=vendor_filter,
            status_filter=status_filter,
            order_by=order_by,  # Kirim order_by ke template
            page=page,
            per_page=per_page,
            total_count=total_count,
            total_pages=total_pages,
            top_city=top_city,
            top_city_total=top_city_total,
            top_vendor=top_vendor,
            top_vendor_total=top_vendor_total,
            destinations=destinations,
            vendors=vendors,
            statuses=statuses
        )
    except Exception as e:
        flash(f"Terjadi kesalahan saat mengambil data: {e}", "danger")
        return redirect(url_for('kmeans_bp.set_cluster'))



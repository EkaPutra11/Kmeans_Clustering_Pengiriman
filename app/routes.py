from flask import Blueprint, request, jsonify
import pymysql
import csv
import io
from app.kmeans_utils import calculate_gap_hours, kmeans_manual

main = Blueprint('main', __name__)

@main.route('/upload-csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File tidak ditemukan!'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File harus dalam format CSV!'}), 400

        # Baca file CSV dan hapus BOM
        stream = io.StringIO(file.stream.read().decode("UTF8").lstrip('\ufeff'), newline=None)
        reader = csv.DictReader(stream)

        if not reader.fieldnames or len(reader.fieldnames) == 0:
            return jsonify({'error': 'Header CSV tidak valid atau file kosong!'}), 400

        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='kmeans_db',
            cursorclass=pymysql.cursors.DictCursor  # Hasil query sebagai dictionary
        )
        cursor = connection.cursor()

        # Flag untuk melacak apakah semua data duplikat
        all_data_duplicate = True

        # Simpan data ke tabel `pengiriman`
        for row in reader:
            stt_number_genesis = row.get('STT Number Genesis')
            booking_date = row.get('Booking Date').split('+')[0].strip()
            origin_city = row.get('Origin City')
            destination_city = row.get('Destination City')
            vendor = row.get('Vendor')
            pod_at = row.get('POD at').split('+')[0].strip()

            # Cek apakah data sudah ada di tabel `pengiriman`
            check_query = """
                SELECT COUNT(*) AS count FROM pengiriman
                WHERE stt_number_genesis = %s
            """
            cursor.execute(check_query, (stt_number_genesis,))
            result = cursor.fetchone()

            if result['count'] > 0:
                # Data duplikat, lewati
                continue

            # Jika data baru ditemukan, ubah flag
            all_data_duplicate = False

            # Simpan data baru ke tabel `pengiriman`
            query = """
                INSERT INTO pengiriman (stt_number_genesis, booking_date, origin_city, destination_city, vendor, pod_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (stt_number_genesis, booking_date, origin_city, destination_city, vendor, pod_at))

        connection.commit()

        # Reset pembacaan CSV untuk proses berikutnya
        stream.seek(0)
        reader = csv.DictReader(stream)

        # Simpan hasil gap waktu dan clustering
        destination_cities = {}
        for row in reader:
            booking_date = row.get('Booking Date').split('+')[0].strip()
            pod_at = row.get('POD at').split('+')[0].strip()
            destination_city = row.get('Destination City')
            vendor = row.get('Vendor')  # Ambil vendor dari data CSV

            # Hitung gap dalam jam
            gap_hours = calculate_gap_hours(booking_date, pod_at)

            if destination_city not in destination_cities:
                destination_cities[destination_city] = []
            destination_cities[destination_city].append((gap_hours, vendor))  # Simpan dengan vendor

        # Daftar kota yang dilewatkan
        skipped_cities = []

        # Proses K-Means per kota
        city_comparison = {}
        for destination_city, gaps_vendors in destination_cities.items():
            if len(gaps_vendors) < 3:  # Lewatkan kota dengan kurang dari tiga data
                skipped_cities.append(destination_city)
                continue

            # Pisahkan gap_hours dan vendor
            gaps = [gv[0] for gv in gaps_vendors]
            vendors = [gv[1] for gv in gaps_vendors]

            # Gunakan centroid awal min dan max gap
            min_gap = min(gaps)
            max_gap = max(gaps)
            centroids = [min_gap, max_gap]

            # Lakukan clustering
            clusters = kmeans_manual(gaps, k=2)

            # Hitung jumlah data di setiap cluster
            cluster_0_count = sum(1 for i in range(len(clusters)) if clusters[i] == 0)
            cluster_1_count = sum(1 for i in range(len(clusters)) if clusters[i] == 1)
            total = len(gaps)
            late_percentage = (cluster_1_count / total) * 100 if total > 0 else 0

            # Simpan hasil ke tabel `city_comparison`
            city_comparison[destination_city] = (cluster_1_count, late_percentage)

            # Simpan hasil clustering dan vendor ke tabel `kmeans_results`
            for i in range(len(gaps)):
                is_late = clusters[i] == 1  # Cluster 1 dianggap telat
                query = """
                    INSERT INTO kmeans_results (destination_city, gap_hours, cluster_label, is_late, vendor)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(query, (destination_city, gaps[i], clusters[i], is_late, vendors[i]))

        # Simpan data ke tabel city_comparison
        for city, (late_count, late_percentage) in city_comparison.items():
            query = """
                INSERT INTO city_comparison (destination_city, late_count, late_percentage)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE late_count = VALUES(late_count), late_percentage = VALUES(late_percentage)
            """
            cursor.execute(query, (city, late_count, late_percentage))

        # Berikan umpan balik ke pengguna
        if skipped_cities:
            print(f"Kota yang dilewatkan karena jumlah data kurang dari 3: {', '.join(skipped_cities)}")

        # Jika semua data duplikat
        if all_data_duplicate:
            return jsonify({'message': 'Semua data yang Anda masukkan sudah pernah diunggah sebelumnya!'}), 200

        connection.commit()
        return jsonify({'message': 'File CSV berhasil diproses dan hasil disimpan!'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@main.route('/analyze-vendor-all', methods=['GET'])
def analyze_vendor_all():
    connection = None
    cursor = None
    try:
        # Koneksi ke database
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='kmeans_db',
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = connection.cursor()

        # Ambil semua kota dari tabel `kmeans_results`
        query_cities = "SELECT DISTINCT destination_city FROM kmeans_results"
        cursor.execute(query_cities)
        cities = cursor.fetchall()

        if not cities:
            return jsonify({'message': 'Tidak ada data kota ditemukan.'}), 404

        results = []

        # Proses setiap kota
        for city in cities:
            city_name = city['destination_city']

            # Hitung jumlah keterlambatan berdasarkan cluster_label = 1 untuk setiap vendor
            query_vendor_late = """
                SELECT vendor, COUNT(*) AS late_count
                FROM kmeans_results
                WHERE destination_city = %s AND cluster_label = 1
                GROUP BY vendor
                ORDER BY late_count DESC
            """
            cursor.execute(query_vendor_late, (city_name,))
            vendor_late_counts = cursor.fetchall()

            if vendor_late_counts:
                # Simpan hasil analisis ke tabel vendor_analysis
                for vendor_data in vendor_late_counts:
                    query_insert = """
                        INSERT INTO vendor_analysis (vendor, destination_city, late_count, total_count, late_percentage)
                        VALUES (%s, %s, %s, 
                                (SELECT COUNT(*) FROM kmeans_results WHERE destination_city = %s AND vendor = %s),
                                (%s / (SELECT COUNT(*) FROM kmeans_results WHERE destination_city = %s AND cluster_label = 1)) * 100)
                        ON DUPLICATE KEY UPDATE
                            late_count = VALUES(late_count),
                            total_count = VALUES(total_count),
                            late_percentage = VALUES(late_percentage)
                    """
                    cursor.execute(query_insert, (
                        vendor_data['vendor'], city_name, vendor_data['late_count'],
                        city_name, vendor_data['vendor'],
                        vendor_data['late_count'], city_name
                    ))

                # Tambahkan hasil ke dalam daftar untuk respons
                results.append({
                    'city': city_name,
                    'vendors': vendor_late_counts
                })

        connection.commit()

        # Kembalikan hasil analisis
        return jsonify(results), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Tutup cursor dan koneksi jika berhasil dibuat
        if cursor:
            cursor.close()
        if connection:
            connection.close()










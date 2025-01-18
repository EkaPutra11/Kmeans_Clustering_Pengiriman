from flask import Flask
import pymysql

def create_app():
    """
    Inisialisasi aplikasi Flask dengan koneksi MySQL dan registrasi blueprint.
    """
    app = Flask(__name__)
    app.secret_key = 'your_secret_key'  # Kunci rahasia untuk keamanan sesi

    # Konfigurasi MySQL
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = ''
    app.config['MYSQL_DB'] = 'kmeans_db'

    # Test koneksi MySQL
    try:
        connection = pymysql.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB'],
            cursorclass=pymysql.cursors.DictCursor
        )
        connection.close()  # Tutup koneksi setelah pengujian berhasil
        print("Koneksi ke database berhasil!")
    except pymysql.MySQLError as e:
        print("Gagal terhubung ke database:", e)

    # Import dan register Blueprint di sini
    from app.routes.kmeans_routes import kmeans_bp  # Pastikan path ini benar
    app.register_blueprint(kmeans_bp, url_prefix='/kmeans')

    return app

if __name__ == "__main__":
    """
    Jalankan aplikasi menggunakan Flask bawaan untuk debugging.
    """
    app = create_app()
    app.run(debug=True)  # Mode debugging aktif untuk pengembangan

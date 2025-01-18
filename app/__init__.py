from flask import Flask
import pymysql

def create_app():
    """
    Inisialisasi aplikasi Flask dengan konfigurasi database dan blueprint.
    """
    app = Flask(__name__)

    # Konfigurasi aplikasi
    app.config['SECRET_KEY'] = 'your_secret_key'  # Tambahkan kunci rahasia untuk keamanan
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = ''
    app.config['MYSQL_DB'] = 'kmeans_db'

    # Tes koneksi MySQL saat aplikasi diinisialisasi
    try:
        connection = pymysql.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB'],
            cursorclass=pymysql.cursors.DictCursor  # Mengaktifkan mode dictionary
        )
        print("Koneksi ke database berhasil!")
        connection.close()
    except pymysql.MySQLError as e:
        print(f"Gagal terhubung ke database: {e}")

    # Registrasi blueprint
    from .routes.kmeans_routes import kmeans_bp
    app.register_blueprint(kmeans_bp, url_prefix='/kmeans')

    return app

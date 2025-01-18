from flask import Flask, render_template
import os
from app.routes.kmeans_routes import kmeans_bp  # Import blueprint K-Means


def create_app():
    app = Flask(__name__)
    app.secret_key = 'your_secret_key'

    # Konfigurasi folder untuk upload
    UPLOAD_FOLDER = 'uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register blueprint K-Means
    app.register_blueprint(kmeans_bp, url_prefix='/kmeans')

    @app.route('/')
    def index():
        """
        Halaman utama aplikasi.
        """
        return render_template('home.html')

    return app


if __name__ == '__main__':
    app = create_app()

    # Debug: Cetak URL yang terdaftar
    print(app.url_map)

    app.run(debug=True)

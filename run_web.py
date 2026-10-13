import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from web.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Servidor educativo de clasificacion de simbolos musicales iniciado en http://localhost:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


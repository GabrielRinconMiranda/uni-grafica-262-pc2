import os
import sys
import json
import uuid
import time
import threading
from flask import Flask, request, jsonify, render_template

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.predict import MusicalSymbolPredictor

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)

# Inicializacion de predictor con cerrojo de inferencia
predictor = None
predictor_lock = threading.Lock()

class SessionSpace:
    """Espacio y contexto de ejecución aislado para cada sesión de usuario."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.lock = threading.RLock()
        self.stats = {
            "correctas": 0,
            "incorrectas": 0,
            "invalidas": 0,
            "total": 0,
            "tasa_acierto": 0.0,
            "consultas": 0
        }
        self.historial = []
        self.ultima_prediccion = None

    def registrar_consulta(self, prediccion_data: dict):
        with self.lock:
            self.stats["consultas"] += 1
            self.ultima_prediccion = prediccion_data
            self.historial.append({
                "tiempo": time.time(),
                "simbolo": prediccion_data.get("prediction", {}).get("name"),
                "porcentaje": prediccion_data.get("prediction", {}).get("percentage")
            })

    def registrar_feedback(self, tipo: str):
        with self.lock:
            if tipo == "correcta":
                self.stats["correctas"] += 1
            elif tipo == "incorrecta":
                self.stats["incorrectas"] += 1
            elif tipo == "invalida":
                self.stats["invalidas"] += 1
            self.stats["total"] += 1
            validos = self.stats["correctas"] + self.stats["incorrectas"]
            self.stats["tasa_acierto"] = round((self.stats["correctas"] / validos * 100), 1) if validos > 0 else 0.0
            return self.obtener_resumen()

    def reiniciar(self):
        with self.lock:
            self.stats = {
                "correctas": 0,
                "incorrectas": 0,
                "invalidas": 0,
                "total": 0,
                "tasa_acierto": 0.0,
                "consultas": 0
            }
            self.historial.clear()
            self.ultima_prediccion = None
            return self.obtener_resumen()

    def obtener_resumen(self):
        with self.lock:
            res = self.stats.copy()
            res["session_id"] = self.session_id
            return res

class SessionRegistry:
    """Gestor de instancias de espacios de sesión independientes."""
    def __init__(self):
        self._espacios = {}
        self._lock = threading.Lock()

    def obtener_o_crear(self, session_id: str) -> SessionSpace:
        if not session_id:
            session_id = f"ses_{uuid.uuid4().hex[:10]}"
        with self._lock:
            if session_id not in self._espacios:
                self._espacios[session_id] = SessionSpace(session_id)
            return self._espacios[session_id]

    def crear_nueva_sesion(self) -> SessionSpace:
        session_id = f"ses_{uuid.uuid4().hex[:10]}"
        with self._lock:
            espacio = SessionSpace(session_id)
            self._espacios[session_id] = espacio
            return espacio

# Registro global en memoria para aislamiento por cliente/pestaña
session_registry = SessionRegistry()

def get_session_id_from_request():
    # 1. Cabecera personalizada de la pestaña
    sid = request.headers.get("X-Session-ID")
    # 2. Parametro en cuerpo JSON
    if not sid and request.is_json:
        data = request.get_json(silent=True) or {}
        sid = data.get("session_id")
    # 3. Parametro en URL
    if not sid:
        sid = request.args.get("session_id")
    # 4. Asignacion si no se proporciono
    if not sid:
        sid = f"ses_{uuid.uuid4().hex[:10]}"
    return sid

def get_predictor():
    global predictor
    if predictor is None:
        with predictor_lock:
            if predictor is None:
                model_path = os.path.join(base_dir, "models", "densenet_musical.pth")
                predictor = MusicalSymbolPredictor(model_path=model_path)
    return predictor

@app.route("/")
def index():
    classes_path = os.path.join(base_dir, "dataset", "classes.json")
    with open(classes_path, "r", encoding="utf-8") as f:
        classes_data = json.load(f)["classes"]

    metrics_path = os.path.join(base_dir, "models", "test_metrics.json")
    metrics_data = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    # Cada carga de pagina genera un nuevo identificador de sesion independiente
    nueva_sesion = session_registry.crear_nueva_sesion()
    cache_id = int(time.time() * 1000)

    return render_template(
        "index.html",
        classes=classes_data,
        metrics=metrics_data,
        session_id=nueva_sesion.session_id,
        cache_id=cache_id
    )

@app.route("/classes", methods=["GET"])
def get_classes():
    classes_path = os.path.join(base_dir, "dataset", "classes.json")
    with open(classes_path, "r", encoding="utf-8") as f:
        classes_data = json.load(f)["classes"]
    return jsonify({"classes": classes_data})

@app.route("/metrics", methods=["GET"])
def get_metrics():
    metrics_path = os.path.join(base_dir, "models", "test_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No hay metricas disponibles"}), 404

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sid = get_session_id_from_request()
        espacio = session_registry.obtener_o_crear(sid)

        p = get_predictor()
        data = request.get_json(silent=True) or {}
        image_data = data.get("image")

        if not image_data and "file" in request.files:
            preprocessed = p.preprocess_pil(request.files["file"])
        elif image_data:
            preprocessed = p.preprocess_base64(image_data)
        else:
            return jsonify({"success": False, "error": "No se recibio imagen valida"}), 400

        # Inferencia segura protegida contra colisiones de memoria en CPU
        with predictor_lock:
            result = p.predict(preprocessed, top_k=4, generate_cam=False, return_layers=True)

        with espacio.lock:
            espacio.registrar_consulta(result)

        result["success"] = True
        result["prediccion_id"] = uuid.uuid4().hex
        result["session_id"] = sid
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    sid = get_session_id_from_request()
    espacio = session_registry.obtener_o_crear(sid)

    if request.method == "GET":
        return jsonify(espacio.obtener_resumen())

    data = request.get_json(silent=True) or {}
    tipo = data.get("tipo")
    if tipo not in ["correcta", "incorrecta", "invalida"]:
        return jsonify({"error": "Tipo invalido"}), 400

    stats = espacio.registrar_feedback(tipo)
    return jsonify(stats)

@app.route("/feedback/reset", methods=["POST"])
def feedback_reset():
    sid = get_session_id_from_request()
    espacio = session_registry.obtener_o_crear(sid)
    stats = espacio.reiniciar()
    return jsonify(stats)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

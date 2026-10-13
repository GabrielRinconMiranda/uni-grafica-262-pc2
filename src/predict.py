import os
import sys
import json
import base64
import io
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.model import DenseNetMusical
from dataset.build_dataset import preprocess_symbol_image

class MusicalSymbolPredictor:
    # Motor de inferencia para procesar trazos manuales o archivos de imagen,
    # evaluar la red convolucional densa y devolver distribuciones de probabilidad e imagenes intermedias.
    def __init__(self, model_path=None, config_path=None, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config_path is None:
            config_path = os.path.join(base_dir, "dataset", "classes.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.classes_info = json.load(f)["classes"]

        self.classes_by_id = {c["id"]: c for c in self.classes_info}
        num_classes = len(self.classes_info)

        if model_path is None:
            model_path = os.path.join(base_dir, "models", "densenet_musical.pth")

        # Carga del estado y configuracion estructural del checkpoint serializado
        checkpoint = torch.load(model_path, map_location=self.device)
        growth_rate = checkpoint.get("growth_rate", 16)
        block_config = checkpoint.get("block_config", (3, 3, 3))

        self.model = DenseNetMusical(
            in_channels=1,
            num_classes=num_classes,
            growth_rate=growth_rate,
            block_config=block_config
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def preprocess_image_file(self, file_path):
        return preprocess_symbol_image(file_path, target_size=(64, 64))

    def preprocess_pil(self, pil_image):
        with io.BytesIO() as bio:
            pil_image.save(bio, format="PNG")
            bio.seek(0)
            return preprocess_symbol_image(bio, target_size=(64, 64))

    def preprocess_base64(self, b64_string):
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_string)
        return preprocess_symbol_image(io.BytesIO(img_bytes), target_size=(64, 64))

    def _tensor_to_base64(self, tensor_map):
        # Convierte el promedio espacial de activaciones de un tensor a imagen PNG base64
        mean_act = tensor_map[0].mean(dim=0).cpu().numpy()
        den = mean_act.max() - mean_act.min() + 1e-8
        norm = ((mean_act - mean_act.min()) / den * 255).astype(np.uint8)
        img = Image.fromarray(norm).resize((96, 96), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def predict(self, preprocessed_64x64, top_k=4, generate_cam=False, return_layers=False):
        # Normalizacion de intensidad al dominio real continuo [0.0, 1.0]
        # y conformacion de tensor de 4 dimensiones: (batch=1, canales=1, alto=64, ancho=64)
        img_normalized = preprocessed_64x64.astype(np.float32) / 255.0
        tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            # Normalizacion softmax sobre los logits para obtener probabilidades validas sum(P)=1
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_results = []
        for idx in top_indices:
            c = self.classes_by_id[int(idx)]
            top_results.append({
                "id": int(idx),
                "name": c["name"],
                "symbol": c["symbol"],
                "category": c["category"],
                "description": c["description"],
                "probability": float(probs[idx]),
                "percentage": round(float(probs[idx]) * 100, 2)
            })

        best = top_results[0]
        result = {
            "prediction": best,
            "top_k": top_results,
            "all_probabilities": [float(p) for p in probs]
        }

        # Extraccion opcional de representaciones intermedias por capa
        if return_layers:
            acts = self.model.get_intermediate_activations(tensor)
            result["capas_intermedias"] = [
                {
                    "nombre": "Capa inicial (convolución 3x3)",
                    "dimension": f"{acts['capa_inicial'].shape[1]} canales, {acts['capa_inicial'].shape[2]}x{acts['capa_inicial'].shape[3]}",
                    "descripcion": "Filtros de primer orden: detectores de bordes, terminaciones y orientación de trazo.",
                    "imagen": f"data:image/png;base64,{self._tensor_to_base64(acts['capa_inicial'])}"
                },
                {
                    "nombre": "Transición 1 (convolución 1x1 + submuestreo)",
                    "dimension": f"{acts['transicion_1'].shape[1]} canales, {acts['transicion_1'].shape[2]}x{acts['transicion_1'].shape[3]}",
                    "descripcion": "Compresión intermedia: condensación de uniones, bucles y grosor.",
                    "imagen": f"data:image/png;base64,{self._tensor_to_base64(acts['transicion_1'])}"
                },
                {
                    "nombre": "Bloque profundo previo a GAP",
                    "dimension": f"{acts['mapas_profundos'].shape[1]} canales, {acts['mapas_profundos'].shape[2]}x{acts['mapas_profundos'].shape[3]}",
                    "descripcion": "Mapas semánticos: activación de geometrías completas previa al agrupamiento global.",
                    "imagen": f"data:image/png;base64,{self._tensor_to_base64(acts['mapas_profundos'])}"
                }
            ]

        if generate_cam:
            cam_map, _, _ = self.model.compute_cam(tensor, class_idx=best["id"])
            cam_uint8 = (cam_map * 255).astype(np.uint8)
            cam_pil = Image.fromarray(cam_uint8)
            buffered = io.BytesIO()
            cam_pil.save(buffered, format="PNG")
            result["cam_base64"] = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return result

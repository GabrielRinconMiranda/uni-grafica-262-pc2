import os
import json
import random
import argparse
import numpy as np
from PIL import Image

def preprocess_symbol_image(image_path, target_size=(64, 64), margin=6):
    im = Image.open(image_path).convert("L")
    arr = np.array(im, dtype=np.float32)
    
    # Invertir si el fondo es claro para tener trazo claro sobre fondo oscuro
    if np.mean(arr) > 127:
        arr = 255.0 - arr
        
    # Obtener caja delimitadora del trazo
    threshold = 30
    coords = np.argwhere(arr > threshold)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        arr = arr[y_min:y_max, x_min:x_max]
        
    h, w = arr.shape
    avail_w = max(1, target_size[0] - 2 * margin)
    avail_h = max(1, target_size[1] - 2 * margin)
    scale = min(avail_w / max(w, 1), avail_h / max(h, 1))
    
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    cropped_img = Image.fromarray(arr.astype(np.uint8))
    resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    canvas = np.zeros(target_size, dtype=np.uint8)
    offset_x = (target_size[0] - new_w) // 2
    offset_y = (target_size[1] - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = np.array(resized_img)
    
    return canvas

def build_dataset(raw_dirs, classes_config_path, output_dir, samples_per_class=800, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    with open(classes_config_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes_info = meta["classes"]
    
    key_to_id = {c["key"]: c["id"] for c in classes_info}
    id_to_key = {c["id"]: c["key"] for c in classes_info}
    
    files_by_key = {k: [] for k in key_to_id}   # base publica (Rebelo)
    own_by_key = {k: [] for k in key_to_id}     # dibujos propios del lienzo

    for rdir in raw_dirs:
        if not os.path.exists(rdir):
            continue
        es_propio = os.path.basename(os.path.normpath(rdir)) == "propios"
        destino = own_by_key if es_propio else files_by_key
        for root, dirs, files in os.walk(rdir):
            folder_name = os.path.basename(root).lower()
            if folder_name in destino:
                for file_name in files:
                    if file_name.endswith(".png"):
                        destino[folder_name].append(os.path.join(root, file_name))

    print("Recoleccion de imagenes por categoria:")
    for k in key_to_id:
        total = len(files_by_key[k]) + len(own_by_key[k])
        print(f"  Clase {k:15s}: {len(own_by_key[k])} propias + {len(files_by_key[k])} publicas")
        if total < 500:
            raise ValueError(f"La clase {k} tiene {total} muestras, que es menor al minimo requerido de 500.")

    images_list = []
    n_propias_total = 0
    labels_list = []
    
    for c_info in classes_info:
        k = c_info["key"]
        c_id = c_info["id"]
        own_files = own_by_key[k]
        public_files = files_by_key[k]
        random.shuffle(public_files)

        # Los dibujos propios SIEMPRE entran; la base publica completa hasta la cuota
        if samples_per_class is not None and samples_per_class > 0:
            faltan = max(0, samples_per_class - len(own_files))
            selected_files = own_files + public_files[:faltan]
        else:
            selected_files = own_files + public_files
        n_propias_total += len(own_files)

        print(f"Procesando clase {c_info['name']} (ID {c_id}): {len(selected_files)} muestras...")
        for file_path in selected_files:
            processed = preprocess_symbol_image(file_path, target_size=(64, 64))
            images_list.append(processed)
            labels_list.append(c_id)
            
    X = np.stack(images_list, axis=0)
    y = np.array(labels_list, dtype=np.int64)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    os.makedirs(output_dir, exist_ok=True)
    np_path = os.path.join(output_dir, "data_musical.npz")
    np.savez_compressed(np_path, X=X, y=y)
    
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    
    print("Dataset generado satisfactoriamente:")
    print(f"  Total de muestras: {len(X)} ({n_propias_total} dibujadas por el equipo)")
    print(f"  Forma de X: {X.shape} (uint8, valores [0, 255])")
    print(f"  Forma de y: {y.shape}")
    print(f"  Archivo comprimido: {np_path}")
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construccion del dataset musical para entrenamiento")
    parser.add_argument("--samples-per-class", type=int, default=800, help="Numero de muestras por clase (minimo 500)")
    parser.add_argument("--all", action="store_true", help="Usar todas las muestras disponibles sin recortar")
    args = parser.parse_args()
    
    samples = None if args.all else args.samples_per_class
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dirs = [
        os.path.join(base_dir, "dataset_raw", "db1"),
        os.path.join(base_dir, "dataset_raw", "db2"),
        os.path.join(base_dir, "dataset_raw", "propios")   # dibujos hechos con recolectar.py
    ]
    cfg_path = os.path.join(base_dir, "dataset", "classes.json")
    out_dir = os.path.join(base_dir, "dataset")
    
    build_dataset(raw_dirs, cfg_path, out_dir, samples_per_class=samples)


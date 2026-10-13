import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MusicalSymbolsDataset(Dataset):
    # Envoltorio PyTorch para colecciones matriciales en memoria.
    # Transforma arreglos bidimensionales (H, W) uint8 [0, 255] en tensores flotantes
    # normalizados en el dominio continuo [0.0, 1.0] con dimension explicita de canal (1, H, W).
    def __init__(self, images, labels, transform=None):
        # Escalamiento lineal al dominio unitario: x' = x / 255.0
        self.images = images.astype(np.float32) / 255.0
        # Etiquetas discretas en formato entero de 64 bits para CrossEntropyLoss
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Incorpora dimension de canal: de (64, 64) a (1, 64, 64) respetando convencion NCHW de PyTorch
        img = self.images[idx]
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_tensor

def load_classes_metadata(config_path=None):
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "dataset", "classes.json")
    with open(config_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["classes"]

def create_dataloaders(data_file_path=None, batch_size=32, test_split=0.2, val_split=0.1, seed=42):
    # Genera particiones deterministas de entrenamiento (70%), validacion (10%) y prueba (20%).
    # El generador con semilla fijada asegura reproducibilidad exacta en la separacion de lotes.
    if data_file_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_file_path = os.path.join(base_dir, "dataset", "data_musical.npz")

    data = np.load(data_file_path)
    X = data["X"]
    y = data["y"]

    total_samples = len(X)
    test_size = int(total_samples * test_split)
    val_size = int(total_samples * val_split)
    train_size = total_samples - test_size - val_size

    full_dataset = MusicalSymbolsDataset(X, y)
    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Lotes con mezcla pseudoaleatoria activa unicamente en fase de entrenamiento
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset

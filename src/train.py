import os
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.model import DenseNetMusical
from src.dataset import create_dataloaders, load_classes_metadata

def evaluate_model(model, data_loader, criterion, device):
    # Evaluacion en modo inferencia: desactiva dropout y congelamiento de estadisticas de BatchNorm
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Acumulacion de perdida ponderada por el tamano del lote actual
            running_loss += loss.item() * inputs.size(0)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_targets, np.array(all_probs)

def compute_detailed_metrics(all_targets, all_preds, all_probs, classes):
    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    top1 = float(np.mean(all_preds == all_targets))
    top2_hits = 0
    for i in range(len(all_targets)):
        top2_idx = np.argsort(all_probs[i])[::-1][:2]
        if all_targets[i] in top2_idx:
            top2_hits += 1
    top2 = float(top2_hits / len(all_targets))

    per_class = []
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class.append({
            "id": c["id"],
            "name": c["name"],
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "muestras": int(np.sum(all_targets == i))
        })

    correct_mask = (all_preds == all_targets)
    max_probs = np.max(all_probs, axis=1)
    conf_correct = float(np.mean(max_probs[correct_mask]) * 100)
    conf_incorrect = float(np.mean(max_probs[~correct_mask]) * 100) if np.any(~correct_mask) else 0.0

    return {
        "top1_accuracy": round(top1 * 100, 2),
        "top2_accuracy": round(top2 * 100, 2),
        "metricas_por_clase": per_class,
        "confianza_aciertos": round(conf_correct, 2),
        "confianza_errores": round(conf_incorrect, 2),
        "margen_certidumbre": round(conf_correct - conf_incorrect, 2),
        "matriz_confusion": cm.tolist()
    }

def train_model(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    growth_rate=16,
    block_config=(3, 3, 3),
    save_path=None,
    data_path=None,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de ejecucion: {device}")

    classes = load_classes_metadata()
    num_classes = len(classes)

    train_loader, val_loader, test_loader, full_ds = create_dataloaders(
        data_file_path=data_path,
        batch_size=batch_size
    )

    print(f"Particiones: entrenamiento={len(train_loader.dataset)}, validacion={len(val_loader.dataset)}, prueba={len(test_loader.dataset)}")

    model = DenseNetMusical(
        in_channels=1,
        num_classes=num_classes,
        growth_rate=growth_rate,
        block_config=block_config
    ).to(device)

    # Funcion de perdida de entropia cruzada multiclase:
    # L_CE = -log(exp(z_y) / sum_j exp(z_j)) combinando log-softmax y NLLLoss
    criterion = nn.CrossEntropyLoss()

    # Optimizador Adam con estimacion adaptativa de momentos de primer (beta1=0.9) y segundo orden (beta2=0.999),
    # mas penalizacion L2 (weight_decay=1e-4) para regularizacion de pesos.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Reduccion de tasa de aprendizaje ante estancamiento de exactitud en validacion (factor gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
        "epoch_times": []
    }

    best_val_acc = 0.0
    if save_path is None:
        save_path = os.path.join(base_dir, "models", "densenet_musical.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reinicio de gradientes acumulados en tensores de parametros
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Retropropagacion del error calculando derivadas parciales dL/dw por regla de la cadena
            loss.backward()
            # Actualizacion de pesos convolucionales y lineales
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(epoch_duration)

        print(
            f"Epoca [{epoch:02d}/{epochs:02d}] "
            f"Perdida Train: {train_loss:.4f} | Acc Train: {train_acc * 100:.2f}% | "
            f"Perdida Val: {val_loss:.4f} | Acc Val: {val_acc * 100:.2f}% | "
            f"Tiempo: {epoch_duration:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "growth_rate": growth_rate,
                "block_config": block_config,
                "num_classes": num_classes
            }, save_path)

    total_duration = time.time() - start_total_time
    print(f"Entrenamiento completado en {total_duration:.1f}s.")

    # Evaluacion final sobre conjunto de prueba independiente con mejor checkpoint
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_preds, test_targets, test_probs = evaluate_model(model, test_loader, criterion, device)

    # Medicion de latencia de inferencia sobre CPU
    x_single = torch.randn(1, 1, 64, 64).to(device)
    for _ in range(10): _ = model(x_single)
    lat_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(x_single)
        lat_times.append(time.perf_counter() - t0)

    detailed = compute_detailed_metrics(np.array(test_targets), np.array(test_preds), test_probs, classes)
    detailed["test_loss"] = round(test_loss, 4)
    detailed["latencia_media_ms"] = round(float(np.mean(lat_times) * 1000), 2)
    detailed["latencia_p95_ms"] = round(float(np.percentile(lat_times, 95) * 1000), 2)
    detailed["throughput_fps"] = round(float(1.0 / np.mean(lat_times)), 1)
    detailed["total_parametros"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics_path = os.path.join(os.path.dirname(save_path), "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)

    history_path = os.path.join(os.path.dirname(save_path), "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Metricas completas guardadas en {metrics_path}")
    return model, history, detailed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo DenseNet para simbolos musicales")
    parser.add_argument("--epochs", type=int, default=8, help="Numero de epocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano del lote")
    parser.add_argument("--lr", type=float, default=0.001, help="Tasa de aprendizaje inicial")
    parser.add_argument("--growth-rate", type=int, default=16, help="Tasa de crecimiento k")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        growth_rate=args.growth_rate
    )

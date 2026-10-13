import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    # Bloque basico H_l: aplica normalizacion por lotes, activacion no lineal ReLU
    # y convolucion bidimensional 3x3 sin sesgo (pues BN ya incluye termino de traslacion beta).
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        # Normalizacion de media y varianza por canal para estabilizar el gradiente
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Filtro espacial 3x3 con padding 1 para mantener invariante el alto y ancho (H, W)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Evaluacion de la transformacion compuesta H_l(x)
        out = self.conv(self.relu(self.bn(x)))
        # Conectividad densa por concatenacion de tensores a lo largo del eje de canales (dim=1)
        # Produce un tensor de salida con dimensionalidad: in_channels + growth_rate
        return torch.cat([x, out], dim=1)

class DenseBlock(nn.Module):
    # Bloque denso que encadena num_layers transformaciones compuestas.
    # Si la entrada posee c canales, tras l capas el volumen alcanza c + l * growth_rate canales.
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            layer = ConvBlock(current_channels, growth_rate)
            self.layers.append(layer)
            current_channels += growth_rate
        self.out_channels = current_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    # Capa de transicion para control de dimensionalidad entre bloques densos consecutivos.
    # Aplica convolucion 1x1 para compresion de canales y promedio espacial 2x2 para reduccion de resolucion.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Reduccion de canales (factor theta = out_channels / in_channels) mediante proyeccion puntual 1x1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Submuestreo espacial: reduce alto y ancho a la mitad mediante promedio (stride 2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))

class DenseNetMusical(nn.Module):
    # Arquitectura convolucional densa completa para clasificacion de simbolos monocromaticos.
    # Entrada: tensor 1x64x64 (imagen en escala de grises centrada).
    # Salida: vector de 8 logits correspondientes a las probabilidades de clase.
    def __init__(self, in_channels=1, num_classes=8, growth_rate=16, block_config=(3, 3, 3)):
        super().__init__()
        init_channels = 32
        self.num_classes = num_classes

        # Etapa convolucional inicial: extrae 32 mapas base preservando resolucion 64x64
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        current_channels = init_channels
        self.features = nn.ModuleList()

        # Construccion de bloques densos alternados con capas de transicion
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(current_channels, num_layers, growth_rate)
            self.features.append(block)
            current_channels = block.out_channels

            if i != len(block_config) - 1:
                # Factor de compresion theta = 0.5: reduce a la mitad el numero de canales acumulados
                out_channels = current_channels // 2
                trans = TransitionLayer(current_channels, out_channels)
                self.features.append(trans)
                current_channels = out_channels

        # Normalizacion previa a la agregacion espacial global
        self.final_bn = nn.BatchNorm2d(current_channels)
        self.final_relu = nn.ReLU(inplace=True)

        # Global Average Pooling (GAP): colapsa las dimensiones espaciales HxW a 1x1
        # eliminando la necesidad de capas densas pesadas y mitigando el sobreajuste.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Proyeccion lineal final hacia el espacio de clases
        self.classifier = nn.Linear(current_channels, num_classes)

    def extract_features(self, x):
        out = self.init_conv(x)
        for layer in self.features:
            out = layer(out)
        out = self.final_relu(self.final_bn(out))
        return out

    def forward(self, x):
        features = self.extract_features(x)
        pooled = self.gap(features)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits

    def get_intermediate_activations(self, x):
        # Extrae representaciones intermedias en tres etapas clave de la red:
        # 1. Salida de la convolucion inicial (64x64)
        # 2. Salida del primer bloque tras la primera transicion (32x32)
        # 3. Mapas profundos finales antes de Global Average Pooling (16x16)
        with torch.no_grad():
            act1 = self.init_conv(x)
            act2 = self.features[1](self.features[0](act1))
            act3 = self.extract_features(x)
            return {
                "capa_inicial": act1,
                "transicion_1": act2,
                "mapas_profundos": act3
            }

    def compute_cam(self, x, class_idx=None):
        self.eval()
        with torch.no_grad():
            features = self.extract_features(x)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)

            if class_idx is None:
                class_idx = torch.argmax(probs, dim=1).item()

            weights = self.classifier.weight[class_idx]
            cam = torch.zeros(features.shape[2:], dtype=torch.float32, device=x.device)

            for i, w in enumerate(weights):
                cam += w * features[0, i, :, :]

            cam = F.relu(cam)
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())

            cam_resized = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(x.shape[2], x.shape[3]),
                mode="bilinear",
                align_corners=False
            ).squeeze().cpu().numpy()

            return cam_resized, class_idx, probs[0].cpu().numpy()

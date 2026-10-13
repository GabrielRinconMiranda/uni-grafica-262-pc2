# Reconocimiento de figuras musicales mediante redes convolucionales densas

Este repositorio documenta el diseño, formulación matemática, recolección de datos, evaluación cuantitativa y despliegue interactivo de un sistema de visión artificial orientado a clasificar símbolos de la notación musical occidental dibujados a mano. El conjunto de entrenamiento combina una base pública de referencia con 1,200 dibujos propios recolectados por el equipo en un lienzo digital. El sistema emplea la arquitectura convolucional densa (*DenseNet*) en PyTorch y una interfaz web desacoplada en Flask.

## Fundamentación matemática y teórica de la red neuronal

El análisis óptico de grafías musicales afronta desafíos particulares derivados de la presencia de trazos delgados (plicas, barras y alteraciones), variaciones locales de grosor y contornos geométricos afines entre distintas categorías. Para abordar estas restricciones con un número reducido de parámetros entrenables, se implementa el principio de conectividad densa (*DenseNet*).

### Conectividad densa y reutilización de características

En una red convolucional secuencial clásica, la capa $l$ recibe únicamente la salida de la capa anterior $x_l = H_l(x_{l-1})$. En contraste, en una red densa cada capa se conecta directamente con todas las capas subsiguientes dentro del mismo bloque mediante la concatenación de tensores a lo largo del eje de canales:

$$x_l = H_l([x_0, x_1, \dots, x_{l-1}])$$

donde:
- $[x_0, x_1, \dots, x_{l-1}] \in \mathbb{R}^{H \times W \times c_{\text{in}}}$ denota el tensor resultante de concatenar los mapas producidos por todas las etapas precedentes.
- $H_l(\cdot)$ representa una transformación no lineal compuesta conformada por la secuencia: **Normalización por lotes (BatchNorm) $\to$ Activación no lineal (ReLU) $\to$ Convolución bidimensional ($3 \times 3$)**:

$$H_l(\mathbf{z}) = \mathbf{W}_l * \sigma(\text{BN}(\mathbf{z}))$$

con pesos $\mathbf{W}_l \in \mathbb{R}^{3 \times 3 \times c_{\text{in}} \times k}$, activación no lineal $\sigma(u) = \max(0, u)$ y relleno (*padding*) de 1 para preservar la resolución espacial $(H, W)$.
- **Tasa de crecimiento ($k=16$):** Cada función $H_l$ añade $k$ nuevos canales al estado global. Si el bloque inicia con $k_0$ canales, la capa $l$ recibe $k_0 + (l - 1)k$ canales de entrada. Esta propiedad permite que los filtros primitivos (orientación de trazos, esquinas) permanezcan accesibles en las capas más profundas sin desvanecimiento de gradiente.

### Control de dimensionalidad y capas de transición

Para regular la acumulación de canales y reducir gradualmente el tamaño espacial entre bloques contiguos, se intercalan capas de transición. Cada capa de transición aplica:
1. Convolución puntual $1 \times 1$ con factor de compresión $\theta = 0.5$, reduciendo a la mitad los canales acumulados:

$$c_{\text{out}} = \lfloor \theta \cdot c_{\text{in}} \rfloor$$

2. Submuestreo espacial de promedio (*Average Pooling*) $2 \times 2$ con paso (*stride*) de 2, reduciendo el ancho y alto del mapa a la mitad.

### Agrupamiento promedio global y clasificación probabilística

Al concluir el último bloque denso, la red dispone de una representación de 92 canales de tamaño $16 \times 16$. En lugar de aplanar la matriz hacia capas totalmente conectadas (lo que incrementaría drásticamente los parámetros y el riesgo de sobreajuste), se aplica un agrupamiento promedio global (*Global Average Pooling*, GAP):

$$\bar{z}_c = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W f_c(i, j), \quad c \in \{1, \dots, C\}$$

Este operador reduce cada canal a una magnitud escalar invariante a traslaciones espaciales locales. La proyección final hacia el espacio de 8 clases se realiza mediante una transformación afín y la función Softmax:

$$\hat{p}_c = \frac{\exp(\mathbf{w}_c^T \bar{\mathbf{z}} + b_c)}{\sum_{j=1}^8 \exp(\mathbf{w}_j^T \bar{\mathbf{z}} + b_j)}$$

La optimización de parámetros se rige por la función de pérdida de entropía cruzada multiclase:

$$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^8 y_c \log(\hat{p}_c)$$

donde $y_c \in \{0, 1\}$ indica la pertenencia real a la clase $c$.

## Metodología y procedimiento experimental

El flujo de trabajo abarca desde la selección de fuentes primarias hasta la evaluación independiente y despliegue:

```
[Dibujos propios en lienzo: 150/clase]   [Repositorio Rebelo db1/db2]
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
[Fusión: 150 propias + 650 públicas = 800 muestras/clase]
            │
            ▼
[Preprocesamiento morfológico: binarización, centrado y 64x64 px]
            │
            ▼
[Partición estratificada: 70% train | 10% val | 20% test]
            │
            ▼
[Entrenamiento DenseNetMusical (15 épocas, Adam, CE, ReduceLROnPlateau)]
            │
            ▼
[Evaluación cuantitativa de métricas, latencia y calibración]
            │
            ▼
[Despliegue interactivo con Flask, canvas y registro concurrente]
```

1. **Recolección de datos propios:** A partir del prototipo de captura del curso (`letras_uni/main.py`) se desarrolló `recolectar.py`, una aplicación Flask con lienzo de 220×220 px, fondo blanco y trazo de 12 px (idéntico al lienzo de predicción). La aplicación solicita en cada envío la categoría con menos muestras, garantizando un balance automático. El equipo dibujó **150 muestras por categoría (1,200 en total)**, variando tamaño, posición e inclinación. Las imágenes se almacenan en `dataset_raw/propios/<clase>/`.
2. **Selección del conjunto público complementario:** Se extrajeron instancias manuscritas del repositorio de referencia Rebelo (`db1` y `db2`) para las mismas 8 categorías: figuras rítmicas (negra, blanca), alteraciones (sostenido, bemol, becuadro), símbolos compuestos (barra de unión), silencios (silencio de negra) y claves (clave de sol).
3. **Fusión y balanceo uniforme:** Se fijó una cuota de 800 muestras por categoría (6,400 en total). Los dibujos propios se incluyen siempre en su totalidad (150) y la base pública completa las 650 restantes, previniendo sesgos de frecuencia a priori.
4. **Estandarización espacial y normalización:** Cada muestra se binarizó con inversión de luminancia (fondo 0, trazo 1), se centró geométricamente calculando su centro de masa con un margen perimetral del 15% y se remuestreó mediante interpolación bilineal a una rejilla fija de $64 \times 64$ puntos.
5. **Partición aleatoria reproducible (semilla 42):**
   - Conjunto de entrenamiento: 4,480 muestras (70%).
   - Conjunto de validación: 640 muestras (10%).
   - Conjunto de prueba independiente: 1,280 muestras (20%).
6. **Procedimiento de optimización:** Entrenamiento durante 15 épocas con optimizador Adam ($\eta_0 = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$), tamaño de lote de 32 y reductor de tasa de aprendizaje ante mesetas en validación.

## Estructura modular del proyecto

El código se organiza en módulos con separación explícita de responsabilidades:

- `dataset/classes.json`: Metadatos de clases, glifos Unicode y categorías musicales.
- `recolectar.py`: Aplicación de recolección de dibujos propios (adaptación del prototipo del curso) que guarda cada trazo en `dataset_raw/propios/<clase>/`.
- `dataset/build_dataset.py`: Rutina que fusiona los dibujos propios con la base pública, centra cada muestra y compila el conjunto en formato comprimido NumPy (`data_musical.npz`).
- `src/model.py`: Implementación de `ConvBlock`, `DenseBlock`, `TransitionLayer` y `DenseNetMusical`, incluyendo extracción de activaciones intermedias (`get_intermediate_activations`).
- `src/dataset.py`: Abstracción `MusicalSymbolsDataset` y generador de particiones `create_dataloaders`.
- `src/train.py`: Rutina de entrenamiento con cálculo y serialización de métricas completas en `models/test_metrics.json`.
- `src/predict.py`: Motor desacoplado de preprocesamiento e inferencia con soporte para imágenes base64 y matrices NumPy.
- `web/app.py`: Servidor HTTP en Flask con gestión aislada de hilos y sesiones por usuario (`X-Session-ID` / cookies), sincronización con cerrojos de exclusión mutua (`threading.Lock`) para inferencia en PyTorch e inicialización independiente de la barra de aciertos en 0.0% para cada nuevo cliente.
- `web/templates/index.html`: Plantilla interactiva con paneles plegables, lienzo digital, visualización de capas internas, botones de evaluación y reinicio personal.
- `web/static/css/style.css`: Estilos visuales minimalistas y adaptativos.
- `web/static/js/canvas.js`: Lógica de dibujo, gestión de sesión por cliente, peticiones asíncronas, despliegue de capas internas y actualización de contadores individuales en tiempo real.
- `clasificador_simbolos_musicales.ipynb`: Cuaderno Jupyter que ejecuta el flujo integral con explicaciones teóricas y visualizaciones.
- `run_web.py`: Script de arranque para el servidor web local.

## Evaluación cuantitativa y análisis de métricas

Para valorar el desempeño del modelo de manera rigurosa, se evalúan múltiples dimensiones cuantitativas en el conjunto de prueba independiente (1,280 muestras):

| Métrica | Dominio | Definición formal | Interpretación en el contexto |
| :--- | :---: | :--- | :--- |
| Exactitud Top-1 | $[0, 100]\%$ | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{i, 1} = y_i)$ | Porcentaje de aciertos exactos donde la clase de mayor probabilidad coincide con la etiqueta verdadera. Valor obtenido: **98.91%** (1,266/1,280). |
| Exactitud Top-2 | $[0, 100]\%$ | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \in \{\hat{y}_{i, 1}, \hat{y}_{i, 2}\})$ | Porcentaje de muestras donde la clase real figura entre las dos opciones más probables. Valor obtenido: **99.69%** (1,276/1,280). |
| Precisión | $[0, 100]\%$ | $\frac{TP}{TP + FP}$ | Grado de pureza de las predicciones positivas para cada categoría. Varía entre 96.55% (silencio de negra) y 100.0% (bemol y barra de unión). |
| Exhaustividad (Recall) | $[0, 100]\%$ | $\frac{TP}{TP + FN}$ | Proporción de instancias reales identificadas con éxito. Varía entre 95.35% (barra de unión) y 100.0% (bemol, becuadro y silencio de negra). |
| F1-Score | $[0, 100]\%$ | $2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}}$ | Media armónica balanceada entre precisión y exhaustividad. Media ponderada: **98.90%**. |
| Latencia media en CPU | $[0, \infty)$ ms | $\bar{t} = \frac{1}{M} \sum_{k=1}^M t_{\text{inferencia}, k}$ | Tiempo de propagación hacia adelante por muestra en procesador estándar. Valor obtenido: **14.57 ms** (P95: **16.01 ms**), medido en el equipo de entrenamiento. |
| Rendimiento (Throughput) | $[0, \infty)$ fps | $\text{Throughput} = \frac{1000}{\bar{t}}$ | Capacidad de procesamiento continuo en CPU. Valor obtenido: **68.6 cuadros/segundo**. |
| Margen de certidumbre | $[0, 100]\%$ | $\Delta P = \bar{P}_{\text{acierto}} - \bar{P}_{\text{error}}$ | Diferencia entre la certeza media en aciertos (98.15%) y errores (69.90%). Valor obtenido: **28.26%**, evidenciando adecuada calibración probabilística. |

### Métricas detalladas por clase

| Clase | Símbolo | Precisión (%) | Recall (%) | F1-Score (%) | Muestras de prueba |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Figura: negra | 𝅘𝅥 | 99.37 | 99.37 | 99.37 | 158 |
| Figura: blanca | 𝅗𝅥 | 98.64 | 99.32 | 98.98 | 146 |
| Armadura: sostenido | ♯ | 98.35 | 98.90 | 98.62 | 181 |
| Armadura: bemol | ♭ | 100.00 | 100.00 | 100.00 | 163 |
| Armadura: becuadro | ♮ | 98.82 | 100.00 | 99.41 | 168 |
| Compuesto: barra de unión | ♫ | 100.00 | 95.35 | 97.62 | 172 |
| Silencio: silencio de negra | 𝄽 | 96.55 | 100.00 | 98.25 | 140 |
| Clave: clave de sol | 𝄞 | 99.34 | 98.68 | 99.01 | 152 |

### Impacto de los datos propios

| Métrica | Solo base pública (8 épocas) | Base pública + dibujos propios (15 épocas) |
| :--- | :---: | :---: |
| Exactitud Top-1 | 96.17% | **98.91%** |
| Exactitud Top-2 | 98.83% | **99.69%** |
| F1 ponderado | 96.19% | **98.90%** |
| Errores en prueba | 49 / 1,280 | **14 / 1,280** |
| Precisión mínima por clase | 83.58% (becuadro) | **96.55%** (silencio de negra) |

La incorporación de trazos capturados en el mismo lienzo que se usa durante la inferencia eliminó la principal fuente de error del modelo anterior: 19 bemoles y 8 sostenidos se clasificaban como becuadro, mientras que ahora el bemol alcanza 100% de aciertos y el becuadro solo recibe 2 falsos positivos. Nota: el conjunto de prueba actual incluye dibujos propios, por lo que la comparación es indicativa.

### Representación gráfica de resultados

![Curvas de aprendizaje](docs/curvas_aprendizaje.png)

![Matriz de confusión](docs/matriz_confusion.png)

![Activaciones intermedias](docs/activaciones_intermedias.png)

## Observaciones sobre casos particulares

1. **Estabilidad del entrenamiento:** Durante las primeras 6 épocas la exactitud de validación oscila (44%–96%) por la alta tasa de aprendizaje inicial. Tras la reducción automática de la tasa (ReduceLROnPlateau, de 0.001 a 0.0005 en la época 6 y a 0.00025 en la época 12), la validación se estabiliza entre 97.5% y 99.1% desde la época 7.

2. **Comportamiento en la clase de compuestos (`Compuesto: barra de unión`):**
   - Es la clase con menor exhaustividad (95.35%): 5 de sus 172 muestras de prueba se confunden con el silencio de negra, ya que ambas presentan trazos inclinados de grosor similar.
   - En el conjunto de datos de entrenamiento (Rebelo), las muestras de barras de unión corresponden a segmentos de vigas aisladas tras la supresión de pentagramas. No contienen las figuras completas con plicas ni cabezas.
   - Si se dibujan únicamente líneas paralelas horizontales o inclinadas, el clasificador predice la clase compuesto con alta certidumbre (>95%). Al incorporar plicas verticales completas, la dominancia de trazos verticales activa preferentemente la clase `Figura: negra`.
3. **Sensibilidad y variabilidad en el silencio de negra (`Silencio: silencio de negra`):**
   - Las muestras de entrenamiento de este símbolo exhiben una acusada dispersión morfológica, irregularidades de trazo y asimetrías caligráficas.
   - La capa de agrupamiento global (GAP) promedia las activaciones espaciales, lo que ante trazos libres poco definidos puede causar dispersión probabilística. Trazos estructurados con cuerpo superior claro y cola inferior muestran estabilidad en la inferencia.

## Conclusiones

1. **Valor de los datos propios:** Complementar la base pública con 1,200 dibujos capturados en el mismo lienzo de inferencia elevó la exactitud de 96.17% a 98.91% y redujo los errores de 49 a 14, porque el modelo aprende la variabilidad real del trazo con mouse.
2. **Eficacia paramétrica de la arquitectura:** La conectividad densa permitió alcanzar una exactitud Top-1 de 98.91% y Top-2 de 99.69% con solo 80,520 parámetros entrenables.
3. **Idoneidad para interacción en tiempo real:** Una latencia promedio de 14.57 ms (68.6 fps) en CPU ordinaria es muy inferior al umbral perceptible por el usuario, lo que demuestra la viabilidad del modelo para aplicaciones web interactivas.
4. **Calibración y confiabilidad predictiva:** El margen de certidumbre de 28.26% entre aciertos y equivocaciones garantiza que las incertidumbres morfológicas no se traduzcan en predicciones erróneas con alta confianza espuria.
5. **Influencia directa del preprocesamiento:** El análisis de los casos particulares resalta la relación directa entre la curación del conjunto de entrenamiento y el comportamiento del modelo ante trazos interactivos humanos.

## Recursos y referencias bibliográficas

1. **Base de datos de símbolos musicales:**
   - Rebelo, A., Capela, G., & Cardoso, J. S. (2010). *Optical recognition of music symbols: A comparative study*. International Journal on Document Analysis and Recognition (IJDAR), 13(1), 19-31.
   - Pacha, A. (2018). *OMR-Datasets: A collection of datasets used for Optical Music Recognition*. Repositorio: https://github.com/apacha/OMR-Datasets (Licencia CC BY-SA 4.0).
2. **Arquitectura DenseNet:**
   - Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). *Densely connected convolutional networks*. En *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4700-4708.
3. **Material conceptual y de referencia:**
   - Presentaciones teóricas del curso (`teoria/`): *Principios de Arquitecturas Convolucionales: DenseNets* y *Redes Convolucionales para Clasificación de Imágenes*.
   - Cuadernos de estudio (`jupyter referencias/`): *Clasificación de Imágenes* y *Notebook Resnets, Global Average Pooling y Class Activation Maps*.
   - Prototipo base de captura sobre lienzo (`letras_uni/`).

## Instrucciones de ejecución

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. (Opcional) Recolectar más dibujos propios en `http://localhost:5001`:
   ```bash
   python recolectar.py
   ```
3. Construir el conjunto de datos y entrenar:
   ```bash
   python dataset/build_dataset.py
   python src/train.py --epochs 15
   ```
4. Ejecutar el servidor web:
   ```bash
   python run_web.py
   ```
   Abrir en el navegador: `http://localhost:5000`
5. Explorar el cuaderno de Jupyter:
   ```bash
   jupyter notebook clasificador_simbolos_musicales.ipynb
   ```

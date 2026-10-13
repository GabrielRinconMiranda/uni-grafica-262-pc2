# Guía paso a paso para despliegue público de la demostración

Este documento detalla el procedimiento para publicar la aplicación web de clasificación de figuras musicales en diferentes plataformas de alojamiento en la nube con acceso público mediante URL o dirección IP.

---

## 1. Preparación previa del proyecto

Antes de publicar en cualquier plataforma en la nube, se recomienda aislar los archivos indispensables (evitando subir los conjuntos de datos sin procesar de más de 80 MB):

1. **Inicializar o preparar el repositorio Git:**
   ```bash
   cd /home/gartur/Downloads/grafica-pc2-nuevo
   git init
   git add .
   git commit -m "Compilación educativa para despliegue de clasificador musical"
   ```

2. **Publicar en GitHub:**
   - Crear un repositorio (público o privado) en [GitHub](https://github.com/new), por ejemplo `clasificador-musical`.
   - Vincular y subir los cambios:
     ```bash
     git remote add origin https://github.com/<tu-usuario>/clasificador-musical.git
     git branch -M main
     git push -u origin main
     ```

---

## 2. Opción 1: Render (Recomendada, gratuita y sin tarjeta)

Render ofrece alojamiento web gratuito para servicios en Python con certificado SSL automático (`https://tu-app.onrender.com`).

### Pasos:
1. Iniciar sesión en [render.com](https://render.com) (se puede ingresar con la cuenta de GitHub).
2. En el panel principal, pulsar **New +** y seleccionar **Web Service**.
3. Seleccionar **Build and deploy from a Git repository** y conectar el repositorio `clasificador-musical`.
4. Configurar los parámetros del servicio:
   - **Name:** `reconocimiento-musical` (o el nombre deseado).
   - **Region:** Ohio (US East) o Frankfurt (cualquiera cercana).
   - **Branch:** `main`
   - **Root Directory:** Dejar en blanco (raíz).
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
     ```
     *(Esta instrucción instala la versión de PyTorch para CPU en menos de 30 segundos y previene que el servidor exceda los límites de memoria durante la compilación).*
   - **Start Command:**
     ```bash
     gunicorn web.app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4
     ```
   - **Instance Type:** `Free` (512 MB RAM, 0.1 CPU).
5. Pulsar **Deploy Web Service**.
6. En 2 a 3 minutos, Render mostrará el estado **Live** y proveerá una URL pública como:
   `https://reconocimiento-musical.onrender.com`

---

## 3. Opción 2: Railway (Despliegue automático en dos pasos)

Railway detecta automáticamente el archivo `Procfile` y el `Dockerfile`.

### Pasos:
1. Iniciar sesión en [railway.app](https://railway.app) con GitHub.
2. Pulsar **New Project** $\to$ **Deploy from GitHub repo**.
3. Seleccionar el repositorio `clasificador-musical`.
4. Railway iniciará el despliegue de inmediato utilizando el `Dockerfile` incluido.
5. Una vez desplegado, ir a la pestaña **Settings** del servicio en Railway y en la sección **Networking** pulsar **Generate Domain**.
6. Se generará una URL pública activa como:
   `https://reconocimiento-musical.up.railway.app`

---

## 4. Opción 3: Hugging Face Spaces (Ideal para inteligencia artificial, 16 GB RAM)

Hugging Face Spaces es gratuito, no apaga el contenedor tras inactividad y ofrece 16 GB de memoria RAM.

### Pasos:
1. Crear una cuenta en [huggingface.co](https://huggingface.co).
2. Ir a **Spaces** y pulsar **Create new Space**.
3. Configurar:
   - **Space name:** `clasificador-figuras-musicales`
   - **License:** `mit` u `openrail`
   - **Space SDK:** Seleccionar **Docker** $\to$ **Blank**.
4. Clonar el repositorio del Space en tu máquina o subir los archivos directamente mediante la interfaz web de Hugging Face:
   - Subir: `Dockerfile`, `requirements.txt`, `run_web.py`, carpetas `src/`, `web/`, `models/` y `dataset/classes.json`.
5. Hugging Face compilará el contenedor automáticamente y expondrá la aplicación de inmediato en:
   `https://huggingface.co/spaces/<tu-usuario>/clasificador-figuras-musicales`

---

## 5. Opción 4: Túnel público inmediato (Para demostración rápida sin subir código)

Si el objetivo es realizar una demostración inmediata para compañeros y profesores directamente desde tu máquina local sin registrarte en nubes externas:

### Vía Cloudflare Tunnel (Gratuito, sin registro y seguro):
1. Descargar o ejecutar `cloudflared`:
   ```bash
   # En Linux Fedora / RedHat / Arch / Ubuntu
   curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
   chmod +x cloudflared
   ```
2. Iniciar el servidor local en una terminal:
   ```bash
   python /home/gartur/Downloads/grafica-pc2-nuevo/run_web.py
   ```
3. En otra terminal, abrir el túnel público:
   ```bash
   ./cloudflared tunnel --url http://localhost:5000
   ```
4. Cloudflare imprimirá en la consola una URL pública temporal segura (`https://<palabras-aleatorias>.trycloudflare.com`) que cualquier compañero o profesor puede abrir desde su celular o computadora.

### Vía Ngrok:
1. Instalar ngrok (`sudo dnf install ngrok` o descargar binario).
2. Ejecutar:
   ```bash
   ngrok http 5000
   ```
3. Compartir el enlace `https://xxxx.ngrok-free.app` generado.

---

## 6. Opción 5: Máquina virtual en la nube (AWS EC2 / Azure VM)

Para instituciones que solicitan desplegar sobre infraestructura propia o máquinas virtuales:

### Pasos en AWS (EC2) o Azure (Virtual Machine Ubuntu 22.04 / 24.04):
1. **Lanzar instancia:**
   - Seleccionar tipo `t2.micro` o `t3.micro` (capa gratuita de AWS) o `B1s` (Azure).
   - En el grupo de seguridad (*Security Group*), abrir los puertos:
     - `22` (SSH para administración).
     - `80` (HTTP).
     - `5000` (puerto directo de la aplicación).
2. **Conectar vía SSH:**
   ```bash
   ssh -i tu-llave.pem ubuntu@<IP-PUBLICA-DE-LA-INSTANCIA>
   ```
3. **Instalar dependencias del sistema:**
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv git
   ```
4. **Clonar y configurar el proyecto:**
   ```bash
   git clone https://github.com/<tu-usuario>/clasificador-musical.git app
   cd app
   python3 -m venv venv
   source venv/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
5. **Configurar servicio en segundo plano con `systemd`:**
   Crear el archivo `/etc/systemd/system/musical.service`:
   ```ini
   [Unit]
   Description=Servicio web del clasificador musical
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/app
   Environment="PATH=/home/ubuntu/app/venv/bin"
   ExecStart=/home/ubuntu/app/venv/bin/gunicorn web.app:app --bind 0.0.0.0:5000 --workers 2 --threads 4
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
6. **Iniciar el servicio:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable musical
   sudo systemctl start musical
   ```
7. Acceso público directo en: `http://<IP-PUBLICA>:5000`

---

## 7. Verificación final tras el despliegue

Al abrir la URL pública asignada:
1. Verificar que el lienzo digital permita dibujar con mouse o pantalla táctil.
2. Comprobar que al pulsar **Clasificar**, la inferencia responda en menos de 50 ms.
3. Verificar el aislamiento de sesiones: solicitar a dos personas distintas que abran el enlace en sus navegadores y constatar que cada una dispone de un identificador de sesión propio (`#ses_...`) con contador y barra de aciertos iniciando en cero.


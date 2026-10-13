FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

WORKDIR /app

RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalacion de PyTorch version CPU para minimizar tamano de imagen y tiempos de despliegue
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY web/ ./web/
COPY dataset/classes.json ./dataset/classes.json
COPY models/ ./models/
COPY run_web.py .
COPY Procfile .

EXPOSE 5000

CMD ["sh", "-c", "gunicorn web.app:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4"]


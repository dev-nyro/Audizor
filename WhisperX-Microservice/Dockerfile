FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instalar PyTorch con CUDA 12.1 primero
RUN pip install --no-cache-dir torch==2.2.0+cu121 torchaudio==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Copiar archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar main.py
COPY main.py .

# Copiar credenciales al contenedor
COPY nyro-450117-84821c1cf7ee.json .

# Comando para ejecutar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
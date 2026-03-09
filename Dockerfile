FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Arrancamos Uvicorn (FastAPI) en el puerto 8000 en segundo plano (&)
# Y luego lanzamos el Worker de LiveKit para que el contenedor no muera
CMD uvicorn main:app --host 0.0.0.0 --port 8000 & python main.py start
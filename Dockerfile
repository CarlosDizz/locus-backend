FROM python:3.11-slim

# Mantenemos ffmpeg por si Gemini o LiveKit necesitan procesar flujos específicos
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Comando para arrancar el Agent Worker en modo producción
# 'start' es el comando nativo del CLI de LiveKit Agents
CMD ["python", "main.py", "start"]
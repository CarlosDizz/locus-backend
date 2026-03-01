FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Eliminamos los pings manuales para que Cloud Run gestione la persistencia
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]